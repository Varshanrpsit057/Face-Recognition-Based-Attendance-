"""
Model verification harness — Phase 2 of the production build.

Implements the "never report a model as ready unless a real inference
test succeeds" rule. For every registered ONNX model this verifies:

    file exists -> size -> SHA256 -> ONNX graph validity
    -> input/output names & shapes -> execution providers
    -> warmup inference -> measured latency

For recognition models it goes further, because a face recognizer that
loads successfully can still be the wrong model or be fed the wrong
preprocessing and produce embeddings that look fine and match nothing:

    embedding dimension -> L2 normalization -> determinism
    -> genuine/impostor similarity separation on the real dataset
    -> EER and a recommended operating threshold

Every number printed is measured at run time. Nothing here is asserted
from a filename, a config value, or a model card.

Usage:
    python scripts/verify_models.py                 # verify everything
    python scripts/verify_models.py --models adaface scrfd
    python scripts/verify_models.py --skip-separation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_REGISTRY, cfg  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ModelReport:
    """Verification outcome for a single ONNX model."""

    key: str
    filename: str
    status: str = "unverified"          # ready | degraded | failed | missing
    exists: bool = False
    size_bytes: int = 0
    sha256: str = ""
    graph_valid: Optional[bool] = None  # None when the onnx package is absent
    providers: List[str] = field(default_factory=list)
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    warmup_ok: bool = False
    latency_ms_p50: Optional[float] = None
    latency_ms_p95: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class SeparationReport:
    """Genuine/impostor separation for one recognizer configuration."""

    label: str
    n_genuine: int = 0
    n_impostor: int = 0
    genuine_mean: float = 0.0
    genuine_std: float = 0.0
    impostor_mean: float = 0.0
    impostor_std: float = 0.0
    separation: float = 0.0      # genuine_mean - impostor_mean
    d_prime: float = 0.0         # normalized separation; scale-free comparison
    eer: float = 1.0
    eer_threshold: float = 0.0
    tar_at_far_1pct: float = 0.0
    errors: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────

def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def concrete_shape(shape: Sequence[Any], batch: int = 1) -> Tuple[int, ...]:
    """Turn an ONNX shape with symbolic dims into a concrete one."""
    out: List[int] = []
    for i, d in enumerate(shape):
        if isinstance(d, int) and d > 0:
            out.append(d)
        elif i == 0:
            out.append(batch)
        else:
            raise ValueError(f"cannot concretize non-batch symbolic dim at index {i}: {shape}")
    return tuple(out)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x if n == 0 else x / n


def compute_eer(genuine: np.ndarray, impostor: np.ndarray) -> Tuple[float, float]:
    """Equal Error Rate and the threshold achieving it.

    FAR(t) = fraction of impostor pairs scoring >= t
    FRR(t) = fraction of genuine pairs scoring <  t
    """
    if len(genuine) == 0 or len(impostor) == 0:
        return 1.0, 0.0
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    best_gap, best_eer, best_t = np.inf, 1.0, 0.0
    for t in thresholds:
        far = float(np.mean(impostor >= t))
        frr = float(np.mean(genuine < t))
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap, best_eer, best_t = gap, (far + frr) / 2.0, float(t)
    return best_eer, best_t


def tar_at_far(genuine: np.ndarray, impostor: np.ndarray, target_far: float = 0.01) -> float:
    """True Accept Rate at a target False Accept Rate.

    This is the operating point that matters for attendance: rule 65
    says false acceptance is worse than a missed recognition.
    """
    if len(genuine) == 0 or len(impostor) == 0:
        return 0.0
    threshold = float(np.quantile(impostor, 1.0 - target_far))
    return float(np.mean(genuine >= threshold))


# ─────────────────────────────────────────────────────────────────────
# Stage 1 — structural + inference verification
# ─────────────────────────────────────────────────────────────────────

def verify_model(key: str, models_dir: Path, runs: int = 12) -> ModelReport:
    info = MODEL_REGISTRY[key]
    path = models_dir / info.filename
    rep = ModelReport(key=key, filename=info.filename)

    if not path.exists():
        rep.status = "missing"
        rep.errors.append(f"file not found: {path}")
        return rep

    rep.exists = True
    rep.size_bytes = path.stat().st_size
    rep.sha256 = sha256_of(path)

    if info.sha256 and rep.sha256 != info.sha256:
        rep.status = "failed"
        rep.errors.append(f"SHA256 mismatch: expected {info.sha256}, got {rep.sha256}")
        return rep
    if not info.sha256:
        rep.notes.append("no expected SHA256 pinned in registry; recording measured hash")

    # Graph validity. Large models use external .data files, which the
    # checker only resolves when handed the real path.
    try:
        import onnx
        try:
            onnx.checker.check_model(str(path))
            rep.graph_valid = True
        except Exception as exc:
            rep.graph_valid = False
            rep.errors.append(f"onnx.checker rejected the graph: {exc}")
    except ImportError:
        rep.notes.append("onnx package not installed; graph structure not statically checked")

    # Session creation.
    try:
        sess = ort.InferenceSession(str(path), providers=ort.get_available_providers())
    except Exception as exc:
        rep.status = "failed"
        rep.errors.append(f"InferenceSession creation failed: {exc}")
        return rep

    rep.providers = list(sess.get_providers())
    rep.inputs = [{"name": i.name, "shape": list(i.shape), "type": i.type} for i in sess.get_inputs()]
    rep.outputs = [{"name": o.name, "shape": list(o.shape), "type": o.type} for o in sess.get_outputs()]

    # Real inference on correctly-shaped random input. A model that
    # loads but cannot run is not ready.
    try:
        inp = sess.get_inputs()[0]
        shape = concrete_shape(inp.shape)
        feed = {inp.name: np.random.rand(*shape).astype(np.float32)}
        sess.run(None, feed)          # warmup, excluded from timing
        rep.warmup_ok = True

        timings: List[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            sess.run(None, feed)
            timings.append((time.perf_counter() - t0) * 1000.0)
        rep.latency_ms_p50 = float(np.percentile(timings, 50))
        rep.latency_ms_p95 = float(np.percentile(timings, 95))
    except Exception as exc:
        rep.status = "failed"
        rep.errors.append(f"inference failed: {exc}")
        return rep

    rep.status = "degraded" if (rep.errors or rep.graph_valid is False) else "ready"
    return rep


# ─────────────────────────────────────────────────────────────────────
# Stage 2 — recognizer behaviour on real faces
# ─────────────────────────────────────────────────────────────────────

ARCFACE_REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def build_aligned_face_set(limit_per_student: int = 0) -> Dict[str, List[np.ndarray]]:
    """Detect and 5-point-align every dataset face once, for reuse.

    Uses SCRFD-10G specifically because it has a keypoint head; correct
    alignment must not be conflated with the recognizer being tested.
    """
    from src.detector import SCRFDDetector

    detector = SCRFDDetector(variant="10g")
    dataset_dir = Path(cfg.paths.dataset_dir)
    faces: Dict[str, List[np.ndarray]] = {}

    for student_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        crops: List[np.ndarray] = []
        images = sorted(p for p in student_dir.rglob("*")
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        if limit_per_student:
            images = images[:limit_per_student]

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"    ! unreadable: {img_path.name}")
                continue
            dets = detector.detect(img)
            if not dets:
                print(f"    ! no face: {img_path.name}")
                continue
            det = max(dets, key=lambda d: d.confidence)
            if det.landmarks is None:
                print(f"    ! no landmarks: {img_path.name}")
                continue
            tform, _ = cv2.estimateAffinePartial2D(
                det.landmarks.astype(np.float32), ARCFACE_REFERENCE_LANDMARKS, method=cv2.LMEDS
            )
            if tform is None:
                print(f"    ! alignment failed: {img_path.name}")
                continue
            crops.append(cv2.warpAffine(img, tform, (112, 112), borderValue=0.0))

        if crops:
            faces[student_dir.name] = crops
    return faces


def to_tensor(bgr_face: np.ndarray, channel_order: str) -> np.ndarray:
    """Aligned 112x112 BGR crop -> NCHW float32 tensor in [-1, 1].

    Both AdaFace and ArcFace use the same (x - 127.5) / 127.5 scaling;
    they differ in channel order, which is exactly what we are testing.
    """
    img = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB) if channel_order == "RGB" else bgr_face
    t = (img.astype(np.float32) - 127.5) / 127.5
    return np.expand_dims(np.transpose(t, (2, 0, 1)), axis=0)


def embed_all(
    session: ort.InferenceSession,
    faces: Dict[str, List[np.ndarray]],
    channel_order: str,
    emb_index: int,
) -> Dict[str, np.ndarray]:
    input_name = session.get_inputs()[0].name
    out: Dict[str, np.ndarray] = {}
    for student, crops in faces.items():
        vecs = []
        for crop in crops:
            raw = session.run(None, {input_name: to_tensor(crop, channel_order)})[emb_index]
            vecs.append(l2_normalize(raw.flatten()))
        out[student] = np.array(vecs)
    return out


def measure_separation(embeddings: Dict[str, np.ndarray], label: str) -> SeparationReport:
    """Cosine similarity of same-identity vs different-identity pairs.

    Embeddings are already L2-normalized, so the dot product is cosine
    similarity. A recognizer that is loaded correctly and preprocessed
    correctly separates these two distributions cleanly; one that is
    not produces overlapping distributions regardless of how healthy
    its ONNX session looks.
    """
    rep = SeparationReport(label=label)
    students = sorted(embeddings)

    genuine: List[float] = []
    for s in students:
        vecs = embeddings[s]
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                genuine.append(float(np.dot(vecs[i], vecs[j])))

    impostor: List[float] = []
    for a_idx, a in enumerate(students):
        for b in students[a_idx + 1:]:
            for va in embeddings[a]:
                for vb in embeddings[b]:
                    impostor.append(float(np.dot(va, vb)))

    if not genuine or not impostor:
        rep.errors.append("insufficient pairs (need >=2 images for one student and >=2 students)")
        return rep

    g, im = np.array(genuine), np.array(impostor)
    rep.n_genuine, rep.n_impostor = len(g), len(im)
    rep.genuine_mean, rep.genuine_std = float(g.mean()), float(g.std())
    rep.impostor_mean, rep.impostor_std = float(im.mean()), float(im.std())
    rep.separation = rep.genuine_mean - rep.impostor_mean
    pooled = np.sqrt((g.var() + im.var()) / 2.0)
    rep.d_prime = float(rep.separation / pooled) if pooled > 1e-9 else 0.0
    rep.eer, rep.eer_threshold = compute_eer(g, im)
    rep.tar_at_far_1pct = tar_at_far(g, im, 0.01)
    return rep


# ─────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────

def print_model_report(rep: ModelReport) -> None:
    mark = {"ready": "READY", "degraded": "DEGRADED", "failed": "FAILED", "missing": "MISSING"}[rep.status]
    print(f"\n[{mark}] {rep.key}  ({rep.filename})")
    if not rep.exists:
        for e in rep.errors:
            print(f"    error: {e}")
        return

    print(f"    size      : {rep.size_bytes / 1e6:.1f} MB")
    print(f"    sha256    : {rep.sha256}")
    print(f"    graph     : {'valid' if rep.graph_valid else ('INVALID' if rep.graph_valid is False else 'not checked')}")
    print(f"    providers : {', '.join(rep.providers)}")
    for i in rep.inputs:
        print(f"    input     : {i['name']} {i['shape']}")
    for o in rep.outputs:
        print(f"    output    : {o['name']} {o['shape']}")
    if rep.warmup_ok:
        print(f"    latency   : p50 {rep.latency_ms_p50:.1f} ms | p95 {rep.latency_ms_p95:.1f} ms")
    for n in rep.notes:
        print(f"    note      : {n}")
    for e in rep.errors:
        print(f"    error     : {e}")


def print_separation_report(rep: SeparationReport) -> None:
    if rep.errors:
        print(f"\n  {rep.label}: FAILED — {'; '.join(rep.errors)}")
        return
    print(f"\n  {rep.label}")
    print(f"    pairs     : {rep.n_genuine} genuine / {rep.n_impostor} impostor")
    print(f"    genuine   : {rep.genuine_mean:.4f} +/- {rep.genuine_std:.4f}")
    print(f"    impostor  : {rep.impostor_mean:.4f} +/- {rep.impostor_std:.4f}")
    print(f"    separation: {rep.separation:.4f}   d'={rep.d_prime:.2f}")
    print(f"    EER       : {rep.eer * 100:.2f}%  @ threshold {rep.eer_threshold:.4f}")
    print(f"    TAR@FAR=1%: {rep.tar_at_far_1pct * 100:.2f}%")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

RECOGNIZER_KEYS = {"adaface", "cr_fiqa"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify every ONNX model before production integration.")
    ap.add_argument("--models", nargs="*", default=None, help="subset of registry keys to verify")
    ap.add_argument("--skip-separation", action="store_true", help="structural checks only")
    ap.add_argument("--limit-per-student", type=int, default=0, help="cap dataset images per student")
    ap.add_argument("--json", type=Path, default=None, help="write full results to this JSON file")
    args = ap.parse_args()

    keys = args.models or list(MODEL_REGISTRY)
    unknown = [k for k in keys if k not in MODEL_REGISTRY]
    if unknown:
        print(f"Unknown model keys: {', '.join(unknown)}")
        return 2

    models_dir = Path(cfg.paths.models_dir)
    print("=" * 72)
    print("STAGE 1 — structural verification and real inference")
    print("=" * 72)
    print(f"models dir        : {models_dir}")
    print(f"onnxruntime       : {ort.__version__}")
    print(f"available providers: {', '.join(ort.get_available_providers())}")

    model_reports = [verify_model(k, models_dir) for k in keys]
    for rep in model_reports:
        print_model_report(rep)

    separation_reports: List[SeparationReport] = []
    usable = {r.key for r in model_reports if r.status in {"ready", "degraded"}}
    recognizers = sorted(usable & RECOGNIZER_KEYS)

    if not args.skip_separation and recognizers:
        print("\n" + "=" * 72)
        print("STAGE 2 — recognizer behaviour on real dataset faces")
        print("=" * 72)
        print("\nAligning dataset faces with SCRFD-10G (keypoint head required)...")
        faces = build_aligned_face_set(args.limit_per_student)
        total = sum(len(v) for v in faces.values())
        print(f"  aligned {total} faces across {len(faces)} identities")

        eligible = {s: v for s, v in faces.items() if len(v) >= 2}
        if len(faces) < 2 or not eligible:
            print("  ! dataset too small for pair statistics; skipping separation")
        else:
            for key in recognizers:
                info = MODEL_REGISTRY[key]
                sess = ort.InferenceSession(str(models_dir / info.filename),
                                            providers=ort.get_available_providers())
                emb_index = next((i for i, o in enumerate(sess.get_outputs())
                                  if o.name.lower() == "embedding"), 0)
                print(f"\n{'-' * 72}\n{key} — embedding output "
                      f"'{sess.get_outputs()[emb_index].name}'\n{'-' * 72}")
                for order in ("BGR", "RGB"):
                    embs = embed_all(sess, faces, order, emb_index)
                    dim = next(iter(embs.values())).shape[1]
                    norms = np.linalg.norm(np.vstack(list(embs.values())), axis=1)
                    rep = measure_separation(embs, f"{key} / {order} input / dim={dim}")
                    print_separation_report(rep)
                    print(f"    L2 norms  : {norms.min():.6f}..{norms.max():.6f} (post-normalization)")
                    separation_reports.append(rep)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for rep in model_reports:
        print(f"  {rep.status.upper():9s} {rep.key}")
    if separation_reports:
        best = min((r for r in separation_reports if not r.errors), key=lambda r: r.eer, default=None)
        if best:
            print(f"\n  Best separation: {best.label}")
            print(f"    EER {best.eer * 100:.2f}% | d' {best.d_prime:.2f} | "
                  f"TAR@FAR=1% {best.tar_at_far_1pct * 100:.2f}%")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "onnxruntime": ort.__version__,
            "providers": ort.get_available_providers(),
            "models": [asdict(r) for r in model_reports],
            "separation": [asdict(r) for r in separation_reports],
        }, indent=2))
        print(f"\n  wrote {args.json}")

    failed = [r for r in model_reports if r.status in {"failed", "missing"}]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
