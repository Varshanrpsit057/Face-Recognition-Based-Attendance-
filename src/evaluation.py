"""
Evaluation Engine — Full benchmark pipeline.

Splits dataset 80/20 into gallery/probe, enrolls gallery, evaluates probes,
computes metrics, generates visualizations, and produces reports.
"""

from __future__ import annotations

import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

from config import cfg
from src.logger import get_logger
from src.utils import get_student_dirs, get_image_files, load_image
from src.detector import FaceDetectorFactory, FaceDetectorInterface
from src.preprocessing import FacePreprocessor
from src.quality_gate import QualityGate
from src.recognizer import RecognizerFactory, RecognizerInterface
from src.vector_db import VectorDatabase
from src.profiler import get_profiler
from src.metrics import MetricsCalculator
from src.visualization import Visualizer

logger = get_logger(__name__, "inference")
profiler = get_profiler()


class EvaluationEngine:
    """End-to-end evaluation and benchmarking engine."""

    def __init__(
        self,
        detector: Optional[FaceDetectorInterface] = None,
        preprocessor: Optional[FacePreprocessor] = None,
        quality_gate: Optional[QualityGate] = None,
        recognizer: Optional[RecognizerInterface] = None,
        vector_db: Optional[VectorDatabase] = None,
    ) -> None:
        self.detector = detector or FaceDetectorFactory.create_with_fallback()
        self.preprocessor = preprocessor or FacePreprocessor()
        self.quality_gate = quality_gate or QualityGate()
        self.recognizer = recognizer or RecognizerFactory.create()
        self.vector_db = vector_db or VectorDatabase()
        self.last_results: Optional[Dict[str, Any]] = None

    # ── Dataset Splitting ──────────────────────────────────────────

    def _split_dataset(self, dataset_dir: Path) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
        """Split each student's images into gallery (80%) and probe (20%) sets."""
        random.seed(cfg.evaluation.random_seed)
        gallery_dict: Dict[str, List[Path]] = {}
        probe_dict: Dict[str, List[Path]] = {}

        for student_id, s_dir in get_student_dirs(dataset_dir):
            images = get_image_files(s_dir)
            if not images:
                continue
            random.shuffle(images)
            split_idx = max(1, int(len(images) * cfg.evaluation.gallery_ratio))
            gallery_dict[student_id] = images[:split_idx]
            probe_dict[student_id] = images[split_idx:]

        return gallery_dict, probe_dict

    # ── Gallery Enrollment ─────────────────────────────────────────

    def _enroll_gallery(
        self, gallery_dict: Dict[str, List[Path]]
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract embeddings for gallery images and build FAISS index."""
        self.vector_db.clear()
        all_emb: List[np.ndarray] = []
        all_labels: List[str] = []

        for student_id, image_paths in gallery_dict.items():
            student_embs: List[np.ndarray] = []
            for img_path in image_paths:
                try:
                    img = load_image(img_path)
                    if img is None:
                        continue

                    profiler.start("eval_detect")
                    detections = self.detector.detect(img)
                    profiler.stop("eval_detect")

                    if not detections:
                        continue
                    det = detections[0]

                    face_img = img[
                        max(0, int(det.bbox[1])): int(det.bbox[3]),
                        max(0, int(det.bbox[0])): int(det.bbox[2]),
                    ]
                    if face_img.size == 0:
                        continue

                    qr = self.quality_gate.assess(face_img, det.bbox, det.landmarks)
                    if not qr.passed:
                        continue

                    profiler.start("eval_preprocess")
                    tensor = self.preprocessor.preprocess(img, det.bbox, det.landmarks)
                    profiler.stop("eval_preprocess")

                    profiler.start("eval_extract")
                    emb = self.recognizer.extract(tensor[0])
                    profiler.stop("eval_extract")

                    if emb is not None:
                        student_embs.append(emb)
                except Exception as exc:
                    logger.warning(f"Gallery enroll error for {img_path}: {exc}")

            if student_embs:
                mean_emb = np.mean(student_embs, axis=0)
                from src.utils import l2_normalize
                mean_emb = l2_normalize(mean_emb)
                all_emb.append(mean_emb)
                all_labels.append(student_id)

        if all_emb:
            emb_array = np.array(all_emb, dtype=np.float32)
            self.vector_db.build_index(emb_array, all_labels)
            return emb_array, all_labels
        return np.array([]), []

    # ── Probe Evaluation ───────────────────────────────────────────

    def _evaluate_probes(
        self,
        probe_dict: Dict[str, List[Path]],
        gallery_embeddings: np.ndarray,
        gallery_labels: List[str],
    ) -> Tuple[List[str], List[str], List[float], List[np.ndarray]]:
        """Evaluate probe images against the gallery. Returns (y_true, y_pred, y_scores, probe_embeddings)."""
        y_true: List[str] = []
        y_pred: List[str] = []
        y_scores: List[float] = []
        probe_embs: List[np.ndarray] = []

        for student_id, image_paths in probe_dict.items():
            for img_path in image_paths:
                try:
                    img = load_image(img_path)
                    if img is None:
                        continue

                    profiler.start("eval_probe_detect")
                    detections = self.detector.detect(img)
                    profiler.stop("eval_probe_detect")

                    if not detections:
                        continue
                    det = detections[0]

                    profiler.start("eval_probe_preprocess")
                    tensor = self.preprocessor.preprocess(img, det.bbox, det.landmarks)
                    profiler.stop("eval_probe_preprocess")

                    profiler.start("eval_probe_extract")
                    emb = self.recognizer.extract(tensor[0])
                    profiler.stop("eval_probe_extract")

                    if emb is None:
                        continue

                    probe_embs.append(emb)

                    profiler.start("eval_probe_search")
                    results = self.vector_db.search(emb, top_k=1)
                    profiler.stop("eval_probe_search")

                    if results:
                        pred_id, score = results[0]
                        y_true.append(student_id)
                        y_pred.append(pred_id)
                        y_scores.append(float(score))
                    else:
                        y_true.append(student_id)
                        y_pred.append("unknown")
                        y_scores.append(0.0)
                except Exception as exc:
                    logger.warning(f"Probe evaluation error for {img_path}: {exc}")

        return y_true, y_pred, y_scores, probe_embs

    # ── Results Generation ─────────────────────────────────────────

    def _generate_results(
        self,
        y_true: List[str],
        y_pred: List[str],
        y_scores: List[float],
        gallery_emb: np.ndarray,
        gallery_labels: List[str],
        probe_embs: List[np.ndarray],
    ) -> Dict[str, Any]:
        """Compute all metrics and return a flat results dict."""
        labels = sorted(set(y_true))

        # Core metrics
        results: Dict[str, Any] = {
            "accuracy": float(MetricsCalculator.accuracy(y_true, y_pred)),
            "precision": float(MetricsCalculator.precision(y_true, y_pred)),
            "recall": float(MetricsCalculator.recall(y_true, y_pred)),
            "f1": float(MetricsCalculator.f1_score(y_true, y_pred)),
            "balanced_accuracy": float(MetricsCalculator.balanced_accuracy(y_true, y_pred)),
            "mcc": float(MetricsCalculator.matthews_corrcoef(y_true, y_pred)),
            "total_gallery": len(gallery_labels),
            "total_probes": len(y_true),
            "labels": labels,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_scores": y_scores,
        }

        # Confusion matrix
        try:
            cm = MetricsCalculator.confusion_matrix(y_true, y_pred, labels)
            results["confusion_matrix"] = cm.tolist()
        except Exception:
            results["confusion_matrix"] = []

        # Per-class metrics
        try:
            results["per_class"] = MetricsCalculator.per_class_metrics(y_true, y_pred, labels)
        except Exception:
            results["per_class"] = {}

        # CMC curve
        try:
            if len(probe_embs) > 0 and len(gallery_emb) > 0:
                probe_labels = y_true[:len(probe_embs)]
                cmc = MetricsCalculator.cmc_curve(
                    gallery_emb, gallery_labels,
                    probe_embs, probe_labels,
                    ranks=[1, 5, 10],
                )
                results["cmc"] = {str(k): v for k, v in cmc.items()}
        except Exception:
            results["cmc"] = {}

        # Profiler stats
        results["profiler"] = profiler.to_dict()

        return results

    # ── Public API ─────────────────────────────────────────────────

    def run_full_benchmark(
        self, dataset_dir: Optional[Path] = None, progress_callback=None
    ) -> Dict[str, Any]:
        """Run the complete evaluation benchmark."""
        dataset_dir = dataset_dir or Path(cfg.paths.dataset_dir)
        logger.info(f"Starting full benchmark on {dataset_dir}")

        if progress_callback:
            progress_callback(0.1, "Splitting dataset…")
        gallery_dict, probe_dict = self._split_dataset(dataset_dir)

        if progress_callback:
            progress_callback(0.2, "Enrolling gallery…")
        gallery_emb, gallery_labels = self._enroll_gallery(gallery_dict)

        if not gallery_labels:
            logger.error("No gallery embeddings could be extracted.")
            return {"error": "No gallery embeddings extracted"}

        if progress_callback:
            progress_callback(0.5, "Evaluating probes…")
        y_true, y_pred, y_scores, probe_embs = self._evaluate_probes(
            probe_dict, gallery_emb, gallery_labels
        )

        if not y_true:
            logger.error("No probe results obtained.")
            return {"error": "No probe results obtained"}

        if progress_callback:
            progress_callback(0.8, "Computing metrics…")
        results = self._generate_results(
            y_true, y_pred, y_scores, gallery_emb, gallery_labels, probe_embs
        )

        # Generate visualizations
        try:
            if progress_callback:
                progress_callback(0.9, "Generating charts…")
            viz = Visualizer()
            labels = results.get("labels", [])
            cm = results.get("confusion_matrix")
            if cm:
                viz.plot_confusion_matrix(np.array(cm), labels)
        except Exception as exc:
            logger.warning(f"Chart generation error: {exc}")

        self.last_results = results

        if progress_callback:
            progress_callback(1.0, "Benchmark complete!")
        logger.info("Benchmark complete.")
        return results

    def get_last_results(self) -> Optional[Dict[str, Any]]:
        """Return results from the most recent benchmark."""
        return self.last_results

    def export_results(self, results: Dict[str, Any], fmt: str = "json") -> Path:
        """Export results to a file."""
        path = Path(cfg.paths.reports_dir) / f"eval_results.{fmt}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "json":
            # Convert numpy types to native Python
            def _convert(obj: Any) -> Any:
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            clean = json.loads(json.dumps(results, default=_convert))
            with open(path, "w", encoding="utf-8") as f:
                json.dump(clean, f, indent=2)
        return path
