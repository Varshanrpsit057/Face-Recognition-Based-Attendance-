"""
Facial Recognition Attendance System - Streamlit Dashboard
11-page interactive web application for enrollment, recognition,
evaluation, and attendance management.
"""

from __future__ import annotations

import sys
import time
import cv2
import numpy as np
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import cfg, MODEL_REGISTRY
from src.logger import get_logger
from src.device_manager import DeviceManager
from src.quality_gate import QualityGate
from src.preprocessing import FacePreprocessor
from src.vector_db import VectorDatabase
from src.enrollment import EnrollmentEngine
from src.attendance import AttendanceEngine
from src.profiler import get_profiler
from src.detector import FaceDetectorFactory, FaceDetectorInterface
from src.tracker import TrackerFactory
from src.recognizer import RecognizerFactory
from src.evaluation import EvaluationEngine
from src.dataset_validator import DatasetValidator
from src.utils import get_student_dirs, get_image_files
from src.quality_gate import get_ml_quality_scorer
from src.liveness import get_liveness_detector
from src.camera import CameraStream, make_preview

logger = get_logger("app", "system")

# ── Page Config ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Smart Attendance",
    page_icon="face",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme CSS ──────────────────────────────────────────────────────

THEME_CSS = """
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        margin-bottom: 10px;
    }
    .metric-card-title {
        color: #8892b0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card-value {
        color: #64ffda;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 8px;
    }
    .stSidebar {
        background: rgba(15, 12, 41, 0.95) !important;
    }
    [data-testid="stSidebar"] h1 {
        color: #64ffda !important;
    }
</style>
"""


def _metric_card(title: str, value: Any) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-card-title">{title}</div>
        <div class="metric-card-value">{value}</div>
    </div>"""


# ── Component Initialization ──────────────────────────────────────

def _safe_init_components(force: bool = False) -> Dict[str, Any]:
    """Initialise all components and cache in session state."""
    if not force and "components" in st.session_state:
        return st.session_state["components"]

    if force:
        try:
            import importlib
            import src.detector
            import src.tracker
            import src.recognizer
            import src.quality_gate
            import src.preprocessing
            import src.vector_db
            import src.enrollment
            import src.evaluation
            importlib.reload(src.detector)
            importlib.reload(src.tracker)
            importlib.reload(src.recognizer)
            importlib.reload(src.quality_gate)
            importlib.reload(src.preprocessing)
            importlib.reload(src.vector_db)
            importlib.reload(src.enrollment)
            importlib.reload(src.evaluation)
        except Exception:
            pass

    comps: Dict[str, Any] = {}
    init_errors: List[str] = []

    # Each component initializes independently
    try:
        comps["detector"] = FaceDetectorFactory.create_with_fallback()
    except Exception as e:
        init_errors.append(f"Detector: {e}")
        comps["detector"] = None

    try:
        comps["tracker"] = TrackerFactory.create()
    except Exception as e:
        init_errors.append(f"Tracker: {e}")
        comps["tracker"] = None

    try:
        comps["recognizer"] = RecognizerFactory.create()
    except Exception as e:
        init_errors.append(f"Recognizer: {e}")
        comps["recognizer"] = None

    comps["quality_gate"] = QualityGate()
    comps["preprocessor"] = FacePreprocessor()
    comps["attendance_engine"] = AttendanceEngine()
    comps["profiler"] = get_profiler()
    comps["ml_quality_scorer"] = get_ml_quality_scorer()
    comps["liveness_detector"] = get_liveness_detector()

    try:
        comps["vector_db"] = VectorDatabase()
        db_path = Path(cfg.paths.faiss_index_path)
        if db_path.with_suffix(".pkl").exists():
            comps["vector_db"].load(db_path)
    except Exception as e:
        init_errors.append(f"VectorDB: {e}")
        comps["vector_db"] = None

    # Composite components
    try:
        comps["enrollment_engine"] = EnrollmentEngine(
            detector=comps.get("detector"),
            preprocessor=comps.get("preprocessor"),
            quality_gate=comps.get("quality_gate"),
            recognizer=comps.get("recognizer"),
            vector_db=comps.get("vector_db"),
        )
    except Exception as e:
        init_errors.append(f"EnrollmentEngine: {e}")
        comps["enrollment_engine"] = None

    try:
        comps["evaluation_engine"] = EvaluationEngine(
            detector=comps.get("detector"),
            preprocessor=comps.get("preprocessor"),
            quality_gate=comps.get("quality_gate"),
            recognizer=comps.get("recognizer"),
            vector_db=comps.get("vector_db"),
        )
    except Exception as e:
        init_errors.append(f"EvaluationEngine: {e}")
        comps["evaluation_engine"] = None

    if init_errors:
        comps["_init_errors"] = init_errors

    st.session_state["components"] = comps
    return comps


# =====================================================================
#  PAGES
# =====================================================================

def page_dashboard() -> None:
    st.title("System Dashboard")

    comps = _safe_init_components()

    # Show initialization warnings
    init_errors = comps.get("_init_errors", [])
    if init_errors:
        with st.expander("Some components failed to load (click to expand)", expanded=False):
            for err in init_errors:
                st.warning(err)
            st.info("Run model download first: `python run.py` to download ONNX models.")

    vdb = comps.get("vector_db")
    att = comps.get("attendance_engine")
    prof = comps.get("profiler")

    enrolled = vdb.size() if vdb else 0
    unique = len(vdb.labels_list()) if vdb and enrolled else 0
    today = len(att.get_today_records()) if att else 0
    fps = prof.get_fps() if prof else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_metric_card("Students Enrolled", unique), unsafe_allow_html=True)
    c2.markdown(_metric_card("Total Embeddings", enrolled), unsafe_allow_html=True)
    c3.markdown(_metric_card("Avg FPS", f"{fps:.1f}"), unsafe_allow_html=True)
    c4.markdown(_metric_card("Today's Attendance", today), unsafe_allow_html=True)

    st.subheader("System Information")
    try:
        dev = DeviceManager().get_device_info()
        providers = DeviceManager().get_best_providers()
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**OS:** {dev.os_name} {dev.os_version}")
            st.write(f"**CPU:** {dev.cpu_name} ({dev.cpu_cores} cores, {dev.cpu_threads} threads)")
            st.write(f"**RAM:** {dev.total_ram_gb:.1f} GB (available: {dev.available_ram_gb:.1f} GB)")
        with col2:
            gpu_text = dev.gpu_name if dev.gpu_available else "None (CPU-only)"
            st.write(f"**GPU:** {gpu_text}")
            st.write(f"**Execution Provider:** {providers[0] if providers else 'Unknown'}")
            st.write(f"**Python:** {dev.python_version}")
    except Exception as exc:
        st.warning(f"Device info unavailable: {exc}")

    st.subheader("Loaded Models")
    try:
        from src.model_manager import get_model_manager
        mm = get_model_manager()
        loaded = mm.get_loaded_models()
        if loaded:
            for m in loaded:
                lt = mm.get_load_time(m)
                st.write(f"- **{m}** (loaded in {lt:.2f}s)" if lt else f"- **{m}**")
        else:
            st.info("No models loaded yet.")
    except Exception:
        st.info("Model manager not available.")


def page_dataset_manager() -> None:
    st.title("Dataset Manager")

    dataset_dir = Path(cfg.paths.dataset_dir)
    if not dataset_dir.exists():
        st.warning(f"Dataset directory not found: `{dataset_dir}`")
        return

    student_dirs = get_student_dirs(dataset_dir)
    st.write(f"Found **{len(student_dirs)}** students in `{dataset_dir}`")

    if student_dirs:
        names = [sid for sid, _ in student_dirs]
        selected = st.selectbox("Select Student", names)
        if selected:
            sdir = dataset_dir / selected
            images = get_image_files(sdir)
            st.write(f"**{len(images)}** images for `{selected}`")

            if images:
                cols = st.columns(4)
                for idx, img_path in enumerate(images[:12]):
                    with cols[idx % 4]:
                        st.image(str(img_path), use_container_width=True)
            else:
                st.info("No images found for this student.")

    st.divider()
    st.subheader("Dataset Validation")
    if st.button("Run Validation", type="primary"):
        with st.spinner("Validating dataset..."):
            try:
                validator = DatasetValidator()
                results = validator.validate_dataset(dataset_dir)
                score = validator.compute_dataset_health_score(results)
                recs = validator.get_recommendations(results)

                st.success(f"Health Score: **{score:.1f} / 100**")
                for rec in recs:
                    st.write(rec)
                with st.expander("Full Report"):
                    st.json(results)
            except Exception as exc:
                st.error(f"Validation failed: {exc}")


def page_enrollment() -> None:
    st.title("Enrollment")
    comps = _safe_init_components()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Enroll All Students")
        st.write("Processes images in `dataset/`, detects faces, extracts embeddings, and builds the vector database.")
        engine = comps.get("enrollment_engine")
        if engine is None:
            st.error("Enrollment engine not available. Models may not be downloaded yet. Run `python run.py` first.")
        elif st.button("Start Enrollment", type="primary"):
            comps = _safe_init_components(force=True)
            engine = comps.get("enrollment_engine")
            with st.spinner("Enrolling... this may take a while."):
                try:
                    stats = engine.enroll_all()
                    st.success(
                        f"Enrolled: {stats.get('successful_enrollments', 0)} students "
                        f"| Failed: {stats.get('failed_enrollments', 0)}"
                    )
                    with st.expander("Details"):
                        st.json(stats)
                except Exception as exc:
                    st.error(f"Enrollment error: {exc}")

    with col2:
        st.subheader("Database Info")
        vdb = comps.get("vector_db")
        if vdb:
            st.metric("Total Embeddings", vdb.size())
            st.metric("Unique Students", len(vdb.labels_list()))
        else:
            st.info("Vector database not initialised.")


def page_evaluation() -> None:
    st.title("Evaluation Benchmark")
    comps = _safe_init_components()

    st.write("Split dataset into gallery (80%) and probe (20%), enroll gallery, evaluate probes, compute metrics.")

    engine = comps.get("evaluation_engine")
    if engine is None:
        st.error("Evaluation engine not available. Models may not be downloaded. Run `python run.py` first.")
        return

    if st.button("Run Full Benchmark", type="primary"):
        progress = st.progress(0.0)
        status = st.empty()

        def _cb(frac: float, msg: str):
            progress.progress(frac)
            status.text(msg)

        with st.spinner("Benchmarking..."):
            try:
                results = engine.run_full_benchmark(progress_callback=_cb)

                if "error" in results:
                    st.error(results["error"])
                    return

                st.success("Benchmark complete!")
                mc = st.columns(4)
                mc[0].metric("Accuracy", f"{results.get('accuracy', 0):.4f}")
                mc[1].metric("Precision", f"{results.get('precision', 0):.4f}")
                mc[2].metric("Recall", f"{results.get('recall', 0):.4f}")
                mc[3].metric("F1 Score", f"{results.get('f1', 0):.4f}")

                mc2 = st.columns(3)
                mc2[0].metric("Balanced Accuracy", f"{results.get('balanced_accuracy', 0):.4f}")
                mc2[1].metric("MCC", f"{results.get('mcc', 0):.4f}")
                mc2[2].metric("Probes Evaluated", results.get("total_probes", 0))

                # Show charts
                charts_dir = Path(cfg.paths.outputs_dir)
                charts = list(charts_dir.glob("*.png"))
                if charts:
                    st.subheader("Charts")
                    for chart in charts:
                        st.image(str(chart), caption=chart.stem, width=600)

                with st.expander("Full Results JSON"):
                    safe = {k: v for k, v in results.items() if k not in ("y_true", "y_pred", "y_scores", "confusion_matrix")}
                    st.json(safe)

            except Exception as exc:
                st.error(f"Benchmark failed: {exc}")


def page_realtime_camera() -> None:
    st.title("Real-time Recognition")
    comps = _safe_init_components()

    col_main, col_side = st.columns([3, 1])

    with col_side:
        st.subheader("Controls")
        cam_source = st.text_input("Camera Source (0, 1, 2... or RTSP URL)", value="1")
        st.caption("💡 Select **1** for Iriun Webcam / USB Camera")
        start = st.button("Start Camera", type="primary")
        stop = st.button("Stop Camera")

        st.subheader("Live Stats")
        fps_ph = st.empty()
        det_ph = st.empty()
        st.subheader("Recent Logs")
        logs_ph = st.empty()

    if stop:
        st.session_state["camera_running"] = False
    if start:
        st.session_state["camera_running"] = True

    if st.session_state.get("camera_running", False):
        with col_main:
            frame_ph = st.empty()

        try:
            source = int(cam_source) if cam_source.isdigit() else cam_source
        except Exception:
            source = 1

        cap = None
        if isinstance(source, int) and sys.platform.startswith("win"):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            st.error(f"Cannot open camera source '{source}'. Please check Iriun Webcam app & USB connection.")
            st.session_state["camera_running"] = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)

        detector = comps.get("detector")
        if detector is None:
            st.error("Face detector not loaded. Download models first by running `python run.py`.")
            st.session_state["camera_running"] = False
            cap.release()
            return

        tracker = comps.get("tracker")
        qg = comps.get("quality_gate")
        preproc = comps.get("preprocessor")
        recognizer = comps.get("recognizer")
        vdb = comps.get("vector_db")
        att_engine = comps.get("attendance_engine")
        profiler = comps.get("profiler")

        frame_count = 0
        recent_logs: List[str] = []

        while st.session_state.get("camera_running", False):
            ret, frame = cap.read()
            if not ret:
                st.warning("Lost camera feed.")
                break

            frame_count += 1
            t0 = time.perf_counter()

            try:
                detections = detector.detect(frame)
            except Exception:
                detections = []

            # Draw bounding boxes
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Recognition pipeline
                if recognizer and vdb and vdb.size() > 0 and preproc:
                    try:
                        tensor = preproc.preprocess(frame, det.bbox, det.landmarks)
                        emb = recognizer.extract(tensor[0])
                        results = vdb.search(emb, top_k=1)
                        if results:
                            label, score = results[0]
                            thresh = cfg.recognition.similarity_threshold
                            if score >= thresh:
                                cv2.putText(frame, f"{label} ({score:.2f})", (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                latency = time.perf_counter() - t0
                                if att_engine:
                                    track_id = getattr(det, "track_id", frame_count)
                                    record = att_engine.process_recognition(track_id, label, score, latency)
                                    if record:
                                        recent_logs.append(f"{record.timestamp}: {record.student_id}")
                            else:
                                cv2.putText(frame, "Unknown", (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    except Exception:
                        pass

            elapsed = time.perf_counter() - t0
            cur_fps = 1.0 / max(elapsed, 0.001)

            frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            fps_ph.metric("FPS", f"{cur_fps:.1f}")
            det_ph.metric("Detections", len(detections))
            logs_ph.text("\n".join(recent_logs[-10:]))

        cap.release()
    else:
        with col_main:
            st.info("Click 'Start Camera' to begin real-time recognition.")


def page_video_upload() -> None:
    st.title("Video Upload")
    comps = _safe_init_components()

    uploaded = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded is not None:
        detector = comps.get("detector")
        recognizer = comps.get("recognizer")
        preproc = comps.get("preprocessor")
        vdb = comps.get("vector_db")

        if detector is None or recognizer is None:
            st.error("Models not loaded. Run `python run.py` first.")
            return

        # Save uploaded file
        temp_path = Path(cfg.paths.cache_dir) / "uploaded_video.mp4"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded.read())

        cap = cv2.VideoCapture(str(temp_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.write(f"Total frames: {total_frames}")

        if st.button("Process Video", type="primary"):
            progress = st.progress(0.0)
            frame_ph = st.empty()
            results_list = []

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % 5 != 0:  # Process every 5th frame
                    continue

                progress.progress(min(frame_idx / max(total_frames, 1), 1.0))

                try:
                    detections = detector.detect(frame)
                    for det in detections:
                        x1, y1, x2, y2 = det.bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        if vdb and vdb.size() > 0 and preproc:
                            tensor = preproc.preprocess(frame, det.bbox, det.landmarks)
                            emb = recognizer.extract(tensor[0])
                            search_results = vdb.search(emb, top_k=1)
                            if search_results:
                                label, score = search_results[0]
                                if score >= cfg.recognition.similarity_threshold:
                                    results_list.append({"frame": frame_idx, "student": label, "score": score})
                                    cv2.putText(frame, f"{label}", (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception:
                    pass

                if frame_idx % 15 == 0:
                    frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            cap.release()
            progress.progress(1.0)
            st.success(f"Processed {frame_idx} frames, {len(results_list)} recognitions.")

            if results_list:
                df = pd.DataFrame(results_list)
                st.dataframe(df, use_container_width=True)


def page_attendance_logs() -> None:
    st.title("Attendance Logs")
    comps = _safe_init_components()

    att = comps.get("attendance_engine")
    if att is None:
        st.info("Attendance engine not initialised.")
        return

    records = att.get_records()

    if not records:
        st.info("No attendance records yet. Use the Real-time Camera or Video Upload to start recognizing.")
        return

    data = [{"Student ID": r.student_id, "Name": r.name, "Timestamp": r.timestamp,
             "Confidence": f"{r.confidence:.3f}", "Camera": r.camera_id} for r in records]
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    st.subheader("Export")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Export CSV"):
            p = att.export_csv()
            st.success(f"Saved to `{p}`")
    with col2:
        if st.button("Export Excel"):
            try:
                p = att.export_excel()
                st.success(f"Saved to `{p}`")
            except Exception as e:
                st.error(f"Excel export failed: {e}")
    with col3:
        if st.button("Export JSON"):
            p = att.export_json()
            st.success(f"Saved to `{p}`")
    with col4:
        if st.button("Export SQLite"):
            p = att.export_sqlite()
            st.success(f"Saved to `{p}`")


def page_reports() -> None:
    st.title("Reports")

    reports_dir = Path(cfg.paths.reports_dir)
    outputs_dir = Path(cfg.paths.outputs_dir)

    charts = []
    for d in [reports_dir, outputs_dir]:
        if d.exists():
            charts.extend(d.glob("*.png"))
            charts.extend(d.glob("*.jpg"))

    if not charts:
        st.info("No report charts generated yet. Run an Evaluation Benchmark first.")
        return

    st.write(f"Found **{len(charts)}** charts")
    for chart in sorted(charts):
        st.image(str(chart), caption=chart.stem, width=700)


def page_settings() -> None:
    st.title("Settings")

    st.subheader("Detection")
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Confidence Threshold", value=cfg.detection.confidence_threshold,
                         min_value=0.1, max_value=1.0, step=0.05, key="det_conf")
        st.number_input("NMS Threshold", value=cfg.detection.nms_threshold,
                         min_value=0.1, max_value=1.0, step=0.05, key="det_nms")
    with col2:
        st.number_input("Max Faces", value=cfg.detection.max_faces,
                         min_value=1, max_value=200, step=1, key="det_max")

    st.subheader("Recognition")
    st.number_input("Similarity Threshold", value=cfg.recognition.similarity_threshold,
                     min_value=0.1, max_value=1.0, step=0.05, key="rec_sim")
    st.number_input("Recognition Threshold", value=cfg.recognition.recognition_threshold,
                     min_value=0.1, max_value=1.0, step=0.05, key="rec_thresh")

    st.subheader("Temporal Voting")
    col3, col4 = st.columns(2)
    with col3:
        st.number_input("M (minimum votes)", value=cfg.voting.m, min_value=1, max_value=20, key="vote_m")
    with col4:
        st.number_input("N (window size)", value=cfg.voting.n, min_value=1, max_value=30, key="vote_n")

    st.subheader("Quality Gate")
    st.number_input("Min Face Width (px)", value=cfg.quality.min_face_width,
                     min_value=10, max_value=200, key="q_minw")
    st.number_input("Min Laplacian (blur)", value=cfg.quality.min_laplacian_variance,
                     min_value=1.0, max_value=500.0, step=5.0, key="q_blur")

    st.subheader("Attendance")
    st.number_input("Cooldown (seconds)", value=cfg.attendance.cooldown_seconds,
                     min_value=0.0, max_value=3600.0, step=30.0, key="att_cool")

    st.info("Settings changes only apply to the current session. Edit `config.py` for persistent changes.")


def page_system_status() -> None:
    st.title("System Status")
    comps = _safe_init_components()

    # Device info
    try:
        dm = DeviceManager()
        dev = dm.get_device_info()

        col1, col2, col3 = st.columns(3)
        col1.metric("CPU Cores", dev.cpu_threads)
        col2.metric("RAM", f"{dev.total_ram_gb:.1f} GB")
        gpu_text = dev.gpu_name if dev.gpu_available else "CPU Only"
        col3.metric("GPU", gpu_text)

        st.write(f"**Provider:** {dev.best_provider}")
        st.write(f"**Available Providers:** {', '.join(dev.onnx_providers)}")
    except Exception as e:
        st.warning(f"Device info error: {e}")

    # Profiler stats
    st.subheader("Performance Profiler")
    prof = comps.get("profiler")
    if prof:
        try:
            report = prof.get_report()
            if report:
                rows = []
                for op, stats in report.items():
                    rows.append({
                        "Operation": op,
                        "Count": stats.get("count", 0),
                        "Mean (ms)": f"{stats.get('mean', 0)*1000:.2f}",
                        "P95 (ms)": f"{stats.get('p95', 0)*1000:.2f}",
                    })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info("No profiler data yet. Run some operations first.")
            else:
                st.info("No profiler data yet.")
        except Exception:
            st.info("Profiler report unavailable.")
    else:
        st.info("Profiler not initialised.")

    # Loaded models
    st.subheader("Models")
    try:
        from src.model_manager import get_model_manager
        mm = get_model_manager()
        loaded = mm.get_loaded_models()
        for key in MODEL_REGISTRY:
            status = "Loaded" if key in loaded else "Not Loaded"
            info = MODEL_REGISTRY[key]
            st.write(f"- **{info.name}** (`{info.filename}`): {status}")
    except Exception:
        st.info("Model manager not available.")

    # Memory
    st.subheader("Memory Usage")
    try:
        mem = dm.get_memory_usage()
        st.write(f"**Process RSS:** {mem['rss_mb']:.1f} MB")
        st.write(f"**Process VMS:** {mem['vms_mb']:.1f} MB")
    except Exception:
        pass


def page_about() -> None:
    st.title("About")

    st.markdown(f"""
### {cfg.app_name}
**Version:** {cfg.version}

A production-grade facial recognition attendance system implementing modern computer vision literature.

**Pipeline:**
- Face Detection: SCRFD (Sample and Computation Redistribution)
- Face Tracking: ByteTrack (Kalman + two-stage IoU)
- Quality Assessment: Blur, brightness, pose, size, EAR
- Face Recognition: AdaFace / GhostFaceNet / MobileFaceNet
- Vector Search: FAISS (with numpy fallback)
- Temporal Voting: M-of-N consensus
- Inference: ONNX Runtime (CPU-first, GPU-ready)

**Technology Stack:**
- Python, OpenCV, ONNX Runtime, FAISS, NumPy, Streamlit
- Factory pattern for runtime backend switching
- Automatic GPU detection and provider fallback

**Author:** {cfg.author}
""")


# =====================================================================
#  MAIN
# =====================================================================

def main() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.title("Smart Attendance")
        st.divider()

        st.subheader("Navigation")
        selection = st.radio(
            "Navigation",
            options=[
                "Dashboard",
                "Dataset Manager",
                "Enrollment",
                "Evaluation",
                "Real-time Camera",
                "Video Upload",
                "Attendance Logs",
                "Reports",
                "Settings",
                "System Status",
                "About",
            ],
            label_visibility="collapsed",
        )

        st.divider()

        # Footer
        try:
            dm = DeviceManager()
            dev = dm.get_device_info()
            st.caption(f"Provider: {dev.best_provider}")
            st.caption(f"RAM: {dev.available_ram_gb:.1f}/{dev.total_ram_gb:.1f} GB")
        except Exception:
            pass

    pages = {
        "Dashboard": page_dashboard,
        "Dataset Manager": page_dataset_manager,
        "Enrollment": page_enrollment,
        "Evaluation": page_evaluation,
        "Real-time Camera": page_realtime_camera,
        "Video Upload": page_video_upload,
        "Attendance Logs": page_attendance_logs,
        "Reports": page_reports,
        "Settings": page_settings,
        "System Status": page_system_status,
        "About": page_about,
    }

    pages[selection]()


if __name__ == "__main__":
    main()
