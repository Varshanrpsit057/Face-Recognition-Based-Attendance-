"""
Centralized Configuration for the Attendance System.

All thresholds, paths, model names, camera configuration,
tracking parameters, recognition thresholds, evaluation settings,
and export formats are stored here.

Usage:
    from config import cfg
    print(cfg.RECOGNITION_THRESHOLD)
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────

class ExecutionProvider(Enum):
    """ONNX Runtime execution providers in priority order."""
    TENSORRT = "TensorrtExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    OPENVINO = "OpenVINOExecutionProvider"
    DIRECTML = "DmlExecutionProvider"
    COREML = "CoreMLExecutionProvider"
    CPU = "CPUExecutionProvider"


class RecognizerBackend(Enum):
    """Available face recognition backends."""
    ADAFACE = "adaface"
    GHOSTFACENET = "ghostfacenet"
    MOBILEFACENET = "mobilefacenet"


class DetectorBackend(Enum):
    """Available face detection backends."""
    SCRFD = "scrfd"
    YOLOFACE = "yoloface"
    INSIGHTFACE = "insightface"
    OPENCV = "opencv"


class TrackerBackend(Enum):
    """Available tracking backends."""
    BYTETRACK = "bytetrack"
    SORT = "sort"


class ExportFormat(Enum):
    """Supported export formats."""
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    SQLITE = "sqlite"
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"


class LogLevel(Enum):
    """Logging verbosity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ─────────────────────────────────────────────────────────────────────
# Path Configuration
# ─────────────────────────────────────────────────────────────────────

# Base directory is the project root
BASE_DIR: Path = Path(__file__).resolve().parent


@dataclass
class PathConfig:
    """All file-system paths used by the system."""

    base_dir: Path = BASE_DIR
    dataset_dir: Path = BASE_DIR / "dataset"
    models_dir: Path = BASE_DIR / "models"
    outputs_dir: Path = BASE_DIR / "outputs"
    embeddings_dir: Path = BASE_DIR / "outputs" / "embeddings"
    logs_dir: Path = BASE_DIR / "outputs" / "logs"
    attendance_dir: Path = BASE_DIR / "attendance_logs"
    reports_dir: Path = BASE_DIR / "reports"
    cache_dir: Path = BASE_DIR / "cache"
    sample_video_dir: Path = BASE_DIR / "sample_video"

    # Model files
    scrfd_model: Path = BASE_DIR / "models" / "scrfd_10g.onnx"
    scrfd_34g_model: Path = BASE_DIR / "models" / "34g.onnx"
    adaface_model: Path = BASE_DIR / "models" / "adaface_ir101.onnx"
    ghostfacenet_model: Path = BASE_DIR / "models" / "ghostfacenet.onnx"
    mobilefacenet_model: Path = BASE_DIR / "models" / "mobilefacenet.onnx"
    cr_fiqa_model: Path = BASE_DIR / "models" / "cr_fiqa_l.onnx"
    minifasnet_model: Path = BASE_DIR / "models" / "minifasnet_v2.onnx"

    # Haar cascade (OpenCV fallback)
    haar_cascade: str = "haarcascade_frontalface_default.xml"

    # FAISS index
    faiss_index_path: Path = BASE_DIR / "cache" / "faiss_index.bin"
    faiss_labels_path: Path = BASE_DIR / "cache" / "faiss_labels.pkl"

    # Attendance DB
    attendance_db: Path = BASE_DIR / "attendance_logs" / "attendance.db"

    def ensure_dirs(self) -> None:
        """Create all directories if they do not exist."""
        for attr_name in vars(self):
            val = getattr(self, attr_name)
            if isinstance(val, Path) and "dir" in attr_name:
                val.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# Model Download Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ModelInfo:
    """Metadata for a downloadable model."""

    name: str
    filename: str
    url: str
    sha256: str
    description: str
    input_size: Tuple[int, int] = (112, 112)
    embedding_dim: int = 512


# Model registry. Entries with url="" are local-only assets: they must
# already exist in models/ (as provided) because no verified public
# direct-download URL is configured for them — the downloader will
# raise rather than silently fetching an unverified file.
# SHA256 values will be verified after download; set to empty string to
# skip verification when the hash is not yet known.
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    "scrfd": ModelInfo(
        name="SCRFD-10GF",
        filename="scrfd_10g.onnx",
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_10g_bnkps.onnx",
        sha256="",
        description="SCRFD face detector (10 GFlops, with key-points)",
        input_size=(640, 640),
        embedding_dim=0,
    ),
    "scrfd_34g": ModelInfo(
        name="SCRFD-34GF",
        filename="34g.onnx",
        url="",
        sha256="",
        description="SCRFD face detector (34 GFlops, no key-points) — stronger small/far-face recall for large rooms",
        input_size=(640, 640),
        embedding_dim=0,
    ),
    "adaface": ModelInfo(
        name="AdaFace-IR101",
        filename="adaface_ir101.onnx",
        url="",
        sha256="",
        description="AdaFace recognition model (IR-101, 512-d embedding + norm outputs)",
        input_size=(112, 112),
        embedding_dim=512,
    ),
    "cr_fiqa": ModelInfo(
        name="CR-FIQA-L",
        filename="cr_fiqa_l.onnx",
        url="",
        sha256="",
        description="CR-FIQA learned face image quality model (512-d embedding + quality_score)",
        input_size=(112, 112),
        embedding_dim=512,
    ),
    "minifasnet_v2": ModelInfo(
        name="MiniFASNet-V2",
        filename="minifasnet_v2.onnx",
        url="",
        sha256="",
        description="MiniFASNet-V2 anti-spoofing / liveness classifier (3-class)",
        input_size=(80, 80),
        embedding_dim=0,
    ),
    "ghostfacenet": ModelInfo(
        name="GhostFaceNet",
        filename="ghostfacenet.onnx",
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/glint360k_r50.onnx",
        sha256="",
        description="GhostFaceNet lightweight recognition (512-d)",
        input_size=(112, 112),
        embedding_dim=512,
    ),
    "mobilefacenet": ModelInfo(
        name="MobileFaceNet",
        filename="mobilefacenet.onnx",
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/w600k_mbf.onnx",
        sha256="",
        description="MobileFaceNet ultra-light recognition (512-d)",
        input_size=(112, 112),
        embedding_dim=512,
    ),
}


# ─────────────────────────────────────────────────────────────────────
# Detection Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class DetectionConfig:
    """Face detection parameters."""

    backend: DetectorBackend = DetectorBackend.SCRFD
    # SCRFD checkpoint variant: "10g" (10 GFlops, has 5-point landmarks,
    # faster) or "34g" (34 GFlops, no landmarks in this export, notably
    # stronger recall on small/far faces). Defaults to "34g" because
    # recovering far-row faces is the binding constraint for a single
    # 8MP camera covering a full 30x40 ft classroom; switch to "10g" for
    # smaller rooms/webcams where landmark-based alignment matters more
    # than squeezing out extra small-face recall.
    scrfd_variant: str = "34g"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    input_size: Tuple[int, int] = (640, 640)
    max_faces: int = 100  # headroom above the 65-student room capacity
    fallback_order: List[DetectorBackend] = field(
        default_factory=lambda: [
            DetectorBackend.SCRFD,
            DetectorBackend.INSIGHTFACE,
            DetectorBackend.OPENCV,
        ]
    )

    # ── High-resolution tiled detection ─────────────────────────────
    # A single SCRFD pass resizes the whole frame down to `input_size`
    # (640x640). For an 8MP frame (e.g. 3840x2160) showing a full
    # 30x40 ft classroom, that crushes far-row faces to a few pixels —
    # well below what any recognizer can match. Tiling instead runs
    # detection on overlapping crops of the native-resolution frame so
    # each tile only loses a little resolution, then merges results.
    use_tiling: bool = True
    # (cols, rows) grid of overlapping tiles across the frame. 3x2 = 6
    # tiles is a good default for an 8MP frame from the back/front of a
    # 30x40 ft room; increase (e.g. 4x3) if far-row faces are still
    # being missed, at the cost of more inference passes per frame.
    tile_grid: Tuple[int, int] = (3, 2)
    # Fractional overlap between adjacent tiles so a face straddling a
    # tile boundary is still whole in at least one tile.
    tile_overlap: float = 0.20
    # Only tile when the frame is meaningfully larger than the
    # detector's native input; small frames (e.g. a laptop webcam) are
    # detected in a single pass.
    tile_min_dimension: int = 1280
    # IoU threshold used to de-duplicate detections of the same face
    # that were found in more than one overlapping tile.
    tile_merge_iou: float = 0.3


# ─────────────────────────────────────────────────────────────────────
# Tracking Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TrackingConfig:
    """Object tracker parameters."""

    backend: TrackerBackend = TrackerBackend.BYTETRACK
    max_lost_frames: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    high_threshold: float = 0.6
    low_threshold: float = 0.1
    match_threshold: float = 0.8
    track_buffer: int = 30
    frame_rate: int = 30


# ─────────────────────────────────────────────────────────────────────
# Quality Gate Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class QualityConfig:
    """Face quality assessment thresholds."""

    # 28px keeps back-row faces from a tiled 8MP capture usable while
    # still rejecting noise; 35px (the old default) discarded too many
    # legitimate far-row detections in a 30x40 ft room.
    min_face_width: int = 28
    min_face_height: int = 28
    min_laplacian_variance: float = 50.0
    max_yaw: float = 40.0
    max_pitch: float = 30.0
    max_roll: float = 25.0
    min_brightness: float = 40.0
    max_brightness: float = 220.0
    min_eye_aspect_ratio: float = 0.15
    occlusion_threshold: float = 0.5
    mask_detection_enabled: bool = False

    # ── Learned quality (CR-FIQA) ────────────────────────────────────
    # Fused with the heuristic score above rather than replacing it, so
    # a missing/failed model degrades gracefully to heuristics only.
    use_ml_quality: bool = True
    ml_quality_weight: float = 0.4  # 0 = heuristic only, 1 = ML only
    # CR-FIQA's raw regression output has no fixed scale. Measured
    # against this project's own enrolled dataset (68 images, all
    # decent-quality frontal photos): raw scores ranged 0.68-2.31,
    # mean 1.60. Bounds below give that range headroom on both ends
    # (worse live-camera captures scoring lower, excellent ones
    # scoring higher) while mapping to 0..1 via clipped min-max
    # normalization. Re-run scratchpad calibration after swapping the
    # CR-FIQA checkpoint or dataset.
    cr_fiqa_score_min: float = 0.3
    cr_fiqa_score_max: float = 2.6

    # ── Liveness / anti-spoofing (MiniFASNet-V2) ─────────────────────
    # Only ever applied to the live camera feed — never to dataset
    # enrollment photos, which are themselves static images and would
    # always score as "spoof".
    liveness_enabled: bool = True
    liveness_threshold: float = 0.5
    liveness_crop_scale: float = 2.7  # face-bbox padding factor fed to MiniFASNet


# ─────────────────────────────────────────────────────────────────────
# Recognition Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class RecognitionConfig:
    """Face recognition parameters."""

    backend: RecognizerBackend = RecognizerBackend.ADAFACE
    embedding_dim: int = 512
    similarity_threshold: float = 0.45
    recognition_threshold: float = 0.50
    batch_size: int = 8
    fallback_order: List[RecognizerBackend] = field(
        default_factory=lambda: [
            RecognizerBackend.ADAFACE,
            RecognizerBackend.GHOSTFACENET,
            RecognizerBackend.MOBILEFACENET,
        ]
    )


# ─────────────────────────────────────────────────────────────────────
# FAISS / Vector DB Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class VectorDBConfig:
    """FAISS vector database parameters."""

    embedding_dim: int = 512
    index_type: str = "Flat"  # "Flat", "IVFFlat", "HNSW"
    nprobe: int = 10
    top_k: int = 5
    use_gpu: bool = False


# ─────────────────────────────────────────────────────────────────────
# Temporal Voting Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class VotingConfig:
    """M-of-N temporal voting parameters."""

    m: int = 5
    n: int = 7
    confidence_averaging: bool = True
    min_confidence: float = 0.45


# ─────────────────────────────────────────────────────────────────────
# Attendance Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class AttendanceConfig:
    """Attendance engine parameters."""

    cooldown_seconds: float = 300.0  # 5 minutes
    camera_id: str = "CAM_01"
    # Expected headcount for the monitored room — drives the "present /
    # capacity" gauge and lets Settings warn if the enrolled roster
    # exceeds what a single camera + tiling config was tuned for.
    room_capacity: int = 65
    export_formats: List[ExportFormat] = field(
        default_factory=lambda: [
            ExportFormat.CSV,
            ExportFormat.EXCEL,
            ExportFormat.JSON,
            ExportFormat.SQLITE,
        ]
    )


# ─────────────────────────────────────────────────────────────────────
# Camera Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class CameraConfig:
    """Camera / video source parameters.

    Defaults target a single 8MP IP camera (3840x2160, i.e. "4K"/8.3MP,
    the common sensor resolution for 8MP security/IP cameras) covering
    a 30x40 ft classroom of up to `AttendanceConfig.room_capacity`
    students. Native resolution is kept for detection (tiling handles
    the downscale loss); only the on-screen preview is shrunk.
    """

    source: int = 0  # Default webcam; overridden by ip_camera_url when set
    width: int = 3840
    height: int = 2160
    fps: int = 15  # IP cameras rarely need >15fps for attendance scanning
    buffer_size: int = 1  # keep OpenCV's internal queue shallow to cut latency

    # ── IP camera (RTSP/HTTP) ───────────────────────────────────────
    # e.g. "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101"
    ip_camera_url: Optional[str] = None
    rtsp_transport: str = "tcp"  # "tcp" is far more reliable than UDP over Wi-Fi/LAN hops
    connect_timeout_sec: float = 8.0
    read_timeout_sec: float = 5.0
    reconnect_delay_sec: float = 2.0
    max_reconnect_delay_sec: float = 30.0

    # ── Threaded capture ─────────────────────────────────────────────
    # Grabs frames in a background thread and always exposes only the
    # latest one, so a detection pipeline slower than the camera's fps
    # (expected here, given tiling) never processes a growing backlog
    # of stale frames.
    use_threaded_capture: bool = True

    # Display-only downscale so the Streamlit preview stays responsive;
    # detection always runs on the full native frame.
    preview_max_width: int = 1280


# ─────────────────────────────────────────────────────────────────────
# Evaluation Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationConfig:
    """Evaluation / benchmarking parameters."""

    gallery_ratio: float = 0.80
    probe_ratio: float = 0.20
    random_seed: int = 42
    threshold_sweep_start: float = 0.1
    threshold_sweep_end: float = 0.9
    threshold_sweep_steps: int = 50
    dpi: int = 150
    figure_formats: List[str] = field(
        default_factory=lambda: ["png", "pdf", "svg"]
    )


# ─────────────────────────────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class LoggingConfig:
    """Structured logging parameters."""

    level: LogLevel = LogLevel.INFO
    log_to_file: bool = True
    log_to_console: bool = True
    max_file_size_mb: int = 10
    backup_count: int = 5
    colored_output: bool = True
    separate_categories: bool = True  # Separate log files per category


# ─────────────────────────────────────────────────────────────────────
# Preprocessing Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PreprocessingConfig:
    """Image preprocessing parameters."""

    target_size: Tuple[int, int] = (112, 112)
    use_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    gamma: float = 1.0
    normalize: bool = True
    mean: Tuple[float, float, float] = (127.5, 127.5, 127.5)
    std: Tuple[float, float, float] = (127.5, 127.5, 127.5)


# ─────────────────────────────────────────────────────────────────────
# Execution Provider Priority
# ─────────────────────────────────────────────────────────────────────

EXECUTION_PROVIDER_PRIORITY: List[ExecutionProvider] = [
    ExecutionProvider.TENSORRT,
    ExecutionProvider.CUDA,
    ExecutionProvider.OPENVINO,
    ExecutionProvider.DIRECTML,
    ExecutionProvider.COREML,
    ExecutionProvider.CPU,
]


# ─────────────────────────────────────────────────────────────────────
# Master Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class AppConfig:
    """Master configuration aggregating all sub-configs."""

    paths: PathConfig = field(default_factory=PathConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    voting: VotingConfig = field(default_factory=VotingConfig)
    attendance: AttendanceConfig = field(default_factory=AttendanceConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    # Application metadata
    app_name: str = "Facial Recognition Attendance System"
    version: str = "1.0.0"
    author: str = "Attendance System Team"

    def __post_init__(self) -> None:
        """Ensure all required directories exist."""
        self.paths.ensure_dirs()


# Global singleton
cfg = AppConfig()
