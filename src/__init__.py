"""
Facial Recognition Attendance System — Source Package.

This package contains all core modules:
    - device_manager: Hardware/GPU detection and execution provider selection
    - model_manager: Centralised ONNX model loading and caching
    - logger: Structured, rotating, coloured logging
    - config_loader: Configuration loading and validation
    - utils: Cross-platform helpers
    - downloader: Model download with checksum verification
    - profiler: Pipeline performance profiling
    - detector: Face detection (SCRFD, OpenCV, InsightFace)
    - tracker: Object tracking (ByteTrack, SORT)
    - quality_gate: Face quality assessment
    - preprocessing: Image preprocessing and alignment
    - recognizer: Face recognition (AdaFace, GhostFaceNet, MobileFaceNet)
    - vector_db: FAISS vector search
    - enrollment: Student enrollment pipeline
    - attendance: Attendance engine with temporal voting
    - metrics: Evaluation metrics computation
    - evaluation: Full benchmark pipeline
    - visualization: Chart and figure generation
    - dataset_validator: Dataset integrity checking
"""

__all__: list[str] = [
    "device_manager",
    "model_manager",
    "logger",
    "config_loader",
    "utils",
    "downloader",
    "profiler",
    "detector",
    "tracker",
    "quality_gate",
    "preprocessing",
    "recognizer",
    "vector_db",
    "enrollment",
    "attendance",
    "metrics",
    "evaluation",
    "visualization",
    "dataset_validator",
]
