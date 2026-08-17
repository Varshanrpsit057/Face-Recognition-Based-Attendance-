# 🎓 Facial Recognition Attendance System

A **production-grade, research-quality** real-time facial recognition attendance system built with Python. Designed as an engineering capstone project implementing modern computer vision literature recommendations.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.16+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Key Features

| Feature | Technology |
|---------|-----------|
| **Face Detection** | SCRFD (Sample and Computation Redistribution for Efficient Face Detection) |
| **Face Tracking** | ByteTrack (Kalman filter + two-stage IoU matching) |
| **Quality Assessment** | Blur, brightness, pose, face size, eye aspect ratio |
| **Face Recognition** | AdaFace (primary) / GhostFaceNet / MobileFaceNet |
| **Vector Search** | FAISS (with numpy fallback) |
| **Temporal Voting** | M-of-N consensus voting for robust identification |
| **Dashboard** | 11-page interactive Streamlit web app |
| **Benchmarking** | Full evaluation suite (accuracy, precision, recall, F1, ROC, CMC, EER) |
| **Inference Runtime** | ONNX Runtime (CPU-first, GPU-ready) |

---

## 🏗️ Architecture

```
Camera → SCRFD Detector → ByteTrack Tracker → Quality Gate → Face Alignment
         → AdaFace Recognizer → FAISS Search → Temporal Voting → Attendance Log
```

### CPU-First, GPU-Ready

The system **runs immediately on CPU** with zero configuration. When deployed to GPU systems, it automatically detects and uses the best available execution provider:

```
TensorRT → CUDA → OpenVINO → DirectML → CoreML → CPU
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone <repo-url> && cd Attendance_System_testing_phase

# One command to set up everything and launch:
python run.py
```

The `run.py` script automatically:
1. ✅ Checks Python ≥ 3.9
2. ✅ Creates a virtual environment (`.venv`)
3. ✅ Installs all dependencies
4. ✅ Creates project directories
5. ✅ Downloads ONNX models (~200 MB)
6. ✅ Launches the Streamlit dashboard at `http://localhost:8501`

---

## 📁 Project Structure

```
Attendance_System_testing_phase/
├── config.py                # Centralized configuration (12 sub-configs)
├── run.py                   # Zero-config bootstrap script
├── app.py                   # Streamlit dashboard (11 pages)
├── requirements.txt         # All dependencies
│
├── src/                     # Core modules
│   ├── detector.py          # SCRFD face detection + OpenCV fallback
│   ├── tracker.py           # ByteTrack + SORT tracking
│   ├── quality_gate.py      # Multi-factor quality assessment
│   ├── preprocessing.py     # ArcFace alignment (112×112)
│   ├── recognizer.py        # AdaFace/GhostFaceNet/MobileFaceNet
│   ├── vector_db.py         # FAISS vector search + numpy fallback
│   ├── enrollment.py        # Student enrollment pipeline
│   ├── attendance.py        # Temporal voting + attendance logging
│   ├── evaluation.py        # Benchmarking pipeline
│   ├── metrics.py           # Evaluation metrics (ROC, CMC, EER, etc.)
│   ├── visualization.py     # Chart generation (12 plot types)
│   ├── dataset_validator.py # Dataset integrity checking
│   ├── device_manager.py    # Hardware detection + provider selection
│   ├── model_manager.py     # ONNX session caching
│   ├── downloader.py        # Model download with retry + SHA256
│   ├── profiler.py          # Performance profiling
│   ├── logger.py            # Colored logging + rotating files
│   ├── config_loader.py     # YAML config I/O
│   └── utils.py             # Common utilities
│
├── dataset/                 # Student images (one folder per student)
│   ├── student_001/
│   ├── student_002/
│   └── ...
├── models/                  # ONNX model files (auto-downloaded)
├── outputs/                 # Embeddings, logs, charts
├── attendance_logs/         # Attendance records + exports
├── reports/                 # Evaluation reports
└── cache/                   # FAISS index + labels
```

---

## 📸 Dataset Preparation

Create a folder for each student inside `dataset/`:

```
dataset/
├── john_doe/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── img_003.jpg
├── jane_smith/
│   ├── photo1.png
│   └── photo2.png
└── ...
```

**Requirements per student:**
- Minimum **3 images** (5+ recommended)
- Clear, frontal face shots
- Good lighting (avoid extreme shadows)
- Various angles are beneficial for robustness

---

## 📊 Dashboard Pages

| # | Page | Description |
|---|------|-------------|
| 1 | 📊 Dashboard | System overview, enrolled students, today's attendance |
| 2 | 📁 Dataset Manager | Browse students, preview images, validate dataset |
| 3 | 📝 Enrollment | Batch enroll students, build FAISS index |
| 4 | 📈 Evaluation | Run gallery/probe benchmark with full metrics |
| 5 | 🎥 Real-time Camera | Live webcam recognition + automatic attendance |
| 6 | 🎬 Video Upload | Process recorded video files |
| 7 | 📋 Attendance Logs | View records, export to CSV/Excel/JSON/SQLite |
| 8 | 📊 Reports | View charts and evaluation results |
| 9 | ⚙️ Settings | Adjust thresholds and parameters |
| 10 | 🖥️ System Status | Hardware info, profiler stats, loaded models |
| 11 | ℹ️ About | System information and tech stack |

---

## ⚙️ Configuration

All settings are in [`config.py`](config.py):

| Section | Key Parameters |
|---------|---------------|
| **Detection** | `confidence_threshold=0.5`, `nms_threshold=0.4`, `input_size=(640,640)` |
| **Recognition** | `similarity_threshold=0.45`, `embedding_dim=512` |
| **Quality** | `min_laplacian_variance=50`, `min_face_width=35`, `max_yaw=40°` |
| **Voting** | `m=5`, `n=7` (5-of-7 consensus required) |
| **Attendance** | `cooldown_seconds=300` (5 min between re-marks) |
| **Camera** | `source=0`, `width=1280`, `height=720`, `fps=30` |

---

## 📦 Dependencies

- **Core:** numpy, scipy, Pillow, pyyaml
- **Vision:** opencv-python, onnxruntime
- **ML:** scikit-learn, faiss-cpu
- **Tracking:** filterpy, lap
- **Viz:** matplotlib, seaborn
- **Data:** pandas, openpyxl, xlsxwriter
- **Web:** streamlit
- **Utils:** tqdm, psutil, requests

---

## 🔬 Evaluation Metrics

The benchmarking suite computes:

- **Classification:** Accuracy, Precision, Recall, F1, Balanced Accuracy, MCC
- **Biometric:** FAR (False Accept Rate), FRR (False Reject Rate), EER (Equal Error Rate)
- **Curves:** ROC, Precision-Recall, DET, CMC
- **Per-class:** Individual student metrics
- **Performance:** FPS, latency percentiles (P50, P95, P99), memory usage

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) — SCRFD detection models
- [AdaFace](https://github.com/mk-minchul/AdaFace) — Quality-adaptive face recognition
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — Multi-object tracking
- [FAISS](https://github.com/facebookresearch/faiss) — Efficient similarity search
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — Cross-platform inference
