# 06 --- Implementation Plan

## 1. Development Strategy

Build the system in layers rather than attempting the complete
application simultaneously.

Order:

``` text
AI Validation
→ Local Attendance
→ Local Database
→ Student/Face Enrollment
→ Camera Management
→ Sync
→ Home PC Backend
→ Web UI
→ Roles
→ Reports
→ XAI
→ Security
→ Testing
→ Deployment
```

## Phase 0 --- Environment Verification

### Laptop

Verify: - Windows. - NVIDIA driver. - CUDA compatibility. - ONNX Runtime
GPU provider. - RTX 3050 6 GB availability. - Python environment.

Test:

``` text
GPU detected
→ SCRFD loads
→ AdaFace loads
→ FIQA loads
→ MiniFASNet loads
→ FAISS loads
```

Do not proceed to full application development until each model's
input/output contract is verified.

## Phase 1 --- Model Verification

For each model:

### SCRFD

-   Load ONNX.
-   Run test image.
-   Verify bounding boxes.
-   Verify landmarks.
-   Measure latency.

### AdaFace

-   Load ONNX and `.data`.
-   Run known face.
-   Verify embedding dimension.
-   Normalize embedding.
-   Verify repeatability.

### FIQA

-   Inspect README/model metadata.
-   Determine expected input.
-   Determine output quality score.
-   Test good/bad images.

### MiniFASNet

-   Verify model purpose and input/output.
-   Test genuine live face.
-   Test replay/printed sample if available.
-   Define score interpretation.

## Phase 2 --- Face Dataset Enrollment

Implement:

``` text
Student profile
→ Select student
→ Upload/capture photos
→ Validate
→ SCRFD
→ Quality
→ Save sample
→ AdaFace
→ Embedding
→ FAISS
```

Requirements: - Multiple samples. - Sample quality display. - Delete
invalid sample. - Re-enrollment. - FAISS rebuild/update.

## Phase 3 --- Local Database

Install PostgreSQL on the laptop.

Create: - students. - face_samples. - face_embeddings. - classrooms. -
cameras. - subjects. - sessions. - attendance. - recognition_events. -
sync_events.

Use migrations.

## Phase 4 --- Local Camera Engine

Implement a generic camera service.

``` text
CameraManager
→ RTSP connector
→ Frame reader
→ reconnect logic
→ FPS monitor
```

Support: - Manual RTSP. - ONVIF discovery where possible. - Multiple
camera records. - Camera selection.

## Phase 5 --- Local AI Attendance Engine

Implement:

``` text
Frame
→ SCRFD
→ Quality
→ Anti-spoof
→ AdaFace
→ FAISS
→ Decision
→ Attendance
```

Add: - Frame skipping. - Batch inference where beneficial. - GPU
execution. - Duplicate suppression. - Multi-frame confirmation. -
Session validation.

## Phase 6 --- Attendance Rules

Implement: - Start session. - Stop session. - Present. - Absent. -
Optional late state. - Duplicate protection. - Correction request. -
Audit trail.

Do not allow a student to be repeatedly marked present because they
remain visible.

## Phase 7 --- Offline Queue

Implement:

``` text
Local attendance
→ Sync queue
→ Retry
→ Acknowledgement
```

Requirements: - Durable queue. - Unique event IDs. - Retry with
backoff. - No duplicate inserts. - Status monitoring.

## Phase 8 --- Home PC Backend

Install on Windows home PC: - Python. - FastAPI. - PostgreSQL. -
WebSocket support. - Reverse proxy if required. - Docker optionally.

Implement: - Authentication. - Student synchronized records. -
Attendance synchronized records. - Reports. - Notifications. - Audit.

## Phase 9 --- Synchronization API

Build:

``` text
Laptop Sync Client
        ↓
HTTPS
        ↓
FastAPI
        ↓
Home PC Database
        ↓
ACK
```

Test: - Normal sync. - Duplicate event. - Internet disconnect. -
Internet reconnect. - Server restart. - Laptop restart. - Partial
failure. - Large pending queue.

## Phase 10 --- Frontend

Build React + TypeScript UI based on Figma.

Recommended sequence:

1.  Role Selection.
2.  Login.
3.  Dashboard.
4.  Live Attendance.
5.  Attendance Sessions.
6.  Students.
7.  Face Dataset.
8.  Face Database.
9.  Cameras.
10. Classrooms.
11. Reports.
12. Analytics.
13. XAI.
14. Verification.
15. Notifications.
16. Audit Logs.
17. Settings.
18. Student Portal.

## Phase 11 --- Developer UI

Developer-only pages: - AI model status. - GPU diagnostics. - FAISS. -
Database. - Synchronization. - Camera diagnostics. - Logs.

Example:

``` text
SCRFD       LOADED
AdaFace     LOADED
FIQA        LOADED
MiniFASNet  LOADED
FAISS       READY
CUDA        AVAILABLE
Camera      ONLINE
Database    CONNECTED
Sync        CONNECTED
```

## Phase 12 --- HOD UI

HOD: - Manage students. - Manage datasets. - Manage faculty. - Manage
classrooms. - Manage cameras. - Review attendance. - Correct
attendance. - Reports. - Analytics. - Audit.

## Phase 13 --- Student UI

Student: - Login. - Dashboard. - Attendance. - Subject-wise
attendance. - History. - Notifications. - Correction requests. -
Profile.

Student UI must never expose privileged biometric or system-management
controls.

## Phase 14 --- XAI

Implement XAI after recognition is stable.

Recommended strategy:

``` text
Normal recognition
→ Store recognition metadata

Selected event
→ Generate XAI
→ Store result
→ Display to authorized user
```

Do not run expensive explanations on every classroom frame unless
benchmarking proves the laptop can handle it.

## Phase 15 --- Security

Implement: - Password hashing. - Role-based permissions. - Session
expiration. - HTTPS. - Device authentication. - Audit logging. - Secure
camera credentials. - Restricted biometric access.

Developer mode: - Keep passwordless only for local development. -
Disable or restrict before real deployment.

## Phase 16 --- Testing

### Unit tests

-   Student CRUD.
-   Attendance logic.
-   Duplicate detection.
-   Sync event handling.
-   Permissions.

### AI tests

-   Detection.
-   Recognition.
-   Quality.
-   Anti-spoof.
-   FAISS search.

### Integration tests

-   Camera → AI.
-   AI → database.
-   Database → sync.
-   Sync → server.
-   Server → WebSocket.
-   WebSocket → browser.

### Failure tests

-   Camera disconnect.
-   Internet disconnect.
-   Home PC shutdown.
-   Laptop restart.
-   Database failure.
-   FAISS failure.
-   GPU unavailable.

## Phase 17 --- Recognition Evaluation

Create a controlled evaluation dataset.

Measure: - Face detection performance. - Recognition accuracy. - False
acceptance rate. - False rejection rate. - Unknown rejection. -
Anti-spoof performance. - Average inference latency. - FPS. - FAISS
search latency.

Do not claim accuracy until measured.

## Phase 18 --- Classroom Evaluation

Test with: - Different distances. - Different face angles. - Different
lighting. - Different student densities. - Occlusion. - Multiple
faces. - Different camera positions.

The actual classroom performance must determine final: - Camera
resolution. - Camera position. - Stream resolution. - Recognition
threshold. - Frame sampling rate.

## Phase 19 --- Deployment

### Laptop

Run: - AI service. - Local database. - Camera service. - Attendance
engine. - Sync client.

### Home PC

Run: - FastAPI. - PostgreSQL. - WebSocket. - Frontend. - Synchronization
receiver.

## Phase 20 --- Operational Workflow

Before class:

``` text
Camera online
→ Laptop AI services ready
→ Database ready
→ FAISS ready
→ Sync ready
```

During class:

``` text
Faculty starts session
→ Camera selected
→ AI processes
→ Students recognized
→ Attendance marked
→ Local DB
→ Sync
→ Website updated
```

After class:

``` text
Session ends
→ Attendance finalized
→ Pending sync completed
→ Report available
```

## Phase 21 --- Recommended Project Folder Structure

``` text
face-attendance/
│
├── laptop/
│   ├── ai/
│   │   ├── scrfd/
│   │   ├── adaface/
│   │   ├── fiqa/
│   │   ├── antispoof/
│   │   └── pipeline/
│   │
│   ├── camera/
│   ├── attendance/
│   ├── enrollment/
│   ├── faiss/
│   ├── database/
│   ├── sync/
│   ├── config/
│   └── main.py
│
├── server/
│   ├── app/
│   │   ├── api/
│   │   ├── auth/
│   │   ├── students/
│   │   ├── attendance/
│   │   ├── cameras/
│   │   ├── reports/
│   │   ├── xai/
│   │   ├── sync/
│   │   └── audit/
│   │
│   └── main.py
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   ├── layouts/
│   │   ├── services/
│   │   ├── hooks/
│   │   └── types/
│
├── models/
│   ├── scrfd_10g_bnkps.onnx
│   ├── adaface_ir101.onnx
│   ├── adaface_ir101.onnx.data
│   ├── cr_fiqa_l.onnx
│   └── minifasnet_v2.onnx
│
├── migrations/
├── tests/
├── docs/
└── README.md
```

## Phase 22 --- Final Milestone

The system is ready for demonstration when the following end-to-end path
works:

``` text
Developer
→ Select Student
→ Register Profile
→ Register Face Dataset
→ Generate AdaFace embeddings
→ Build FAISS
→ Configure IP camera
→ Start attendance
→ SCRFD detects face
→ Quality validation
→ Anti-spoofing
→ AdaFace embedding
→ FAISS match
→ Register number identified
→ Attendance marked on laptop
→ Internet-independent local storage
→ Synchronization
→ Home PC receives event
→ WebSocket updates dashboard
→ HOD sees attendance
→ Student sees own attendance
→ Report generated
→ Audit trail recorded
```

## Final Implementation Principle

The system should be developed as a **local-first AI attendance
platform**:

**Laptop = AI + biometric data + attendance master**

**Home PC = web server + synchronized application data**

**Camera = replaceable RTSP/ONVIF input**

**SCRFD = detection**

**Face-quality model = quality validation**

**MiniFASNet = anti-spoofing, after model verification**

**AdaFace = recognition/embedding**

**FAISS = vector search**

**PostgreSQL = structured application data**

**FastAPI = backend**

**React + TypeScript = frontend**

**WebSocket = real-time updates**

**Synchronization queue = offline reliability**

**HOD = institutional administrator**

**Developer = unrestricted development mode**

**Student = restricted read-only user**
