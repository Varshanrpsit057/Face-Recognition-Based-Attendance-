# 02 --- Technical Requirements Guide

## 1. Recommended Technology Stack

  Layer                Recommended Technology
  -------------------- -------------------------------------------
  OS                   Windows on laptop and home PC
  Language             Python
  Face detection       SCRFD ONNX
  Face recognition     AdaFace ONNX
  Face quality         `cr_fiqa_l.onnx` after verification
  Anti-spoofing        `minifasnet_v2.onnx` after verification
  Vector search        FAISS
  Inference            ONNX Runtime with CUDA where compatible
  Video                OpenCV
  Backend              FastAPI
  Frontend             React + TypeScript
  API                  REST
  Real-time            WebSocket
  Database             PostgreSQL
  Local sync           Durable synchronization queue
  Cache/coordination   Redis if needed
  Camera               RTSP + ONVIF
  Deployment           Docker for home-PC server where practical
  Reports              XLSX/CSV/PDF
  Version control      Git

## 2. Laptop AI Requirements

The laptop has: - Intel Core i5-13450HX. - RTX 3050 Laptop GPU 6 GB. -
16 GB RAM. - Windows. - Approximately 477 GB storage shown in Windows.

The laptop should run the AI pipeline locally.

### GPU responsibilities

-   SCRFD inference.
-   AdaFace inference.
-   Face-quality inference if GPU-compatible.
-   Anti-spoofing inference if GPU-compatible.
-   Other optimized preprocessing/inference operations.

### CPU responsibilities

-   Camera connection.
-   Data orchestration.
-   Database operations.
-   Synchronization.
-   API client.
-   Logging.
-   Non-GPU preprocessing where appropriate.

## 3. Model Requirements

### SCRFD

File:

`scrfd_10g_bnkps.onnx`

Purpose: - Multi-face detection. - Bounding boxes. - Facial landmarks if
supported by the model output.

### AdaFace

Files:

`adaface_ir101.onnx`

`adaface_ir101.onnx.data`

Purpose: - Generate facial embeddings. - Handle recognition from
registered identities.

### Face Quality

File:

`cr_fiqa_l.onnx`

Purpose must be verified from the supplied README/model implementation
before final integration.

Expected role: - Reject low-quality face crops before expensive
recognition.

### Anti-Spoofing

File:

`minifasnet_v2.onnx`

Expected role: - Detect presentation attacks such as printed photographs
or screen replay.

The exact input/output contract must be verified before implementation.

## 4. Model Loading

The application should load models once at process startup rather than
loading them for every frame.

``` text
Application start
→ Load SCRFD
→ Load FIQA
→ Load MiniFASNet
→ Load AdaFace
→ Load FAISS
→ Warm up inference
→ Ready
```

## 5. Camera Abstraction

Do not implement:

``` python
if camera == "dahua":
    ...
elif camera == "hikvision":
    ...
```

Instead implement:

``` text
CameraManager
    ↓
CameraAdapter
    ↓
RTSP / ONVIF
```

A camera record should contain: - `camera_id` - `name` -
`manufacturer` - `model` - `ip_address` - `protocol` -
`rtsp_reference` - `onvif_enabled` - `classroom_id` - `status`

Credentials should be stored securely and never returned to ordinary
frontend clients.

## 6. Stream Strategy

The supplied Dahua reference camera supports multiple streams and up to
3840×2160. For AI processing, the system should use the lowest stream
that still provides sufficient face detail.

Recommended architecture:

``` text
Main stream
→ AI processing if required

Substream
→ Web preview / monitoring
```

The exact stream resolution and FPS must be benchmarked on the actual
classroom geometry.

## 7. AI Pipeline

``` text
RTSP Frame
    ↓
Frame sampling / decode
    ↓
SCRFD
    ↓
Face bounding box + landmarks
    ↓
Face crop/alignment
    ↓
FIQA
    ↓
Anti-spoof
    ↓
AdaFace
    ↓
Embedding normalization
    ↓
FAISS
    ↓
Top-K candidates
    ↓
Decision engine
    ↓
Attendance
```

## 8. FAISS Architecture

FAISS is the vector-search layer, not the primary student database.

PostgreSQL: - Student metadata. - Register number. - Academic data. -
Attendance. - Sessions. - Classes.

FAISS: - Face vectors. - Search index.

Each FAISS identity must map back to a stable student identifier.

Recommended mapping:

``` text
FAISS vector ID
→ internal student UUID
→ register number
```

## 9. Recognition Decision

Recommended logical decision:

``` text
Detection valid
AND quality valid
AND anti-spoof valid
AND FAISS similarity passes threshold
AND multi-frame confirmation passes
AND attendance session is active
→ eligible for attendance
```

Threshold values must be experimentally calibrated. Do not hard-code a
threshold such as 0.80 as a final accuracy claim.

## 10. Multi-Frame Confirmation

To reduce false positives, the system should optionally require repeated
recognition across consecutive frames.

Example configurable parameters: - Minimum confirmation frames. -
Maximum confirmation interval. - Duplicate attendance interval.

These values must be benchmarked.

## 11. Attendance Engine

The attendance engine should be separate from the recognition engine.

``` text
Recognition Engine
→ identity candidate

Attendance Engine
→ session validation
→ duplicate check
→ attendance decision
→ attendance event
```

This separation prevents recognition logic from becoming tightly coupled
to academic attendance rules.

## 12. Local Database

The laptop database should contain: - Students. - Profiles. - Face
samples. - Face enrollment metadata. - Attendance sessions. - Attendance
records. - Recognition events. - Sync queue. - Camera configuration. -
Academic structure.

The biometric tables should be logically separated from normal student
information.

## 13. Synchronization

Use an event-based synchronization design.

Laptop creates:

``` text
sync_event
event_id
entity_type
entity_id
operation
payload
created_at
sync_status
retry_count
```

The home PC acknowledges each event.

Possible states: - `PENDING` - `SENDING` - `SYNCED` - `FAILED` -
`CONFLICT`

For attendance, the laptop remains authoritative.

## 14. Idempotency

Every synchronization event must have a unique ID.

The home server must accept an event only once.

Example:

``` text
event_id = ATT-20260815-000001

First request:
INSERT

Repeated request:
Already exists → ACK
```

This prevents duplicate attendance.

## 15. WebSocket

WebSocket should be used for real-time website updates.

Examples: - New attendance event. - Dashboard counter update. - Camera
status. - Session started/stopped. - Recognition verification event. -
Synchronization status.

## 16. Authentication

Production authentication: - HOD account. - Student account. - Faculty
account if enabled. - Secure sessions/JWT. - Password hashing. -
Role-based authorization.

Developer: - Passwordless role-selection access only for development. -
Must be disabled/restricted before production deployment.

## 17. API Architecture

Suggested API groups:

``` text
/api/auth
/api/students
/api/students/{id}/face-dataset
/api/face-enrollment
/api/recognition
/api/attendance
/api/sessions
/api/classrooms
/api/cameras
/api/reports
/api/analytics
/api/xai
/api/notifications
/api/audit
/api/sync
/api/system
```

## 18. Network Architecture

``` text
Camera
→ PoE Ethernet
→ Laptop
```

and:

``` text
Laptop
→ Phone Wi-Fi
→ Internet
→ Home PC
```

The raw classroom video should not be sent to the home PC unless
specifically required.

Only required application events/data should be synchronized.

## 19. Home PC Server

Responsibilities: - FastAPI backend. - React static application hosting
or reverse proxy. - Synchronized application database. - WebSocket
service. - Reports. - Notifications. - HOD/student access. - Audit logs.

The home PC should not be required for live AI inference.

## 20. Backup

Backups should cover: - Student metadata. - Attendance records. -
Synchronization events. - Configuration. - Reports if required.

Biometric backups require an explicit privacy policy. The default
architecture should minimize copying raw face data away from the laptop.

## 21. Logging

Use structured logs for: - Camera connection. - Model loading. -
Inference errors. - Recognition events. - Attendance events. -
Synchronization. - API errors. - Authentication. - Administrative
actions.

Never log passwords or raw authentication credentials.

## 22. Performance Monitoring

Developer diagnostics should show: - GPU utilization. - GPU memory. -
CPU utilization. - RAM. - Camera FPS. - Detection FPS. - Recognition
FPS. - Average inference latency. - FAISS search latency. -
Synchronization latency. - Queue size.

## 23. Security

Use: - HTTPS where network access requires it. - Password hashing. -
Role-based authorization. - Secure cookies/tokens. - Rate limiting on
authentication. - Audit logs. - Camera credential protection. - Database
access controls. - Restricted biometric access.

## 24. Development and Production Separation

Development:

``` text
Windows Laptop
→ AI + local DB
→ Windows Home PC
→ local web server
```

Production should introduce: - Secure authentication. - HTTPS. -
Restricted Developer mode. - Backup. - Monitoring. - Database migration
strategy. - Secure network configuration.
