# 01 --- Project Requirements Document (PRD)

## Project Title

**Face Recognition Based Attendance System Using SCRFD, AdaFace, FAISS
and Explainable AI**

## 1. Purpose

This project is an AI-powered classroom attendance system. Face
recognition is performed locally on a Windows laptop connected to
classroom IP cameras. The laptop is the master machine for student
biometric data and attendance processing. A Windows home PC hosts the
web application and maintains a synchronized replica of the application
data so that HODs, faculty and students can access the required
information through a browser.

## 2. Core Objectives

-   Detect multiple faces from classroom IP-camera streams.
-   Recognize registered students using AdaFace facial embeddings.
-   Use FAISS for fast similarity search.
-   Use SCRFD for face detection.
-   Use face-quality assessment before recognition.
-   Use MiniFASNet-based anti-spoofing if the supplied model is
    confirmed for this purpose.
-   Mark attendance locally on the laptop.
-   Use the student register number as the institution-facing identity.
-   Synchronize attendance from laptop to the home PC.
-   Continue attendance operation when the internet connection is
    unavailable.
-   Automatically synchronize pending records when connectivity returns.
-   Provide separate student-profile registration and face-dataset
    registration.
-   Provide camera-independent IP-camera support through RTSP/ONVIF
    rather than hard-coding one camera brand.
-   Provide role-based access for Developer, HOD and Student; faculty
    access can be added according to the final permission matrix.
-   Provide dashboards, reports, audit logs, XAI analysis and system
    monitoring.

## 3. Current Hardware Architecture

### AI/Attendance Laptop

-   Operating system: Windows
-   Processor: 13th Gen Intel Core i5-13450HX, 2.40 GHz
-   RAM: 16 GB, 4800 MT/s
-   GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 6 GB
-   Storage shown in Windows: 477 GB
-   Internet: phone Wi-Fi
-   Camera connection: Ethernet through PoE

### Home PC

-   Operating system: Windows
-   Stable Wi-Fi
-   Hosts the backend/web application and synchronized application data
-   Does not perform normal classroom face-recognition inference

## 4. Camera Requirement

The application must not depend on a single camera manufacturer or
model.

The camera layer must support compatible IP cameras using standard
interfaces such as:

-   RTSP
-   ONVIF
-   Network/IP configuration

The currently supplied Dahua reference camera is DH-IPC-HFW2831T-ZS-S2.
Its supplied datasheet states 8 MP resolution, 3840×2160 maximum
resolution, three streams, RTSP, ONVIF Profile S/G/T, RJ-45 networking
and PoE 802.3af. These capabilities make it a suitable reference device
for the generic camera architecture.

## 5. User Roles

### Developer

Developer mode is intended for development and system administration
during project development.

Developer access: - No password in the intended development-only
role-selection interface. - Full access to all system features. - AI
model status and configuration. - Student database. - Face dataset
registration. - FAISS management. - Camera management. - Attendance. -
XAI. - Synchronization. - Database diagnostics. - Logs. - Developer
tools.

This passwordless Developer mode must not be exposed as a production
security mechanism.

### HOD

HOD is the institutional administrator.

Expected permissions: - Student creation/editing/deactivation. - Student
profile management. - Face-dataset management. - Faculty management. -
Classroom management. - Camera management. - Attendance review and
authorized correction. - Reports. - Analytics. - Audit logs. -
Notifications. - System administration according to the final permission
matrix.

HOD authentication should be implemented separately from Developer mode.

### Student

Students have restricted access to their own information.

Students can view: - Own profile. - Register number. - Today's
attendance. - Overall attendance percentage. - Subject-wise
attendance. - Attendance history. - Current classes. - Notifications. -
Attendance correction request status.

Students cannot: - View other students. - Access face datasets. - Access
embeddings or FAISS. - Configure cameras. - Modify attendance
directly. - Modify institutional data. - Access AI configuration.

## 6. Student Identity

The institution-facing student identifier is the **Register Number**.

Example:

`23AD001 → Varshan.C`

An internal UUID may still be used as the database primary key, but
register number is used for operational identity and attendance display.

## 7. Student Registration

Student profile registration and biometric dataset registration are
separate operations.

### Student Profile

Required information may include: - Register number - Full name -
Department - Program - Year - Semester - Section - Email - Phone - One
profile photo - Status

The profile photo is not the complete recognition dataset.

### Face Dataset Registration

Workflow:

``` text
Select existing student
→ Select by register number
→ Face Dataset Registration
→ Capture from laptop webcam/camera OR upload photos
→ Validate images
→ Detect face
→ Check quality
→ Generate AdaFace embeddings
→ Update FAISS
→ Enrollment complete
```

The dataset should support multiple images per student and multiple face
angles.

## 8. AI Pipeline

The intended processing pipeline is:

``` text
IP Camera
→ SCRFD
→ Face Crop / Alignment
→ Face Quality Assessment
→ Anti-Spoofing
→ AdaFace
→ Face Embedding
→ FAISS Similarity Search
→ Identity / Register Number
→ Attendance Decision
→ Local Laptop Database
→ Synchronization
→ Home PC Web Application
```

The exact role/order of `cr_fiqa_l.onnx` and `minifasnet_v2.onnx` must
be verified against their README/model implementation before the
production pipeline is frozen.

## 9. Supplied Model Files

The current model directory contains:

-   `scrfd_10g_bnkps.onnx`
-   `adaface_ir101.onnx`
-   `adaface_ir101.onnx.data`
-   `cr_fiqa_l.onnx`
-   `minifasnet_v2.onnx`
-   `README`

The intended roles are: - SCRFD: face detection. - AdaFace IR101: face
recognition/embedding. - `cr_fiqa_l.onnx`: presumed face-quality model;
verify. - `minifasnet_v2.onnx`: presumed anti-spoofing model; verify.

## 10. Attendance Requirements

Attendance is marked on the laptop.

Recommended logic:

``` text
Camera frame
→ Detect face
→ Quality check
→ Anti-spoofing
→ Recognition
→ Similarity threshold
→ Multi-frame confirmation
→ Check session
→ Check duplicate attendance
→ Mark present
```

A student should only be marked once in a session unless an authorized
correction changes the record.

## 11. Offline Requirement

The laptop must continue marking attendance if the connection to the
home PC or internet fails.

Offline mode:

``` text
AI inference
→ Local attendance database
→ Synchronization queue
```

When connectivity returns:

``` text
Synchronization queue
→ Home PC
→ Acknowledge event
→ Mark event synchronized
```

## 12. Synchronization and Conflict Resolution

The laptop is the attendance master.

Every attendance event receives a globally unique `attendance_event_id`.

If the home PC receives an event whose ID already exists: - Do not
create a duplicate. - Acknowledge the event as already synchronized.

The home PC is a synchronized replica for web access, not the authority
for the original AI attendance event.

Authorized attendance corrections must be represented as separate
correction/audit events rather than silently overwriting the original AI
event.

## 13. Data Location

### Laptop

Stores: - Student master data required for AI operation. - Face
datasets. - Face embeddings. - FAISS index. - Attendance master
records. - AI models. - Local synchronization queue. - Recognition
events required for local operation.

### Home PC

Stores synchronized web/application data required for: - HOD
dashboard. - Faculty dashboard. - Student portal. - Reports. -
Notifications. - Audit logs. - Web access.

Raw biometric data should not be copied to the home PC unless explicitly
required and approved.

## 14. Camera Management

The system must provide: - Camera ID. - Camera name. - Manufacturer. -
Model. - IP address. - Protocol. - RTSP endpoint/reference. - ONVIF
discovery/configuration where available. - Classroom assignment. -
Status. - Stream configuration. - Health monitoring.

A classroom can be configured with one or more cameras if required.

## 15. Dashboard Requirements

The dashboard should include: - Total students. - Present today. -
Absent today. - Attendance rate. - Active classrooms. - Active
cameras. - Recognition success. - Attendance trend. - AI pipeline. -
System health. - Today's classes. - Recent activity. - Recognition
summary.

The supplied dashboard image is a visual reference and uses sample/demo
values.

## 16. XAI Requirements

XAI should explain recognition decisions rather than becoming an
additional recognition stage.

Recommended architecture:

``` text
SCRFD
→ AdaFace
→ FAISS
→ Decision
→ Attendance

Decision
→ XAI explanation
```

Potential explanation data: - Face crop. - Detection bounding box. -
Recognition score. - FAISS similarity/distance. - Top-K candidates. -
Decision threshold. - Decision result. - XAI visualization.

XAI generation may be limited to selected events to avoid unnecessary
computation.

## 17. Attendance Correction

Recommended workflow:

``` text
Student
→ Correction request
→ Faculty/HOD review
→ Approve or reject
→ Correction event
→ Audit log
→ Synchronization
```

Students should not directly modify attendance.

## 18. Security Requirements

The system should include: - Role-based access control. - Secure
authentication for HOD/student accounts. - Password hashing. -
Session/token protection. - HTTPS for network access where applicable. -
Audit logging. - Restricted biometric-data access. - Data
retention/deletion controls. - Secure synchronization. - No unnecessary
exposure of embeddings. - No direct student access to biometric
infrastructure.

## 19. Non-Functional Requirements

### Reliability

Attendance must continue locally if the web server is temporarily
unavailable.

### Performance

AI processing must be optimized for the RTX 3050 6 GB laptop GPU.

### Scalability

The web architecture should support multiple classrooms and multiple
cameras even though the prototype uses a single laptop.

### Maintainability

Camera models, thresholds and AI models must be configurable rather than
hard-coded.

### Auditability

Attendance changes must be traceable.

### Privacy

Biometric data should remain on the laptop by default.

## 20. Success Criteria

The system is considered functionally complete when: - A student can be
registered. - A profile photo can be stored. - A separate face dataset
can be registered. - Embeddings can be generated. - FAISS can search
identities. - An IP camera can be selected/configured. - Attendance can
be processed locally. - Attendance survives internet failure. - Pending
attendance synchronizes automatically. - HOD can manage institutional
data. - Students can see their own attendance. - Unauthorized users
cannot modify protected records. - Reports can be generated. -
Recognition events can be audited.
