# 05 --- Backend Schema and API Specification

## 1. Architecture

``` text
Laptop
├── Local PostgreSQL
├── AI Engine
├── FAISS
├── Local Sync Queue
└── Sync Client

Home PC
├── FastAPI
├── PostgreSQL replica/application DB
├── WebSocket
└── React frontend
```

The laptop is authoritative for AI-generated attendance and biometric
data.

## 2. Identity Strategy

### Internal ID

Every entity should have a UUID or equivalent internal identifier.

### Register Number

Student-facing and attendance-facing identifier.

Example:

``` text
student_uuid = internal technical ID
register_number = 23AD001
```

FAISS mappings should resolve through internal IDs and then register
number.

## 3. Main Tables

### users

Fields: - id - username/email - password_hash - role_id - status -
created_at - updated_at - last_login

### roles

Fields: - id - name

Roles: - developer - hod - faculty - student

### students

Fields: - id - register_number - full_name - department_id -
program_id - year - semester - section - email - phone -
profile_photo_path/reference - status - created_at - updated_at

Unique: - register_number

### face_samples

Fields: - id - student_id - file_reference - capture_method -
quality_score - validation_status - created_at

### face_embeddings

Fields: - id - student_id - face_sample_id - model_name -
model_version - vector_dimension - faiss_vector_id - created_at - status

Raw vectors may be stored locally only if required by the
implementation. FAISS is the search index.

### faiss_indexes

Fields: - id - index_name - index_type - vector_dimension -
vector_count - model_name - version - path/reference - created_at -
updated_at - status

### departments

Fields: - id - name - code

### programs

Fields: - id - department_id - name - code

### faculty

Fields: - id - user_id - employee_id - name - department_id - status

### subjects

Fields: - id - code - name - department_id - semester

### classrooms

Fields: - id - name - building - floor - capacity - status

### cameras

Fields: - id - camera_code - name - manufacturer - model - ip_address -
protocol - classroom_id - onvif_enabled - status - created_at -
updated_at

Credentials must be stored securely and not returned to clients.

### camera_streams

Fields: - id - camera_id - stream_type - rtsp_reference - resolution -
fps - codec - enabled

### timetables

Fields: - id - classroom_id - subject_id - faculty_id - section - day -
start_time - end_time - status

### attendance_sessions

Fields: - id - session_code - classroom_id - subject_id - faculty_id -
camera_id - start_time - end_time - status - created_at

### attendance_records

Fields: - id - attendance_event_id - session_id - student_id -
register_number - timestamp - status - source - recognition_confidence -
similarity_score - synchronized_at - created_at

`attendance_event_id` must be unique.

### recognition_events

Fields: - id - event_id - session_id - camera_id - timestamp -
detected_face_reference - predicted_student_id - confidence -
similarity - decision - processing_latency - spoof_result -
quality_result

Raw frames should not be retained indefinitely.

### recognition_candidates

Fields: - id - recognition_event_id - student_id - rank - similarity -
distance

### xai_explanations

Fields: - id - recognition_event_id - method - result_reference -
summary - created_at

### notifications

Fields: - id - user_id - type - title - message - read_at - created_at

### attendance_correction_requests

Fields: - id - attendance_record_id - requested_by - reason - status -
reviewed_by - reviewed_at - review_note - created_at

### audit_logs

Fields: - id - user_id - role - action - entity_type - entity_id -
description - timestamp - status - client_reference

### sync_events

Fields: - id - event_id - source_device - entity_type - entity_id -
operation - payload - status - retry_count - created_at - synced_at

## 4. Relationships

``` text
Department
→ Programs
→ Students

Student
→ Face Samples
→ Face Embeddings
→ Attendance Records
→ Recognition Events

Classroom
→ Cameras
→ Timetable
→ Attendance Sessions

Attendance Session
→ Attendance Records
→ Recognition Events

Recognition Event
→ Candidates
→ XAI Explanation
```

## 5. API Groups

### Authentication

``` http
POST /api/auth/hod/login
POST /api/auth/student/login
POST /api/auth/logout
GET  /api/auth/me
```

Developer mode is local development access and should not be exposed as
an unsecured public endpoint.

### Students

``` http
GET    /api/students
POST   /api/students
GET    /api/students/{id}
PATCH  /api/students/{id}
DELETE /api/students/{id}
```

### Face Dataset

``` http
GET  /api/students/{id}/face-dataset
POST /api/students/{id}/face-dataset
POST /api/students/{id}/face-dataset/capture
POST /api/students/{id}/face-dataset/validate
DELETE /api/face-samples/{id}
POST /api/face-enrollment/{student_id}/generate
```

### FAISS

``` http
GET  /api/faiss/status
POST /api/faiss/rebuild
POST /api/faiss/validate
POST /api/faiss/synchronize
```

### Cameras

``` http
GET  /api/cameras
POST /api/cameras
GET  /api/cameras/{id}
PATCH /api/cameras/{id}
POST /api/cameras/{id}/test
POST /api/cameras/{id}/discover
```

### Attendance

``` http
GET  /api/attendance
GET  /api/attendance/{id}
POST /api/attendance/sessions
POST /api/attendance/sessions/{id}/start
POST /api/attendance/sessions/{id}/stop
POST /api/attendance/{id}/correction
```

### Reports

``` http
GET /api/reports/daily
GET /api/reports/monthly
GET /api/reports/student/{id}
GET /api/reports/subject/{id}
```

### XAI

``` http
GET  /api/xai/{recognition_event_id}
POST /api/xai/{recognition_event_id}/generate
```

### Synchronization

``` http
POST /api/sync/events
GET  /api/sync/status
POST /api/sync/ack
```

## 6. Synchronization Contract

Laptop sends:

``` json
{
  "event_id": "unique-event-id",
  "entity_type": "attendance",
  "entity_id": "local-id",
  "operation": "CREATE",
  "created_at": "timestamp",
  "payload": {}
}
```

Home PC response:

``` json
{
  "event_id": "unique-event-id",
  "status": "SYNCED"
}
```

If already present:

``` json
{
  "event_id": "unique-event-id",
  "status": "ALREADY_EXISTS"
}
```

## 7. Attendance Authority

Laptop: - Creates original AI attendance event. - Owns biometric
decision. - Maintains local master record.

Home PC: - Receives synchronized event. - Displays it. - Does not
silently overwrite it.

HOD correction: - Creates an auditable correction event.

## 8. WebSocket Events

Suggested events:

``` text
attendance.created
attendance.corrected
session.started
session.stopped
camera.status_changed
recognition.detected
recognition.verified
sync.completed
sync.failed
notification.created
```

## 9. Access Control

### Developer

Full local development access.

### HOD

Institutional administration.

### Faculty

Assigned attendance/classes if enabled.

### Student

Own records only.

Every protected API must enforce role permissions on the backend, not
merely hide UI controls.

## 10. Database Rules

-   Register number unique.
-   Attendance event ID unique.
-   Foreign keys enforced.
-   Soft deletion preferred for institutional records where audit
    history is required.
-   Attendance records should not be physically deleted through ordinary
    UI.
-   Corrections should be auditable.
-   Face samples should have explicit lifecycle states.

## 11. Privacy

Raw face images and embeddings should remain on the laptop by default.

The web server should receive only the minimum data needed for: -
Attendance. - Student portal. - Reports. - Notifications. - Audit.

## 12. Sync Security

Use: - HTTPS/TLS. - Device authentication. - Signed/validated
requests. - API tokens or device credentials. - Replay protection
through unique event IDs. - Request timestamps where appropriate.

## 13. API Error Handling

Use consistent responses:

``` json
{
  "success": false,
  "error_code": "CAMERA_OFFLINE",
  "message": "Camera is unavailable"
}
```

Never return internal stack traces to ordinary users.
