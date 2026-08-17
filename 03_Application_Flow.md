# 03 --- Application Flow

## 1. Overall System Flow

``` text
Role Selection
    ↓
Developer / HOD / Student
    ↓
Role-specific Dashboard
```

AI flow:

``` text
Camera
→ RTSP
→ SCRFD
→ Face Quality
→ Anti-Spoofing
→ AdaFace
→ FAISS
→ Register Number
→ Attendance Engine
→ Laptop Master DB
→ Sync Queue
→ Home PC
→ WebSocket
→ Web Clients
```

## 2. Developer Entry Flow

``` text
Application start
→ Select Developer
→ Direct access
→ Developer Dashboard
```

No password is required for the development-only Developer role.

Developer can access all features.

## 3. HOD Entry Flow

``` text
Application start
→ Select HOD
→ HOD authentication
→ HOD Dashboard
```

HOD authentication is separate from Developer mode.

## 4. Student Entry Flow

``` text
Application start
→ Select Student
→ Student authentication
→ Student Dashboard
```

Student can only access authorized personal information.

## 5. Student Profile Registration

``` text
Developer/HOD
→ Students
→ Add Student
→ Enter register number
→ Enter student details
→ Upload one profile photo
→ Validate
→ Save student profile
→ Student available for dataset registration
```

## 6. Face Dataset Registration

This is separate from student profile registration.

``` text
Developer/HOD
→ Students
→ Select student
→ Register number
→ Face Dataset
→ Choose Upload or Capture
```

### Capture workflow

``` text
Open laptop webcam
→ Capture face
→ SCRFD detection
→ Face quality
→ Accept/reject
→ Save valid sample
→ Repeat
```

### Upload workflow

``` text
Select student
→ Upload multiple photos
→ Validate each image
→ Detect face
→ Reject invalid images
→ Save accepted samples
```

## 7. Enrollment Processing

``` text
Accepted samples
→ Face crop
→ Alignment
→ AdaFace
→ Embedding
→ Normalize
→ Add/update FAISS
→ Store enrollment metadata
→ Enrollment complete
```

## 8. Camera Registration

``` text
HOD/Developer
→ Cameras
→ Add Camera
→ Select discovery or manual RTSP
→ Test connection
→ Assign classroom
→ Configure stream
→ Save
```

## 9. Camera Selection

``` text
Attendance Session
→ Select classroom
→ List assigned/available cameras
→ Select camera
→ Test stream
→ Start
```

The application must not assume one fixed camera.

## 10. Attendance Session

``` text
Faculty/Developer/HOD
→ Attendance Sessions
→ Select classroom
→ Select subject
→ Select session
→ Select camera
→ Start Attendance
```

Once started:

``` text
Camera
→ Laptop
→ AI pipeline
→ Recognition
→ Attendance engine
```

## 11. Recognition Logic

``` text
Frame
→ SCRFD
```

If no face:

``` text
Ignore frame
```

If face detected:

``` text
Face crop
→ Quality check
```

If poor quality:

``` text
Ignore / record quality event
```

If quality acceptable:

``` text
Anti-spoof
```

If spoof detected:

``` text
Reject
→ Security event
→ No attendance
```

If live:

``` text
AdaFace
→ Embedding
→ FAISS Top-K
→ Similarity decision
```

## 12. Attendance Decision

``` text
Identity candidate
        ↓
Similarity passes threshold?
        │
     ┌──┴──┐
     No    Yes
     ↓      ↓
Unknown   Session active?
              │
           ┌──┴──┐
          No    Yes
          ↓      ↓
        Ignore  Multi-frame confirmation
                     ↓
                 Confirmed?
                  ┌──┴──┐
                 No    Yes
                 ↓      ↓
              Continue  Duplicate?
                           │
                        ┌──┴──┐
                       Yes   No
                       ↓      ↓
                    Ignore   Mark Present
```

## 13. Attendance Record

Each attendance event should contain: - Event ID. - Student/register
number. - Session. - Classroom. - Subject. - Timestamp. - Status. -
Recognition metadata required for audit. - Source laptop. -
Synchronization status.

## 14. Offline Attendance

If internet is unavailable:

``` text
Recognition
→ Attendance
→ Local DB
→ Sync queue
```

No attendance should be lost.

When connection returns:

``` text
Sync queue
→ Home PC
→ Idempotent insert
→ Acknowledge
→ Mark synced
```

## 15. Website Update Flow

``` text
Laptop marks attendance
→ Sync API
→ Home PC database
→ WebSocket event
→ Dashboard update
```

The page should update without requiring manual refresh.

## 16. Student Attendance Flow

``` text
Student logs in
→ Dashboard
→ Today's attendance
→ Overall attendance
→ Subject-wise attendance
→ History
```

Only that student's authorized records are returned.

## 17. Attendance Correction Flow

``` text
Student
→ Attendance
→ Request correction
→ Enter reason
→ Submit
→ Faculty/HOD review
→ Approve/reject
→ Create correction event
→ Audit log
→ Synchronize
```

Direct student editing is prohibited.

## 18. XAI Flow

Recommended:

``` text
Recognition event
→ Decision
→ XAI requested
→ Explanation generated
→ Store explanation metadata
→ Display to authorized user
```

XAI can be generated on demand for selected recognition events rather
than every frame.

## 19. Report Flow

``` text
HOD/Faculty
→ Reports
→ Select date/class/subject/student
→ Query synchronized database
→ Generate report
→ Export XLSX/CSV/PDF
```

## 20. Camera Failure Flow

``` text
Camera stream fails
→ Camera status OFFLINE
→ Dashboard warning
→ Log event
→ Faculty notified
```

If multiple cameras are configured, an optional backup-camera policy can
be implemented.

## 21. Synchronization Failure Flow

``` text
Sync request fails
→ Keep event in local queue
→ Retry with backoff
→ Do not duplicate
→ Continue attendance
```

## 22. System Startup Flow

Laptop:

``` text
Start application
→ Check GPU
→ Load SCRFD
→ Load quality model
→ Load anti-spoof model
→ Load AdaFace
→ Load FAISS
→ Check local DB
→ Check camera configuration
→ Check sync service
→ Ready
```

Home PC:

``` text
Start server
→ Load configuration
→ Connect DB
→ Start API
→ Start WebSocket
→ Start sync receiver
→ Start web frontend
→ Ready
```
