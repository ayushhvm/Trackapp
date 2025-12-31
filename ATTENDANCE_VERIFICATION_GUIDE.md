# Multi-Capture Attendance Verification System

## Overview

The system now implements **strict attendance verification** to prevent students from being marked present if they only appear briefly during a session. Students must be present throughout the session to be marked as present.

## How It Works

### 1. Continuous Photo Capture

During an automated attendance session, the system:

- Captures photos at regular intervals (default: 30 seconds)
- Each capture is numbered sequentially (Capture #1, #2, #3, etc.)
- Detects and records all students in each photo
- Stores location data with each capture

### 2. Student Detection Tracking

For each capture, the system creates a `StudentCapture` record that tracks:

- Which student was detected
- Confidence score of the detection
- Bounding box coordinates
- Timestamp of detection

### 3. Attendance Verification Criteria

At the end of the session, attendance is verified based on THREE requirements:

#### ✅ Requirement 1: Present at START

Student must appear in at least one of the **first 20% of captures**

- Example: If there are 10 captures, must be in at least one of captures #1-2

#### ✅ Requirement 2: Present at END

Student must appear in at least one of the **last 20% of captures**

- Example: If there are 10 captures, must be in at least one of captures #9-10

#### ✅ Requirement 3: Majority Presence

Student must appear in **more than 50% of total captures**

- Example: If there are 10 captures, must be in at least 6 captures

### 4. Verification Results

**✓ MARKED PRESENT** if ALL three requirements are met:

- Present in start captures ✓
- Present in end captures ✓
- Present in majority of captures ✓
- Status: `present`
- Verification notes show the attendance details

**✗ MARKED ABSENT** if ANY requirement fails:

- Missing from start captures ✗
- Missing from end captures ✗
- Present in less than 50% of captures ✗
- Status: `absent`
- Verification notes explain why attendance was rejected

## Database Models

### CaptureRecord

Tracks individual photo captures during a session:

```python
- session: Link to AttendanceSession
- capture_number: Sequential number (1, 2, 3...)
- image_path: Path to captured image
- captured_at: Timestamp
- latitude, longitude, location_name: GPS data
- is_processed: Whether faces have been detected
- faces_detected: Number of faces found
```

### StudentCapture

Tracks which students appeared in each capture:

```python
- capture: Link to CaptureRecord
- student: Link to Student
- confidence_score: Face recognition confidence
- bbox_x, bbox_y, bbox_w, bbox_h: Face location in image
- detected_at: Timestamp
```

### AttendanceRecord (Updated)

Now includes verification fields:

```python
- status: 'pending' | 'present' | 'absent' | 'late'
- is_verified: Boolean
- total_captures: Number of captures student appeared in
- present_in_start: Boolean
- present_in_end: Boolean
- verification_notes: Detailed explanation
```

## Usage

### Automated Sessions

When you create a session with automated attendance enabled:

1. **During the session:**

   - Photos are captured automatically
   - Students are detected and tracked
   - Records remain in `pending` status

2. **At session end:**

   - Automatic verification runs
   - Each student's presence is analyzed
   - Final attendance status is determined

3. **Console output shows:**

   ```
   ✓ S001 - John Doe: PRESENT
     └─ 8/10 captures (80.0%)
     └─ Start: ✓, End: ✓

   ✗ S002 - Jane Smith: ABSENT
     └─ 3/10 captures (30.0%)
     └─ Start: ✓, End: ✗
     └─ Reason: not present at end, only 3/10 captures
   ```

### Manual Verification Command

Teachers can manually verify attendance for any session:

```bash
python manage.py verify_attendance <session_id>
```

Example:

```bash
python manage.py verify_attendance 6
```

This will:

- Analyze all captures for that session
- Re-verify all pending attendance records
- Update attendance status based on verification rules
- Display detailed results

## Viewing Results

### In Admin Panel

Navigate to:

- **Attendance Records**: See `is_verified`, `total_captures`, verification notes
- **Capture Records**: View all captures for a session
- **Student Captures**: See which students appeared in each capture

### In Teacher Dashboard

- View attendance records with verification status
- See verification notes explaining why each student was marked present/absent
- Filter by verified/unverified attendance

## Configuration

### Adjusting Verification Thresholds

Edit `attendance/utils/automated_attendance.py` in the `verify_attendance()` method:

```python
# Default: First and last 20% of captures
start_range = max(1, int(total_captures * 0.2))  # Change 0.2 to adjust
end_range = max(1, int(total_captures * 0.2))    # Change 0.2 to adjust

# Default: More than 50% of captures
majority_threshold = total_captures * 0.5  # Change 0.5 to adjust
```

**Examples:**

- `0.2` = 20% (stricter)
- `0.1` = 10% (more lenient)
- `0.5` = 50% majority
- `0.6` = 60% majority (stricter)

## Benefits

1. **Prevents Gaming**: Students can't just show up briefly and leave
2. **Ensures Participation**: Students must stay for the entire session
3. **Fair Assessment**: Only students genuinely present get marked
4. **Detailed Tracking**: Complete record of when each student was detected
5. **Transparency**: Clear reasons for attendance decisions
6. **Location Tracking**: GPS data stored with each capture

## Example Scenarios

### Scenario 1: Student Present Entire Class ✓

- Session has 10 captures (every 30 seconds over 5 minutes)
- Student appears in captures: #1, #2, #3, #4, #5, #6, #7, #8, #9, #10
- Result: **PRESENT** (100% attendance)

### Scenario 2: Student Leaves Early ✗

- Session has 10 captures
- Student appears in captures: #1, #2, #3, #4, #5
- Result: **ABSENT** (not present at end, only 50% attendance)

### Scenario 3: Student Arrives Late ✗

- Session has 10 captures
- Student appears in captures: #6, #7, #8, #9, #10
- Result: **ABSENT** (not present at start, only 50% attendance)

### Scenario 4: Student Leaves and Returns ✓

- Session has 10 captures
- Student appears in captures: #1, #2, #3, #7, #8, #9, #10
- Result: **PRESENT** (in start ✓, in end ✓, 7/10 = 70% ✓)

### Scenario 5: Student Steps Out Briefly ✓

- Session has 10 captures
- Student appears in captures: #1, #2, #3, #4, #5, #6, #8, #9, #10
- Result: **PRESENT** (in start ✓, in end ✓, 9/10 = 90% ✓)

## Troubleshooting

**Issue**: All students marked absent despite being present

- **Solution**: Check if session has enough captures (minimum 3-5 recommended)

**Issue**: Students marked present who left early

- **Solution**: Increase `end_range` percentage or `majority_threshold`

**Issue**: Legitimate students marked absent

- **Solution**: Decrease thresholds or check face recognition confidence scores

## Migration from Old System

Existing attendance records without verification will have:

- `is_verified = False`
- `total_captures = 0`
- `status = 'present'` or `'absent'` (unchanged)

Only new automated sessions will use the verification system.
