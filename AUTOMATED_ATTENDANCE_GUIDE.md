# Automated Attendance System - User Guide

## Overview
The system now includes three major automated features:
1. **Automated Photo Capture** - Automatically captures photos during sessions
2. **Location Tracking** - Records GPS coordinates with each attendance entry
3. **Auto-marking at 0.5 Confidence** - Automatically marks attendance at 50% confidence threshold

---

## Feature 1: Automated Attendance Capture

### How It Works
When you create a session with automated attendance enabled:
- The system starts capturing photos automatically when the session begins
- Photos are captured at your specified interval (e.g., every 30 seconds)
- Face recognition runs on each captured photo
- Students are automatically marked present if detected with ≥0.5 confidence
- All captures are saved with timestamps

### Using Automated Attendance

#### Option A: Via Web Interface
1. Navigate to **Create Session** page
2. Fill in session details (name, course, date, times)
3. Check **"Enable Automated Attendance"** checkbox
4. Set **Capture Interval** (default: 30 seconds, range: 10-300)
5. Click **Create Session**

The system will:
- Start capturing automatically at session start time
- Continue until session end time
- Mark attendance for all recognized students

#### Option B: Via Command Line
```bash
# Start automated capture for session ID 1
python manage.py start_auto_attendance 1

# Custom interval (every 60 seconds)
python manage.py start_auto_attendance 1 --interval 60

# Different camera (camera index 1)
python manage.py start_auto_attendance 1 --camera 1

# Custom interval and camera
python manage.py start_auto_attendance 1 --interval 45 --camera 0
```

**To stop:** Press `Ctrl+C`

### Important Notes
- Ensure your camera is connected and accessible
- The system uses camera index 0 by default (built-in webcam)
- All captures are saved in `media/attendance_images/`
- Students won't be marked twice (duplicate prevention)

---

## Feature 2: Location Tracking

### How It Works
- GPS coordinates are captured when marking attendance
- Location name is resolved via reverse geocoding
- Location data is stored with each attendance record
- Location appears in all reports and exports

### Capturing Location

#### Manual Attendance Marking
1. Go to **Mark Attendance** page
2. Select session and upload image
3. Click **"📍 Get Current Location"** button
4. Browser will request location permission
5. Location will be auto-filled
6. Submit the form

The system captures:
- **Latitude** (e.g., 37.7749)
- **Longitude** (e.g., -122.4194)
- **Location Name** (e.g., "Main Campus Building, University...")

#### Automated Capture
Location can be programmatically added:
```python
from attendance.utils.automated_attendance import start_automated_attendance

location_data = {
    'latitude': 37.7749,
    'longitude': -122.4194,
    'location_name': 'Main Classroom',
    'device_id': 'Camera-1'
}

start_automated_attendance(
    session_id=1,
    location_data=location_data
)
```

### Viewing Location Data
- **Attendance Records**: Location column shows GPS pin with name
- **Hover**: See full coordinates (Lat/Lon)
- **Reports**: Location included in all exports

---

## Feature 3: 0.5 Confidence Threshold

### What Changed
- **Previous**: Default threshold was 0.6 (60%)
- **Current**: Default threshold is 0.5 (50%)
- **Auto-marking**: Students automatically marked at ≥0.5 confidence

### Why 0.5?
- More lenient recognition (fewer false negatives)
- Better for varying lighting conditions
- Captures students at different angles
- Reduces manual intervention needed

### Confidence Levels
- **0.8 - 1.0**: Very high confidence (ideal lighting, direct face)
- **0.6 - 0.8**: High confidence (good quality recognition)
- **0.5 - 0.6**: Moderate confidence (acceptable for auto-marking)
- **< 0.5**: Low confidence (not marked automatically)

### Adjusting Threshold
You can still adjust the threshold when manually marking:
1. Go to **Mark Attendance**
2. Change **Confidence Threshold** field (0.0 - 1.0)
3. Lower = more lenient, Higher = more strict

---

## Complete Workflow Example

### Scenario: Morning Class Session

**Step 1: Create Session**
- Session Name: "CS101 - Lecture 5"
- Course: "Computer Science 101"
- Date: Today
- Time: 9:00 AM - 10:30 AM
- ✅ Enable Automated Attendance
- Interval: 30 seconds

**Step 2: System Behavior**
- 9:00 AM: First capture taken
- 9:00:30 AM: Second capture
- 9:01:00 AM: Third capture
- ... continues every 30 seconds until 10:30 AM

**Step 3: For Each Capture**
1. Photo captured from camera
2. Face detection runs on image
3. Face recognition identifies students
4. Students with ≥0.5 confidence marked present
5. Location data saved (if configured)
6. Image saved with timestamp

**Step 4: Results**
- All present students automatically marked
- Confidence scores recorded
- Capture timestamps saved
- Location data included
- No duplicate entries

---

## Database Schema Updates

### AttendanceRecord Model
New fields added:
```python
latitude = FloatField()           # GPS latitude
longitude = FloatField()          # GPS longitude  
location_name = CharField()       # Human-readable location
device_id = CharField()           # Camera/device identifier
photo_captured_at = DateTimeField()  # Exact capture time
```

---

## API Usage (Programmatic)

### Start Automated Capture
```python
from attendance.utils.automated_attendance import AutomatedAttendanceCapture

# Create instance
capture = AutomatedAttendanceCapture(
    session_id=1,
    camera_index=0,
    capture_interval=30
)

# Start in background
capture.start()

# Stop when needed
capture.stop()
```

### With Location
```python
location_data = {
    'latitude': 12.9716,
    'longitude': 77.5946,
    'location_name': 'Bangalore Campus, Room 101',
    'device_id': 'RPI-Camera-1'
}

capture = AutomatedAttendanceCapture(session_id=1)
capture.start(location_data=location_data)
```

---

## Troubleshooting

### Camera Not Working
- Check camera is connected and permissions granted
- Try different camera index: `--camera 1`
- Verify camera works: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

### No Faces Detected
- Ensure good lighting
- Students should face camera
- Adjust camera position/angle
- Check camera resolution

### Low Confidence Scores
- Improve lighting conditions
- Ensure students registered with quality images
- Re-train model with more images
- Consider using higher quality camera

### Location Not Capturing
- Enable location permissions in browser
- Use HTTPS (required for geolocation API)
- Check internet connection (for reverse geocoding)

---

## Best Practices

1. **Session Setup**
   - Set appropriate capture intervals (30-60 seconds recommended)
   - Ensure session times are accurate
   - Test camera before session starts

2. **Camera Placement**
   - Position to capture most students
   - Good lighting, avoid backlighting
   - Stable mount (no movement)

3. **Training Data**
   - Use 5-10 images per student
   - Vary angles and lighting
   - Update regularly for accuracy

4. **Monitoring**
   - Check capture logs during session
   - Review attendance records after
   - Verify location data accuracy

---

## Security & Privacy

- All images stored locally in `media/attendance_images/`
- GPS coordinates only captured with explicit action
- Location services require user permission
- Confidence scores logged for audit trail
- No third-party data sharing

---

## Future Enhancements

Potential additions:
- Email notifications when attendance marked
- Real-time dashboard for active sessions
- Multi-camera support
- Mobile app integration
- Attendance analytics and insights

---

## Support

For issues or questions:
1. Check this guide
2. Review Django logs: `tail -f logs/django.log`
3. Check camera logs in automated capture output
4. Verify model is trained and active

