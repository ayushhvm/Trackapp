# Raspberry Pi Camera Integration

This directory contains scripts and configuration for integrating Raspberry Pi camera module with the Face Recognition Attendance System.

## Setup Instructions

### 1. Hardware Requirements

- Raspberry Pi (3 or newer recommended)
- Raspberry Pi Camera Module (v1 or v2)
- Camera module connected to Raspberry Pi CSI port

### 2. Software Installation

On your Raspberry Pi, install the required Python packages:

```bash
# Install system dependencies (if needed)
sudo apt update
sudo apt install -y python3-picamera2 python3-pip

# Install Python packages
pip3 install -r requirements.txt
```

### 3. Configuration

Edit `config.json` with your settings:

```json
{
    "api_url": "http://your-server-ip:8000/api/mark-attendance/",
    "session_id": 1,
    "location_name": "Room 101",
    "threshold": 0.6,
    "device_id": "RPI_001",
    "save_local_copy": true,
    "output_path": "captured_photo.jpg"
}
```

**Configuration Options:**
- `api_url`: Full URL to the Django API endpoint
- `session_id`: ID of the attendance session (get from Django admin or web interface)
- `location_name`: Name of the location (e.g., "Room 101", "Lab A")
- `threshold`: Face recognition confidence threshold (0.0 to 1.0)
- `device_id`: Unique identifier for this Raspberry Pi
- `save_local_copy`: Whether to save photos locally on the Pi
- `output_path`: Path where captured photos are saved

### 4. Usage

Run the script to capture a photo and send it to the API:

```bash
python3 capture_and_send.py
```

The script will:
1. Capture a photo using the camera module
2. Send it to the Django API
3. Display recognition results
4. Save a local copy (if configured)

### 5. Network Configuration

Ensure your Raspberry Pi can reach the Django server:
- If Django is on same network: Use local IP (e.g., `http://192.168.1.100:8000`)
- If Django is remote: Use public IP or domain name
- For production: Configure firewall rules and use HTTPS

### 6. Testing Without Camera (Development)

If you're testing without a Raspberry Pi camera:
1. The script will attempt to use an existing image file if `picamera2` is not available
2. Place a test image named `captured_photo.jpg` in the script directory
3. Run the script - it will use the existing image

### 7. Automation (Optional)

To run automatically at intervals, use cron:

```bash
# Edit crontab
crontab -e

# Add line to run every 5 minutes (example)
*/5 * * * * cd /path/to/raspberry_pi && /usr/bin/python3 capture_and_send.py >> /var/log/attendance.log 2>&1
```

## Troubleshooting

### Camera Not Detected
- Check camera module connection
- Enable camera in `raspi-config`: `sudo raspi-config` → Interface Options → Camera → Enable
- Reboot after enabling camera

### API Connection Failed
- Verify API URL is correct
- Check network connectivity: `ping your-server-ip`
- Verify Django server is running
- Check firewall settings

### No Faces Detected
- Ensure good lighting
- Students should face the camera
- Check image quality in saved photos
- Adjust camera focus if needed

### Recognition Errors
- Verify face recognition model is trained
- Check that students are registered with face images
- Try adjusting the threshold value in config.json

## API Response Format

The API returns JSON with the following structure:

```json
{
    "success": true,
    "message": "Attendance marked for 2 student(s)",
    "recognitions": [
        {
            "student_id": "S001",
            "name": "John Doe",
            "confidence": 0.85,
            "status": "marked"
        }
    ],
    "marked_count": 2
}
```

Status values:
- `marked`: Successfully marked attendance
- `already_marked`: Student already marked for this session
- `not_found`: Recognized but student ID not in database
- `unrecognized`: Face detected but not recognized

## Security Notes (MVP vs Production)

**Current Implementation (MVP):**
- No authentication (CSRF exempt)
- HTTP communication (not HTTPS)
- No rate limiting

**For Production:**
- Add API key authentication
- Use HTTPS
- Implement rate limiting
- Add device registration/whitelist
- Add request logging

