# Face Recognition Attendance Management System

A Django-based attendance management system using facial recognition with MTCNN for face detection and InsightFace (ArcFace) for face embeddings.

## Features

- Student registration with face images
- Face recognition model training using SVM
- Real-time attendance marking
- Session management
- High accuracy face recognition using ArcFace embeddings

## Installation

### 1. Virtual Environment Setup
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Database Setup
```bash
python manage.py migrate
python manage.py createsuperuser
```

### 3. Create Media Directories
The system will automatically create these directories, but you can also create them manually:
```bash
mkdir -p media/faces media/models media/attendance_images
```

## Usage

### 1. Register Students

#### Option A: Using Management Command
```bash
python manage.py register_student <student_id> <first_name> <last_name> <email> --images-dir /path/to/images --department "Computer Science" --year 2
```

Example:
```bash
python manage.py register_student S001 John Doe john@example.com --images-dir ./student_photos/john --department "CS" --year 3
```

#### Option B: Using Django Admin
1. Go to http://localhost:8000/admin
2. Navigate to Students
3. Add student details
4. Then add face embeddings separately

### 2. Prepare Face Images

Organize student face images in this structure:
```
faces/
├── student_id_1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
├── student_id_2/
│   ├── img1.jpg
│   └── img2.jpg
└── ...
```

**Tips for best results:**
- Use 5-10 images per student
- Include different angles and lighting conditions
- Ensure clear, frontal face images
- Resolution: 160x160 pixels or higher

### 3. Train the Model

#### Option A: Train from Directory
```bash
python manage.py train_model --from-directory ./faces --set-active
```

#### Option B: Train from Database
```bash
python manage.py train_model --model-name my_model_v1 --set-active
```

This will:
- Load all face embeddings from registered students
- Train an SVM classifier
- Save the model to `media/models/`
- Display training accuracy

### 4. Test Recognition

Test the model on a single image:
```bash
python manage.py test_recognition /path/to/test_image.jpg --threshold 0.6
```

### 5. Create Attendance Session

You can create sessions through Django admin or programmatically:

**Via Django Admin:**
1. Go to http://localhost:8000/admin
2. Navigate to "Attendance Sessions"
3. Add a new session with:
   - Session name
   - Course name
   - Date, start time, end time

**Via Django Shell:**
```python
python manage.py shell

from attendance.models import AttendanceSession
from datetime import date, time

session = AttendanceSession.objects.create(
    session_name="Morning Lecture",
    course_name="Machine Learning",
    session_date=date.today(),
    start_time=time(9, 0),
    end_time=time(11, 0)
)
```

### 6. Mark Attendance

Mark attendance using face recognition:
```bash
python manage.py mark_attendance /path/to/student_image.jpg <session_id> --threshold 0.6
```

Example:
```bash
python manage.py mark_attendance ./captured_face.jpg 1 --threshold 0.6
```

## Management Commands Reference

### `register_student`
Register a new student with face images.

**Arguments:**
- `student_id`: Unique student identifier
- `first_name`: Student's first name
- `last_name`: Student's last name
- `email`: Student's email

**Options:**
- `--images-dir`: Directory containing face images
- `--department`: Department name
- `--year`: Year of study
- `--phone`: Phone number

### `train_model`
Train the face recognition model.

**Options:**
- `--model-name`: Name for the model (default: auto-generated)
- `--from-directory`: Train from directory structure
- `--set-active`: Set as active model after training

### `test_recognition`
Test face recognition on an image.

**Arguments:**
- `image_path`: Path to test image

**Options:**
- `--threshold`: Confidence threshold (default: 0.5)
- `--model-id`: Specific model ID to use

### `mark_attendance`
Mark attendance from an image.

**Arguments:**
- `image_path`: Path to student image
- `session_id`: Session ID

**Options:**
- `--threshold`: Confidence threshold (default: 0.6)

## Django Models

### Student
- `student_id`: Unique identifier
- `first_name`, `last_name`: Name
- `email`: Contact email
- `department`, `year`: Academic info
- `is_active`: Active status

### FaceEmbedding
- `student`: Foreign key to Student
- `embedding`: 512-dimensional face embedding
- `image_path`: Path to original image

### AttendanceSession
- `session_name`: Session identifier
- `course_name`: Course name
- `session_date`: Date of session
- `start_time`, `end_time`: Time range

### AttendanceRecord
- `student`: Foreign key to Student
- `session`: Foreign key to AttendanceSession
- `status`: present/absent/late
- `confidence_score`: Recognition confidence
- `marked_at`: Timestamp

## API Integration (Future)

The system is designed to support REST API integration for:
- Real-time webcam attendance
- Mobile app integration
- Bulk attendance processing

## Troubleshooting

### No face detected
- Ensure image has clear frontal face
- Check lighting conditions
- Verify image quality

### Low confidence scores
- Add more training images per student
- Retrain model with better quality images
- Adjust threshold parameter

### Model accuracy issues
- Collect more diverse training images
- Ensure minimum 5 images per student
- Check for duplicate/mislabeled images

## Technologies Used

- **Django 5.2.9**: Web framework
- **MTCNN**: Face detection
- **InsightFace (ArcFace)**: Face embeddings
- **scikit-learn**: SVM classifier
- **OpenCV**: Image processing
- **NumPy**: Numerical operations

## Directory Structure

```
TrueLegend/
├── attendance/              # Main Django app
│   ├── models.py           # Database models
│   ├── admin.py            # Admin configuration
│   ├── management/         # Management commands
│   │   └── commands/
│   │       ├── register_student.py
│   │       ├── train_model.py
│   │       ├── test_recognition.py
│   │       └── mark_attendance.py
│   └── utils/              # Utilities
│       └── face_recognition.py
├── attendance_system/      # Django project settings
├── media/                  # Media files
│   ├── faces/             # Student face images
│   ├── models/            # Trained models
│   └── attendance_images/ # Captured attendance images
├── db.sqlite3             # Database
├── manage.py              # Django management script
└── requirements.txt       # Python dependencies
```

## Running the Development Server

```bash
python manage.py runserver
```

Access admin panel at: http://localhost:8000/admin

## Next Steps

1. Implement web interface for attendance marking
2. Add REST API endpoints
3. Real-time webcam integration
4. Attendance reports and analytics
5. Email notifications
6. Mobile app integration

## License

MIT License
