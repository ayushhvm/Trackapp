from django.db import models
from django.utils import timezone
from django.contrib.auth.hashers import make_password, check_password
import numpy as np

# Create your models here.

class Student(models.Model):
    """Model to store student information"""
    student_id = models.CharField(max_length=50, unique=True)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=15, blank=True, null=True)
    department = models.CharField(max_length=100, blank=True, null=True)
    year = models.IntegerField(blank=True, null=True)
    password = models.CharField(max_length=128, blank=True)  # Hashed password
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['student_id']

    def set_password(self, raw_password):
        """Hash and set password"""
        self.password = make_password(raw_password)
    
    def check_password(self, raw_password):
        """Check if password is correct"""
        if not self.password:
            return False
        return check_password(raw_password, self.password)

    def __str__(self):
        return f"{self.student_id} - {self.first_name} {self.last_name}"


class Teacher(models.Model):
    """Model to store teacher information"""
    teacher_id = models.CharField(max_length=50, unique=True)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=15, blank=True, null=True)
    password = models.CharField(max_length=128)  # Hashed password
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['teacher_id']

    def set_password(self, raw_password):
        """Hash and set password"""
        self.password = make_password(raw_password)
    
    def check_password(self, raw_password):
        """Check if password is correct"""
        return check_password(raw_password, self.password)

    def __str__(self):
        return f"{self.teacher_id} - {self.first_name} {self.last_name}"


class FaceEmbedding(models.Model):
    """Model to store face embeddings for each student"""
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='face_embeddings')
    embedding = models.BinaryField()  # Store numpy array as binary
    image_path = models.CharField(max_length=500, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def set_embedding(self, embedding_array):
        """Convert numpy array to binary for storage"""
        self.embedding = embedding_array.tobytes()

    def get_embedding(self):
        """Convert binary back to numpy array"""
        return np.frombuffer(self.embedding, dtype=np.float32)

    def __str__(self):
        return f"Embedding for {self.student.student_id}"


class AttendanceSession(models.Model):
    """Model to represent a class/session"""
    session_name = models.CharField(max_length=200)
    course_name = models.CharField(max_length=200)
    session_date = models.DateField()
    start_time = models.TimeField()
    end_time = models.TimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['-session_date', '-start_time']

    def __str__(self):
        return f"{self.course_name} - {self.session_date}"


class AttendanceRecord(models.Model):
    """Model to store attendance records"""
    STATUS_CHOICES = [
        ('present', 'Present'),
        ('absent', 'Absent'),
        ('late', 'Late'),
    ]

    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='attendance_records')
    session = models.ForeignKey(AttendanceSession, on_delete=models.CASCADE, related_name='attendance_records')
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='absent')
    marked_at = models.DateTimeField(default=timezone.now)  # Server timestamp
    photo_captured_at = models.DateTimeField(blank=True, null=True)  # When photo was taken on device
    confidence_score = models.FloatField(blank=True, null=True)  # Confidence from face recognition
    image_path = models.CharField(max_length=500, blank=True, null=True)  # Path to captured image
    marked_by = models.CharField(max_length=100, default='system')  # 'system' for auto, or admin name
    device_id = models.CharField(max_length=100, blank=True, null=True)  # Raspberry Pi identifier
    location_name = models.CharField(max_length=200, blank=True, null=True)  # Manual location name
    latitude = models.FloatField(blank=True, null=True)  # GPS latitude
    longitude = models.FloatField(blank=True, null=True)  # GPS longitude
    notes = models.TextField(blank=True, null=True)

    class Meta:
        unique_together = ['student', 'session']
        ordering = ['-marked_at']

    def __str__(self):
        return f"{self.student.student_id} - {self.session.course_name} - {self.status}"


class FaceRecognitionModel(models.Model):
    """Model to store trained model information"""
    model_name = models.CharField(max_length=100)
    model_file = models.CharField(max_length=500)  # Path to .pkl file
    encoder_file = models.CharField(max_length=500)  # Path to encoder .pkl file
    embeddings_file = models.CharField(max_length=500, blank=True, null=True)  # Path to .npz file
    accuracy = models.FloatField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.model_name} - Accuracy: {self.accuracy}"
