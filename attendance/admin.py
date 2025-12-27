from django.contrib import admin
from .models import Student, Teacher, FaceEmbedding, AttendanceSession, AttendanceRecord, FaceRecognitionModel

# Register your models here.

@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ['student_id', 'first_name', 'last_name', 'email', 'department', 'year', 'is_active']
    list_filter = ['department', 'year', 'is_active']
    search_fields = ['student_id', 'first_name', 'last_name', 'email']
    ordering = ['student_id']


@admin.register(FaceEmbedding)
class FaceEmbeddingAdmin(admin.ModelAdmin):
    list_display = ['student', 'image_path', 'created_at']
    list_filter = ['created_at']
    search_fields = ['student__student_id', 'student__first_name', 'student__last_name']
    ordering = ['-created_at']


@admin.register(AttendanceSession)
class AttendanceSessionAdmin(admin.ModelAdmin):
    list_display = ['session_name', 'course_name', 'session_date', 'start_time', 'end_time', 'is_active']
    list_filter = ['session_date', 'is_active']
    search_fields = ['session_name', 'course_name']
    ordering = ['-session_date', '-start_time']


@admin.register(AttendanceRecord)
class AttendanceRecordAdmin(admin.ModelAdmin):
    list_display = ['student', 'session', 'status', 'marked_at', 'confidence_score', 'marked_by']
    list_filter = ['status', 'marked_at', 'session__session_date']
    search_fields = ['student__student_id', 'student__first_name', 'student__last_name', 'session__course_name']
    ordering = ['-marked_at']


@admin.register(Teacher)
class TeacherAdmin(admin.ModelAdmin):
    list_display = ['teacher_id', 'first_name', 'last_name', 'email', 'is_active']
    list_filter = ['is_active']
    search_fields = ['teacher_id', 'first_name', 'last_name', 'email']
    ordering = ['teacher_id']


@admin.register(FaceRecognitionModel)
class FaceRecognitionModelAdmin(admin.ModelAdmin):
    list_display = ['model_name', 'accuracy', 'created_at', 'is_active']
    list_filter = ['is_active', 'created_at']
    ordering = ['-created_at']
