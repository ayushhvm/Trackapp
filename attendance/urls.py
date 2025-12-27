from django.urls import path
from . import views
from . import api_views

urlpatterns = [
    # Authentication
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    # Student routes
    path('student/dashboard/', views.student_dashboard, name='student_dashboard'),
    path('student/attendance/', views.student_attendance, name='student_attendance'),
    
    # Teacher routes
    path('teacher/dashboard/', views.teacher_dashboard, name='teacher_dashboard'),
    path('teacher/students/register/', views.register_student, name='register_student'),
    path('teacher/students/list/', views.view_students, name='view_students'),
    path('teacher/attendance/mark/', views.mark_attendance, name='mark_attendance'),
    path('teacher/attendance/records/', views.view_records, name='view_records'),
    path('teacher/sessions/', views.view_sessions, name='view_sessions'),
    path('teacher/sessions/create/', views.create_session, name='create_session'),
    path('teacher/train-model/', views.train_model, name='train_model'),
    
    # API routes
    path('api/mark-attendance/', api_views.mark_attendance_api, name='api_mark_attendance'),
]

