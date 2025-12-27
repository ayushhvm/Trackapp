from django import forms
from .models import Student, Teacher, AttendanceSession


class MultipleFileInput(forms.FileInput):
    """Custom widget for multiple file uploads"""
    def __init__(self, attrs=None):
        super().__init__(attrs)
        if attrs is not None:
            self.attrs.update(attrs)
        self.attrs['multiple'] = True


class LoginForm(forms.Form):
    """Login form for both students and teachers"""
    user_id = forms.CharField(
        label='User ID',
        max_length=50,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter Student ID or Teacher ID'
        })
    )
    password = forms.CharField(
        label='Password',
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter password'
        })
    )
    user_type = forms.ChoiceField(
        label='Login as',
        choices=[('student', 'Student'), ('teacher', 'Teacher')],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )


class StudentRegistrationForm(forms.Form):
    """Form for registering a new student"""
    student_id = forms.CharField(
        max_length=50,
        widget=forms.TextInput(attrs={'class': 'form-control'}),
        label='Student ID'
    )
    first_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control'}),
        label='First Name'
    )
    last_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control'}),
        label='Last Name'
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={'class': 'form-control'}),
        label='Email'
    )
    phone = forms.CharField(
        max_length=15,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control'}),
        label='Phone'
    )
    department = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control'}),
        label='Department'
    )
    year = forms.IntegerField(
        required=False,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        label='Year'
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control'}),
        label='Password',
        min_length=6,
        help_text='Password must be at least 6 characters long.'
    )
    images = forms.FileField(
        label='Face Images',
        widget=MultipleFileInput(attrs={
            'accept': 'image/*',
            'class': 'form-control',
            'id': 'id_images'
        }),
        help_text='Upload 5-10 face images (JPG, PNG). Include different angles and lighting.'
    )


class AttendanceMarkingForm(forms.Form):
    """Form for marking attendance"""
    session = forms.ModelChoiceField(
        label='Attendance Session',
        queryset=AttendanceSession.objects.filter(is_active=True).order_by('-session_date', '-start_time'),
        widget=forms.Select(attrs={'class': 'form-control'}),
        empty_label='Select a session'
    )
    image = forms.ImageField(
        label='Upload Image',
        widget=forms.FileInput(attrs={
            'accept': 'image/*',
            'class': 'form-control',
            'id': 'attendance-image-input'
        }),
        help_text='Upload an image containing one or more student faces'
    )
    threshold = forms.FloatField(
        label='Confidence Threshold',
        initial=0.6,
        min_value=0.0,
        max_value=1.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'min': '0',
            'max': '1'
        }),
        help_text='Minimum confidence score (0.0 - 1.0)'
    )


class SessionForm(forms.ModelForm):
    """Form for creating/editing attendance sessions"""
    class Meta:
        model = AttendanceSession
        fields = ['session_name', 'course_name', 'session_date', 'start_time', 'end_time']
        widgets = {
            'session_name': forms.TextInput(attrs={'class': 'form-control'}),
            'course_name': forms.TextInput(attrs={'class': 'form-control'}),
            'session_date': forms.DateInput(attrs={
                'class': 'form-control',
                'type': 'date'
            }),
            'start_time': forms.TimeInput(attrs={
                'class': 'form-control',
                'type': 'time'
            }),
            'end_time': forms.TimeInput(attrs={
                'class': 'form-control',
                'type': 'time'
            }),
        }
        labels = {
            'session_name': 'Session Name',
            'course_name': 'Course Name',
            'session_date': 'Date',
            'start_time': 'Start Time',
            'end_time': 'End Time',
        }

