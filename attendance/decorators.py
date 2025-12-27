from functools import wraps
from django.shortcuts import redirect
from django.contrib import messages


def student_required(view_func):
    """Decorator to ensure user is logged in as a student"""
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if 'student_id' not in request.session:
            messages.error(request, 'Please login as a student to access this page.')
            return redirect('login')
        return view_func(request, *args, **kwargs)
    return _wrapped_view


def teacher_required(view_func):
    """Decorator to ensure user is logged in as a teacher"""
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if 'teacher_id' not in request.session:
            messages.error(request, 'Please login as a teacher to access this page.')
            return redirect('login')
        return view_func(request, *args, **kwargs)
    return _wrapped_view

