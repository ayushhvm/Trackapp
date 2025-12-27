from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.dateparse import parse_datetime
from django.conf import settings
import os
import json

from .models import Student, AttendanceSession, AttendanceRecord, FaceRecognitionModel
from .utils.face_recognition import FaceRecognitionSystem


@csrf_exempt
@require_http_methods(["POST"])
def mark_attendance_api(request):
    """
    API endpoint for Raspberry Pi to mark attendance via image upload.
    
    Accepts:
    - image (file): Photo from Raspberry Pi
    - session_id (int): Attendance session ID
    - captured_at (string): ISO datetime string when photo was taken
    - location_name (string, optional): Manual location name
    - latitude (float, optional): GPS latitude
    - longitude (float, optional): GPS longitude
    - device_id (string, optional): Pi identifier
    - threshold (float, optional): Recognition threshold (default 0.6)
    
    Returns JSON response with recognition results.
    """
    try:
        # Validate required fields
        if 'image' not in request.FILES:
            return JsonResponse({
                'success': False,
                'message': 'No image file provided',
                'recognitions': [],
                'marked_count': 0
            }, status=400)
        
        if 'session_id' not in request.POST:
            return JsonResponse({
                'success': False,
                'message': 'session_id is required',
                'recognitions': [],
                'marked_count': 0
            }, status=400)
        
        # Get session
        try:
            session_id = int(request.POST['session_id'])
            session = AttendanceSession.objects.get(id=session_id)
        except (ValueError, AttendanceSession.DoesNotExist):
            return JsonResponse({
                'success': False,
                'message': f'Invalid session_id: {request.POST.get("session_id")}',
                'recognitions': [],
                'marked_count': 0
            }, status=400)
        
        # Get optional parameters
        threshold = float(request.POST.get('threshold', 0.6))
        device_id = request.POST.get('device_id', '')
        location_name = request.POST.get('location_name', '')
        latitude = request.POST.get('latitude')
        longitude = request.POST.get('longitude')
        captured_at_str = request.POST.get('captured_at', '')
        
        # Parse captured_at datetime
        photo_captured_at = None
        if captured_at_str:
            try:
                photo_captured_at = parse_datetime(captured_at_str)
            except (ValueError, TypeError):
                pass
        
        # Parse latitude/longitude
        latitude_float = None
        longitude_float = None
        if latitude:
            try:
                latitude_float = float(latitude)
            except (ValueError, TypeError):
                pass
        if longitude:
            try:
                longitude_float = float(longitude)
            except (ValueError, TypeError):
                pass
        
        # Save uploaded image
        image_file = request.FILES['image']
        image_filename = f"{session_id}_{device_id}_{image_file.name}" if device_id else f"{session_id}_{image_file.name}"
        image_path = os.path.join(settings.ATTENDANCE_IMAGES_DIR, image_filename)
        os.makedirs(settings.ATTENDANCE_IMAGES_DIR, exist_ok=True)
        
        with open(image_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)
        
        # Load active model
        try:
            face_model = FaceRecognitionModel.objects.get(is_active=True)
        except FaceRecognitionModel.DoesNotExist:
            return JsonResponse({
                'success': False,
                'message': 'No active face recognition model found. Please train a model first.',
                'recognitions': [],
                'marked_count': 0
            }, status=400)
        
        # Initialize face recognition system
        fr_system = FaceRecognitionSystem()
        fr_system.initialize_arcface()
        fr_system.load_model(face_model.model_file)
        
        # Recognize all faces in the image
        recognitions = fr_system.recognize_all_faces_from_image(image_path, threshold)
        
        if not recognitions:
            return JsonResponse({
                'success': True,
                'message': 'No faces detected in the image',
                'recognitions': [],
                'marked_count': 0
            })
        
        # Mark attendance for recognized students
        marked_count = 0
        results = []
        
        for name, confidence, bbox, message in recognitions:
            if name:
                try:
                    student = Student.objects.get(student_id=name)
                    
                    # Check if already marked
                    existing = AttendanceRecord.objects.filter(
                        student=student,
                        session=session
                    ).first()
                    
                    if existing:
                        results.append({
                            'student_id': student.student_id,
                            'name': f"{student.first_name} {student.last_name}",
                            'confidence': float(confidence),
                            'status': 'already_marked'
                        })
                    else:
                        # Mark attendance with metadata
                        attendance = AttendanceRecord.objects.create(
                            student=student,
                            session=session,
                            status='present',
                            confidence_score=confidence,
                            image_path=image_path,
                            marked_by='raspberry_pi',
                            device_id=device_id if device_id else None,
                            location_name=location_name if location_name else None,
                            latitude=latitude_float,
                            longitude=longitude_float,
                            photo_captured_at=photo_captured_at
                        )
                        marked_count += 1
                        results.append({
                            'student_id': student.student_id,
                            'name': f"{student.first_name} {student.last_name}",
                            'confidence': float(confidence),
                            'status': 'marked'
                        })
                except Student.DoesNotExist:
                    results.append({
                        'student_id': name,
                        'name': 'Unknown',
                        'confidence': float(confidence),
                        'status': 'not_found'
                    })
            else:
                results.append({
                    'student_id': None,
                    'name': 'Unknown',
                    'confidence': float(confidence),
                    'status': 'unrecognized'
                })
        
        return JsonResponse({
            'success': True,
            'message': f'Attendance marked for {marked_count} student(s)',
            'recognitions': results,
            'marked_count': marked_count
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'Error processing request: {str(e)}',
            'recognitions': [],
            'marked_count': 0
        }, status=500)

