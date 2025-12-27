from django.core.management.base import BaseCommand
from attendance.models import Student, AttendanceSession, AttendanceRecord, FaceRecognitionModel
from attendance.utils.face_recognition import FaceRecognitionSystem
from django.utils import timezone
from datetime import datetime
import os


class Command(BaseCommand):
    help = 'Mark attendance from an image'

    def add_arguments(self, parser):
        parser.add_argument('image_path', type=str, help='Path to the attendance image')
        parser.add_argument('session_id', type=int, help='Session ID')
        parser.add_argument(
            '--threshold',
            type=float,
            default=0.6,
            help='Confidence threshold (default: 0.6)'
        )

    def handle(self, *args, **options):
        image_path = options['image_path']
        session_id = options['session_id']
        threshold = options['threshold']

        if not os.path.exists(image_path):
            self.stdout.write(self.style.ERROR(f'Image not found: {image_path}'))
            return

        # Get session
        try:
            session = AttendanceSession.objects.get(id=session_id)
        except AttendanceSession.DoesNotExist:
            self.stdout.write(self.style.ERROR(f'Session {session_id} not found'))
            return

        # Load active model
        try:
            face_model = FaceRecognitionModel.objects.get(is_active=True)
        except FaceRecognitionModel.DoesNotExist:
            self.stdout.write(self.style.ERROR('No active model found. Train a model first.'))
            return

        self.stdout.write(f'Session: {session.course_name} - {session.session_date}')
        self.stdout.write(f'Using model: {face_model.model_name}')

        # Initialize face recognition system
        fr_system = FaceRecognitionSystem()
        fr_system.initialize_arcface()
        fr_system.load_model(face_model.model_file)

        # Perform recognition
        self.stdout.write(f'Processing image: {image_path}')
        name, confidence, message = fr_system.recognize_from_image(image_path, threshold)

        if name:
            # Find student
            try:
                student = Student.objects.get(student_id=name)
                
                # Check if already marked
                existing = AttendanceRecord.objects.filter(
                    student=student,
                    session=session
                ).first()

                if existing:
                    self.stdout.write(self.style.WARNING(
                        f'Attendance already marked for {student.student_id}'
                    ))
                    return

                # Mark attendance
                attendance = AttendanceRecord.objects.create(
                    student=student,
                    session=session,
                    status='present',
                    confidence_score=confidence,
                    image_path=image_path,
                    marked_by='system'
                )

                self.stdout.write(self.style.SUCCESS(
                    f'\n✓ Attendance Marked!\n'
                    f'  Student: {student.first_name} {student.last_name} ({student.student_id})\n'
                    f'  Session: {session.course_name}\n'
                    f'  Confidence: {confidence * 100:.2f}%\n'
                    f'  Time: {attendance.marked_at.strftime("%Y-%m-%d %H:%M:%S")}'
                ))

            except Student.DoesNotExist:
                self.stdout.write(self.style.ERROR(
                    f'Student {name} recognized but not found in database'
                ))
        else:
            self.stdout.write(self.style.WARNING(
                f'\n✗ Recognition Failed\n'
                f'  Confidence: {confidence * 100:.2f}%\n'
                f'  Status: {message}'
            ))
