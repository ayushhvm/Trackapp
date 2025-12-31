from django.core.management.base import BaseCommand
from attendance.models import AttendanceSession
from attendance.utils.automated_attendance import AutomatedAttendanceCapture
from datetime import datetime


class Command(BaseCommand):
    help = 'Start automated attendance capture for a session'

    def add_arguments(self, parser):
        parser.add_argument('session_id', type=int, help='Session ID to start automated capture')
        parser.add_argument('--interval', type=int, default=30, help='Capture interval in seconds (default: 30)')
        parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')

    def handle(self, *args, **options):
        session_id = options['session_id']
        interval = options['interval']
        camera = options['camera']
        
        try:
            session = AttendanceSession.objects.get(id=session_id, is_active=True)
        except AttendanceSession.DoesNotExist:
            self.stdout.write(self.style.ERROR(f'Session {session_id} not found or inactive'))
            return
        
        self.stdout.write(self.style.SUCCESS(f'Starting automated attendance for: {session.session_name}'))
        self.stdout.write(f'Course: {session.course_name}')
        self.stdout.write(f'Date: {session.session_date}')
        self.stdout.write(f'Time: {session.start_time} - {session.end_time}')
        self.stdout.write(f'Capture interval: {interval} seconds')
        self.stdout.write(f'Camera index: {camera}')
        self.stdout.write('')
        self.stdout.write(self.style.WARNING('Press Ctrl+C to stop'))
        self.stdout.write('')
        
        # Create capture instance (don't use background thread - run directly)
        capture = AutomatedAttendanceCapture(
            session_id=session_id,
            camera_index=camera,
            capture_interval=interval
        )
        
        # Run directly (blocking) instead of starting background thread
        try:
            capture.run_session_capture()
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING('\nStopping automated capture...'))
            capture.stop()
            self.stdout.write(self.style.SUCCESS('Automated capture stopped'))
