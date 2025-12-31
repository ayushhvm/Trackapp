from django.core.management.base import BaseCommand
from attendance.models import AttendanceSession, AttendanceRecord, CaptureRecord, StudentCapture


class Command(BaseCommand):
    help = 'Verify attendance for a session based on capture records'

    def add_arguments(self, parser):
        parser.add_argument('session_id', type=int, help='Session ID to verify')

    def handle(self, *args, **options):
        session_id = options['session_id']
        
        try:
            session = AttendanceSession.objects.get(id=session_id)
        except AttendanceSession.DoesNotExist:
            self.stdout.write(self.style.ERROR(f'Session {session_id} not found'))
            return
        
        self.stdout.write(self.style.SUCCESS(f'\n{"="*60}'))
        self.stdout.write(self.style.SUCCESS(f'VERIFYING ATTENDANCE FOR SESSION'))
        self.stdout.write(self.style.SUCCESS(f'{"="*60}'))
        self.stdout.write(f'Session: {session.session_name}')
        self.stdout.write(f'Course: {session.course_name}')
        self.stdout.write(f'Date: {session.session_date}\n')
        
        # Get all captures for this session
        captures = CaptureRecord.objects.filter(session=session).order_by('capture_number')
        total_captures = captures.count()
        
        if total_captures == 0:
            self.stdout.write(self.style.WARNING('No captures found. Cannot verify attendance.'))
            return
        
        self.stdout.write(f'Total captures: {total_captures}')
        
        # Define start and end ranges (20% of captures)
        start_range = max(1, int(total_captures * 0.2))
        end_range = max(1, int(total_captures * 0.2))
        majority_threshold = total_captures * 0.5
        
        self.stdout.write(f'Start range: First {start_range} captures')
        self.stdout.write(f'End range: Last {end_range} captures')
        self.stdout.write(f'Majority threshold: >{majority_threshold:.1f} captures\n')
        
        # Get all pending attendance records for this session
        pending_records = AttendanceRecord.objects.filter(session=session, status='pending')
        
        verified_count = 0
        rejected_count = 0
        
        for record in pending_records:
            student = record.student
            
            # Count total appearances
            total_appearances = StudentCapture.objects.filter(
                capture__session=session,
                student=student
            ).count()
            
            # Check if present in start captures
            start_captures = captures[:start_range]
            present_in_start = StudentCapture.objects.filter(
                capture__in=start_captures,
                student=student
            ).exists()
            
            # Check if present in end captures
            end_captures = captures[total_captures - end_range:]
            present_in_end = StudentCapture.objects.filter(
                capture__in=end_captures,
                student=student
            ).exists()
            
            # Calculate presence percentage
            presence_percentage = (total_appearances / total_captures) * 100
            
            # Verification logic
            is_verified = (
                present_in_start and 
                present_in_end and 
                total_appearances > majority_threshold
            )
            
            # Update attendance record
            record.total_captures = total_appearances
            record.present_in_start = present_in_start
            record.present_in_end = present_in_end
            record.is_verified = is_verified
            
            if is_verified:
                record.status = 'present'
                record.verification_notes = f"Verified: Present in {total_appearances}/{total_captures} captures ({presence_percentage:.1f}%)"
                verified_count += 1
                self.stdout.write(self.style.SUCCESS(f'✓ {student.student_id} - {student.first_name} {student.last_name}: PRESENT'))
                self.stdout.write(f'  └─ {total_appearances}/{total_captures} captures ({presence_percentage:.1f}%)')
                self.stdout.write(f'  └─ Start: {"✓" if present_in_start else "✗"}, End: {"✓" if present_in_end else "✗"}')
            else:
                record.status = 'absent'
                reasons = []
                if not present_in_start:
                    reasons.append("not present at start")
                if not present_in_end:
                    reasons.append("not present at end")
                if total_appearances <= majority_threshold:
                    reasons.append(f"only {total_appearances}/{total_captures} captures")
                
                record.verification_notes = f"Rejected: {', '.join(reasons)}"
                rejected_count += 1
                self.stdout.write(self.style.ERROR(f'✗ {student.student_id} - {student.first_name} {student.last_name}: ABSENT'))
                self.stdout.write(f'  └─ {total_appearances}/{total_captures} captures ({presence_percentage:.1f}%)')
                self.stdout.write(f'  └─ Start: {"✓" if present_in_start else "✗"}, End: {"✓" if present_in_end else "✗"}')
                self.stdout.write(f'  └─ Reason: {", ".join(reasons)}')
            
            record.save()
        
        self.stdout.write(f'\n{"-"*60}')
        self.stdout.write(self.style.SUCCESS(f'✓ Verified Present: {verified_count}'))
        self.stdout.write(self.style.ERROR(f'✗ Marked Absent: {rejected_count}'))
        self.stdout.write(self.style.SUCCESS(f'{"="*60}\n'))
