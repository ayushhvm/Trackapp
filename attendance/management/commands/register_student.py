from django.core.management.base import BaseCommand
from attendance.models import Student, FaceEmbedding
from attendance.utils.face_recognition import FaceRecognitionSystem
from django.conf import settings
import os
import shutil


class Command(BaseCommand):
    help = 'Register a new student with face images'

    def add_arguments(self, parser):
        parser.add_argument('student_id', type=str, help='Student ID')
        parser.add_argument('first_name', type=str, help='First Name')
        parser.add_argument('last_name', type=str, help='Last Name')
        parser.add_argument('email', type=str, help='Email')
        parser.add_argument('--images-dir', type=str, help='Directory containing student face images')
        parser.add_argument('--department', type=str, default='', help='Department')
        parser.add_argument('--year', type=int, default=None, help='Year')
        parser.add_argument('--phone', type=str, default='', help='Phone number')

    def handle(self, *args, **options):
        student_id = options['student_id']
        first_name = options['first_name']
        last_name = options['last_name']
        email = options['email']
        images_dir = options.get('images_dir')
        department = options.get('department', '')
        year = options.get('year')
        phone = options.get('phone', '')

        # Check if student already exists
        if Student.objects.filter(student_id=student_id).exists():
            self.stdout.write(self.style.ERROR(f'Student {student_id} already exists!'))
            return

        # Create student
        student = Student.objects.create(
            student_id=student_id,
            first_name=first_name,
            last_name=last_name,
            email=email,
            department=department,
            year=year,
            phone=phone
        )
        self.stdout.write(self.style.SUCCESS(f'Created student: {student}'))

        if images_dir and os.path.exists(images_dir):
            # Initialize face recognition system
            fr_system = FaceRecognitionSystem()
            fr_system.initialize_arcface()

            # Create directory for student faces
            student_faces_dir = settings.FACE_IMAGES_DIR / student_id
            os.makedirs(student_faces_dir, exist_ok=True)

            # Process each image
            processed_count = 0
            for img_file in os.listdir(images_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                img_path = os.path.join(images_dir, img_file)
                
                # Extract face and generate embedding
                face = fr_system.extract_face(img_path)
                if face is None:
                    self.stdout.write(self.style.WARNING(f'No face detected in {img_file}'))
                    continue

                # Generate embedding
                embedding = fr_system.get_embedding(face)
                if embedding is None:
                    self.stdout.write(self.style.WARNING(f'Failed to generate embedding for {img_file}'))
                    continue

                # Copy image to student directory
                dest_path = student_faces_dir / img_file
                shutil.copy2(img_path, dest_path)

                # Save embedding to database
                face_embedding = FaceEmbedding(
                    student=student,
                    image_path=str(dest_path)
                )
                face_embedding.set_embedding(embedding)
                face_embedding.save()

                processed_count += 1
                self.stdout.write(self.style.SUCCESS(f'Processed {img_file}'))

            self.stdout.write(self.style.SUCCESS(
                f'Successfully registered {student_id} with {processed_count} face images'
            ))
        else:
            self.stdout.write(self.style.WARNING(
                'No images directory provided or directory does not exist. '
                'Student created without face embeddings.'
            ))
