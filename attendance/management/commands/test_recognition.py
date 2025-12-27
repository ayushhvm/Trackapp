from django.core.management.base import BaseCommand
from attendance.models import FaceRecognitionModel
from attendance.utils.face_recognition import FaceRecognitionSystem
from django.conf import settings
import os


class Command(BaseCommand):
    help = 'Test face recognition on an image'

    def add_arguments(self, parser):
        parser.add_argument('image_path', type=str, help='Path to the test image')
        parser.add_argument(
            '--threshold',
            type=float,
            default=0.5,
            help='Confidence threshold (default: 0.5)'
        )
        parser.add_argument(
            '--model-id',
            type=int,
            help='Specific model ID to use (default: active model)'
        )

    def handle(self, *args, **options):
        image_path = options['image_path']
        threshold = options['threshold']
        model_id = options.get('model_id')

        if not os.path.exists(image_path):
            self.stdout.write(self.style.ERROR(f'Image not found: {image_path}'))
            return

        # Load the model
        if model_id:
            try:
                face_model = FaceRecognitionModel.objects.get(id=model_id)
            except FaceRecognitionModel.DoesNotExist:
                self.stdout.write(self.style.ERROR(f'Model with ID {model_id} not found'))
                return
        else:
            try:
                face_model = FaceRecognitionModel.objects.get(is_active=True)
            except FaceRecognitionModel.DoesNotExist:
                self.stdout.write(self.style.ERROR('No active model found'))
                return

        self.stdout.write(f'Using model: {face_model.model_name}')

        # Initialize face recognition system
        fr_system = FaceRecognitionSystem()
        fr_system.initialize_arcface()
        fr_system.load_model(face_model.model_file)

        # Perform recognition on all faces
        self.stdout.write(f'Testing image: {image_path}')
        recognitions = fr_system.recognize_all_faces_from_image(image_path, threshold)

        if not recognitions:
            self.stdout.write(self.style.ERROR('\n✗ No faces detected in image'))
            return
        
        self.stdout.write(f'\nFound {len(recognitions)} face(s) in image:\n')
        self.stdout.write('=' * 60)
        
        for idx, (name, confidence, bbox, message) in enumerate(recognitions, 1):
            x, y, w, h = bbox
            self.stdout.write(f'\nFace {idx}:')
            self.stdout.write(f'  Location: ({x}, {y}) - Size: {w}x{h}')
            
            if name:
                self.stdout.write(self.style.SUCCESS(
                    f'  ✓ Recognition Success!\n'
                    f'    Student ID: {name}\n'
                    f'    Confidence: {confidence * 100:.2f}%\n'
                    f'    Status: {message}'
                ))
            else:
                self.stdout.write(self.style.WARNING(
                    f'  ✗ Recognition Failed\n'
                    f'    Confidence: {confidence * 100:.2f}%\n'
                    f'    Status: {message}'
                ))
        
        self.stdout.write('\n' + '=' * 60)
