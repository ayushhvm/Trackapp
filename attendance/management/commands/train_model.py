from django.core.management.base import BaseCommand
from attendance.models import Student, FaceEmbedding, FaceRecognitionModel
from attendance.utils.face_recognition import FaceRecognitionSystem
from django.conf import settings
import numpy as np
import os
from datetime import datetime


class Command(BaseCommand):
    help = 'Train face recognition model using registered student face embeddings'

    def add_arguments(self, parser):
        parser.add_argument(
            '--model-name',
            type=str,
            default=f'face_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            help='Name for the trained model'
        )
        parser.add_argument(
            '--from-directory',
            type=str,
            help='Train from a directory structure (faces/student_name/*.jpg)'
        )
        parser.add_argument(
            '--set-active',
            action='store_true',
            help='Set this model as active after training'
        )

    def handle(self, *args, **options):
        model_name = options['model_name']
        from_directory = options.get('from_directory')
        set_active = options['set_active']

        # Initialize face recognition system
        self.stdout.write('Initializing Face Recognition System...')
        fr_system = FaceRecognitionSystem()
        fr_system.initialize_arcface()

        if from_directory:
            # Train from directory
            self.stdout.write(f'Loading faces from directory: {from_directory}')
            X, Y = fr_system.load_faces_from_directory(from_directory)
            
            if len(X) == 0:
                self.stdout.write(self.style.ERROR('No faces found in directory!'))
                return

            self.stdout.write(self.style.SUCCESS(
                f'Loaded {len(X)} faces from {len(set(Y))} people'
            ))

            # Generate embeddings
            self.stdout.write('Generating face embeddings...')
            embeddings = fr_system.generate_embeddings(X)
            
        else:
            # Train from database
            self.stdout.write('Loading face embeddings from database...')
            
            # Get all students with face embeddings
            students_with_faces = Student.objects.filter(
                face_embeddings__isnull=False
            ).distinct()

            if not students_with_faces.exists():
                self.stdout.write(self.style.ERROR(
                    'No students with face embeddings found in database!'
                ))
                return

            embeddings_list = []
            labels_list = []

            for student in students_with_faces:
                face_embeddings = student.face_embeddings.all()
                for face_emb in face_embeddings:
                    embedding = face_emb.get_embedding()
                    embeddings_list.append(embedding)
                    labels_list.append(student.student_id)

            embeddings = np.array(embeddings_list)
            Y = np.array(labels_list)

            self.stdout.write(self.style.SUCCESS(
                f'Loaded {len(embeddings)} embeddings from {len(students_with_faces)} students'
            ))

        # Train the model
        self.stdout.write('Training SVM classifier...')
        
        # Create models directory if it doesn't exist
        os.makedirs(settings.FACE_MODELS_DIR, exist_ok=True)
        
        model_path = settings.FACE_MODELS_DIR / f'{model_name}.pkl'
        accuracy = fr_system.train_model(embeddings, Y, str(model_path))

        # Save model information to database
        encoder_path = str(model_path).replace('.pkl', '_encoder.pkl')
        
        face_model = FaceRecognitionModel.objects.create(
            model_name=model_name,
            model_file=str(model_path),
            encoder_file=encoder_path,
            accuracy=accuracy,
            is_active=set_active
        )

        if set_active:
            # Deactivate all other models
            FaceRecognitionModel.objects.exclude(id=face_model.id).update(is_active=False)
            self.stdout.write(self.style.SUCCESS(f'Set {model_name} as active model'))

        self.stdout.write(self.style.SUCCESS(
            f'\nTraining complete!\n'
            f'Model: {model_name}\n'
            f'Accuracy: {accuracy * 100:.2f}%\n'
            f'Model file: {model_path}\n'
            f'Encoder file: {encoder_path}'
        ))
