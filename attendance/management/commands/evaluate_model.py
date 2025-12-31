import os
import numpy as np
from django.core.management.base import BaseCommand
from attendance.models import FaceRecognitionModel, Student, FaceEmbedding
from attendance.utils.face_recognition import FaceRecognitionSystem
from django.conf import settings
import time
from collections import defaultdict


class Command(BaseCommand):
    help = 'Evaluate face recognition model performance with detailed metrics'

    def add_arguments(self, parser):
        parser.add_argument(
            '--test-split',
            type=float,
            default=0.3,
            help='Percentage of data to use for testing (default: 0.3 = 30%%)'
        )
        parser.add_argument(
            '--threshold',
            type=float,
            default=0.5,
            help='Confidence threshold for recognition (default: 0.5)'
        )

    def handle(self, *args, **options):
        test_split = options['test_split']
        threshold = options['threshold']
        
        self.stdout.write(self.style.SUCCESS('\n' + '='*70))
        self.stdout.write(self.style.SUCCESS('FACE RECOGNITION MODEL PERFORMANCE EVALUATION'))
        self.stdout.write(self.style.SUCCESS('='*70))
        
        # Get active model
        try:
            active_model = FaceRecognitionModel.objects.get(is_active=True)
        except FaceRecognitionModel.DoesNotExist:
            self.stdout.write(self.style.ERROR('No active model found!'))
            return
        
        self.stdout.write(f'\nModel: {active_model.model_name}')
        self.stdout.write(f'Trained: {active_model.created_at}')
        self.stdout.write(f'Test Split: {test_split*100:.0f}%')
        self.stdout.write(f'Confidence Threshold: {threshold}\n')
        
        # Initialize face recognition system
        fr_system = FaceRecognitionSystem()
        fr_system.initialize_arcface()
        fr_system.load_model(active_model.model_file)
        
        # Load test data
        self.stdout.write(self.style.WARNING('\n📁 Loading test data...'))
        faces_dir = settings.FACES_DIR
        
        X_train, Y_train, X_test, Y_test = self.load_train_test_split(
            fr_system, faces_dir, test_split
        )
        
        if len(X_test) == 0:
            self.stdout.write(self.style.ERROR('No test data available!'))
            return
        
        self.stdout.write(f'✓ Training samples: {len(X_train)}')
        self.stdout.write(f'✓ Testing samples: {len(X_test)}')
        self.stdout.write(f'✓ Number of people: {len(set(Y_test))}')
        
        # Generate embeddings
        self.stdout.write(self.style.WARNING('\n🔄 Generating embeddings...'))
        start_time = time.time()
        X_test_embeddings = fr_system.generate_embeddings(X_test)
        embedding_time = time.time() - start_time
        
        avg_embedding_time = embedding_time / len(X_test) * 1000
        self.stdout.write(f'✓ Average embedding time: {avg_embedding_time:.2f} ms/image')
        
        # Make predictions
        self.stdout.write(self.style.WARNING('\n🎯 Running predictions...'))
        start_time = time.time()
        predictions = []
        confidences = []
        
        for embedding in X_test_embeddings:
            name, confidence = fr_system.predict(embedding, threshold)
            predictions.append(name)
            confidences.append(confidence)
        
        prediction_time = time.time() - start_time
        avg_prediction_time = prediction_time / len(X_test) * 1000
        self.stdout.write(f'✓ Average prediction time: {avg_prediction_time:.2f} ms/prediction')
        
        # Calculate metrics
        self.stdout.write(self.style.SUCCESS('\n' + '='*70))
        self.stdout.write(self.style.SUCCESS('PERFORMANCE METRICS'))
        self.stdout.write(self.style.SUCCESS('='*70))
        
        # Overall accuracy
        correct = sum(1 for pred, true in zip(predictions, Y_test) if pred == true)
        total = len(Y_test)
        accuracy = (correct / total) * 100
        
        # False positives, false negatives, true positives, true negatives
        tp = sum(1 for pred, true in zip(predictions, Y_test) if pred == true and pred is not None)
        fp = sum(1 for pred, true in zip(predictions, Y_test) if pred != true and pred is not None)
        fn = sum(1 for pred, true in zip(predictions, Y_test) if pred is None and true is not None)
        tn = sum(1 for pred, true in zip(predictions, Y_test) if pred is None and true is None)
        
        # Calculate rates
        far = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0  # False Acceptance Rate
        frr = (fn / (fn + tp) * 100) if (fn + tp) > 0 else 0  # False Rejection Rate
        
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        self.stdout.write(f'\n📊 Overall Metrics:')
        self.stdout.write(f'  ├─ Recognition Accuracy: {accuracy:.2f}%')
        self.stdout.write(f'  ├─ Precision: {precision:.2f}%')
        self.stdout.write(f'  ├─ Recall: {recall:.2f}%')
        self.stdout.write(f'  └─ F1-Score: {f1_score:.2f}%')
        
        self.stdout.write(f'\n❌ Error Rates:')
        self.stdout.write(f'  ├─ False Acceptance Rate (FAR): {far:.2f}%')
        self.stdout.write(f'  └─ False Rejection Rate (FRR): {frr:.2f}%')
        
        self.stdout.write(f'\n🔢 Confusion Matrix:')
        self.stdout.write(f'  ├─ True Positives (TP): {tp}')
        self.stdout.write(f'  ├─ False Positives (FP): {fp}')
        self.stdout.write(f'  ├─ True Negatives (TN): {tn}')
        self.stdout.write(f'  └─ False Negatives (FN): {fn}')
        
        # Confidence statistics
        valid_confidences = [c for c in confidences if c > 0]
        if valid_confidences:
            avg_confidence = np.mean(valid_confidences)
            min_confidence = np.min(valid_confidences)
            max_confidence = np.max(valid_confidences)
            std_confidence = np.std(valid_confidences)
            
            self.stdout.write(f'\n📈 Confidence Statistics:')
            self.stdout.write(f'  ├─ Average: {avg_confidence:.4f}')
            self.stdout.write(f'  ├─ Minimum: {min_confidence:.4f}')
            self.stdout.write(f'  ├─ Maximum: {max_confidence:.4f}')
            self.stdout.write(f'  └─ Std Dev: {std_confidence:.4f}')
        
        # Per-person accuracy
        self.stdout.write(f'\n👤 Per-Person Performance:')
        person_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidences': []})
        
        for pred, true, conf in zip(predictions, Y_test, confidences):
            person_stats[true]['total'] += 1
            if pred == true:
                person_stats[true]['correct'] += 1
                person_stats[true]['confidences'].append(conf)
        
        for person in sorted(person_stats.keys()):
            stats = person_stats[person]
            person_acc = (stats['correct'] / stats['total']) * 100
            avg_conf = np.mean(stats['confidences']) if stats['confidences'] else 0
            self.stdout.write(
                f'  ├─ {person}: {person_acc:.1f}% ({stats["correct"]}/{stats["total"]}) '
                f'avg_conf: {avg_conf:.3f}'
            )
        
        # Performance timing
        self.stdout.write(f'\n⏱️  Performance Timing:')
        total_time = avg_embedding_time + avg_prediction_time
        self.stdout.write(f'  ├─ Face Detection + Embedding: {avg_embedding_time:.2f} ms')
        self.stdout.write(f'  ├─ Recognition: {avg_prediction_time:.2f} ms')
        self.stdout.write(f'  └─ Total per Image: {total_time:.2f} ms ({1000/total_time:.1f} FPS)')
        
        # Threshold analysis
        self.stdout.write(f'\n🎚️  Threshold Analysis:')
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        self.stdout.write(f'\n  Threshold | Accuracy | FAR    | FRR')
        self.stdout.write(f'  ----------|----------|--------|--------')
        
        for thresh in thresholds:
            thresh_preds = []
            for embedding in X_test_embeddings:
                name, _ = fr_system.predict(embedding, thresh)
                thresh_preds.append(name)
            
            thresh_correct = sum(1 for pred, true in zip(thresh_preds, Y_test) if pred == true)
            thresh_acc = (thresh_correct / total) * 100
            
            thresh_tp = sum(1 for pred, true in zip(thresh_preds, Y_test) if pred == true and pred is not None)
            thresh_fp = sum(1 for pred, true in zip(thresh_preds, Y_test) if pred != true and pred is not None)
            thresh_fn = sum(1 for pred, true in zip(thresh_preds, Y_test) if pred is None)
            thresh_tn = sum(1 for pred, true in zip(thresh_preds, Y_test) if pred is None and true is None)
            
            thresh_far = (thresh_fp / (thresh_fp + thresh_tn) * 100) if (thresh_fp + thresh_tn) > 0 else 0
            thresh_frr = (thresh_fn / (thresh_fn + thresh_tp) * 100) if (thresh_fn + thresh_tp) > 0 else 0
            
            marker = ' ←' if thresh == threshold else ''
            self.stdout.write(f'  {thresh:>8.1f}  | {thresh_acc:>6.2f}% | {thresh_far:>5.2f}% | {thresh_frr:>5.2f}%{marker}')
        
        # Summary
        self.stdout.write(self.style.SUCCESS('\n' + '='*70))
        self.stdout.write(self.style.SUCCESS('SUMMARY'))
        self.stdout.write(self.style.SUCCESS('='*70))
        
        if accuracy >= 95:
            status = self.style.SUCCESS('EXCELLENT ✓')
        elif accuracy >= 90:
            status = self.style.WARNING('GOOD ⚠')
        elif accuracy >= 80:
            status = self.style.WARNING('ACCEPTABLE ⚠')
        else:
            status = self.style.ERROR('NEEDS IMPROVEMENT ✗')
        
        self.stdout.write(f'\nModel Performance: {status}')
        self.stdout.write(f'Recognition Accuracy: {accuracy:.2f}%')
        self.stdout.write(f'F1-Score: {f1_score:.2f}%')
        self.stdout.write(f'Processing Speed: {1000/total_time:.1f} FPS\n')
        
        # Recommendations
        self.stdout.write(self.style.WARNING('💡 Recommendations:'))
        if accuracy < 90:
            self.stdout.write('  • Add more training images per person (5-10 recommended)')
            self.stdout.write('  • Ensure good lighting in training images')
            self.stdout.write('  • Capture images from different angles')
        if far > 5:
            self.stdout.write(f'  • High FAR ({far:.2f}%): Consider increasing threshold to {threshold + 0.1:.1f}')
        if frr > 10:
            self.stdout.write(f'  • High FRR ({frr:.2f}%): Consider decreasing threshold to {threshold - 0.1:.1f}')
        if total_time > 1000:
            self.stdout.write('  • Processing is slow: Consider using GPU acceleration')
        
        self.stdout.write(self.style.SUCCESS('\n' + '='*70 + '\n'))

    def load_train_test_split(self, fr_system, faces_dir, test_split):
        """Load data and split into train/test sets"""
        X_all = []
        Y_all = []
        
        for person_name in os.listdir(faces_dir):
            person_path = os.path.join(faces_dir, person_name)
            
            if not os.path.isdir(person_path):
                continue
            
            person_faces = []
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                face = fr_system.extract_face(img_path)
                if face is not None:
                    person_faces.append(face)
            
            # Split this person's faces into train/test
            n_test = max(1, int(len(person_faces) * test_split))
            indices = np.random.permutation(len(person_faces))
            
            test_indices = indices[:n_test]
            train_indices = indices[n_test:]
            
            for idx in train_indices:
                X_all.append(('train', person_faces[idx], person_name))
            
            for idx in test_indices:
                X_all.append(('test', person_faces[idx], person_name))
        
        # Separate train and test
        X_train = [item[1] for item in X_all if item[0] == 'train']
        Y_train = [item[2] for item in X_all if item[0] == 'train']
        X_test = [item[1] for item in X_all if item[0] == 'test']
        Y_test = [item[2] for item in X_all if item[0] == 'test']
        
        return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)
