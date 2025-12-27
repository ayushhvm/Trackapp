import cv2 as cv
import os
import numpy as np
from mtcnn.mtcnn import MTCNN
from insightface.app import FaceAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
from django.conf import settings


class FaceRecognitionSystem:
    """
    Complete face recognition system using MTCNN for detection 
    and InsightFace (ArcFace) for embeddings
    """
    
    def __init__(self):
        self.target_size = (160, 160)
        self.detector = MTCNN()
        self.arcface_app = None
        self.model = None
        self.encoder = None
        
    def initialize_arcface(self):
        """Initialize ArcFace model for face embeddings"""
        if self.arcface_app is None:
            self.arcface_app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.arcface_app.prepare(ctx_id=0, det_size=(160, 160))
        return self.arcface_app
    
    def extract_face(self, image_path):
        """
        Extract and align face from image using MTCNN
        Returns normalized face array of size (160, 160, 3)
        """
        try:
            img = cv.imread(image_path)
            if img is None:
                return None
                
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            results = self.detector.detect_faces(img)
            
            if not results or len(results) == 0:
                return None
                
            x, y, w, h = results[0]['box']
            x, y = abs(x), abs(y)
            face = img[y:y+h, x:x+w]
            face_arr = cv.resize(face, self.target_size)
            return face_arr
            
        except Exception as e:
            print(f"Error extracting face from {image_path}: {str(e)}")
            return None
    
    def extract_face_from_array(self, img_array):
        """
        Extract face from numpy array (for real-time detection)
        """
        try:
            results = self.detector.detect_faces(img_array)
            
            if not results or len(results) == 0:
                return None, None
                
            x, y, w, h = results[0]['box']
            x, y = abs(x), abs(y)
            face = img_array[y:y+h, x:x+w]
            face_arr = cv.resize(face, self.target_size)
            
            return face_arr, (x, y, w, h)
            
        except Exception as e:
            print(f"Error extracting face: {str(e)}")
            return None, None
    
    def get_embedding(self, face_img):
        """
        Generate face embedding using InsightFace (ArcFace)
        Returns 512-dimensional embedding vector
        """
        try:
            if self.arcface_app is None:
                self.initialize_arcface()
            
            # Convert to BGR for InsightFace
            face_bgr = cv.cvtColor(face_img, cv.COLOR_RGB2BGR)
            faces = self.arcface_app.get(face_bgr)
            
            if faces and len(faces) > 0:
                return faces[0].embedding
            return None
            
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return None
    
    def load_faces_from_directory(self, directory):
        """
        Load all faces from directory structure
        Expected structure: directory/person_name/*.jpg
        Returns: X (face arrays), Y (labels)
        """
        X = []
        Y = []
        
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist")
            return np.array([]), np.array([])
        
        for person_name in os.listdir(directory):
            person_path = os.path.join(directory, person_name)
            
            if not os.path.isdir(person_path):
                continue
                
            faces_loaded = 0
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                face = self.extract_face(img_path)
                if face is not None:
                    X.append(face)
                    Y.append(person_name)
                    faces_loaded += 1
            
            print(f"Loaded {faces_loaded} faces for {person_name}")
        
        return np.asarray(X), np.asarray(Y)
    
    def generate_embeddings(self, faces):
        """
        Generate embeddings for array of face images
        """
        embeddings = []
        
        if self.arcface_app is None:
            self.initialize_arcface()
        
        for face in faces:
            embedding = self.get_embedding(face)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                # Add zero vector if embedding fails
                embeddings.append(np.zeros(512))
        
        return np.asarray(embeddings)
    
    def train_model(self, X_embeddings, Y_labels, save_path=None):
        """
        Train SVM classifier on face embeddings
        """
        # Encode labels
        self.encoder = LabelEncoder()
        Y_encoded = self.encoder.fit_transform(Y_labels)
        
        # Train SVM
        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(X_embeddings, Y_encoded)
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
        
        # Calculate accuracy
        predictions = self.model.predict(X_embeddings)
        accuracy = np.mean(predictions == Y_encoded)
        print(f"Training accuracy: {accuracy * 100:.2f}%")
        
        return accuracy
    
    def save_model(self, model_path):
        """
        Save trained model and encoder
        """
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save encoder
        encoder_path = model_path.replace('.pkl', '_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.encoder, f)
        
        print(f"Model saved to {model_path}")
        print(f"Encoder saved to {encoder_path}")
    
    def load_model(self, model_path):
        """
        Load trained model and encoder
        """
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load encoder
        encoder_path = model_path.replace('.pkl', '_encoder.pkl')
        with open(encoder_path, 'rb') as f:
            self.encoder = pickle.load(f)
        
        print(f"Model loaded from {model_path}")
        return True
    
    def predict(self, face_embedding, threshold=0.5):
        """
        Predict identity from face embedding
        Returns: (name, confidence) or (None, 0) if below threshold
        """
        if self.model is None or self.encoder is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        # Reshape embedding for prediction
        embedding = face_embedding.reshape(1, -1)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(embedding)[0]
        max_prob = np.max(probabilities)
        
        if max_prob < threshold:
            return None, max_prob
        
        # Get predicted class
        prediction = self.model.predict(embedding)[0]
        name = self.encoder.inverse_transform([prediction])[0]
        
        return name, max_prob
    
    def recognize_from_image(self, image_path, threshold=0.5):
        """
        Complete recognition pipeline from image path
        Returns recognition for the first face detected (for backward compatibility)
        """
        # Extract face
        face = self.extract_face(image_path)
        if face is None:
            return None, 0, "No face detected"
        
        # Generate embedding
        embedding = self.get_embedding(face)
        if embedding is None:
            return None, 0, "Failed to generate embedding"
        
        # Predict
        name, confidence = self.predict(embedding, threshold)
        
        if name is None:
            return None, confidence, "Confidence too low"
        
        return name, confidence, "Success"
    
    def recognize_all_faces_from_image(self, image_path, threshold=0.5):
        """
        Recognize all faces in an image
        Returns list of tuples: [(name, confidence, bbox, message), ...]
        """
        try:
            img = cv.imread(image_path)
            if img is None:
                return []
                
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            results = self.detector.detect_faces(img_rgb)
            
            if not results or len(results) == 0:
                return []
            
            recognitions = []
            for face_info in results:
                x, y, w, h = face_info['box']
                x, y = abs(x), abs(y)
                face = img_rgb[y:y+h, x:x+w]
                face_arr = cv.resize(face, self.target_size)
                
                # Generate embedding
                embedding = self.get_embedding(face_arr)
                if embedding is None:
                    recognitions.append((None, 0.0, (x, y, w, h), "Failed to generate embedding"))
                    continue
                
                # Predict
                name, confidence = self.predict(embedding, threshold)
                
                if name is None:
                    recognitions.append((None, confidence, (x, y, w, h), "Confidence too low"))
                else:
                    recognitions.append((name, confidence, (x, y, w, h), "Success"))
            
            return recognitions
            
        except Exception as e:
            print(f"Error recognizing faces from {image_path}: {str(e)}")
            return []
    
    def recognize_from_frame(self, frame, threshold=0.5):
        """
        Recognize face from video frame (numpy array)
        Returns: (name, confidence, bbox, message)
        """
        # Extract face
        face, bbox = self.extract_face_from_array(frame)
        if face is None:
            return None, 0, None, "No face detected"
        
        # Generate embedding
        embedding = self.get_embedding(face)
        if embedding is None:
            return None, 0, bbox, "Failed to generate embedding"
        
        # Predict
        name, confidence = self.predict(embedding, threshold)
        
        if name is None:
            return None, confidence, bbox, "Confidence too low"
        
        return name, confidence, bbox, "Success"


def train_face_recognition_model(faces_directory, output_model_path):
    """
    Helper function to train a new face recognition model
    """
    print("Initializing Face Recognition System...")
    fr_system = FaceRecognitionSystem()
    
    print(f"Loading faces from {faces_directory}...")
    X, Y = fr_system.load_faces_from_directory(faces_directory)
    
    if len(X) == 0:
        print("No faces found!")
        return None
    
    print(f"Loaded {len(X)} faces from {len(set(Y))} people")
    
    print("Generating embeddings...")
    embeddings = fr_system.generate_embeddings(X)
    
    print("Training model...")
    accuracy = fr_system.train_model(embeddings, Y, output_model_path)
    
    print(f"Training complete! Model saved to {output_model_path}")
    return fr_system
