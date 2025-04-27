import torch
import numpy as np
import joblib
import os
from models import BYOL, StudentModel

class FraudDetector:
    def __init__(self, model_dir="app/models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        
        # Load models
        self.byol = self._load_byol_model()
        self.classifier = self._load_classifier()
        self.student = self._load_student_model()
        
        print("✅ All models loaded successfully")
        
    def _load_byol_model(self):
        model = BYOL(input_dim=9).to(self.device)
        model.load_state_dict(torch.load(os.path.join(self.model_dir, 'byol.pth'), 
                                         map_location=self.device))
        model.eval()
        return model
        
    def _load_classifier(self):
        return joblib.load(os.path.join(self.model_dir, 'best_classifier_byol.pkl'))
        
    def _load_student_model(self):
        # Changed input dimension from 32 to 16 to match the saved model
        model = StudentModel(input_dim=16).to(self.device)
        model.load_state_dict(torch.load(os.path.join(self.model_dir, 'student_classifier_1.pth'), 
                                         map_location=self.device))
        model.eval()
        return model
    
    def preprocess(self, features):
        # Convert to tensor
        features_tensor = torch.FloatTensor([features]).to(self.device)
        return features_tensor
    
    def predict(self, features):
        # Preprocess input
        features_tensor = self.preprocess(features)
        
        # Get BYOL embeddings
        with torch.no_grad():
            byol_embeddings = self.byol.get_embeddings(features_tensor).cpu().numpy()
        
        # Get classifier prediction
        classifier_prob = self.classifier.predict_proba(byol_embeddings)[:, 1]
        
        # Since student model expects 16 dimensions, we need to adjust
        # Option 1: Use only the first 16 dimensions of BYOL embeddings
        student_input_data = byol_embeddings[:, :16]
        
        # Get student prediction
        with torch.no_grad():
            student_input = torch.FloatTensor(student_input_data).to(self.device)
            student_prob = self.student(student_input).cpu().numpy().flatten()[0]
        
        # Ensemble prediction (average of classifier and student)
        ensemble_prob = (classifier_prob[0] + student_prob) / 2
        
        # Decision
        is_fraud = bool(ensemble_prob > 0.5)
        
        return {
            "fraud_probability": float(ensemble_prob),
            "is_fraud": is_fraud,
            "classifier_prob": float(classifier_prob[0]),
            "student_prob": float(student_prob)
        }
