from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import json
import os
from utils import load_models, preprocess_input
import torch
import numpy as np
import io

app = Flask(__name__)

# Load models at startup
byol, best_classifier_byol, student_classifier_1 = load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_single', methods=['POST'])
def predict_single():
    try:
        # Get form data
        features = []
        for i in range(9):
            features.append(float(request.form.get(f'feature_{i}')))
        
        # Preprocess input data
        input_tensor = preprocess_input(features)
        
        # Generate BYOL embeddings
        with torch.no_grad():
            embeddings = byol.get_embeddings(input_tensor)
            embeddings_np = embeddings.cpu().numpy()
        
            # Get predictions from both classifiers
            byol_probs = best_classifier_byol.predict_proba(embeddings_np)
        
            # Since student model expects 16 dimensions, we need to adjust
            student_input_data = embeddings[:, :16]
            student_pred = student_classifier_1(student_input_data)
            student_probs = student_pred.detach().cpu().numpy()
        
        # Average the predictions (ensemble)
        byol_fraud_prob = byol_probs[:, 1] if byol_probs.shape[1] > 1 else byol_probs[:, 0]
        student_fraud_prob = student_probs.flatten()
        
        # Calculate ensemble probability
        ensemble_prob = (byol_fraud_prob + student_fraud_prob) / 2
        
        # Make prediction based on threshold
        predicted_class = int(ensemble_prob[0] > 0.5)
        
        result = {
            'prediction': int(predicted_class),
            'fraud_probability': float(ensemble_prob[0]),
            'byol_prob': float(byol_fraud_prob[0]),
            'student_prob': float(student_fraud_prob[0]),
            'features': features,
            'fraud_probability_width': f"{float(ensemble_prob[0]) * 100:.2f}",
            'byol_prob_width': f"{float(byol_fraud_prob[0]) * 100:.2f}",
            'student_prob_width': f"{float(student_fraud_prob[0]) * 100:.2f}"
        }
        
        return render_template('results.html', result=result, single=True)
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return render_template('index.html', error=str(e))

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error="No file selected")
        
        # Read CSV file
        if file.filename.endswith('.csv'):
            # Read the file content
            stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
            csv_data = pd.read_csv(stream, header=None)
            
            # Process each row
            results = []
            for _, row in csv_data.iterrows():
                # Extract features (assuming the first 9 columns are features)
                features = row.values[:9].tolist()
                
                # Preprocess input data
                input_tensor = preprocess_input(features)
                
                # Generate BYOL embeddings
                with torch.no_grad():
                    embeddings = byol.get_embeddings(input_tensor)
                    embeddings_np = embeddings.cpu().numpy()
                
                    # Get predictions from both classifiers
                    byol_probs = best_classifier_byol.predict_proba(embeddings_np)
                
                    # Since student model expects 16 dimensions, we need to adjust
                    student_input_data = embeddings[:, :16]
                    student_pred = student_classifier_1(student_input_data)
                    student_probs = student_pred.detach().cpu().numpy()
                
                # Average the predictions (ensemble)
                byol_fraud_prob = byol_probs[:, 1] if byol_probs.shape[1] > 1 else byol_probs[:, 0]
                student_fraud_prob = student_probs.flatten()
                
                # Calculate ensemble probability
                ensemble_prob = (byol_fraud_prob + student_fraud_prob) / 2
                
                # Make prediction based on threshold
                predicted_class = int(ensemble_prob[0] > 0.5)
                
                results.append({
                    'prediction': int(predicted_class),
                    'fraud_probability': float(ensemble_prob[0]),
                    'byol_prob': float(byol_fraud_prob[0]),
                    'student_prob': float(student_fraud_prob[0]),
                    'features': features,
                    'fraud_probability_width': f"{float(ensemble_prob[0]) * 100:.2f}",
                    'byol_prob_width': f"{float(byol_fraud_prob[0]) * 100:.2f}",
                    'student_prob_width': f"{float(student_fraud_prob[0]) * 100:.2f}"
                })
            
            return render_template('results.html', results=results, single=False)
        else:
            return render_template('index.html', error="Please upload a CSV file")
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
