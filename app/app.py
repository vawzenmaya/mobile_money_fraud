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
        
        # Generate BYOL embeddings (feature extraction only)
        with torch.no_grad():
            embeddings = byol.get_embeddings(input_tensor)
            
            # Since student model expects 16 dimensions, we need to adjust
            student_input_data = embeddings[:, :16]
            
            # Get prediction from student model only
            student_pred = student_classifier_1(student_input_data)
            student_probs = student_pred.detach().cpu().numpy()
        
        # Use only student model prediction
        fraud_probability = float(student_probs.flatten()[0])
        
        # Make prediction based on threshold
        predicted_class = int(fraud_probability > 0.5)
        
        result = {
            'prediction': predicted_class,
            'fraud_probability': fraud_probability,
            'byol_prob': 0.0,  # Not using BYOL for prediction
            'student_prob': fraud_probability,
            'features': features,
            'fraud_probability_width': f"{fraud_probability * 100:.2f}",
            'byol_prob_width': "0.00",  # Not using BYOL for prediction
            'student_prob_width': f"{fraud_probability * 100:.2f}"
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
                
                # Generate BYOL embeddings (feature extraction only)
                with torch.no_grad():
                    embeddings = byol.get_embeddings(input_tensor)
                    
                    # Since student model expects 16 dimensions, we need to adjust
                    student_input_data = embeddings[:, :16]
                    
                    # Get prediction from student model only
                    student_pred = student_classifier_1(student_input_data)
                    student_probs = student_pred.detach().cpu().numpy()
                
                # Use only student model prediction
                fraud_probability = float(student_probs.flatten()[0])
                
                # Make prediction based on threshold
                predicted_class = int(fraud_probability > 0.5)
                
                results.append({
                    'prediction': predicted_class,
                    'fraud_probability': fraud_probability,
                    'byol_prob': 0.0,  # Not using BYOL for prediction
                    'student_prob': fraud_probability,
                    'features': features,
                    'fraud_probability_width': f"{fraud_probability * 100:.2f}",
                    'byol_prob_width': "0.00",  # Not using BYOL for prediction
                    'student_prob_width': f"{fraud_probability * 100:.2f}"
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
