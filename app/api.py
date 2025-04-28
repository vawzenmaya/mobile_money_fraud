from flask import Flask, request, jsonify
from utils import load_models, preprocess_input
import torch
import numpy as np

app = Flask(__name__)
# Load models at startup
byol, best_classifier_byol, student_classifier_1 = load_models()

@app.route('/')
def home():
    return """
    <h1>Fraud Detection API</h1>
    <p>Use POST /predict endpoint with JSON data to get predictions</p>
    """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
                
        if not data or not isinstance(data, dict) or 'features' not in data:
            return jsonify({'error': 'Invalid input. Please provide a JSON with "features" field.'}), 400
                
        # Preprocess input data
        features = data['features']
        input_tensor = preprocess_input(features)
                
        # Generate BYOL embeddings
        with torch.no_grad():
            embeddings = byol.get_embeddings(input_tensor)
            embeddings_np = embeddings.cpu().numpy()
                        
            # Get predictions from both classifiers
            # Use predict_proba for sklearn classifiers
            byol_probs = best_classifier_byol.predict_proba(embeddings_np)
            
            # Since student model expects 16 dimensions, we need to adjust
            student_input_data = embeddings[:, :16]
            student_pred = student_classifier_1(student_input_data)
            student_probs = student_pred.cpu().numpy()
                        
            # Average the predictions (ensemble)
            # Convert to numpy arrays for easier handling
            byol_fraud_prob = byol_probs[:, 1] if byol_probs.shape[1] > 1 else byol_probs[:, 0]
            student_fraud_prob = student_probs.flatten()
            
            # Calculate ensemble probability
            ensemble_prob = (byol_fraud_prob + student_fraud_prob) / 2
            
            # Make prediction based on threshold
            predicted_class = int(ensemble_prob[0] > 0.5)
                
        # Return prediction result
        return jsonify({
            'prediction': int(predicted_class),
            'fraud_probability': float(ensemble_prob[0]),
            'byol_prob': float(byol_fraud_prob[0]),
            'student_prob': float(student_fraud_prob[0]),
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Health check endpoint for Render
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
