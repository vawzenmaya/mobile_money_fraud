from flask import Flask, request, jsonify
import traceback
from utils import FraudDetector

app = Flask(__name__)
detector = FraudDetector()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = data.get('features', [])
                
        if len(features) != 9:
            return jsonify({"error": "Expected 9 features"}), 400
                    
        result = detector.predict(features)
        return jsonify(result)
            
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
