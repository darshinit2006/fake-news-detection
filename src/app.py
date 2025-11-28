"""
Flask API for fake news detection.
Provides REST endpoints for predictions.
"""

import os
from flask import Flask, request, jsonify

from train import FakeNewsClassifier


app = Flask(__name__)

# Global model variable
model = None

# Default model path
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/model.pkl')


def load_model():
    """Load the trained model."""
    global model
    if os.path.exists(MODEL_PATH):
        model = FakeNewsClassifier.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")


@app.route('/')
def home():
    """Home endpoint with API information."""
    return jsonify({
        'name': 'Fake News Detection API',
        'version': '1.0.0',
        'endpoints': {
            '/predict': 'POST - Predict if news is fake or real',
            '/batch_predict': 'POST - Predict multiple news articles',
            '/health': 'GET - Health check endpoint'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if news is fake or real.
    
    Request body:
        {
            "text": "News article text to analyze"
        }
    
    Response:
        {
            "prediction": "fake" or "real",
            "label": 0 (real) or 1 (fake),
            "confidence": probability score (if available)
        }
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train a model first.'
        }), 503
    
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({
            'error': 'Missing required field: text'
        }), 400
    
    text = data['text']
    
    if not text or not isinstance(text, str):
        return jsonify({
            'error': 'Text must be a non-empty string'
        }), 400
    
    # Make prediction
    prediction = model.predict([text])[0]
    
    response = {
        'prediction': 'fake' if prediction == 1 else 'real',
        'label': int(prediction)
    }
    
    # Add confidence score if available
    try:
        proba = model.predict_proba([text])[0]
        response['confidence'] = float(max(proba))
    except (ValueError, AttributeError):
        pass
    
    return jsonify(response)


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict multiple news articles.
    
    Request body:
        {
            "texts": ["Article 1", "Article 2", ...]
        }
    
    Response:
        {
            "predictions": [
                {"text": "...", "prediction": "fake/real", "label": 0/1},
                ...
            ]
        }
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train a model first.'
        }), 503
    
    data = request.get_json()
    
    if not data or 'texts' not in data:
        return jsonify({
            'error': 'Missing required field: texts'
        }), 400
    
    texts = data['texts']
    
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({
            'error': 'Texts must be a non-empty list'
        }), 400
    
    # Make predictions
    predictions = model.predict(texts)
    
    results = []
    for text, pred in zip(texts, predictions):
        results.append({
            'text': text[:100] + '...' if len(text) > 100 else text,
            'prediction': 'fake' if pred == 1 else 'real',
            'label': int(pred)
        })
    
    return jsonify({'predictions': results})


def create_app():
    """Create and configure the Flask application."""
    load_model()
    return app


if __name__ == '__main__':
    load_model()
    # Debug mode should only be enabled for development
    # In production, use a WSGI server like gunicorn
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
