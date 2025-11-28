"""
Tests for the Flask API.
"""

import pytest
import sys
import os
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import app, load_model
from train import FakeNewsClassifier
import app as app_module


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def trained_model():
    """Create and save a trained model."""
    texts = ["Real news story about science", "Fake conspiracy theory"] * 10
    labels = [0, 1] * 10
    
    classifier = FakeNewsClassifier()
    classifier.train(texts, labels)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        filepath = f.name
    
    classifier.save(filepath)
    
    yield filepath, classifier
    
    # Cleanup
    if os.path.exists(filepath):
        os.unlink(filepath)


class TestHomeEndpoint:
    """Tests for home endpoint."""
    
    def test_home_returns_api_info(self, client):
        """Test that home endpoint returns API info."""
        response = client.get('/')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'name' in data
        assert 'version' in data
        assert 'endpoints' in data


class TestHealthEndpoint:
    """Tests for health endpoint."""
    
    def test_health_returns_status(self, client):
        """Test that health endpoint returns status."""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'status' in data
        assert data['status'] == 'healthy'


class TestPredictEndpoint:
    """Tests for predict endpoint."""
    
    def test_predict_without_model(self, client):
        """Test predict returns error when model not loaded."""
        # Ensure no model is loaded
        app_module.model = None
        
        response = client.post('/predict', 
                              json={'text': 'Test news article'})
        assert response.status_code == 503
    
    def test_predict_missing_text(self, client, trained_model):
        """Test predict returns error when text is missing."""
        filepath, classifier = trained_model
        app_module.model = classifier
        
        response = client.post('/predict', json={})
        assert response.status_code == 400
        
        data = response.get_json()
        assert 'error' in data
    
    def test_predict_success(self, client, trained_model):
        """Test successful prediction."""
        filepath, classifier = trained_model
        app_module.model = classifier
        
        response = client.post('/predict',
                              json={'text': 'Scientists discover new findings'})
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'prediction' in data
        assert 'label' in data
        assert data['prediction'] in ['fake', 'real']
        assert data['label'] in [0, 1]


class TestBatchPredictEndpoint:
    """Tests for batch predict endpoint."""
    
    def test_batch_predict_success(self, client, trained_model):
        """Test successful batch prediction."""
        filepath, classifier = trained_model
        app_module.model = classifier
        
        response = client.post('/batch_predict',
                              json={'texts': ['Article 1', 'Article 2']})
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'predictions' in data
        assert len(data['predictions']) == 2
    
    def test_batch_predict_missing_texts(self, client, trained_model):
        """Test batch predict with missing texts field."""
        filepath, classifier = trained_model
        app_module.model = classifier
        
        response = client.post('/batch_predict', json={})
        assert response.status_code == 400
