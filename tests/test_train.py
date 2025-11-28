"""
Tests for the training module.
"""

import pytest
import sys
import os
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import FakeNewsClassifier


class TestFakeNewsClassifier:
    """Tests for FakeNewsClassifier class."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample training data."""
        texts = [
            "Scientists discover new cure for disease",
            "Breaking news: important research findings",
            "Study shows positive health outcomes",
            "Aliens invade earth and take over government",
            "Secret conspiracy revealed by insider",
            "Miracle cure hidden by big pharma"
        ]
        labels = [0, 0, 0, 1, 1, 1]  # 0 = real, 1 = fake
        return texts, labels
    
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        classifier = FakeNewsClassifier()
        assert classifier.classifier_type == 'logistic_regression'
        assert classifier.is_trained is False
    
    def test_classifier_types(self):
        """Test different classifier types."""
        for clf_type in ['logistic_regression', 'naive_bayes', 'svm']:
            classifier = FakeNewsClassifier(classifier_type=clf_type)
            assert classifier.classifier_type == clf_type
    
    def test_invalid_classifier_type(self):
        """Test that invalid classifier type raises error."""
        with pytest.raises(ValueError):
            FakeNewsClassifier(classifier_type='invalid')
    
    def test_train(self, sample_data):
        """Test training the classifier."""
        texts, labels = sample_data
        classifier = FakeNewsClassifier()
        classifier.train(texts, labels)
        assert classifier.is_trained is True
    
    def test_predict_before_training(self):
        """Test that prediction before training raises error."""
        classifier = FakeNewsClassifier()
        with pytest.raises(RuntimeError):
            classifier.predict(["Test text"])
    
    def test_predict_after_training(self, sample_data):
        """Test prediction after training."""
        texts, labels = sample_data
        classifier = FakeNewsClassifier()
        classifier.train(texts, labels)
        
        predictions = classifier.predict(["Scientists publish research findings"])
        assert len(predictions) == 1
        assert predictions[0] in [0, 1]
    
    def test_predict_single_text(self, sample_data):
        """Test prediction with single text string."""
        texts, labels = sample_data
        classifier = FakeNewsClassifier()
        classifier.train(texts, labels)
        
        prediction = classifier.predict("Test news article")
        assert len(prediction) == 1
    
    def test_save_and_load(self, sample_data):
        """Test saving and loading the model."""
        texts, labels = sample_data
        classifier = FakeNewsClassifier()
        classifier.train(texts, labels)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name
        
        try:
            # Save
            classifier.save(filepath)
            assert os.path.exists(filepath)
            
            # Load
            loaded_classifier = FakeNewsClassifier.load(filepath)
            assert loaded_classifier.is_trained is True
            
            # Predictions should match
            original_pred = classifier.predict(["Test text"])
            loaded_pred = loaded_classifier.predict(["Test text"])
            assert original_pred[0] == loaded_pred[0]
        finally:
            os.unlink(filepath)
    
    def test_save_untrained_model_raises_error(self):
        """Test that saving untrained model raises error."""
        classifier = FakeNewsClassifier()
        with pytest.raises(RuntimeError):
            classifier.save("test.pkl")


class TestLogisticRegression:
    """Tests specific to logistic regression classifier."""
    
    @pytest.fixture
    def trained_classifier(self):
        """Return a trained logistic regression classifier."""
        texts = ["Real news article", "Fake conspiracy theory"] * 10
        labels = [0, 1] * 10
        classifier = FakeNewsClassifier(classifier_type='logistic_regression')
        classifier.train(texts, labels)
        return classifier
    
    def test_predict_proba(self, trained_classifier):
        """Test probability prediction."""
        proba = trained_classifier.predict_proba(["Test article"])
        assert len(proba) == 1
        assert len(proba[0]) == 2  # Two classes
        assert abs(sum(proba[0]) - 1.0) < 0.01  # Probabilities sum to 1
