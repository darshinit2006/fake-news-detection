"""
Tests for the evaluation module.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluate import evaluate_model, get_classification_report


class TestEvaluateModel:
    """Tests for evaluate_model function."""
    
    def test_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        
        metrics = evaluate_model(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_imperfect_predictions(self):
        """Test evaluation with imperfect predictions."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 0]  # 50% accuracy
        
        metrics = evaluate_model(y_true, y_pred)
        
        assert metrics['accuracy'] == 0.5
    
    def test_confusion_matrix_format(self):
        """Test that confusion matrix is in correct format."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        
        metrics = evaluate_model(y_true, y_pred)
        
        assert isinstance(metrics['confusion_matrix'], list)
        assert len(metrics['confusion_matrix']) == 2
        assert len(metrics['confusion_matrix'][0]) == 2


class TestClassificationReport:
    """Tests for get_classification_report function."""
    
    def test_returns_string(self):
        """Test that classification report is a string."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        
        report = get_classification_report(y_true, y_pred)
        
        assert isinstance(report, str)
    
    def test_contains_class_names(self):
        """Test that report contains class names."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        
        report = get_classification_report(y_true, y_pred)
        
        assert 'Real' in report
        assert 'Fake' in report
    
    def test_custom_target_names(self):
        """Test report with custom target names."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        
        report = get_classification_report(y_true, y_pred, target_names=['Class0', 'Class1'])
        
        assert 'Class0' in report
        assert 'Class1' in report
