"""
Evaluation module for fake news detection.
Provides metrics and analysis utilities.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary containing evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    return metrics


def get_classification_report(y_true, y_pred, target_names=None):
    """
    Generate a detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: List of label names
        
    Returns:
        Classification report as string
    """
    if target_names is None:
        target_names = ['Real', 'Fake']
    
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def print_evaluation(y_true, y_pred, target_names=None):
    """
    Print evaluation results to console.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: List of label names
    """
    metrics = evaluate_model(y_true, y_pred)
    
    print("=" * 50)
    print("Model Evaluation Results")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(f"  {metrics['confusion_matrix']}")
    print("\nDetailed Classification Report:")
    print(get_classification_report(y_true, y_pred, target_names))
