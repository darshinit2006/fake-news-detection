"""
Training module for fake news detection.
Uses TF-IDF vectorization and machine learning classifiers.
"""

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_text, preprocess_dataframe, download_nltk_data


class FakeNewsClassifier:
    """
    Fake news classifier using TF-IDF and machine learning.
    """
    
    CLASSIFIERS = {
        'logistic_regression': LogisticRegression,
        'naive_bayes': MultinomialNB,
        'svm': LinearSVC
    }
    
    def __init__(self, classifier_type='logistic_regression', max_features=5000):
        """
        Initialize the classifier.
        
        Args:
            classifier_type: Type of classifier to use
            max_features: Maximum number of TF-IDF features
        """
        self.classifier_type = classifier_type
        self.max_features = max_features
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Initialize classifier
        if classifier_type not in self.CLASSIFIERS:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        if classifier_type == 'logistic_regression':
            self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        elif classifier_type == 'naive_bayes':
            self.classifier = MultinomialNB()
        else:
            self.classifier = LinearSVC(random_state=42, max_iter=1000)
        
        self.is_trained = False
    
    def train(self, texts, labels):
        """
        Train the classifier on text data.
        
        Args:
            texts: List or Series of text samples
            labels: List or Series of labels (0=real, 1=fake)
            
        Returns:
            self for chaining
        """
        # Preprocess texts
        processed_texts = [preprocess_text(text) for text in texts]
        
        # Vectorize
        X = self.vectorizer.fit_transform(processed_texts)
        
        # Train classifier
        self.classifier.fit(X, labels)
        self.is_trained = True
        
        return self
    
    def predict(self, texts):
        """
        Predict labels for text samples.
        
        Args:
            texts: List or Series of text samples
            
        Returns:
            Array of predicted labels
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before prediction")
        
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess and vectorize
        processed_texts = [preprocess_text(text) for text in texts]
        X = self.vectorizer.transform(processed_texts)
        
        return self.classifier.predict(X)
    
    def predict_proba(self, texts):
        """
        Predict probability scores for text samples.
        
        Args:
            texts: List or Series of text samples
            
        Returns:
            Array of probability scores
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before prediction")
        
        if not hasattr(self.classifier, 'predict_proba'):
            raise ValueError(f"{self.classifier_type} does not support probability prediction")
        
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess and vectorize
        processed_texts = [preprocess_text(text) for text in texts]
        X = self.vectorizer.transform(processed_texts)
        
        return self.classifier.predict_proba(X)
    
    def save(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'classifier_type': self.classifier_type,
            'max_features': self.max_features
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded FakeNewsClassifier instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(
            classifier_type=model_data['classifier_type'],
            max_features=model_data['max_features']
        )
        instance.vectorizer = model_data['vectorizer']
        instance.classifier = model_data['classifier']
        instance.is_trained = True
        
        return instance


def train_model(data_path, text_column='text', label_column='label',
                classifier_type='logistic_regression', test_size=0.2,
                save_path=None):
    """
    Train a fake news classifier from a CSV file.
    
    Args:
        data_path: Path to the CSV file
        text_column: Name of the text column
        label_column: Name of the label column
        classifier_type: Type of classifier
        test_size: Fraction of data for testing
        save_path: Path to save the trained model
        
    Returns:
        Tuple of (classifier, X_train, X_test, y_train, y_test)
    """
    # Download NLTK data
    download_nltk_data()
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Clean data
    df = df.dropna(subset=[text_column, label_column])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_column], df[label_column],
        test_size=test_size, random_state=42
    )
    
    # Create and train classifier
    classifier = FakeNewsClassifier(classifier_type=classifier_type)
    classifier.train(X_train, y_train)
    
    # Save if path provided
    if save_path:
        classifier.save(save_path)
    
    return classifier, X_train, X_test, y_train, y_test


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python train.py <data_path> [model_save_path]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else 'models/model.pkl'
    
    print(f"Training model from {data_path}...")
    classifier, X_train, X_test, y_train, y_test = train_model(
        data_path, save_path=save_path
    )
    
    # Print training accuracy
    train_pred = classifier.predict(X_train)
    test_pred = classifier.predict(X_test)
    
    train_acc = (train_pred == y_train).mean()
    test_acc = (test_pred == y_test).mean()
    
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Model saved to {save_path}")
