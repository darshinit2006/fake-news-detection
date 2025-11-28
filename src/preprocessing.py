"""
Preprocessing module for fake news detection.
Handles text cleaning, tokenization, and feature extraction.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def download_nltk_data():
    """Download required NLTK data."""
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)


def clean_text(text):
    """
    Clean and preprocess text for NLP analysis.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def remove_stopwords(text):
    """
    Remove stopwords from text.
    
    Args:
        text: Input text string
        
    Returns:
        Text with stopwords removed
    """
    if not isinstance(text, str) or not text:
        return ""
    
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        download_nltk_data()
        stop_words = set(stopwords.words('english'))
    
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    
    return ' '.join(filtered_words)


def stem_text(text):
    """
    Apply stemming to text.
    
    Args:
        text: Input text string
        
    Returns:
        Stemmed text
    """
    if not isinstance(text, str) or not text:
        return ""
    
    stemmer = PorterStemmer()
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    
    return ' '.join(stemmed_words)


def preprocess_text(text, remove_stops=True, apply_stemming=True):
    """
    Apply full preprocessing pipeline to text.
    
    Args:
        text: Input text string
        remove_stops: Whether to remove stopwords
        apply_stemming: Whether to apply stemming
        
    Returns:
        Preprocessed text
    """
    text = clean_text(text)
    
    if remove_stops:
        text = remove_stopwords(text)
    
    if apply_stemming:
        text = stem_text(text)
    
    return text


def preprocess_dataframe(df, text_column, label_column=None):
    """
    Preprocess a dataframe containing text data.
    
    Args:
        df: Pandas DataFrame
        text_column: Name of the column containing text
        label_column: Name of the column containing labels (optional)
        
    Returns:
        DataFrame with preprocessed text column added
    """
    df = df.copy()
    df['processed_text'] = df[text_column].apply(preprocess_text)
    
    # Remove empty processed texts
    df = df[df['processed_text'].str.len() > 0]
    
    return df
