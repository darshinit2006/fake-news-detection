"""
Tests for the preprocessing module.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import (
    clean_text,
    remove_stopwords,
    stem_text,
    preprocess_text
)


class TestCleanText:
    """Tests for clean_text function."""
    
    def test_lowercase(self):
        """Test that text is converted to lowercase."""
        result = clean_text("HELLO WORLD")
        assert result == "hello world"
    
    def test_remove_urls(self):
        """Test that URLs are removed."""
        result = clean_text("Check this http://example.com out")
        assert "http" not in result
        assert "example" not in result
    
    def test_remove_html(self):
        """Test that HTML tags are removed."""
        result = clean_text("<p>Hello</p> <b>World</b>")
        assert "<" not in result
        assert ">" not in result
    
    def test_remove_punctuation(self):
        """Test that punctuation is removed."""
        result = clean_text("Hello, world! How are you?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result
    
    def test_remove_numbers(self):
        """Test that numbers are removed."""
        result = clean_text("There are 123 apples")
        assert "123" not in result
    
    def test_empty_input(self):
        """Test handling of empty input."""
        assert clean_text("") == ""
        assert clean_text(None) == ""
    
    def test_whitespace_normalization(self):
        """Test that extra whitespace is normalized."""
        result = clean_text("Hello    world   test")
        assert "    " not in result


class TestRemoveStopwords:
    """Tests for remove_stopwords function."""
    
    def test_removes_common_stopwords(self):
        """Test that common stopwords are removed."""
        result = remove_stopwords("this is a test of the system")
        assert "this" not in result.split()
        assert "is" not in result.split()
        assert "a" not in result.split()
        assert "the" not in result.split()
    
    def test_preserves_content_words(self):
        """Test that content words are preserved."""
        result = remove_stopwords("the cat sat on the mat")
        assert "cat" in result
        assert "sat" in result
        assert "mat" in result
    
    def test_empty_input(self):
        """Test handling of empty input."""
        assert remove_stopwords("") == ""
        assert remove_stopwords(None) == ""


class TestStemText:
    """Tests for stem_text function."""
    
    def test_stems_words(self):
        """Test that words are stemmed."""
        result = stem_text("running runner runs")
        assert "run" in result
    
    def test_empty_input(self):
        """Test handling of empty input."""
        assert stem_text("") == ""
        assert stem_text(None) == ""


class TestPreprocessText:
    """Tests for preprocess_text function."""
    
    def test_full_pipeline(self):
        """Test the full preprocessing pipeline."""
        text = "Check this URL http://example.com! There are 5 people running."
        result = preprocess_text(text)
        
        # Should be lowercase
        assert result == result.lower()
        
        # Should not have URLs
        assert "http" not in result
        
        # Should not have punctuation
        assert "!" not in result
        
        # Should not have numbers
        assert "5" not in result
    
    def test_without_stopwords_removal(self):
        """Test preprocessing without stopword removal."""
        result = preprocess_text("this is a test", remove_stops=False)
        # Stopwords should still be present
        assert len(result) > 0
    
    def test_without_stemming(self):
        """Test preprocessing without stemming."""
        result = preprocess_text("running", apply_stemming=False)
        assert "running" in result or "run" in result
