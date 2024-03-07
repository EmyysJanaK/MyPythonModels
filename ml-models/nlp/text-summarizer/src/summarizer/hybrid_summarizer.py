"""
Hybrid Text Summarizer combining extractive and abstractive approaches.
Provides intelligent selection between methods based on text characteristics.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from .extractive_summarizer import ExtractiveSummarizer
from .abstractive_summarizer import AbstractiveSummarizer
import nltk
import textstat
import re


class HybridSummarizer:
    """Advanced hybrid summarizer that combines extractive and abstractive methods."""
    
    def __init__(self, 
                 abstractive_model: str = 'bart-large-cnn',
                 device: Optional[str] = None,
                 enable_abstractive: bool = True):
        """
        Initialize hybrid summarizer.
        
        Args:
            abstractive_model: Model to use for abstractive summarization
            device: Device for abstractive model
            enable_abstractive: Whether to enable abstractive summarization
        """
        self.extractive = ExtractiveSummarizer()
        self.enable_abstractive = enable_abstractive
        
        if enable_abstractive:
            try:
                self.abstractive = AbstractiveSummarizer(abstractive_model, device)
                print("Hybrid summarizer initialized with both extractive and abstractive capabilities")
            except Exception as e:
                print(f"Warning: Could not initialize abstractive summarizer: {e}")
                print("Falling back to extractive-only mode")
                self.enable_abstractive = False
                self.abstractive = None
        else:
            self.abstractive = None
            print("Hybrid summarizer initialized in extractive-only mode")
    
    def analyze_text_characteristics(self, text: str) -> Dict[str, Union[float, int, str]]:
        """
        Analyze text characteristics to determine optimal summarization approach.
        
        Returns:
            Dictionary with text analysis metrics
        """
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        analysis = {
            # Basic metrics
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'char_count': len(text),
            
            # Readability metrics
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'automated_readability_index': textstat.automated_readability_index(text),
            
            # Text complexity
            'avg_syllables_per_word': textstat.avg_sentence_length(text),
            'difficult_words': textstat.difficult_words(text),
            'text_standard': textstat.text_standard(text, float_output=False),
            
            # Structure analysis
            'has_headings': bool(re.search(r'^#+\s+', text, re.MULTILINE)),
            'has_bullet_points': bool(re.search(r'^\s*[•\-\*]\s+', text, re.MULTILINE)),
            'has_numbers': bool(re.search(r'\d+', text)),
            'has_quotes': bool(re.search(r'["\'].*["\']', text)),
            
            # Content type indicators
            'question_ratio': len(re.findall(r'\?', text)) / len(sentences) if sentences else 0,
            'exclamation_ratio': len(re.findall(r'!', text)) / len(sentences) if sentences else 0,
            'capital_ratio': len(re.findall(r'[A-Z]', text)) / len(text) if text else 0,
        }
        
        return analysis