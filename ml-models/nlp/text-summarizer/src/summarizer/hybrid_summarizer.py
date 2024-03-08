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
    def recommend_approach(self, text_analysis: Dict) -> Tuple[str, float, str]:
        """
        Recommend the best summarization approach based on text analysis.
        
        Returns:
            Tuple of (approach, confidence, reasoning)
        """
        if not self.enable_abstractive:
            return 'extractive', 1.0, 'Abstractive summarization not available'
        
        score_extractive = 0
        score_abstractive = 0
        reasons = []
        
        # Length-based scoring
        word_count = text_analysis['word_count']
        if word_count < 200:
            score_extractive += 2
            reasons.append("Short text favors extractive")
        elif word_count > 1000:
            score_abstractive += 1
            reasons.append("Long text benefits from abstractive")
        
        # Readability scoring
        flesch_score = text_analysis['flesch_reading_ease']
        if flesch_score < 30:  # Very difficult
            score_abstractive += 2
            reasons.append("Complex text benefits from abstractive rephrasing")
        elif flesch_score > 70:  # Easy
            score_extractive += 1
            reasons.append("Simple text works well with extractive")
        
        # Structure scoring
        if text_analysis['has_headings'] or text_analysis['has_bullet_points']:
            score_extractive += 1
            reasons.append("Structured text suits extractive approach")
        
        # Content type scoring
        if text_analysis['question_ratio'] > 0.1:
            score_extractive += 1
            reasons.append("Question-heavy text suits extractive")
        
        # Technical content indicators
        if text_analysis['has_numbers'] and text_analysis['difficult_words'] > word_count * 0.1:
            score_extractive += 1
            reasons.append("Technical content with numbers favors extractive")
        
        # Sentence complexity
        avg_sentence_length = text_analysis['avg_sentence_length']
        if avg_sentence_length > 20:
            score_abstractive += 1
            reasons.append("Long sentences benefit from abstractive simplification")
        elif avg_sentence_length < 10:
            score_extractive += 1
            reasons.append("Short sentences work well with extractive")
        
        # Make recommendation
        if score_abstractive > score_extractive:
            approach = 'abstractive'
            confidence = min(0.9, 0.5 + (score_abstractive - score_extractive) * 0.1)
        elif score_extractive > score_abstractive:
            approach = 'extractive' 
            confidence = min(0.9, 0.5 + (score_extractive - score_abstractive) * 0.1)
        else:
            approach = 'hybrid'
            confidence = 0.6
            reasons.append("Mixed indicators suggest hybrid approach")
        
        reasoning = "; ".join(reasons)
        return approach, confidence, reasoning
    
    def summarize(self,
                  text: str,
                  approach: str = 'auto',
                  summary_ratio: float = 0.3,
                  max_length: int = 150,
                  min_length: int = 30,
                  extractive_algorithm: str = 'hybrid',
                  abstractive_style: str = 'balanced',
                  combine_method: str = 'weighted',
                  **kwargs) -> Dict[str, Union[str, Dict]]:
        """
        Generate summary using hybrid approach.
        
        Args:
            text: Input text
            approach: 'auto', 'extractive', 'abstractive', or 'hybrid'
            summary_ratio: Ratio for extractive summarization
            max_length: Max length for abstractive summarization
            min_length: Min length for abstractive summarization
            extractive_algorithm: Algorithm for extractive summarization
            abstractive_style: Style for abstractive summarization
            combine_method: How to combine approaches ('weighted', 'best', 'concatenate')
            
        Returns:
            Dictionary with summary and metadata
        """
        # Analyze text
        analysis = self.analyze_text_characteristics(text)
        
        # Determine approach
        if approach == 'auto':
            recommended_approach, confidence, reasoning = self.recommend_approach(analysis)
        else:
            recommended_approach = approach
            confidence = 1.0
            reasoning = f"User specified {approach} approach"
        
        results = {
            'analysis': analysis,
            'recommended_approach': recommended_approach,
            'confidence': confidence,
            'reasoning': reasoning,
            'summaries': {}
        }