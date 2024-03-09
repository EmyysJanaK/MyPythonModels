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
        # Generate summaries based on approach
        if recommended_approach == 'extractive' or not self.enable_abstractive:
            extractive_summary = self.extractive.summarize(
                text, 
                summary_ratio=summary_ratio,
                algorithm=extractive_algorithm
            )
            results['summaries']['extractive'] = extractive_summary
            results['final_summary'] = extractive_summary
            results['method_used'] = 'extractive'
            
        elif recommended_approach == 'abstractive':
            abstractive_summary = self.abstractive.summarize(
                text,
                max_length=max_length,
                min_length=min_length,
                summary_style=abstractive_style,
                **kwargs
            )
            results['summaries']['abstractive'] = abstractive_summary
            results['final_summary'] = abstractive_summary
            results['method_used'] = 'abstractive'
            
        else:  # hybrid approach
            # Generate both summaries
            extractive_summary = self.extractive.summarize(
                text,
                summary_ratio=summary_ratio,
                algorithm=extractive_algorithm
            )
            
            if self.enable_abstractive:
                abstractive_summary = self.abstractive.summarize(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    summary_style=abstractive_style,
                    **kwargs
                )
            else:
                abstractive_summary = extractive_summary
            
            results['summaries']['extractive'] = extractive_summary
            results['summaries']['abstractive'] = abstractive_summary
            
            # Combine summaries
            combined_summary = self._combine_summaries(
                extractive_summary,
                abstractive_summary,
                method=combine_method,
                analysis=analysis
            )
            
            results['final_summary'] = combined_summary
            results['method_used'] = 'hybrid'
        
        return results
    
    def _combine_summaries(self, 
                          extractive: str, 
                          abstractive: str,
                          method: str = 'weighted',
                          analysis: Dict = None) -> str:
        """Combine extractive and abstractive summaries."""
        
        if method == 'best':
            # Choose the better summary based on length and coherence
            ext_sentences = len(nltk.sent_tokenize(extractive))
            abs_sentences = len(nltk.sent_tokenize(abstractive))
            
            # Prefer summary with better sentence structure
            if abs_sentences > 0 and len(abstractive) > len(extractive) * 0.7:
                return abstractive
            else:
                return extractive
                
        elif method == 'concatenate':
            # Simple concatenation with transition
            return f"{extractive} Additionally, {abstractive.lower()}"
            
        else:  # weighted combination
            # Create a hybrid by selecting sentences from both
            ext_sentences = nltk.sent_tokenize(extractive)
            abs_sentences = nltk.sent_tokenize(abstractive)
            
            # Weight based on text characteristics
            if analysis and analysis.get('word_count', 0) > 500:
                # For longer texts, prefer abstractive
                weight_abs = 0.7
            else:
                # For shorter texts, prefer extractive
                weight_abs = 0.3
            
            combined_sentences = []
            
            # Take weighted selection from both
            num_ext = max(1, int(len(ext_sentences) * (1 - weight_abs)))
            num_abs = max(1, int(len(abs_sentences) * weight_abs))
            
            combined_sentences.extend(ext_sentences[:num_ext])
            combined_sentences.extend(abs_sentences[:num_abs])
            
            return ' '.join(combined_sentences)
    
    def compare_approaches(self, text: str, **kwargs) -> Dict[str, str]:
        """Compare all available summarization approaches."""
        results = {}
        
        # Extractive summary
        results['extractive'] = self.extractive.summarize(text, **kwargs)
        
        # Abstractive summary (if available)
        if self.enable_abstractive:
            try:
                results['abstractive'] = self.abstractive.summarize(text, **kwargs)
            except Exception as e:
                results['abstractive'] = f"Error: {str(e)}"
        else:
            results['abstractive'] = "Not available"
        
        # Hybrid summary
        hybrid_result = self.summarize(text, approach='hybrid', **kwargs)
        results['hybrid'] = hybrid_result.get('final_summary', 'Error in hybrid summarization')
        
        return results

    def get_summary_statistics(self, original_text: str, summary: str) -> Dict[str, float]:
        """Calculate statistics comparing original text and summary."""
        orig_words = len(nltk.word_tokenize(original_text))
        orig_sentences = len(nltk.sent_tokenize(original_text))
        orig_chars = len(original_text)

        summ_words = len(nltk.word_tokenize(summary))
        summ_sentences = len(nltk.sent_tokenize(summary))
        summ_chars = len(summary)

        compression_ratio = summ_words / orig_words if orig_words > 0 else 0
        stats = {
            'original_word_count': orig_words,
            'original_sentence_count': orig_sentences,
            'original_char_count': orig_chars,
            'summary_word_count': summ_words,
            'summary_sentence_count': summ_sentences,
            'summary_char_count': summ_chars,
            'compression_ratio': compression_ratio,
            'word_reduction': orig_words - summ_words,
            'sentence_reduction': orig_sentences - summ_sentences
        }
        return stats