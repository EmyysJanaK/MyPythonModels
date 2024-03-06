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
    
    