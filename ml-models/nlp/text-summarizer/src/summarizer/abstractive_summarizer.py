"""
Abstractive Text Summarizer using transformer models.
Supports multiple pre-trained models and custom fine-tuning.
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration,
    PegasusTokenizer, PegasusForConditionalGeneration,
    pipeline
)
import logging
from typing import List, Dict, Optional, Union
import re
import warnings

class AbstractiveSummarizer:
    """Advanced abstractive text summarizer using transformer models."""
    
    # Available models and their configurations
    MODELS = {
        'bart-large-cnn': {
            'tokenizer': BartTokenizer,
            'model': BartForConditionalGeneration,
            'max_input': 1024,
            'description': 'BART model fine-tuned on CNN/DailyMail (good for news)'
        },
        't5-base': {
            'tokenizer': T5Tokenizer,
            'model': T5ForConditionalGeneration,
            'max_input': 512,
            'description': 'T5 base model (versatile, smaller)'
        },
        't5-large': {
            'tokenizer': T5Tokenizer,
            'model': T5ForConditionalGeneration,
            'max_input': 512,
            'description': 'T5 large model (better quality, slower)'
        },
        'pegasus-xsum': {
            'tokenizer': PegasusTokenizer,
            'model': PegasusForConditionalGeneration,
            'max_input': 512,
            'description': 'Pegasus model trained on XSum (good for short summaries)'
        },
        'pegasus-cnn_dailymail': {
            'tokenizer': PegasusTokenizer,
            'model': PegasusForConditionalGeneration,
            'max_input': 1024,
            'description': 'Pegasus model trained on CNN/DailyMail'
        }
    }

    def __init__(self, 
                 model_name: str = 'bart-large-cnn',
                 device: Optional[str] = None,
                 use_pipeline: bool = True):
        """
        Initialize the abstractive summarizer.
        
        Args:
            model_name: Name of the model to use
            device: Device to run the model on ('cpu', 'cuda', or None for auto)
            use_pipeline: Whether to use HuggingFace pipeline (easier) or raw models
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_pipeline = use_pipeline

        if model_name not in self.MODELS:
            raise ValueError(f"Model {model_name} not supported. Available: {list(self.MODELS.keys())}")
        
        self.model_config = self.MODELS[model_name]
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        self._load_model()