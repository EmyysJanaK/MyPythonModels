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

    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            print(f"Loading {self.model_name} model...")
            
            if self.use_pipeline:
                # Use HuggingFace pipeline (simpler)
                self.pipeline = pipeline(
                    "summarization",
                    model=f"facebook/{self.model_name}" if 'bart' in self.model_name else self.model_name,
                    device=0 if self.device == 'cuda' else -1,
                    framework="pt"
                )
            else:
                # Load model and tokenizer separately
                model_path = f"facebook/{self.model_name}" if 'bart' in self.model_name else self.model_name
                
                self.tokenizer = self.model_config['tokenizer'].from_pretrained(model_path)
                self.model = self.model_config['model'].from_pretrained(model_path)
                
                if self.device == 'cuda' and torch.cuda.is_available():
                    self.model = self.model.cuda()
                
            print(f"Model {self.model_name} loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to default summarization pipeline...")
            self.pipeline = pipeline("summarization", device=0 if self.device == 'cuda' else -1)
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for summarization."""
        # Clean up text
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s\.\!\?\;\:\,\-\(\)]', ' ', text)
        
        # Add special tokens for T5 models
        if 't5' in self.model_name.lower():
            text = f"summarize: {text}"
        
        return text
    
    def chunk_text(self, text: str, max_length: int, overlap: int = 50) -> List[str]:
        """Split text into chunks that fit model's max input length."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_length - overlap):
            chunk = ' '.join(words[i:i + max_length])
            chunks.append(chunk)
            
        return chunks
    
    def summarize_with_pipeline(self, 
                              text: str,
                              max_length: int = 150,
                              min_length: int = 30,
                              do_sample: bool = False,
                              temperature: float = 1.0,
                              top_p: float = 1.0) -> str:
        """Summarize using HuggingFace pipeline."""
        try:
            result = self.pipeline(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                truncation=True
            )
            
            return result[0]['summary_text']
            
        except Exception as e:
            print(f"Error in pipeline summarization: {e}")
            return self._fallback_summary(text, max_length)

    def summarize_with_model(self,
                           text: str,
                           max_length: int = 150,
                           min_length: int = 30,
                           num_beams: int = 4,
                           length_penalty: float = 2.0,
                           early_stopping: bool = True,
                           do_sample: bool = False,
                           temperature: float = 1.0,
                           top_p: float = 1.0) -> str:
        """Summarize using raw model (more control over parameters)."""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(
                text,
                return_tensors="pt",
                max_length=self.model_config['max_input'],
                truncation=True
            )
            if self.device == 'cuda':
                inputs = inputs.cuda()
            
            # Generate summary
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    early_stopping=early_stopping,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else 1.0,
                    top_p=top_p if do_sample else 1.0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode summary
            summary = self.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return summary
        
        except Exception as e:
            print(f"Error in model summarization: {e}")
            return self._fallback_summary(text, max_length)