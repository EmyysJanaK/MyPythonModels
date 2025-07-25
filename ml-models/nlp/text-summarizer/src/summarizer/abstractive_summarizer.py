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

    def _fallback_summary(self, text: str, max_length: int) -> str:
        """Simple extractive fallback when models fail."""

        sentences = text.split('. ')
        if len(sentences) <= 3:
            return text
            
        # Return first few sentences as fallback
        words_count = 0
        summary_sentences = []
            
        for sentence in sentences:
            words_count += len(sentence.split())
            summary_sentences.append(sentence)
            
            if words_count >= max_length * 0.7:  # Approximate word count
                break
        
        return '. '.join(summary_sentences) + '.'   
    
    def summarize(self,
                  text: str,
                  max_length: int = 150,
                  min_length: int = 30,
                  summary_style: str = 'balanced',
                  **kwargs) -> str:
        """
        Generate abstractive summary with customizable parameters.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            summary_style: 'factual', 'balanced', or 'creative'
            **kwargs: Additional parameters for generation
            
        Returns:
            Generated summary
        """
        # Preprocess text
        clean_text = self.preprocess_text(text)

        # Handle long texts by chunking
        max_input = self.model_config['max_input']
        if len(clean_text.split()) > max_input:
            chunks = self.chunk_text(clean_text, max_input - 100)  # Leave room for special tokens
            summaries = []

            for chunk in chunks:
                if self.use_pipeline:
                    chunk_summary = self.summarize_with_pipeline(
                        chunk, 
                        max_length=max_length // len(chunks) + 50,
                        min_length=min_length // len(chunks),
                        **kwargs
                    )
                else:
                    chunk_summary = self.summarize_with_model(
                        chunk,
                        max_length=max_length // len(chunks) + 50,
                        min_length=min_length // len(chunks),
                        **kwargs
                    )
                summaries.append(chunk_summary)
            
             # Combine chunk summaries
            combined_summary = ' '.join(summaries)
            
            # Summarize the combined summary if it's still too long
            if len(combined_summary.split()) > max_length:
                return self.summarize(combined_summary, max_length, min_length, summary_style, **kwargs)
            
            return combined_summary
        
        # Set parameters based on style
        style_params = self._get_style_parameters(summary_style)
        kwargs.update(style_params)
        
        # Generate summary
        if self.use_pipeline:
            return self.summarize_with_pipeline(clean_text, max_length, min_length, **kwargs)
        else:
            return self.summarize_with_model(clean_text, max_length, min_length, **kwargs)
    
    def _get_style_parameters(self, style: str) -> Dict:
        """Get generation parameters based on summary style."""
        if style == 'factual':
            return {
                'do_sample': False,
                'num_beams': 5,
                'length_penalty': 2.0,
                'early_stopping': True
            }
        elif style == 'creative':
            return {
                'do_sample': True,
                'temperature': 1.2,
                'top_p': 0.9,
                'num_beams': 1,
                'length_penalty': 1.0
            }
        else:  # balanced
            return {
                'do_sample': False,
                'num_beams': 4,
                'length_penalty': 2.0,
                'early_stopping': True
            }
    def summarize_batch(self,
                       texts: List[str],
                       max_length: int = 150,
                       min_length: int = 30,
                       summary_style: str = 'balanced',
                       **kwargs) -> List[str]:
        """
        Generate summaries for a batch of texts.
        
        Args:
            texts: List of input texts to summarize
            max_length: Maximum length of each summary
            min_length: Minimum length of each summary
            summary_style: 'factual', 'balanced', or 'creative'
            **kwargs: Additional parameters for generation
            
        Returns:
            List of generated summaries
        """
        summaries = []
        for text in texts:
            summary = self.summarize(text, max_length, min_length, summary_style, **kwargs)
            summaries.append(summary)
        return summaries
    
    def compare_models(self, text: str, models: List[str] = None) -> Dict[str, str]:
        """Compare summaries from different models."""
        if models is None:
            models = ['bart-large-cnn', 't5-base']
        
        results = {}
        original_model = self.model_name
        
        for model_name in models:
            if model_name in self.MODELS:
                print(f"Testing {model_name}...")
                try:
                    # Temporarily switch model
                    self.model_name = model_name
                    self.model_config = self.MODELS[model_name]
                    self._load_model()
                    
                    # Generate summary
                    summary = self.summarize(text)
                    results[model_name] = summary
                    
                except Exception as e:
                    results[model_name] = f"Error: {str(e)}"
        # Restore original model
            self.model_name = original_model
            self.model_config = self.MODELS[original_model]
            self._load_model()
            
            return results
    
    @staticmethod
    def list_available_models() -> Dict[str, str]:
        """List all available models with descriptions."""
        return {name: config['description'] for name, config in AbstractiveSummarizer.MODELS.items()}
    
    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'max_input_length': self.model_config['max_input'],
            'description': self.model_config['description'],
            'use_pipeline': self.use_pipeline
        }