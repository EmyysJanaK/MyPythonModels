#!/usr/bin/env python3
"""
Command Line Interface for Text Summarizer
Provides a powerful CLI for text summarization with various options.
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from summarizer.extractive_summarizer import ExtractiveSummarizer
    from summarizer.abstractive_summarizer import AbstractiveSummarizer
    from summarizer.hybrid_summarizer import HybridSummarizer
except ImportError as e:
    print(f"Error importing summarizers: {e}")
    print("Please install required dependencies: pip install -r requirements.txt")
    sys.exit(1)


class TextSummarizerCLI:
    """Command line interface for text summarization."""
    
    def __init__(self):
        self.summarizers = {}
        self.initialize_summarizers()
    
    def initialize_summarizers(self):
        """Initialize available summarizers."""
        try:
            self.summarizers['extractive'] = ExtractiveSummarizer()
            print("✓ Extractive summarizer loaded")
        except Exception as e:
            print(f"✗ Failed to load extractive summarizer: {e}")
        
        try:
            self.summarizers['abstractive'] = AbstractiveSummarizer()
            print("✓ Abstractive summarizer loaded")
        except Exception as e:
            print(f"✗ Failed to load abstractive summarizer: {e}")
        
        try:
            self.summarizers['hybrid'] = HybridSummarizer()
            print("✓ Hybrid summarizer loaded")
        except Exception as e:
            print(f"✗ Failed to load hybrid summarizer: {e}")
    
    def read_text_file(self, file_path: str) -> str:
        """Read text from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {e}")
    
    def save_summary(self, summary: str, output_path: str):
        """Save summary to file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"✓ Summary saved to: {output_path}")
        except Exception as e:
            print(f"✗ Error saving summary: {e}")
    
    def print_summary_stats(self, original: str, summary: str, processing_time: float):
        """Print summary statistics."""
        original_words = len(original.split())
        summary_words = len(summary.split())
        compression_ratio = summary_words / original_words if original_words > 0 else 0
        
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"Original text:     {original_words:,} words")
        print(f"Summary:           {summary_words:,} words")
        print(f"Compression ratio: {compression_ratio:.2%}")
        print(f"Processing time:   {processing_time:.2f} seconds")
        print("="*50)
    
    def extractive_summarize(self, args):
        """Perform extractive summarization."""
        if 'extractive' not in self.summarizers:
            raise ValueError("Extractive summarizer not available")
        
        summarizer = self.summarizers['extractive']
        
        summary = summarizer.summarize(
            args.text,
            summary_ratio=args.ratio,
            max_sentences=args.max_sentences,
            algorithm=args.algorithm
        )
        
        return summary
    
    def abstractive_summarize(self, args):
        """Perform abstractive summarization."""
        if 'abstractive' not in self.summarizers:
            raise ValueError("Abstractive summarizer not available")
        
        summarizer = self.summarizers['abstractive']
        
        summary = summarizer.summarize(
            args.text,
            max_length=args.max_length,
            min_length=args.min_length,
            summary_style=args.style
        )
        
        return summary
    
    def hybrid_summarize(self, args):
        """Perform hybrid summarization."""
        if 'hybrid' not in self.summarizers:
            raise ValueError("Hybrid summarizer not available")
        
        summarizer = self.summarizers['hybrid']
        
        result = summarizer.summarize(
            args.text,
            approach=args.approach,
            summary_ratio=args.ratio,
            max_length=args.max_length,
            min_length=args.min_length
        )
        
        if args.verbose:
            print(f"\nRecommended approach: {result['recommended_approach']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Reasoning: {result['reasoning']}")
            print(f"Method used: {result['method_used']}")
        
        return result['final_summary']
    
    def compare_methods(self, args):
        """Compare different summarization methods."""
        print("Comparing summarization methods...\n")
        
        results = {}
        methods = ['extractive', 'abstractive', 'hybrid']
        
        for method in methods:
            if method in self.summarizers:
                print(f"Generating {method} summary...")
                start_time = time.time()