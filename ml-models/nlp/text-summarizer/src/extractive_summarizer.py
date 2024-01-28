
"""
Extractive Text Summarizer using multiple algorithms.
Combines TF-IDF, TextRank, and sentence position scoring.
"""

""" TF-IDF and sentence 
    ranking based summarization """	

# import numpy as np
import nltk
import re

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class ExtractiveSummarizer:
    """ Advanced Extractive Summarizer using multiple algorithms. """

    def __int__(self, language: str = 'english'):
        """ Initialize the summarizer with a specified language.
        Args:
            language (str): Language for stopwords and tokenization.
        """

        self.language = language
        self.stopwords = set(nltk.corpus.stopwords.words(language))

    def preprocess_text(self, text: str) -> str:
        """ Preprocess the input text by removing special characters and extra spaces.
        Args:
            text (str): Input text to preprocess.
        Returns:
            str: Cleaned text.
        """
        text = re.sub(r'\s+', ' ', text.strip())