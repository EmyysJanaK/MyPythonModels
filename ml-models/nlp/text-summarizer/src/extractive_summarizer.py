
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

        #Remove URLs, emails, and special characters
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S*@\S*\s?', '', text)
        text = re.sub(r'[^\w\s\.\!\?\;\:]', ' ', text)

        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text using NLTK."""

        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def calculate_sentences(self, text: str) -> List[str]:
        """Calculate sentences from the text."""
        
        text = self.preprocess_text(text)
        sentences = self.extract_sentences(text)
        return sentences
    
    def calculate_word_frequencies(self, sentences: List[str]) -> Dict[str, float]:
        """Calculate word frequencies for scoring."""
        word_freq = Counter()
        
        for sentence in sentences:
            words = nltk.word_tokenize(sentence.lower())
            words = [word for word in words if word.isalnum() and word not in self.stopwords]
            word_freq.update(words)
        
        # Normalize frequencies
        max_freq = max(word_freq.values()) if word_freq else 1
        return {word: freq / max_freq for word, freq in word_freq.items()}

    def score_sentences_frequency(self, sentences: List[str], word_freq: Dict[str, float]) -> List[float]:
        """Score sentences based on word frequency."""
        scores = []
        
        for sentence in sentences:
            words = nltk.word_tokenize(sentence.lower())
            words = [word for word in words if word.isalnum() and word not in self.stopwords]
            
            if not words:
                scores.append(0.0)
                continue
                
            sentence_score = sum(word_freq.get(word, 0) for word in words) / len(words)
            scores.append(sentence_score)