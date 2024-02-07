
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
        return scores

    def score_sentences_tfidf(self, sentences: List[str]) -> List[float]:
        """Score sentences using TF-IDF."""
        if len(sentences) < 2:
            return [1.0] * len(sentences)
        try:
            vectorizer = TfidfVectorizer(
                stop_words='english',
                lowercase=True,
                max_features=1000,
                ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            
            # Normalize scores
            if sentence_scores.max() != 0:
                sentence_scores = sentence_scores / sentence_scores.max()
                
            return sentence_scores.tolist()

        except ValueError:
            # Fallback to frequency-based scoring
            word_freq = self.calculate_word_frequencies(sentences)
            return self.score_sentences_frequency(sentences, word_freq)
    
    def score_sentences_position(self, sentences: List[str]) -> List[float]:
        """Score sentences based on position (first and last sentences get higher scores)."""
        n = len(sentences)
        if n <= 2:
            return [1.0] * n
            
        scores = []
        for i in range(n):
            if i < 2:  # First two sentences
                score = 1.0
            elif i >= n - 2:  # Last two sentences
                score = 0.8
            elif i < n * 0.3:  # First third
                score = 0.6
            else:  # Middle sentences
                score = 0.3
            scores.append(score)
            
        return scores

    def score_sentences_length(self, sentences: List[str]) -> List[float]:
        """Score sentences based on length (prefer medium-length sentences)."""
        lengths = [len(nltk.word_tokenize(s)) for s in sentences]
        avg_length = np.mean(lengths)
        
        scores = []
        for length in lengths:
            if 10 <= length <= 25:  # Optimal range
                score = 1.0
            elif 5 <= length <= 35:  # Good range
                score = 0.8
            elif length < 5:  # Too short
                score = 0.3
            else:  # Too long
                score = 0.6
            scores.append(score)
            
        return scores
    
    def textrank_algorithm(self, sentences: List[str], similarity_threshold: float = 0.1) -> List[float]:
        """Implement TextRank algorithm for sentence ranking."""
        if len(sentences) < 2:
            return [1.0] * len(sentences)
            
        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            # Apply threshold
            similarity_matrix[similarity_matrix < similarity_threshold] = 0
            
            # Normalize the matrix
            for i in range(len(similarity_matrix)):
                if similarity_matrix[i].sum() != 0:
                    similarity_matrix[i] = similarity_matrix[i] / similarity_matrix[i].sum()
            
            # PageRank algorithm
            scores = np.ones(len(sentences))
            damping = 0.85
            
            for _ in range(50):  # iterations
                new_scores = (1 - damping) + damping * similarity_matrix.T.dot(scores)
                if np.allclose(scores, new_scores, atol=1e-6):
                    break
                scores = new_scores
            
            # Normalize scores
            if scores.max() != 0:
                scores = scores / scores.max()
                
            return scores.tolist()
            
        except Exception:
            # Fallback to TF-IDF scoring
            return self.score_sentences_tfidf(sentences)
    
    def combine_scores(self, score_lists: List[List[float]], weights: List[float]) -> List[float]:
        """Combine multiple scoring methods with weights."""
        combined_scores = []
        n_sentences = len(score_lists[0])
        
        for i in range(n_sentences):
            weighted_score = sum(
                scores[i] * weight 
                for scores, weight in zip(score_lists, weights)
            )
            combined_scores.append(weighted_score)
            
        return combined_scores
    
    def summarize(self, 
                  text: str, 
                  summary_ratio: float = 0.3,
                  max_sentences: Optional[int] = None,
                  min_sentences: int = 2,
                  algorithm: str = 'hybrid') -> str:
        """
        Generate extractive summary.
        
        Args:
            text: Input text to summarize
            summary_ratio: Ratio of sentences to keep (0.1 to 0.8)
            max_sentences: Maximum number of sentences in summary
            min_sentences: Minimum number of sentences in summary
            algorithm: 'frequency', 'tfidf', 'textrank', or 'hybrid'
            
        Returns:
            Summarized text
        """
        # Preprocess and extract sentences
        clean_text = self.preprocess_text(text)
        sentences = self.extract_sentences(clean_text)
        
        if len(sentences) <= min_sentences:
            return ' '.join(sentences)
        
        # Calculate number of sentences for summary
        num_sentences = max(
            min_sentences,
            min(
                int(len(sentences) * summary_ratio),
                max_sentences if max_sentences else len(sentences)
            )
        )
        
        # Score sentences based on algorithm
        if algorithm == 'frequency':
            word_freq = self.calculate_word_frequencies(sentences)
            scores = self.score_sentences_frequency(sentences, word_freq)
        elif algorithm == 'tfidf':
            scores = self.score_sentences_tfidf(sentences)
        elif algorithm == 'textrank':
            scores = self.textrank_algorithm(sentences)
        else:  # hybrid
            freq_scores = self.score_sentences_frequency(
                sentences, 
                self.calculate_word_frequencies(sentences)
            )
            tfidf_scores = self.score_sentences_tfidf(sentences)
            textrank_scores = self.textrank_algorithm(sentences)
            position_scores = self.score_sentences_position(sentences)
            length_scores = self.score_sentences_length(sentences)
            
            # Combine with weights
            scores = self.combine_scores(
                [freq_scores, tfidf_scores, textrank_scores, position_scores, length_scores],
                [0.2, 0.3, 0.3, 0.15, 0.05]  # Weights for each method
            )
        
        # Select top sentences
        sentence_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:num_sentences]
        
        # Sort selected sentences by original order
        sentence_indices.sort()
        
        # Generate summary
        summary_sentences = [sentences[i] for i in sentence_indices]
        return ' '.join(summary_sentences)