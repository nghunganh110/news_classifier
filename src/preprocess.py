import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


class TextPreprocessor:
    """Text preprocessing pipeline for news articles."""

    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """Lowercase, remove HTML tags, URLs, and special characters."""
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'<[^>]+>', ' ', text)           # Remove HTML tags
        text = re.sub(r'http\S+|www\.\S+', ' ', text)  # Remove URLs
        text = re.sub(r'\d+', ' ', text)                # Remove numbers
        text = re.sub(r'[^\w\s]', ' ', text)            # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()        # Collapse whitespace
        return text

    def tokenize(self, text: str) -> List[str]:
        """Basic tokenization."""
        try:
            tokens = word_tokenize(text)
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            tokens = word_tokenize(text)
        return tokens

    def remove_stopwords(self, text: str) -> str:
        """Remove English stopwords."""
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        return ' '.join(tokens)

    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline."""
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        return text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Process a list of texts."""
        return [self.preprocess(t) for t in texts]


if __name__ == '__main__':
    p = TextPreprocessor()
    sample = "Breaking News! Visit http://example.com for more. The <b>stock market</b> rose 3% today."
    print(p.preprocess(sample))
