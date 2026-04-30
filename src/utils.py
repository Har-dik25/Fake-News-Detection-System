# src/utils.py

import re
import string
import numpy as np
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure resources (safe to call multiple times)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

stop_words = set(stopwords.words('english')) - {'not'}
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

def preprocess_text(text: str) -> str:
    """Clean, normalize, lemmatize."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    words = text.split()
    return " ".join(
        lemmatizer.lemmatize(w)
        for w in words
        if w not in stop_words or w == 'not'
    )

def extract_features(text: str) -> np.ndarray:
    """Return 5 numerical features."""
    if not isinstance(text, str):
        return np.zeros(5)

    words = text.split()
    return np.array([
        len(words),                                  # word count
        len(text),                                   # char count
        sum(c.isupper() for c in text),              # uppercase count
        sum(c in string.punctuation for c in text),  # punctuation count
        sia.polarity_scores(text)['compound']        # sentiment
    ], dtype=float)