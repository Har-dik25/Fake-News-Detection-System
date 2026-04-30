import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

class ClassicalFeatureExtractor:
    def __init__(self, max_tfidf_features=100):
        self.sid = SentimentIntensityAnalyzer()
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=max_tfidf_features)
        
    def fit_tfidf(self, texts):
        self.tfidf.fit(texts)
        
    def get_tfidf_features(self, texts):
        return self.tfidf.transform(texts).toarray()
        
    def get_pos_features(self, text):
        tokens = nltk.word_tokenize(text.lower())
        tags = nltk.pos_tag(tokens)
        
        adj_count = len([w for w, t in tags if t.startswith('JJ')])
        noun_count = len([w for w, t in tags if t.startswith('NN')])
        
        # Adjective-to-noun ratio
        ratio = adj_count / (noun_count + 1e-6)
        return [ratio, adj_count, noun_count]
        
    def get_sentiment_features(self, text):
        scores = self.sid.polarity_scores(text)
        return [scores['compound'], scores['pos'], scores['neg'], scores['neu']]
        
    def extract_all(self, text):
        pos = self.get_pos_features(text)
        sent = self.get_sentiment_features(text)
        return np.array(pos + sent)

    def save(self, filepath):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.tfidf, f)
            
    def load(self, filepath):
        import pickle
        with open(filepath, 'rb') as f:
            self.tfidf = pickle.load(f)

def extract_classical_features(df, text_col='content', max_tfidf=100, extractor=None):
    if extractor is None:
        extractor = ClassicalFeatureExtractor(max_tfidf_features=max_tfidf)
        print("Fitting TF-IDF...")
        extractor.fit_tfidf(df[text_col])
    
    print("Extracting TF-IDF features...")
    tfidf_feats = extractor.get_tfidf_features(df[text_col])
    
    print("Extracting POS and Sentiment features...")
    other_feats = []
    for text in df[text_col]:
        pos = extractor.get_pos_features(text)
        sent = extractor.get_sentiment_features(text)
        other_feats.append(pos + sent)
        
    other_feats = np.array(other_feats)
    # Concatenate all classical features
    combined = np.hstack([tfidf_feats, other_feats])
    return combined, extractor
