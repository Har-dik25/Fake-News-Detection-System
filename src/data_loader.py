import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os

class NewsDataset:
    def __init__(self, filepath, max_len=200):
        self.filepath = filepath
        self.max_len = max_len
        self.df = None
        
    def load_data(self):
        print(f"Loading data from {self.filepath}...")
        self.df = pd.read_csv(self.filepath)
        # WELFake columns: [index, title, text, label]
        # Label: 0 for real, 1 for fake
        self.df = self.df.dropna(subset=['text', 'label'])
        self.df['content'] = self.df['title'].fillna('') + " " + self.df['text']
        return self.df

    @staticmethod
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        # 1. Lowercase
        text = text.lower()
        # 2. Remove HTML
        text = re.sub(r'<.*?>', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess(self, text):
        cleaned = self.clean_text(text)
        # 3. Tokenize with NLTK
        tokens = word_tokenize(cleaned)
        return tokens

    def get_binary_labels(self):
        return self.df['label'].values

def prepare_sequences(tokens_list, word_to_idx, max_len=200):
    sequences = []
    for tokens in tokens_list:
        seq = [word_to_idx.get(t, word_to_idx['<UNK>']) for t in tokens]
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq = seq + [word_to_idx['<PAD>']] * (max_len - len(seq))
        sequences.append(seq)
    return np.array(sequences)

if __name__ == "__main__":
    # Test data loader
    loader = NewsDataset("WELFake_Dataset.csv")
    df = loader.load_data()
    print(f"Loaded {len(df)} records.")
    sample = df['content'].iloc[0]
    print(f"Original: {sample[:100]}...")
    cleaned = loader.clean_text(sample)
    print(f"Cleaned: {cleaned[:100]}...")
    tokens = loader.preprocess(sample)
    print(f"Tokens: {tokens[:10]}")
