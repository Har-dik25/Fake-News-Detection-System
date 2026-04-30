import numpy as np
import re, pickle, string
import nltk

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from utils import preprocess_text, extract_features

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# LOAD
model = load_model("output/lstm_model.h5")
tokenizer = pickle.load(open("output/tokenizer.pkl", "rb"))
scaler = pickle.load(open("output/scaler.pkl", "rb"))



def predict(text):
    clean = preprocess_text(text)

    seq = tokenizer.texts_to_sequences([clean])
    pad = pad_sequences(seq, maxlen=100)

    num = scaler.transform([extract_features(text)])

    prob = model.predict([pad, num])[0][0]
    return "Real" if prob > 0.5 else "Fake"

while True:
    text = input("Enter text (or exit): ")
    if text.lower() == "exit":
        break
    print("Prediction:", predict(text))