import numpy as np
import pandas as pd
import re, string, os, pickle
import nltk

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from utils import preprocess_text, extract_features

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# =====================
# NLTK SETUP
# =====================
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("WELFake_Dataset.csv")
df = df.dropna(subset=['title', 'text', 'label'])
df['text'] = df['title'] + " " + df['text']

# =====================
# PREPROCESS
# =====================
df['clean'] = df['text'].apply(preprocess_text)

# =====================
# FEATURES
# =====================
X_num = np.array(df['clean'].apply(extract_features).tolist())
y = df['label'].values

# =====================
# TOKENIZATION
# =====================
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(df['clean'])

seqs = tokenizer.texts_to_sequences(df['clean'])
X_pad = pad_sequences(seqs, maxlen=100)

# =====================
# SCALING
# =====================
scaler = StandardScaler()
X_num = scaler.fit_transform(X_num)

# =====================
# SPLIT
# =====================
X_text_tr, X_text_te, X_num_tr, X_num_te, y_tr, y_te = train_test_split(
    X_pad, X_num, y, test_size=0.2, random_state=42
)

# =====================
# BASELINE MODELS
# =====================
print("\n==== BASELINE MODELS ====")

# Convert text to simple features for ML
X_flat_tr = X_text_tr.reshape(X_text_tr.shape[0], -1)
X_flat_te = X_text_te.reshape(X_text_te.shape[0], -1)

# Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(X_flat_tr, y_tr)
lr_pred = lr.predict(X_flat_te)
lr_prob = lr.predict_proba(X_flat_te)[:,1]

print("\nLogistic Regression:")
print(classification_report(y_te, lr_pred))
print("ROC-AUC:", roc_auc_score(y_te, lr_prob))

# Naive Bayes
nb = MultinomialNB()
nb.fit(np.abs(X_flat_tr), y_tr)  # NB needs non-negative
nb_pred = nb.predict(np.abs(X_flat_te))
nb_prob = nb.predict_proba(np.abs(X_flat_te))[:,1]

print("\nNaive Bayes:")
print(classification_report(y_te, nb_pred))
print("ROC-AUC:", roc_auc_score(y_te, nb_prob))

# =====================
# LSTM MODEL
# =====================
print("\n==== LSTM MODEL ====")

text_in = Input(shape=(100,))
x = Embedding(20000, 100)(text_in)
x = LSTM(128)(x)

num_in = Input(shape=(5,))
y2 = Dense(32, activation="relu")(num_in)

merged = Concatenate()([x, y2])
out = Dense(1, activation="sigmoid")(merged)

model = Model([text_in, num_in], out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(
    [X_text_tr, X_num_tr],
    y_tr,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

# =====================
# EVALUATION
# =====================
y_pred_prob = model.predict([X_text_te, X_num_te]).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nLSTM Results:")
print(classification_report(y_te, y_pred))
print("ROC-AUC:", roc_auc_score(y_te, y_pred_prob))

# =====================
# SAVE
# =====================
os.makedirs("output", exist_ok=True)

model.save("output/lstm_model.h5")
pickle.dump(tokenizer, open("output/tokenizer.pkl", "wb"))
pickle.dump(scaler, open("output/scaler.pkl", "wb"))

print("\nAll models evaluated. LSTM saved as final model.")