"""
src/predict.py — PyTorch inference for BiLSTM with Attention model.

Loads the trained BiLSTM model and vocabulary, then performs
interactive fake news prediction from the command line.
"""

import os
import sys
import pickle
import torch
import nltk
from nltk.tokenize import word_tokenize

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bilstm_attention import BiLSTMWithAttention
from src.data_loader import NewsDataset, prepare_sequences

# =====================
# CONFIGURATION
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "bilstm_attn.pth")
VOCAB_PATH = os.path.join(OUTPUT_DIR, "word_to_idx.pkl")
MAX_LEN = 512
EMBED_DIM = 128
HIDDEN_DIM = 256
N_LAYERS = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Ensure NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


def load_model():
    """Load trained BiLSTM model and vocabulary."""
    with open(VOCAB_PATH, "rb") as f:
        word_to_idx = pickle.load(f)

    model = BiLSTMWithAttention(
        vocab_size=len(word_to_idx),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        use_attention=True
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)

    return model, word_to_idx


def predict(text, model, word_to_idx):
    """
    Predict whether a news article is Real or Fake.

    Args:
        text: Raw news article text
        model: Loaded BiLSTM model
        word_to_idx: Vocabulary mapping

    Returns:
        tuple: (label, probability, attention_weights, tokens)
    """
    # Preprocess: lowercase, remove HTML, tokenize with NLTK
    loader = NewsDataset.__new__(NewsDataset)
    cleaned = NewsDataset.clean_text(text)
    tokens = word_tokenize(cleaned)

    # Pad/truncate to MAX_LEN
    seq = prepare_sequences([tokens], word_to_idx, max_len=MAX_LEN)
    tensor = torch.tensor(seq, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        prob, attn_weights = model(tensor)
        probability = prob.item()

    label = "Fake" if probability > 0.5 else "Real"

    # Extract attention weights for visualization
    weights = None
    if attn_weights is not None:
        weights = torch.mean(attn_weights[0], dim=0).cpu().numpy()
        weights = weights[:len(tokens)]

    return label, probability, weights, tokens


if __name__ == "__main__":
    print("Loading BiLSTM + Attention model...")
    model, word_to_idx = load_model()
    print("Model loaded. Ready for predictions.\n")

    while True:
        text = input("Enter news text (or 'exit' to quit): ").strip()
        if text.lower() == "exit":
            print("Goodbye.")
            break
        if not text:
            print("Please enter some text.\n")
            continue

        label, prob, weights, tokens = predict(text, model, word_to_idx)
        print(f"\n  Prediction: {label}")
        print(f"  Probability (fake): {prob:.4f}")

        if weights is not None:
            # Show top-5 attended tokens
            top_indices = weights.argsort()[-5:][::-1]
            top_tokens = [(tokens[i], weights[i]) for i in top_indices if i < len(tokens)]
            print("  Top attended words:", ", ".join(f"{t}({w:.3f})" for t, w in top_tokens))
        print()