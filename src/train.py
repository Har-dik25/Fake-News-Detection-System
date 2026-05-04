"""
src/train.py — DL Objective (CSR311)
BiLSTM with self-attention for fake news classification (PyTorch)

Requirements fulfilled:
  1. WELFake Dataset (available on Kaggle)
  2. Preprocess: lowercase, remove HTML, tokenize with NLTK, pad to 200 tokens
  3. BiLSTM (hidden=256, 2 layers) + scaled dot-product self-attention
  4. Binary classification (real/fake), dropout=0.3, BCELoss + Adam
  5. Plot attention heatmaps over article tokens
  6. Report: Accuracy, AUC-ROC. Compare BiLSTM+attention vs BiLSTM (no attention)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Add parent directory to path so we can run this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import NewsDataset, prepare_sequences
from src.bilstm_attention import BiLSTMWithAttention, train_model, evaluate_model
from src.visualize_attention import get_attention_and_plot

# =====================
# CONFIGURATION
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "WELFake_Dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SUBSET_SIZE = 5000       # Use a meaningful subset for training
MAX_LEN = 200            # Pad sequences to 200 tokens (assignment requirement)
BATCH_SIZE = 32
EMBED_DIM = 128
HIDDEN_DIM = 256         # Assignment requirement: hidden=256
N_LAYERS = 2             # Assignment requirement: 2 layers
DROPOUT = 0.3            # Assignment requirement: dropout=0.3
EPOCHS = 5
LR = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_vocab(token_lists):
    """Build word-to-index vocabulary from tokenized texts."""
    vocab = set()
    for tokens in token_lists:
        vocab.update(tokens)
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for i, word in enumerate(sorted(vocab)):
        word_to_idx[word] = i + 2
    return word_to_idx


def print_comparison_table(results):
    """Print a formatted comparison table of BiLSTM results."""
    print("\n" + "=" * 65)
    print("  DL OBJECTIVE — BiLSTM Comparison Results")
    print("=" * 65)
    print(f"  {'Model':<30} {'Accuracy':>10} {'AUC-ROC':>10}")
    print("-" * 65)
    for name, metrics in results.items():
        print(f"  {name:<30} {metrics['accuracy']:>10.4f} {metrics['auc']:>10.4f}")
    print("=" * 65)


def run_bilstm_training():
    """
    Full DL Objective training pipeline.
    Trains BiLSTM+Attention and BiLSTM (no attention), compares them,
    and generates attention heatmap visualizations.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATA_PATH}")

    # =====================
    # 1. LOAD AND PREPROCESS
    # =====================
    print("\n[Step 1] Loading and preprocessing data...")
    loader = NewsDataset(DATA_PATH, max_len=MAX_LEN)
    df = loader.load_data()

    # Use subset for tractable training time
    if SUBSET_SIZE and SUBSET_SIZE < len(df):
        df = df.sample(SUBSET_SIZE, random_state=42).reset_index(drop=True)
        print(f"  Using subset of {SUBSET_SIZE} samples.")
    else:
        print(f"  Using full dataset: {len(df)} samples.")

    # Tokenize with NLTK (lowercase + remove HTML handled inside preprocess)
    print("  Tokenizing with NLTK...")
    df['clean_tokens'] = df['content'].apply(loader.preprocess)
    df['clean_text'] = df['clean_tokens'].apply(lambda x: " ".join(x))

    # =====================
    # 2. BUILD VOCABULARY
    # =====================
    print("\n[Step 2] Building vocabulary...")
    word_to_idx = build_vocab(df['clean_tokens'])
    print(f"  Vocabulary size: {len(word_to_idx)}")

    # =====================
    # 3. PREPARE SEQUENCES
    # =====================
    print("\n[Step 3] Preparing padded sequences (max_len={})...".format(MAX_LEN))
    X_seq = prepare_sequences(df['clean_tokens'], word_to_idx, max_len=MAX_LEN)
    y = df['label'].values

    # Train/Test split (80/20)
    X_tr, X_te, y_tr, y_te = train_test_split(X_seq, y, test_size=0.2, random_state=42)

    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.long),
                             torch.tensor(y_tr, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_te, dtype=torch.long),
                            torch.tensor(y_te, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    results = {}

    # =====================
    # 4. TRAIN BiLSTM + ATTENTION
    # =====================
    print("\n[Step 4a] Training BiLSTM + Self-Attention...")
    print(f"  Config: hidden={HIDDEN_DIM}, layers={N_LAYERS}, dropout={DROPOUT}, "
          f"BCELoss + Adam(lr={LR}), epochs={EPOCHS}")

    model_attn = BiLSTMWithAttention(
        vocab_size=len(word_to_idx),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        use_attention=True
    )
    model_attn = train_model(model_attn, train_loader, None,
                             epochs=EPOCHS, lr=LR, device=DEVICE)
    acc_attn, auc_attn = evaluate_model(model_attn, test_loader, device=DEVICE)
    results['BiLSTM + Attention'] = {'accuracy': acc_attn, 'auc': auc_attn}
    print(f"  => Accuracy: {acc_attn:.4f}, AUC-ROC: {auc_attn:.4f}")

    # =====================
    # 5. TRAIN BiLSTM (NO ATTENTION)
    # =====================
    print("\n[Step 4b] Training BiLSTM (No Attention)...")
    model_no_attn = BiLSTMWithAttention(
        vocab_size=len(word_to_idx),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        use_attention=False
    )
    model_no_attn = train_model(model_no_attn, train_loader, None,
                                epochs=EPOCHS, lr=LR, device=DEVICE)
    acc_no, auc_no = evaluate_model(model_no_attn, test_loader, device=DEVICE)
    results['BiLSTM (No Attention)'] = {'accuracy': acc_no, 'auc': auc_no}
    print(f"  => Accuracy: {acc_no:.4f}, AUC-ROC: {auc_no:.4f}")

    # =====================
    # 6. COMPARISON TABLE
    # =====================
    print_comparison_table(results)

    # =====================
    # 7. ATTENTION HEATMAPS
    # =====================
    print("\n[Step 5] Generating attention heatmaps...")

    # Find a correctly-classified Fake example
    model_attn.eval()
    fake_indices = df[df['label'] == 1].index.tolist()
    real_indices = df[df['label'] == 0].index.tolist()

    for label_name, indices, fname in [
        ("Fake", fake_indices, "attention_heatmap_fake.png"),
        ("Real", real_indices, "attention_heatmap_real.png"),
    ]:
        for idx in indices[:20]:  # search within first 20
            sample_tensor = torch.tensor(X_seq[idx]).unsqueeze(0)
            with torch.no_grad():
                prob, _ = model_attn(sample_tensor.to(DEVICE))
            pred = 1 if prob.item() > 0.5 else 0
            if pred == df['label'].iloc[idx]:
                tokens = df['clean_tokens'].iloc[idx][:50]  # first 50 tokens for readability
                get_attention_and_plot(model_attn, sample_tensor, tokens,
                                      device=DEVICE, filename=fname)
                print(f"  Saved {fname} (correctly classified {label_name} article)")
                break

    # =====================
    # 8. SAVE MODELS
    # =====================
    print("\n[Step 6] Saving models and vocabulary...")
    torch.save(model_attn.state_dict(), os.path.join(OUTPUT_DIR, "bilstm_attn.pth"))
    torch.save(model_no_attn.state_dict(), os.path.join(OUTPUT_DIR, "bilstm_no_attn.pth"))

    with open(os.path.join(OUTPUT_DIR, "word_to_idx.pkl"), "wb") as f:
        pickle.dump(word_to_idx, f)

    # Save results to file
    with open(os.path.join(OUTPUT_DIR, "bilstm_results.txt"), "w") as f:
        f.write("DL Objective — BiLSTM Comparison Results\n")
        f.write("=" * 55 + "\n")
        f.write(f"{'Model':<30} {'Accuracy':>10} {'AUC-ROC':>10}\n")
        f.write("-" * 55 + "\n")
        for name, metrics in results.items():
            f.write(f"{name:<30} {metrics['accuracy']:>10.4f} {metrics['auc']:>10.4f}\n")
        f.write("=" * 55 + "\n")

    print(f"\n  All outputs saved to {OUTPUT_DIR}/")
    print("  DL Objective complete.\n")

    return model_attn, word_to_idx, results


if __name__ == "__main__":
    run_bilstm_training()