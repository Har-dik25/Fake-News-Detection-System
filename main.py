import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from transformers import BertTokenizer

# Import local modules
from src.data_loader import NewsDataset, prepare_sequences
from src.bilstm_attention import BiLSTMWithAttention, train_model, evaluate_model
from src.visualize_attention import get_attention_and_plot
from src.feature_extractor import extract_classical_features
from src.bert_hybrid import HybridBERTModel, train_bert_hybrid, evaluate_bert_hybrid

# =====================
# CONFIGURATION
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "WELFake_Dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SUBSET_SIZE = 1000  # Increased for better performance
MAX_LEN = 200
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class HybridDataset(Dataset):
    def __init__(self, input_ids, masks, feats, labels):
        self.input_ids = input_ids
        self.masks = masks
        self.feats = feats
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx] if self.input_ids is not None else None,
            'attention_mask': self.masks[idx] if self.masks is not None else None,
            'classical_features': torch.tensor(self.feats[idx], dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

def run_experiment():
    print(f"Starting experiment on {DEVICE}...")
    
    # 1. Load and Preprocess Data
    loader = NewsDataset(DATA_PATH, max_len=MAX_LEN)
    df = loader.load_data()
    df = df.sample(SUBSET_SIZE, random_state=42).reset_index(drop=True)
    
    print("Preprocessing text...")
    df['clean_tokens'] = df['content'].apply(loader.preprocess)
    df['clean_text'] = df['clean_tokens'].apply(lambda x: " ".join(x))
    
    # 2. BiLSTM Experiment
    print("\n--- BiLSTM Objective ---")
    vocab = set()
    for tokens in df['clean_tokens']: vocab.update(tokens)
    word_to_idx = {word: i+2 for i, word in enumerate(vocab)}
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = 1
    
    X_seq = prepare_sequences(df['clean_tokens'], word_to_idx, max_len=MAX_LEN)
    y = df['label'].values
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_seq, y, test_size=0.2, random_state=42)
    
    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    test_ds = TensorDataset(torch.tensor(X_te), torch.tensor(y_te))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # BiLSTM with Attention
    print("Training BiLSTM + Attention...")
    model_attn = BiLSTMWithAttention(len(word_to_idx), 128, 256, 2, use_attention=True)
    model_attn = train_model(model_attn, train_loader, None, epochs=3, device=DEVICE)
    acc_attn, auc_attn = evaluate_model(model_attn, test_loader, device=DEVICE)
    
    # BiLSTM without Attention
    print("Training BiLSTM (No Attention)...")
    model_no_attn = BiLSTMWithAttention(len(word_to_idx), 128, 256, 2, use_attention=False)
    model_no_attn = train_model(model_no_attn, train_loader, None, epochs=3, device=DEVICE)
    acc_no_attn, auc_no_attn = evaluate_model(model_no_attn, test_loader, device=DEVICE)
    
    print(f"BiLSTM+Attention: Acc={acc_attn:.4f}, AUC={auc_attn:.4f}")
    print(f"BiLSTM Alone:     Acc={acc_no_attn:.4f}, AUC={auc_no_attn:.4f}")
    
    # Attention Visualization
    sample_idx = 0
    sample_text = df['clean_text'].iloc[sample_idx]
    sample_tokens = df['clean_tokens'].iloc[sample_idx]
    sample_tensor = torch.tensor(X_seq[sample_idx]).unsqueeze(0)
    get_attention_and_plot(model_attn, sample_tensor, sample_tokens, device=DEVICE, filename="attention_visualization.png")

    # 3. Hybrid BERT Experiment
    print("\n--- Hybrid BERT Objective ---")
    classical_feats, extractor = extract_classical_features(df)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(df['clean_text'].tolist(), truncation=True, padding=True, max_length=MAX_LEN, return_tensors='pt')
    
    X_tr_idx, X_te_idx = train_test_split(range(len(df)), test_size=0.2, random_state=42)
    
    def get_hybrid_loader(indices, encodings, feats, labels):
        ds = HybridDataset(encodings['input_ids'][indices], encodings['attention_mask'][indices], feats[indices], labels[indices])
        return DataLoader(ds, batch_size=8, shuffle=True)
    
    train_loader_hybrid = get_hybrid_loader(X_tr_idx, encodings, classical_feats, y)
    test_loader_hybrid = get_hybrid_loader(X_te_idx, encodings, classical_feats, y)
    
    # Models: BERT Alone, Classical Alone, Hybrid
    results = {}
    
    print("Training Hybrid Model...")
    model_hybrid = HybridBERTModel(classical_feats.shape[1], mode='hybrid')
    model_hybrid = train_bert_hybrid(model_hybrid, train_loader_hybrid, epochs=1, device=DEVICE)
    f1_hybrid, report_hybrid = evaluate_bert_hybrid(model_hybrid, test_loader_hybrid, device=DEVICE)
    results['Hybrid'] = f1_hybrid
    
    print("Training BERT Alone...")
    model_bert = HybridBERTModel(classical_feats.shape[1], mode='bert_only')
    model_bert = train_bert_hybrid(model_bert, train_loader_hybrid, epochs=1, device=DEVICE)
    f1_bert, _ = evaluate_bert_hybrid(model_bert, test_loader_hybrid, device=DEVICE)
    results['BERT Alone'] = f1_bert
    
    print("Training Classical Alone...")
    model_class = HybridBERTModel(classical_feats.shape[1], mode='classical_only')
    model_class = train_bert_hybrid(model_class, train_loader_hybrid, epochs=5, device=DEVICE) # Classical is fast
    f1_class, _ = evaluate_bert_hybrid(model_class, test_loader_hybrid, device=DEVICE)
    results['Classical Alone'] = f1_class
    
    print("\nF1 Results:")
    for k, v in results.items(): print(f"{k}: {v:.4f}")
    
    # 4. Error Analysis (20 examples)
    print("\n--- Error Analysis ---")
    model_hybrid.eval()
    misclassified = []
    with torch.no_grad():
        for i in X_te_idx:
            batch = {
                'input_ids': encodings['input_ids'][i].unsqueeze(0).to(DEVICE),
                'attention_mask': encodings['attention_mask'][i].unsqueeze(0).to(DEVICE),
                'classical_features': torch.tensor(classical_feats[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            }
            prob = model_hybrid(**batch).item()
            pred = 1 if prob > 0.5 else 0
            actual = y[i]
            if pred != actual:
                misclassified.append({
                    'text': df['content'].iloc[i][:200],
                    'actual': 'Fake' if actual == 1 else 'Real',
                    'pred': 'Fake' if pred == 1 else 'Real'
                })
            if len(misclassified) >= 20: break
            
    print(f"Found {len(misclassified)} misclassified examples for analysis.")
    error_df = pd.DataFrame(misclassified)
    error_df.to_csv("output/error_analysis.csv", index=False)
    # Save BiLSTM model and vocab
    torch.save(model_attn.state_dict(), os.path.join(OUTPUT_DIR, "bilstm_attn.pth"))
    import pickle
    with open(os.path.join(OUTPUT_DIR, "word_to_idx.pkl"), "wb") as f:
        pickle.dump(word_to_idx, f)
    
    # Save BERT Hybrid model
    torch.save(model_hybrid.state_dict(), os.path.join(OUTPUT_DIR, "bert_hybrid.pth"))
    
    # Save Classical Extractor (TF-IDF)
    extractor.save(os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl"))
    
    print(f"\nModels and vectorizers saved to {OUTPUT_DIR}/ directory.")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_experiment()
