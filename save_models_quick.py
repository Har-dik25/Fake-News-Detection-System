import torch
import os
from src.bilstm_attention import BiLSTMWithAttention
from src.bert_hybrid import HybridBERTModel
import pickle

os.makedirs('output', exist_ok=True)

# Define vocab size
VOCAB_SIZE = 5000

print(f"Saving weights with vocab size {VOCAB_SIZE}...")
bilstm = BiLSTMWithAttention(VOCAB_SIZE, 128, 256, 2)
torch.save(bilstm.state_dict(), 'output/bilstm_attn.pth')

bert = HybridBERTModel(107)
torch.save(bert.state_dict(), 'output/bert_hybrid.pth')

# Create a vocab that matches the size
word_to_idx = {f"word_{i}": i+2 for i in range(VOCAB_SIZE - 2)}
word_to_idx['<PAD>'] = 0
word_to_idx['<UNK>'] = 1

with open('output/word_to_idx.pkl', 'wb') as f:
    pickle.dump(word_to_idx, f)
    
print("Done.")
