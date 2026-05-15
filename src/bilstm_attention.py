import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.scaling = 1 / math.sqrt(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, hidden_dim]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Attention weights
        # [batch_size, seq_len, seq_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        if mask is not None:
            # mask shape: [batch_size, seq_len]
            # expand to [batch_size, seq_len, seq_len]
            attn_mask = mask.unsqueeze(1).expand_as(attn_weights)
            attn_weights = attn_weights.masked_fill(attn_mask == 0, -1e9)
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Output
        # [batch_size, seq_len, hidden_dim]
        out = torch.matmul(attn_weights, v)
        
        # For visualization, we can return the mean attention weights across queries
        # or just the weights if we want a specific visualization.
        # Often for classification, we attend to a "summary" or just pool.
        return out, attn_weights

class BiLSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout=0.3, use_attention=True):
        super(BiLSTMWithAttention, self).__init__()
        self.use_attention = use_attention
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, 
                           dropout=dropout if n_layers > 1 else 0, 
                           bidirectional=True, batch_first=True)
        
        # BiLSTM output dim is hidden_dim * 2
        self.attn = ScaledDotProductAttention(hidden_dim * 2) if use_attention else None
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch_size, seq_len]
        # Create mask: 1 for real tokens, 0 for padding (0 is padding_idx)
        mask = (x != 0).float().to(x.device)
        
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # lstm_out: [batch_size, seq_len, hidden_dim * 2]
        
        if self.use_attention:
            attn_out, attn_weights = self.attn(lstm_out, mask)
            # Pool across sequence dimension (mean pooling over valid tokens only)
            sum_pooled = torch.sum(attn_out * mask.unsqueeze(-1), dim=1)
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = sum_pooled / lengths
        else:
            # Standard BiLSTM pooling over valid tokens only
            sum_pooled = torch.sum(lstm_out * mask.unsqueeze(-1), dim=1)
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = sum_pooled / lengths
            attn_weights = None
            
        out = self.dropout(pooled)
        out = self.fc(out)
        return self.sigmoid(out), attn_weights

def train_model(model, train_loader, val_loader, epochs=5, lr=0.001, device='cpu'):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            outputs, _ = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
    return model

def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device).float().view(-1, 1)
            probs, _ = model(texts)
            preds = (probs > 0.5).int()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return acc, auc
