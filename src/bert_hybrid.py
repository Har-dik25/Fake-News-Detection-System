import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from sklearn.metrics import f1_score, classification_report
import numpy as np

class HybridBERTModel(nn.Module):
    def __init__(self, classical_feature_dim, dropout=0.1, mode='hybrid'):
        super(HybridBERTModel, self).__init__()
        self.mode = mode
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        
        if mode == 'hybrid':
            # BERT CLS (768) + Classical features
            self.classifier = nn.Linear(768 + classical_feature_dim, 1)
        elif mode == 'bert_only':
            self.classifier = nn.Linear(768, 1)
        else: # classical_only
            self.classifier = nn.Linear(classical_feature_dim, 1)
            
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids=None, attention_mask=None, classical_features=None):
        if self.mode == 'bert_only' or self.mode == 'hybrid':
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.pooler_output # [batch_size, 768]
            cls_output = self.dropout(cls_output)
            
            if self.mode == 'hybrid':
                combined = torch.cat((cls_output, classical_features), dim=1)
                logits = self.classifier(combined)
            else:
                logits = self.classifier(cls_output)
        else: # classical_only
            logits = self.classifier(classical_features)
            
        return self.sigmoid(logits)

def train_bert_hybrid(model, train_loader, epochs=3, lr=2e-5, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch.get('input_ids')
            mask = batch.get('attention_mask')
            feats = batch.get('classical_features')
            labels = batch.get('labels').float().view(-1, 1)
            
            if input_ids is not None: input_ids = input_ids.to(device)
            if mask is not None: mask = mask.to(device)
            if feats is not None: feats = feats.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids, mask, feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    return model

def evaluate_bert_hybrid(model, test_loader, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch.get('input_ids')
            mask = batch.get('attention_mask')
            feats = batch.get('classical_features')
            labels = batch.get('labels').float().view(-1, 1)
            
            if input_ids is not None: input_ids = input_ids.to(device)
            if mask is not None: mask = mask.to(device)
            if feats is not None: feats = feats.to(device)
            labels = labels.to(device)
            
            probs = model(input_ids, mask, feats)
            preds = (probs > 0.5).int()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    f1 = f1_score(all_labels, all_preds)
    return f1, classification_report(all_labels, all_preds)
