import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import os

def plot_attention_heatmap(tokens, attention_weights, filename="attention_heatmap.png"):
    """
    Plots a heatmap of attention weights over tokens.
    tokens: list of strings
    attention_weights: numpy array of shape (seq_len, seq_len) or (seq_len,)
    """
    # If weights are (seq_len, seq_len), average them over the query dimension to get a single weight per token
    if len(attention_weights.shape) == 2:
        weights = np.mean(attention_weights, axis=0)
    else:
        weights = attention_weights
        
    # Take only the first N tokens (matching the weights length)
    tokens = tokens[:len(weights)]
    weights = weights[:len(tokens)]
    
    # Normalize weights for better visualization
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    
    plt.figure(figsize=(15, 2))
    sns.heatmap([weights], annot=[tokens], fmt="", cmap="YlGnBu", cbar=False, 
                xticklabels=False, yticklabels=False)
    plt.title("Attention Heatmap")
    
    os.makedirs("output", exist_ok=True)
    plt.savefig(os.path.join("output", filename), bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to output/{filename}")

def get_attention_and_plot(model, text_tensor, tokens, device='cpu', filename="attention_heatmap.png"):
    model.eval()
    with torch.no_grad():
        _, attn_weights = model(text_tensor.to(device))
        # attn_weights shape: [1, seq_len, seq_len]
        weights = attn_weights[0].cpu().numpy()
        plot_attention_heatmap(tokens, weights, filename)
