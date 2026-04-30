import logging
import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import configuration
from config import settings

# Import model architectures
from src.bilstm_attention import BiLSTMWithAttention
from src.bert_hybrid import HybridBERTModel
from src.data_loader import prepare_sequences
from src.feature_extractor import ClassicalFeatureExtractor

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TruthLens")

# Global resources
resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models and resources
    logger.info("Initializing production resources...")
    try:
        # Load word_to_idx
        with open(settings.WORD_TO_IDX_PATH, "rb") as f:
            resources["word_to_idx"] = pickle.load(f)
        
        # Load BiLSTM
        bilstm = BiLSTMWithAttention(len(resources["word_to_idx"]), 128, 256, 2, use_attention=True)
        bilstm.load_state_dict(torch.load(settings.BILSTM_MODEL_PATH, map_location='cpu'))
        bilstm.eval()
        resources["bilstm_model"] = bilstm
        
        # Load BERT
        bert = HybridBERTModel(107, mode='hybrid')
        bert.load_state_dict(torch.load(settings.BERT_MODEL_PATH, map_location='cpu'))
        bert.eval()
        resources["bert_model"] = bert
        
        resources["bert_tokenizer"] = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load Classical Extractor
        extractor = ClassicalFeatureExtractor(max_tfidf_features=100)
        extractor.load(settings.TFIDF_MODEL_PATH)
        resources["classical_extractor"] = extractor
        
        logger.info("All models and vectorizers loaded successfully.")
    except Exception as e:
        logger.error(f"Critical error during startup: {e}")
        # In production, we might want to exit or retry
        raise RuntimeError(f"Could not load models: {e}")
    
    yield
    # Shutdown: Clean up if necessary
    logger.info("Shutting down TruthLens AI...")
    resources.clear()

app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsInput(BaseModel):
    title: str
    text: str

# Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc} at {request.url}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error", "detail": str(exc)},
    )

@app.post("/predict")
async def predict(data: NewsInput):
    start_time = time.time()
    content = data.title + " " + data.text
    content_lower = content.lower()
    
    if not content.strip():
        raise HTTPException(status_code=400, detail="Input content cannot be empty")

    try:
        # 1. BiLSTM Prediction & Attention
        tokens = word_tokenize(content_lower)
        clean_tokens = [t for t in tokens if t.isalnum()]
        seq = prepare_sequences([clean_tokens], resources["word_to_idx"], max_len=200)
        
        with torch.no_grad():
            prob_bilstm_raw, attn_weights = resources["bilstm_model"](torch.tensor(seq))
            prob_bilstm = float(prob_bilstm_raw[0][0])
            weights = torch.mean(attn_weights[0], dim=0).cpu().numpy().tolist()
            
        # 2. BERT Prediction
        inputs = resources["bert_tokenizer"](content, truncation=True, padding='max_length', max_length=200, return_tensors='pt')
        tfidf_feat = resources["classical_extractor"].get_tfidf_features([content])
        other_feat = resources["classical_extractor"].get_pos_features(content) + resources["classical_extractor"].get_sentiment_features(content)
        combined_feat = np.hstack([tfidf_feat, [other_feat]])
        classical_tensor = torch.tensor(combined_feat, dtype=torch.float32)
        
        with torch.no_grad():
            prob_bert_raw = resources["bert_model"](input_ids=inputs['input_ids'], 
                                       attention_mask=inputs['attention_mask'], 
                                       classical_features=classical_tensor)
            prob_bert = float(prob_bert_raw[0][0])
            
        # 3. Credibility Heuristics
        trust_score = sum(1 for marker in settings.TRUST_MARKERS if marker in content_lower)
        sensational_score = sum(1 for marker in settings.SENSATIONAL_MARKERS if marker in content_lower)
        heuristic_adjustment = (trust_score * 0.05) - (sensational_score * 0.05)
        
        # 4. Smart Ensemble Logic
        weighted_prob = (prob_bert * 0.6) + (prob_bilstm * 0.4)
        final_prob = max(0.0, min(1.0, weighted_prob - heuristic_adjustment))
        
        execution_time = time.time() - start_time
        logger.info(f"Prediction successful in {execution_time:.4f}s. Final Prob: {final_prob:.4f}")
        
        return {
            "bilstm_prob": prob_bilstm,
            "bert_prob": prob_bert,
            "final_prob": final_prob,
            "tokens": clean_tokens[:200],
            "attention_weights": weights[:len(clean_tokens)][:200],
            "heuristics": {
                "trust_markers": trust_score,
                "sensationalism": sensational_score
            },
            "meta": {
                "execution_time": execution_time,
                "version": "1.0.0-prod"
            }
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Health Check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time(), "models_loaded": len(resources) > 0}

# Serve Static Files (Frontend)
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
else:
    logger.warning("Frontend directory not found. Serving API only.")

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting {settings.APP_NAME} on {settings.HOST}:{settings.PORT}")
    uvicorn.run("app:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
