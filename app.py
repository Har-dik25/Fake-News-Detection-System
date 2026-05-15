import logging
import os
import time
import pickle
import numpy as np
import json
from diskcache import Deque
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
from pydantic import BaseModel, Field
import asyncio
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from tenacity import retry, stop_after_attempt, wait_fixed

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None


# Import configuration
from config import settings

# Import model architectures
from src.bilstm_attention import BiLSTMWithAttention
from src.bert_hybrid import HybridBERTModel
from src.data_loader import prepare_sequences, NewsDataset
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        resources["device"] = device
        
        # Load word_to_idx
        with open(settings.WORD_TO_IDX_PATH, "rb") as f:
            resources["word_to_idx"] = pickle.load(f)
        
        # Load BiLSTM
        bilstm = BiLSTMWithAttention(len(resources["word_to_idx"]), 128, 256, 2, use_attention=True)
        bilstm.load_state_dict(torch.load(settings.BILSTM_MODEL_PATH, map_location=device))
        bilstm.to(device)
        bilstm.eval()
        resources["bilstm_model"] = bilstm
        
        # Load BERT
        bert = HybridBERTModel(107, mode='hybrid')
        bert.load_state_dict(torch.load(settings.BERT_MODEL_PATH, map_location=device))
        bert.to(device)
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

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsInput(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    text: str = Field(..., min_length=1, max_length=20000)

# Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc} at {request.url}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error", "detail": "An unexpected error occurred. Please try again later."},
    )

# ============================================================
# Prediction History for Model Calibration Detection
# ============================================================
_prediction_history = Deque(directory=os.path.join(settings.BASE_DIR, 'cache', 'history'))

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def perform_search(query):
    return list(DDGS().text(query, max_results=5))

async def live_fact_check(title: str, text: str) -> dict:
    """
    Live fact-checking using DuckDuckGo search.
    Searches the headline/core claim and checks if reputable domains are reporting it.
    """
    heuristic_prob = 0.5
    details = {"trust_markers": 0, "sensationalism": 0}
    
    if DDGS is None:
        logger.warning("duckduckgo_search not installed. Falling back to neutral heuristics.")
        details["heuristic_prob"] = heuristic_prob
        return {"score": heuristic_prob, "details": details}
        
    try:
        search_query = title[:200]  # First 200 chars of title
        # Run search in a background thread with retry and timeout to prevent blocking
        results = await asyncio.wait_for(
            asyncio.to_thread(lambda: perform_search(search_query)),
            timeout=10.0
        )
        
        domains_path = os.path.join(settings.BASE_DIR, 'config', 'domains.json')
        try:
            with open(domains_path, 'r') as f:
                domains_config = json.load(f)
                reputable_domains = domains_config.get('reputable_domains', [])
                unreliable_domains = domains_config.get('unreliable_domains', [])
        except Exception as e:
            logger.error(f"Could not load domains config: {e}")
            reputable_domains = []
            unreliable_domains = []
        
        reputable_matches = 0
        unreliable_matches = 0
        
        for res in results:
            url = res.get('href', '').lower()
            if any(domain in url for domain in reputable_domains):
                reputable_matches += 1
            if any(domain in url for domain in unreliable_domains):
                unreliable_matches += 1
                
        details["reputable_sources_found"] = reputable_matches
        details["unreliable_sources_found"] = unreliable_matches
        details["search_completed"] = True
        
        if reputable_matches > 0:
            heuristic_prob = max(0.1, 0.5 - (reputable_matches * 0.15))
        elif unreliable_matches > 0:
            heuristic_prob = min(0.9, 0.5 + (unreliable_matches * 0.15))
        elif len(results) == 0:
            # If no one is reporting it, it's highly suspicious
            heuristic_prob = 0.8
            
    except Exception as e:
        logger.error(f"Live fact check failed: {e}")
        details["search_error"] = str(e)
        
    details["heuristic_prob"] = round(heuristic_prob, 3)
    return {"score": heuristic_prob, "details": details}



def detect_model_calibration() -> float:
    """
    Returns a dynamic confidence weight for ML models based on recent prediction variance.
    If the model keeps predicting the same thing (all fake or all real), trust drops.
    """
    if len(_prediction_history) < 10:
        return 0.85  # Not enough data yet, use default trust
    
    recent = list(_prediction_history)[-50:]
    mean_prob = sum(recent) / len(recent)
    variance = sum((p - mean_prob) ** 2 for p in recent) / len(recent)
    
    # If variance is very low (model stuck on one prediction), reduce trust
    if variance < 0.01:
        return 0.50  # Model seems biased, shift weight to heuristics
    elif variance < 0.05:
        return 0.70
    else:
        return 0.85  # Healthy variance, trust the model


@app.post("/predict")
@limiter.limit("5/minute")
async def predict(request: Request, data: NewsInput):
    start_time = time.time()
    
    # Deep text cleaning to match training pipeline (strips HTML, normalizes spaces)
    cleaned_text = NewsDataset.clean_text(data.text)
    
    content = data.title + " " + cleaned_text
    content_lower = content.lower()
    
    if not content.strip():
        raise HTTPException(status_code=400, detail="Input content cannot be empty")

    try:
        # 1. BiLSTM Prediction & Attention
        tokens = word_tokenize(content_lower)
        clean_tokens = [t for t in tokens if t.isalnum()]
        seq = prepare_sequences([clean_tokens], resources["word_to_idx"], max_len=512)
        
        def run_bilstm():
            with torch.no_grad():
                seq_tensor = torch.tensor(seq).to(resources["device"])
                prob_bilstm_raw, attn_weights = resources["bilstm_model"](seq_tensor)
                prob_bilstm = float(prob_bilstm_raw[0][0])
                weights = torch.mean(attn_weights[0], dim=0).cpu().numpy().tolist()
                return prob_bilstm, weights
                
        prob_bilstm, weights = await asyncio.to_thread(run_bilstm)
            
        # 2. BERT Prediction
        def run_bert():
            inputs = resources["bert_tokenizer"](content, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
            tfidf_feat = resources["classical_extractor"].get_tfidf_features([content])
            other_feat = resources["classical_extractor"].get_pos_features(content) + resources["classical_extractor"].get_sentiment_features(content)
            combined_feat = np.hstack([tfidf_feat, [other_feat]])
            classical_tensor = torch.tensor(combined_feat, dtype=torch.float32).to(resources["device"])
            
            with torch.no_grad():
                prob_bert_raw = resources["bert_model"](
                    input_ids=inputs['input_ids'].to(resources["device"]), 
                    attention_mask=inputs['attention_mask'].to(resources["device"]), 
                    classical_features=classical_tensor
                )
                return float(prob_bert_raw[0][0])
                
        prob_bert = await asyncio.to_thread(run_bert)
            
        # 3. Heuristic Analysis (live fact check)
        heuristic_result = await live_fact_check(data.title, data.text)
        heuristic_prob = heuristic_result["score"]
        
        # 4. Adaptive Ensemble — model trust based on calibration
        raw_model_prob = (prob_bert * 0.6) + (prob_bilstm * 0.4)
        _prediction_history.append(raw_model_prob)
        while len(_prediction_history) > 200:
            try:
                _prediction_history.popleft()
            except IndexError:
                break
        
        model_trust = detect_model_calibration()
        
        # --- Bias Mitigation Layer ---
        # The ML models were trained on political news and may exhibit Entity Bias 
        # against entertainment/celebrity topics, flagging them as tabloid fakes.
        # If heuristics strongly indicate objective, high-integrity journalism (< 0.25)
        # but ML models flag it as fake (> 0.75), we dynamically reduce ML trust.
        if heuristic_prob < 0.25 and raw_model_prob > 0.75:
            model_trust = 0.15  # Shift weight to heuristics for out-of-distribution topics
            
        final_prob = (raw_model_prob * model_trust) + (heuristic_prob * (1 - model_trust))
        
        # 5. Domain confidence disclaimer
        word_count = len(content.split())
        disclaimers = []
        if word_count < 30:
            disclaimers.append("Very short input — prediction confidence may be reduced.")
        # Detect likely non-English content (simple heuristic: high ratio of non-ASCII chars)
        non_ascii_ratio = sum(1 for c in content if ord(c) > 127) / max(len(content), 1)
        if non_ascii_ratio > 0.3:
            disclaimers.append("Non-English content detected — model was trained on English text only.")
        if heuristic_prob < 0.25 and raw_model_prob > 0.75:
            disclaimers.append("Model and fact-checker disagree — content may be outside the trained domain (US political news).")
        
        return {
            "bilstm_prob": prob_bilstm,
            "bert_prob": prob_bert,
            "final_prob": final_prob,
            "tokens": clean_tokens[:512],
            "attention_weights": [round(w, 4) for w in weights[:512]],
            "heuristics": heuristic_result["details"],
            "meta": {
                "execution_time": time.time() - start_time,
                "model_trust": model_trust,
                "heuristic_score": heuristic_prob,
                "version": "3.0.0",
                "model_version": os.path.getmtime(settings.BERT_MODEL_PATH) if os.path.exists(settings.BERT_MODEL_PATH) else None,
                "device": str(resources["device"]),
            },
            "disclaimers": disclaimers if disclaimers else None
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal error.")

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
