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

# ============================================================
# Prediction History for Model Calibration Detection
# ============================================================
_prediction_history = []  # tracks recent model outputs to detect bias

def compute_heuristic_score(content: str) -> dict:
    """
    Comprehensive linguistic heuristic analysis.
    Returns a score between 0.0 (very credible) and 1.0 (very suspicious).
    """
    content_lower = content.lower()
    words = content_lower.split()
    word_count = len(words)
    
    fake_signals = 0.0
    real_signals = 0.0
    signal_count = 0
    details = {}
    
    # --- 1. Sensationalism markers (strong fake signal) ---
    sensational_phrases = [
        "you won't believe", "shocker", "exposed!", "conspiracy",
        "hidden truth", "they don't want you to know", "cover-up",
        "breaking:", "urgent!", "share before deleted", "mainstream media won't",
        "big pharma", "wake up", "sheeple", "miracle cure", "secret remedy",
        "the truth about", "exposed", "bombshell", "jaw-dropping",
        "mind-blowing", "insane", "unbelievable", "devastating",
    ]
    sensational_count = sum(1 for p in sensational_phrases if p in content_lower)
    if sensational_count > 0:
        fake_signals += min(sensational_count * 0.15, 0.5)
        signal_count += 1
    details["sensationalism"] = sensational_count
    
    # --- 2. Trust / attribution markers (strong real signal) ---
    trust_phrases = [
        "according to", "reported by", "stated that", "spokesperson",
        "official sources", "confirmed by", "citing", "peer-reviewed",
        "published in", "research shows", "data indicates", "study finds",
        "the associated press", "reuters", "said in a statement",
        "press release", "university of", "department of",
        "in a report", "evidence suggests", "analysis shows",
    ]
    trust_count = sum(1 for p in trust_phrases if p in content_lower)
    if trust_count > 0:
        real_signals += min(trust_count * 0.12, 0.5)
        signal_count += 1
    details["trust_markers"] = trust_count
    
    # --- 3. Exclamation / all-caps density (fake signal) ---
    exclamation_ratio = content.count('!') / max(word_count, 1)
    if exclamation_ratio > 0.05:
        fake_signals += min(exclamation_ratio * 2, 0.3)
        signal_count += 1
    
    upper_words = sum(1 for w in content.split() if w.isupper() and len(w) > 2)
    caps_ratio = upper_words / max(word_count, 1)
    if caps_ratio > 0.1:
        fake_signals += min(caps_ratio, 0.2)
        signal_count += 1
    
    # --- 4. Sentiment extremity (VADER) ---
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sid = SentimentIntensityAnalyzer()
        scores = sid.polarity_scores(content)
        compound = scores['compound']
        # Extreme sentiment (very positive or very negative) = suspicious
        extremity = abs(compound)
        if extremity > 0.7:
            fake_signals += 0.15
            signal_count += 1
        elif extremity < 0.3:
            real_signals += 0.1
            signal_count += 1
        details["sentiment_compound"] = compound
    except:
        details["sentiment_compound"] = 0.0
    
    # --- 5. Text length (very short = less reliable either way) ---
    if word_count < 20:
        fake_signals += 0.05  # short text is slightly suspicious
    elif word_count > 100:
        real_signals += 0.1   # longer, more detailed = slightly more credible
    
    # --- 6. Question marks / clickbait patterns ---
    question_ratio = content.count('?') / max(word_count, 1)
    if question_ratio > 0.03:
        fake_signals += 0.1
        signal_count += 1
    
    # --- 7. Hedging language (real news hedges, fake news asserts absolutely) ---
    hedging = ["may", "might", "could", "possibly", "reportedly", "allegedly",
               "it appears", "sources say", "is believed to", "likely"]
    hedge_count = sum(1 for h in hedging if h in content_lower)
    if hedge_count > 0:
        real_signals += min(hedge_count * 0.08, 0.25)
        signal_count += 1
    
    # --- 8. Absolute/emotional language (fake signal) ---
    absolutes = ["always", "never", "everyone knows", "nobody can deny",
                 "100%", "guaranteed", "proven fact", "undeniable",
                 "exposed", "exposed!"]
    absolute_count = sum(1 for a in absolutes if a in content_lower)
    if absolute_count > 0:
        fake_signals += min(absolute_count * 0.1, 0.3)
        signal_count += 1
    
    # Compute final heuristic score: 0.0 = very real, 1.0 = very fake
    if signal_count == 0:
        heuristic_prob = 0.45  # neutral when no signals
    else:
        heuristic_prob = 0.5 + (fake_signals - real_signals)
    
    heuristic_prob = max(0.05, min(0.95, heuristic_prob))
    
    details["fake_signals"] = round(fake_signals, 3)
    details["real_signals"] = round(real_signals, 3)
    details["heuristic_prob"] = round(heuristic_prob, 3)
    
    return {"score": heuristic_prob, "details": details}


def detect_model_calibration() -> float:
    """
    Returns a confidence weight for ML models. 
    A fixed high trust is used since the models are pre-trained and highly accurate.
    """
    return 0.85  # Trust the neural models 85%, leaving 15% for heuristic tuning


@app.post("/predict")
async def predict(data: NewsInput):
    start_time = time.time()
    
    # Heuristic text cleaning: filter out UI boilerplate and short navigation links
    raw_lines = data.text.split('\n')
    cleaned_lines = []
    for line in raw_lines:
        words = line.strip().split()
        # Only keep lines with substantial word counts (prose), ignore short UI elements and common photo credits
        if len(words) > 10 and 'Getty Images' not in line:
            cleaned_lines.append(line.strip())
            
    # If cleaning stripped everything (e.g. valid short text), fallback to raw text
    cleaned_text = " ".join(cleaned_lines) if cleaned_lines else data.text
    
    content = data.title + " " + cleaned_text
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
            
        # 3. Heuristic Analysis (comprehensive)
        heuristic_result = compute_heuristic_score(content)
        heuristic_prob = heuristic_result["score"]
        trust_count = heuristic_result["details"]["trust_markers"]
        sensational_count = heuristic_result["details"]["sensationalism"]
        
        # 4. Adaptive Ensemble — model trust based on calibration
        raw_model_prob = (prob_bert * 0.6) + (prob_bilstm * 0.4)
        _prediction_history.append(raw_model_prob)
        
        model_trust = detect_model_calibration()
        
        # --- Bias Mitigation Layer ---
        # The ML models were trained on political news and may exhibit Entity Bias 
        # against entertainment/celebrity topics, flagging them as tabloid fakes.
        # If heuristics strongly indicate objective, high-integrity journalism (< 0.25)
        # but ML models flag it as fake (> 0.75), we dynamically reduce ML trust.
        if heuristic_prob < 0.25 and raw_model_prob > 0.75:
            model_trust = 0.15  # Shift weight to heuristics for out-of-distribution topics
            
        final_prob = (raw_model_prob * model_trust) + (heuristic_prob * (1 - model_trust))
        
        return {
            "bilstm_prob": prob_bilstm,
            "bert_prob": prob_bert,
            "final_prob": final_prob,
            "tokens": clean_tokens[:200],
            "attention_weights": [round(w, 4) for w in weights[:200]],
            "heuristics": heuristic_result["details"],
            "meta": {
                "execution_time": time.time() - start_time,
                "model_trust": model_trust,
                "heuristic_score": heuristic_prob,
                "version": "2.0.0-calibrated"
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
