# TruthLens AI - Advanced Fake News Detection System

TruthLens AI is a high-fidelity, production-grade fake news detection platform. It leverages a state-of-the-art hybrid ensemble architecture combining Transformer models with Recurrent Neural Networks and classical linguistic analysis to provide explainable and highly accurate credibility assessments.

## 🚀 Key Features

- **Hybrid Ensemble Architecture**: Combines BERT (contextual) and BiLSTM (sequential) models for robust prediction.
- **Explainable AI (XAI)**: Integrated attention mechanism visualizes token-level importance, highlighting exactly which words triggered a "Fake" classification.
- **Neural Pipeline Visualization**: Watch the real-time processing flow from preprocessing to final calibration in a cinematic dashboard.
- **Production-Grade Backend**: FastAPI-based server with Pydantic configuration, robust logging, and graceful lifespan management.
- **Dockerized Deployment**: Fully containerized environment with Gunicorn/Uvicorn workers for high availability.
- **Linguistic Heuristics**: Custom calibration using journalistic trust markers and sensationalism scoring to reduce false positives.
- **Premium Glassmorphic UI**: A high-end, responsive dashboard featuring dark mode, neural backgrounds, and fluid animations.



## 🏗️ Model Architecture

### 1. Hybrid BERT Model
A BERT-based transformer model augmented with a classical feature layer. It processes the full context of the news article while simultaneously considering:
- **TF-IDF Vectorization**: Capture key statistical word importance.
- **POS Analysis**: Detect linguistic patterns common in misinformation (e.g., high adjective density).
- **Sentiment Analysis**: Analyze emotional tone and sensationalism.

### 2. BiLSTM with Attention
A bidirectional LSTM network that captures sequential dependencies. The **Self-Attention** layer provides transparency by assigning weights to each token, allowing users to see the model's "thought process."

### 3. Smart Calibration Ensemble
The system uses a weighted ensemble (60% BERT / 40% BiLSTM) which is then dynamically adjusted by heuristic scores:
- **Trust Markers**: Keywords typical of credible journalism.
- **Sensationalism Markers**: Phrases often used in clickbait and misinformation.

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **Deep Learning**: PyTorch, HuggingFace Transformers
- **NLP**: NLTK, Scikit-Learn
- **Frontend**: Vanilla HTML5, CSS3 (Modern Glassmorphism), JavaScript
- **API**: RESTful architecture with CORS support

## ⚡ Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd "Fake News Detection"
```

### 2. Standard Local Setup
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader punkt averaged_perceptron_tagger vader_lexicon
```

### 3. Running with Docker (Recommended for Production)
The system is fully containerized for one-command deployment:
```bash
docker-compose up --build
```
This will:
1. Build the production image.
2. Install all dependencies and NLTK resources.
3. Start the Gunicorn server with 4 workers.
4. Serve the frontend at `http://localhost:8000`.

### 4. Direct Execution
```bash
python app.py
```
*Note: In production, use the Docker setup or Gunicorn directly.*

## 📈 Monitoring & Logs
The system generates an `app.log` file in the root directory. You can monitor it in real-time:
```bash
tail -f app.log
```
Or check the health endpoint: `http://localhost:8000/health`


## 📊 Dataset
The models are trained on the **WELFake Dataset**, a comprehensive collection of 72,000+ news articles categorized as "Real" or "Fake".

---
*Developed with excellence in AI and Design.*