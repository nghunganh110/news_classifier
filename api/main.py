import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CATEGORIES = ["business", "entertainment", "health", "politics", "science", "sports", "technology"]
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend')

model_data = {}


def load_tfidf_model():
    global model_data
    try:
        import joblib
        model_path = os.path.join(MODELS_DIR, 'tfidf_model.pkl')
        data = joblib.load(model_path)
        model_data['pipeline'] = data['pipeline']
        model_data['preprocessor'] = data['preprocessor']
        model_data['label_list'] = data['label_list']
        model_data['model_type'] = 'tfidf'
        logger.info(f"Loaded TF-IDF model ({data.get('best_model_name', 'unknown')})")
        return True
    except Exception as e:
        logger.warning(f"Could not load TF-IDF model: {e}")
        return False

 
def load_transformer_model():
    global model_data
    try:
        import torch
        from src.train_transformer import TransformerTrainer
        
        model_path = os.path.join(MODELS_DIR, 'transformer_model.pt')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = TransformerTrainer(num_classes=len(CATEGORIES), device=device)
        trainer.load(model_path)
        
        model_data['trainer'] = trainer
        model_data['model_type'] = 'transformer'
        logger.info("Loaded Transformer model")
        return True
    except Exception as e:
        logger.warning(f"Could not load transformer model: {e}")
        return False


def load_model():
    """Load default model based on MODEL_TYPE env var (transformer or tfidf)"""
    default_model = os.environ.get('MODEL_TYPE', 'transformer').lower()
    
    if default_model == 'tfidf':
        if load_tfidf_model():
            return
        if load_transformer_model():
            return
    else:  # transformer (default)
        if load_transformer_model():
            return
        if load_tfidf_model():
            return
    
    logger.warning("No model loaded. Using mock predictions.")


def mock_predict(text: str):
    """Mock prediction when model is not available."""
    import random
    random.seed(hash(text) % (2**32))
    scores = {cat: random.random() for cat in CATEGORIES}
    total = sum(scores.values())
    scores = {k: round(v / total, 4) for k, v in scores.items()}
    category = max(scores, key=scores.get)
    return category, scores[category], scores


def predict_text(text: str):
    """Predict using loaded model"""
    if 'model_type' not in model_data:
        return mock_predict(text)
    
    if model_data['model_type'] == 'tfidf':
        # TF-IDF prediction
        preprocessor = model_data['preprocessor']
        pipeline = model_data['pipeline']
        label_list = model_data['label_list']

        processed = preprocessor.preprocess(text)
        clf = pipeline.named_steps['clf']
        vectorizer = pipeline.named_steps['tfidf']
        X = vectorizer.transform([processed])

        if hasattr(clf, 'predict_proba'):
            probs = clf.predict_proba(X)[0]
        elif hasattr(clf, 'decision_function'):
            df = clf.decision_function(X)[0]
            import numpy as np
            e = np.exp(df - df.max())
            probs = e / e.sum()
        else:
            probs = None

        pred = pipeline.predict([processed])[0]

        if probs is not None:
            all_scores = {label_list[i]: round(float(probs[i]), 4) for i in range(len(label_list))}
            confidence = round(float(max(probs)), 4)
        else:
            all_scores = {cat: 0.0 for cat in label_list}
            all_scores[pred] = 1.0
            confidence = 1.0

        return pred, confidence, all_scores
    
    else:  # transformer
        trainer = model_data['trainer']
        preds, probs = trainer.predict([text])
        pred = preds[0]
        prob = probs[0]
        
        all_scores = {CATEGORIES[i]: round(float(prob[i]), 4) for i in range(len(CATEGORIES))}
        confidence = round(float(max(prob)), 4)
        
        return pred, confidence, all_scores


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="News Topic Classifier API",
    description="Classify news articles into categories using ML models",
    version="1.0.0",
    lifespan=lifespan,
)

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    category: str
    confidence: float
    all_scores: dict


class BatchPredictRequest(BaseModel):
    texts: List[str]


class ModelStatusResponse(BaseModel):
    current_model: str
    available_models: list


class SelectModelRequest(BaseModel):
    model_type: str  # "transformer" or "tfidf"


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": 'model_type' in model_data,
        "current_model": model_data.get('model_type', 'none')
    }


@app.get("/model-status", response_model=ModelStatusResponse)
async def model_status():
    """Get current model and available models"""
    return {
        "current_model": model_data.get('model_type', 'none'),
        "available_models": ["transformer", "tfidf"]
    }


@app.post("/select-model")
async def select_model(request: SelectModelRequest):
    """Switch to a different model"""
    model_type = request.model_type.lower()
    
    if model_type not in ['transformer', 'tfidf']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type. Choose 'transformer' or 'tfidf'"
        )
    
    try:
        if model_type == 'tfidf':
            if load_tfidf_model():
                return {
                    "message": f"Successfully switched to TF-IDF model",
                    "current_model": model_data['model_type']
                }
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to load TF-IDF model. Check if models/tfidf_model.pkl exists."
                )
        else:  # transformer
            if load_transformer_model():
                return {
                    "message": f"Successfully switched to Transformer model",
                    "current_model": model_data['model_type']
                }
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to load Transformer model. Check if models/transformer_model.pt exists."
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/categories")
async def categories():
    return {"categories": CATEGORIES}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty")
    try:
        category, confidence, all_scores = predict_text(request.text)
        return PredictResponse(category=category, confidence=confidence, all_scores=all_scores)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictRequest):
    if not request.texts:
        raise HTTPException(status_code=422, detail="texts list cannot be empty")
    results = []
    for text in request.texts:
        try:
            category, confidence, all_scores = predict_text(text)
            results.append({"category": category, "confidence": confidence, "all_scores": all_scores})
        except Exception as e:
            results.append({"error": str(e)})
    return {"predictions": results}


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")