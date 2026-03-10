# 📰 News Topic Classifier

A full-stack machine learning system that classifies news articles into 7 topic
categories: **sports, technology, politics, business, entertainment, health,
and science**.

It ships with:
- **TF-IDF + scikit-learn pipeline** (Logistic Regression / LinearSVC / Random Forest)
- **Custom Transformer classifier** built from scratch with PyTorch
- **FastAPI REST API** with batch prediction support
- **Responsive single-page frontend** (vanilla HTML/CSS/JS)
- **Web scraper** for real-world data collection
- **Jupyter notebook** for exploratory analysis

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        User / Client                         │
│            Browser  ───►  frontend/index.html                │
│            curl     ───►  POST /predict                      │
└─────────────────────────────┬────────────────────────────────┘
                              │ HTTP
┌─────────────────────────────▼────────────────────────────────┐
│                   FastAPI Application                        │
│                     api/main.py                              │
│                                                              │
│   GET  /            → serve frontend                         │
│   GET  /health      → model status                           │
│   GET  /categories  → list of categories                     │
│   POST /predict     → single article prediction              │
│   POST /predict/batch → batch predictions                    │
└─────────────────────────────┬────────────────────────────────┘
                              │
        ┌─────────────────────▼──────────────────────┐
        │              ML Pipeline                    │
        │                                             │
        │  src/preprocess.py  ─►  TextPreprocessor   │
        │  src/train_tfidf.py ─►  TF-IDF + sklearn   │
        │  src/train_transformer.py ► PyTorch model  │
        │  src/evaluate.py    ─►  metrics & plots    │
        └─────────────────────┬──────────────────────┘
                              │
        ┌─────────────────────▼──────────────────────┐
        │                  Data                       │
        │  data/sample_data.csv  (180 articles)       │
        │  src/scraper.py  (collect real articles)    │
        └─────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd news_classifier
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train_tfidf.py
```

This produces `models/tfidf_model.pkl` and a confusion matrix in `outputs/`.

### 3. Start the API

```bash
bash run.sh
# or
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** for the web UI, or **http://localhost:8000/docs**
for the interactive Swagger API docs.

---

## API Reference

### `POST /predict`

Classify a single article.

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "NASA discovers exoplanet with signs of liquid water."}' | python3 -m json.tool
```

**Response**

```json
{
  "category": "science",
  "confidence": 0.8731,
  "all_scores": {
    "business": 0.0121,
    "entertainment": 0.0089,
    "health": 0.0143,
    "politics": 0.0112,
    "science": 0.8731,
    "sports": 0.0198,
    "technology": 0.0606
  }
}
```

### `POST /predict/batch`

Classify multiple articles in one request.

```bash
curl -s -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["The team won the championship.", "Fed raises rates by 25 bps."]}' \
  | python3 -m json.tool
```

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true}
```

### `GET /categories`

```bash
curl http://localhost:8000/categories
# {"categories":["business","entertainment","health","politics","science","sports","technology"]}
```

---

## Models

### TF-IDF Pipeline (`src/train_tfidf.py`)

Three classifiers are trained and the best is kept:

| Classifier | Notes |
|---|---|
| Logistic Regression | Excellent baseline; fast inference |
| Linear SVC | Strong on high-dim sparse data |
| Random Forest | Ensemble; slightly slower |

Features: bigram TF-IDF, 50k max features, sublinear TF scaling.

### Custom Transformer (`src/train_transformer.py`)

A small Transformer Encoder trained from scratch:

| Hyper-parameter | Value |
|---|---|
| Vocabulary | 10 000 tokens |
| Embedding dim | 64 |
| Attention heads | 4 |
| Encoder layers | 2 |
| FF dimension | 128 |
| Max sequence len | 128 |
| Epochs | 10 |
| Batch size | 32 |

Training runs on CPU in under 2 minutes on the sample dataset.

```bash
python src/train_transformer.py
```

---

## Data

`data/sample_data.csv` contains **180 synthetic news article snippets** with
labels across 7 categories (≥ 25 per category).

See [`data/README.md`](data/README.md) for full methodology and web scraping
documentation.

### Collect real data

```python
from src.scraper import NewsScraper

scraper = NewsScraper(delay=1.0)
articles = scraper.collect_training_data(
    urls=["https://example.com/article1", ...],
    category="sports"
)
scraper.save_to_csv(articles, "data/scraped_sports.csv")
```

---

## Directory Structure

```
news_classifier/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore
├── run.sh                     # Start the API server
├── data/
│   ├── sample_data.csv        # 180-row training dataset
│   └── README.md              # Data documentation
├── src/
│   ├── __init__.py
│   ├── preprocess.py          # Text cleaning pipeline
│   ├── train_tfidf.py         # TF-IDF model training
│   ├── train_transformer.py   # Custom Transformer training
│   ├── evaluate.py            # Metrics & confusion matrix
│   └── scraper.py             # Web scraping utilities
├── api/
│   ├── __init__.py
│   └── main.py                # FastAPI application
├── frontend/
│   ├── index.html             # Single-page UI
│   ├── style.css              # Responsive styles
│   └── app.js                 # Fetch-based API client
├── notebooks/
│   └── analysis.ipynb         # EDA + training walkthrough
├── models/                    # Saved model artefacts (git-ignored)
└── outputs/                   # Plots and evaluation images (git-ignored)
```

---

## Development

```bash
# Run preprocessing demo
python src/preprocess.py

# Run full TF-IDF training
python src/train_tfidf.py

# Run Transformer training
python src/train_transformer.py

# Launch Jupyter
jupyter notebook notebooks/analysis.ipynb
```

---

## License

MIT
