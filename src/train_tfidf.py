import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import TextPreprocessor
from src.evaluate import plot_confusion_matrix, print_classification_report, compare_models

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'sample_data.csv')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')


def load_data(path: str):
    df = pd.read_csv(path)
    df = df.dropna(subset=['text', 'category'])
    return df['text'].tolist(), df['category'].tolist()


def train_and_evaluate():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("Loading data...")
    texts, labels = load_data(DATA_PATH)

    print("Preprocessing texts...")
    preprocessor = TextPreprocessor()
    processed_texts = preprocessor.preprocess_batch(texts)

    label_list = sorted(set(labels))
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    classifiers = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LinearSVC': LinearSVC(max_iter=1000, random_state=42),
    }

    results = {}
    best_acc = -1
    best_pipeline = None
    best_name = None

    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)),
            ('clf', clf),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = np.mean(np.array(y_pred) == np.array(y_test))
        print(f"{name} Accuracy: {acc:.4f}")
        print_classification_report(y_test, y_pred, label_list)
        results[name] = acc

        if acc > best_acc:
            best_acc = acc
            best_pipeline = pipeline
            best_name = name

    compare_models(results)

    # Save confusion matrix for best model
    y_pred_best = best_pipeline.predict(X_test)
    plot_confusion_matrix(
        y_test, y_pred_best, label_list,
        f"TF-IDF ({best_name}) Confusion Matrix",
        os.path.join(OUTPUTS_DIR, 'tfidf_confusion_matrix.png')
    )

    # Save best model
    model_path = os.path.join(MODELS_DIR, 'tfidf_model.pkl')
    model_data = {
        'pipeline': best_pipeline,
        'label_list': label_list,
        'preprocessor': preprocessor,
        'best_model_name': best_name,
    }
    joblib.dump(model_data, model_path)
    print(f"\nBest model ({best_name}, acc={best_acc:.4f}) saved to {model_path}")
    return model_data


if __name__ == '__main__':
    train_and_evaluate()
