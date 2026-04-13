"""
train.py
--------
Trains a multi-label toxic comment classifier and saves it to disk.

What this script does:
  1. Loads the Jigsaw dataset
  2. Preprocesses comment text (lowercase, lemmatize, etc.)
  3. Splits into train / test sets
  4. Builds a TF-IDF + Logistic Regression pipeline
  5. Trains with class_weight="balanced" to handle label imbalance
  6. Evaluates with F1, ROC-AUC per label
  7. Saves model + metadata to toxic_model.pkl

Run:
    python train.py
"""

import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from preprocessing import preprocess_series

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "data", "train.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "toxic_model.pkl")

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

TFIDF_CONFIG = {
    "max_features": 50000,
    "ngram_range": (1, 2),      # unigrams + bigrams
    "sublinear_tf": True,        # apply log(1+tf) — helps with very frequent words
    "min_df": 3,                 # ignore terms appearing in fewer than 3 docs
    "analyzer": "word",
}

LR_CONFIG = {
    "max_iter": 1000,
    "C": 1.0,                   # regularization — lower = more regularization
    "class_weight": "balanced", # KEY FIX: compensates for 0.3% threat / 1% severe_toxic
    "solver": "lbfgs",
    "n_jobs": -1,
}
# ───────────────────────────────────────────────────────────────────────────


def load_data(path: str):
    """Load CSV and return features X and label matrix y."""
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}")
    print(f"  Null comments: {df['comment_text'].isnull().sum()}")

    # Show class distribution so you can see the imbalance
    print("\nLabel distribution (% positive):")
    for label in LABELS:
        pct = df[label].mean() * 100
        print(f"  {label:<20} {pct:.2f}%")

    X = df["comment_text"]
    y = df[LABELS]
    return X, y


def preprocess(X):
    """Run NLP cleaning pipeline on raw text series."""
    print("\nPreprocessing text (this takes ~2-3 min for 159k rows)...")
    start = time.time()
    X_clean = preprocess_series(X)
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")
    return X_clean


def build_pipeline():
    """
    Construct the sklearn Pipeline:
      TF-IDF Vectorizer -> MultiOutputClassifier(LogisticRegression)

    Why Pipeline?
      - Ensures the same TF-IDF transform is applied at predict time
      - Prevents data leakage (vectorizer fitted only on train split)
      - Everything saved in one .pkl file
    """
    tfidf = TfidfVectorizer(**TFIDF_CONFIG)
    lr = LogisticRegression(**LR_CONFIG)
    multi_clf = MultiOutputClassifier(lr, n_jobs=-1)

    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("clf", multi_clf),
    ])
    return pipeline


def evaluate(pipeline, X_test, y_test):
    """
    Print per-label evaluation metrics.

    Metrics explained:
      Precision  — of comments flagged as toxic, how many actually were?
      Recall     — of all toxic comments, how many did we catch?
      F1         — harmonic mean of precision & recall (main metric)
      ROC-AUC    — area under ROC curve; 1.0 = perfect, 0.5 = random
    """
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    y_pred = pipeline.predict(X_test)
    y_proba_list = pipeline.predict_proba(X_test)

    for i, label in enumerate(LABELS):
        y_true_label = y_test[label].values
        y_pred_label = y_pred[:, i]
        y_prob_label = y_proba_list[i][:, 1]

        f1 = f1_score(y_true_label, y_pred_label, zero_division=0)
        try:
            auc = roc_auc_score(y_true_label, y_prob_label)
        except ValueError:
            auc = float("nan")  # happens if label has no positives in test

        print(f"\n{label.upper()}")
        print(classification_report(
            y_true_label, y_pred_label,
            target_names=["clean", label],
            zero_division=0,
        ))
        print(f"  ROC-AUC: {auc:.4f}   |   F1: {f1:.4f}")

    # Overall macro F1
    macro_f1 = f1_score(y_test.values, y_pred, average="macro", zero_division=0)
    print(f"\nOverall macro F1: {macro_f1:.4f}")
    print("=" * 60)


def save_model(pipeline, path: str):
    """Save pipeline + label list to a single .pkl file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"pipeline": pipeline, "labels": LABELS}, f)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"\nModel saved to: {path}  ({size_mb:.1f} MB)")


def main():
    print("=" * 60)
    print("AI CONTENT MODERATION — TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load
    X_raw, y = load_data(DATA_PATH)

    # 2. Preprocess
    X_clean = preprocess(X_raw)

    # 3. Train / test split — stratify on 'toxic' label (most common)
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y,
        test_size=0.2,
        random_state=42,
        stratify=y["toxic"],   # ensures same toxic % in both splits
    )
    print(f"\nTrain size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    # 4. Build + train
    print("\nBuilding pipeline...")
    pipeline = build_pipeline()

    print("Training model (this takes ~3-5 min)...")
    start = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  Training done in {elapsed:.1f}s")

    # 5. Evaluate
    evaluate(pipeline, X_test, y_test)

    # 6. Save
    save_model(pipeline, MODEL_PATH)

    print("\nAll done! Run the API with:")
    print("  uvicorn main:app --reload")


if __name__ == "__main__":
    main()
