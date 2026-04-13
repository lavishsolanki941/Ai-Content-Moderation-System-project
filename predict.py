"""
predict.py
----------
Handles model loading and toxicity prediction.

Design decisions:
  - Model loaded ONCE at module import time (not per-request)
  - Uses relative path so it works on any machine
  - Severity mapped to human-readable levels
  - predict_toxicity() is the only function the API needs to call
"""

import os
import pickle

from preprocessing import clean_text

# ── Paths ───────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "..", "model", "toxic_model.pkl")

# ── Severity thresholds (configurable) ─────────────────────────────────────
SEVERITY_LEVELS = {
    "critical":  0.75,   # max_score >= 0.75
    "high":      0.50,   # max_score >= 0.50
    "moderate":  0.25,   # max_score >= 0.25
    "low":       0.0,    # everything else
}
# ───────────────────────────────────────────────────────────────────────────


def _load_model(path: str):
    """Load the saved pipeline and label list from .pkl file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at: {path}\n"
            "Run train.py first to generate the model file."
        )
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Support both old format (tuple) and new format (dict)
    if isinstance(data, dict):
        return data["pipeline"], data["labels"]
    elif isinstance(data, tuple):
        return data[0], data[1]
    else:
        raise ValueError("Unrecognized model file format.")


def _get_severity_level(score: float) -> str:
    """Map a 0-1 probability score to a human-readable severity level."""
    for level, threshold in SEVERITY_LEVELS.items():
        if score >= threshold:
            return level
    return "low"


# Load model once when this module is first imported
print(f"Loading model from: {MODEL_PATH}")
try:
    _pipeline, _labels = _load_model(MODEL_PATH)
    print("Model loaded successfully.")
except FileNotFoundError as e:
    print(f"WARNING: {e}")
    _pipeline, _labels = None, []


def predict_toxicity(text: str) -> dict:
    """
    Run the full prediction pipeline on a single comment.

    Args:
        text: raw comment string (will be preprocessed internally)

    Returns:
        {
          "is_toxic": bool,
          "severity_level": "low" | "moderate" | "high" | "critical",
          "severity_score": float (0.0 to 1.0),
          "breakdown": {
            "toxic": 0.87,
            "severe_toxic": 0.12,
            "obscene": 0.65,
            "threat": 0.03,
            "insult": 0.71,
            "identity_hate": 0.08
          },
          "flagged_labels": ["toxic", "obscene", "insult"],
          "input_text": "original text"
        }
    """
    if _pipeline is None:
        raise RuntimeError(
            "Model is not loaded. Run train.py first to generate the model."
        )

    # Preprocess — same cleaning as during training (critical for consistency)
    cleaned = clean_text(text)

    # Predict probabilities for all 6 labels
    # predict_proba returns a list of arrays — one array per label
    # Each array: [[prob_class_0, prob_class_1], ...]
    proba_list = _pipeline.predict_proba([cleaned])

    # Build per-label breakdown
    breakdown = {}
    for i, label in enumerate(_labels):
        # index [0] = first (and only) sample, [1] = probability of positive class
        breakdown[label] = round(float(proba_list[i][0][1]), 4)

    # Severity = highest score across all labels
    severity_score = max(breakdown.values())
    severity_level = _get_severity_level(severity_score)

    # A comment is toxic if any label exceeds 0.5 threshold
    is_toxic = any(v > 0.5 for v in breakdown.values())

    # List of labels that crossed the 0.5 threshold
    flagged_labels = [label for label, score in breakdown.items() if score > 0.5]

    return {
        "is_toxic": is_toxic,
        "severity_level": severity_level,
        "severity_score": round(severity_score, 4),
        "breakdown": breakdown,
        "flagged_labels": flagged_labels,
        "input_text": text,
    }
