# AI Content Moderation System — Backend

Multi-label toxic comment classifier built with scikit-learn + FastAPI.

## Project Structure

```
content_moderation/
├── requirements.txt
├── model/
│   ├── data/
│   │   └── train.csv          ← put Kaggle dataset here
│   └── toxic_model.pkl        ← generated after training
└── backend/
    ├── preprocessing.py       ← NLP cleaning pipeline
    ├── train.py               ← model training + evaluation
    ├── predict.py             ← prediction logic
    ├── main.py                ← FastAPI application
    └── test_api.py            ← test suite
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download NLTK data (auto-runs on first import, or manually)
```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
```

### 3. Add dataset
Download `train.csv` from Kaggle (Jigsaw Toxic Comment Classification Challenge)
and place it at: `model/data/train.csv`

### 4. Train the model
```bash
cd backend
python train.py
```
Training takes ~5-8 minutes. Saves `model/toxic_model.pkl`.

### 5. Start the API
```bash
cd backend
uvicorn main:app --reload --port 8000
```

Visit http://localhost:8000/docs for the interactive Swagger UI.

### 6. Run tests
```bash
cd backend
python test_api.py
```

---

## API Endpoints

### `GET /`
Health check.

### `GET /health`
Detailed status including model load state.

### `POST /moderate`
Moderate a single comment.

**Request:**
```json
{ "text": "your comment here" }
```

**Response:**
```json
{
  "is_toxic": true,
  "severity_level": "high",
  "severity_score": 0.82,
  "breakdown": {
    "toxic": 0.82,
    "severe_toxic": 0.14,
    "obscene": 0.67,
    "threat": 0.05,
    "insult": 0.74,
    "identity_hate": 0.03
  },
  "flagged_labels": ["toxic", "obscene", "insult"],
  "input_text": "your comment here",
  "processing_time_ms": 12.5
}
```

### `POST /moderate/batch`
Moderate up to 100 comments at once.

**Request:**
```json
{ "texts": ["comment 1", "comment 2", "comment 3"] }
```

---

## Severity Levels

| Level    | Max score range | Action suggested         |
|----------|-----------------|--------------------------|
| low      | 0.00 – 0.25     | Allow                    |
| moderate | 0.25 – 0.50     | Flag for human review    |
| high     | 0.50 – 0.75     | Auto-hide + notify       |
| critical | 0.75 – 1.00     | Auto-remove + escalate   |

---

## Labels

| Label         | What it detects                        |
|---------------|----------------------------------------|
| toxic         | General abusive language               |
| severe_toxic  | Extreme, targeted abuse                |
| obscene       | Vulgar / explicit content              |
| threat        | Direct threats of violence             |
| insult        | Personal attacks                       |
| identity_hate | Hate based on race, gender, religion   |


