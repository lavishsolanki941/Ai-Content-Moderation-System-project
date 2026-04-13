"""
main.py
-------
FastAPI application for the AI Content Moderation System.

Endpoints:
  GET  /           — health check
  GET  /health     — detailed health + model status
  POST /moderate   — single comment moderation
  POST /moderate/batch — moderate multiple comments at once

Run:
    uvicorn main:app --reload --port 8000

Then visit: http://localhost:8000/docs  (auto-generated Swagger UI)
"""

import time
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from predict import predict_toxicity, _pipeline

# ── App setup ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Content Moderation API",
    description=(
        "Multi-label toxicity classifier for social media comments. "
        "Detects: toxic, severe_toxic, obscene, threat, insult, identity_hate."
    ),
    version="1.0.0",
)

# CORS — restrict to known origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ───────────────────────────────────────────────────────────────────────────


# ── Request / Response models ───────────────────────────────────────────────

class TextInput(BaseModel):
    """Input schema for a single moderation request."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The comment text to moderate.",
        examples=["You are an amazing person!"],
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty or whitespace only.")
        return v.strip()


class BatchInput(BaseModel):
    """Input schema for batch moderation."""
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of comment texts (max 100 per request).",
    )


class ToxicityBreakdown(BaseModel):
    toxic: float
    severe_toxic: float
    obscene: float
    threat: float
    insult: float
    identity_hate: float


class ModerationResult(BaseModel):
    """Output schema for a single moderated comment."""
    is_toxic: bool
    severity_level: str     # "low" | "moderate" | "high" | "critical"
    severity_score: float   # 0.0 to 1.0
    breakdown: ToxicityBreakdown
    flagged_labels: List[str]
    input_text: str
    processing_time_ms: float


class BatchModerationResult(BaseModel):
    results: List[ModerationResult]
    total: int
    toxic_count: int
    processing_time_ms: float
# ───────────────────────────────────────────────────────────────────────────


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Basic health check — confirms the API is running."""
    return {
        "status": "running",
        "api": "AI Content Moderation System",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """
    Detailed health check — shows model load status.
    Use this to verify the API is ready before sending requests.
    """
    model_loaded = _pipeline is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "message": (
            "Model ready." if model_loaded
            else "Model not loaded. Run train.py first."
        ),
    }


@app.post("/moderate", response_model=ModerationResult, tags=["Moderation"])
def moderate_single(input: TextInput, request: Request):
    """
    Moderate a single comment.

    Returns per-label toxicity scores, severity level, and flagged labels.

    **Severity levels:**
    - `low`      — max score < 0.25
    - `moderate` — max score 0.25 – 0.50
    - `high`     — max score 0.50 – 0.75
    - `critical` — max score >= 0.75
    """
    start = time.perf_counter()

    try:
        result = predict_toxicity(input.text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    return ModerationResult(
        is_toxic=result["is_toxic"],
        severity_level=result["severity_level"],
        severity_score=result["severity_score"],
        breakdown=ToxicityBreakdown(**result["breakdown"]),
        flagged_labels=result["flagged_labels"],
        input_text=result["input_text"],
        processing_time_ms=elapsed_ms,
    )


@app.post("/moderate/batch", response_model=BatchModerationResult, tags=["Moderation"])
def moderate_batch(input: BatchInput):
    """
    Moderate multiple comments in a single request (max 100).

    Useful for bulk processing existing content.
    Returns individual results plus summary statistics.
    """
    start = time.perf_counter()

    results = []
    for text in input.texts:
        req_start = time.perf_counter()
        try:
            result = predict_toxicity(text)
        except Exception as e:
            # Don't fail entire batch — return error for that item
            result = {
                "is_toxic": False,
                "severity_level": "error",
                "severity_score": 0.0,
                "breakdown": {l: 0.0 for l in [
                    "toxic","severe_toxic","obscene","threat","insult","identity_hate"
                ]},
                "flagged_labels": [],
                "input_text": text,
            }
        req_ms = round((time.perf_counter() - req_start) * 1000, 2)
        results.append(ModerationResult(
            **{k: v for k, v in result.items()
               if k != "breakdown"},
            breakdown=ToxicityBreakdown(**result["breakdown"]),
            processing_time_ms=req_ms,
        ))

    total_ms = round((time.perf_counter() - start) * 1000, 2)
    toxic_count = sum(1 for r in results if r.is_toxic)

    return BatchModerationResult(
        results=results,
        total=len(results),
        toxic_count=toxic_count,
        processing_time_ms=total_ms,
    )
