from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from utils import load_model_artifacts, prepare_window_features


MODEL_PATH = Path("models/gesture_model.pkl")
ENCODER_PATH = Path("models/label_encoder.pkl")

app = FastAPI(title="Sign Language Gesture API", version="1.0.0")


class PredictRequest(BaseModel):
    # Shape expected: [window_size, 11], e.g., [40, 11]
    window: List[List[float]] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    gesture: str
    confidence: float
    probabilities: Optional[dict] = None


@app.on_event("startup")
def startup_event() -> None:
    model, encoder = load_model_artifacts(MODEL_PATH, ENCODER_PATH)
    app.state.model = model
    app.state.encoder = encoder


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        feature = prepare_window_features(payload.window).reshape(1, -1)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid window payload: {exc}") from exc

    model = app.state.model
    encoder = app.state.encoder

    pred_idx = int(model.predict(feature)[0])
    label = str(encoder.inverse_transform([pred_idx])[0])

    confidence = 1.0
    probabilities = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feature)[0]
        confidence = float(np.max(proba))
        probabilities = {
            str(encoder.inverse_transform([i])[0]): float(score)
            for i, score in enumerate(proba)
        }

    return PredictResponse(gesture=label, confidence=confidence, probabilities=probabilities)
