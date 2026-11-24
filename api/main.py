"""API REST qui expose le modèle de détection d'anomalies."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.anomaly.features import build_features
from src.anomaly.model import anomaly_scores, load, predict_labels

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "isolation_forest.joblib"

app = FastAPI(title="Sensor Anomaly Detection API", version="1.0.0")
_model = None


def get_model():
    """Chargement paresseux du modèle (une seule fois)."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Modèle non entraîné : lancez python -m src.train")
        _model = load(MODEL_PATH)
    return _model


class Reading(BaseModel):
    timestamp: str
    machine_id: str
    temperature: float
    vibration: float = Field(ge=0)
    pressure: float
    current: float


class PredictionRequest(BaseModel):
    # On envoie une petite série (idéalement les 15 dernières minutes) pour calculer les fenêtres
    readings: list[Reading] = Field(min_length=1)


class Prediction(BaseModel):
    timestamp: str
    machine_id: str
    is_anomaly: bool
    score: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=list[Prediction])
def predict(req: PredictionRequest) -> list[Prediction]:
    model = get_model()
    df = pd.DataFrame([r.model_dump() for r in req.readings])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    feats = build_features(df)

    labels = predict_labels(model, feats)
    scores = anomaly_scores(model, feats)

    return [
        Prediction(
            timestamp=row.timestamp.isoformat(),
            machine_id=row.machine_id,
            is_anomaly=bool(label),
            score=round(float(score), 4),
        )
        for row, label, score in zip(feats.itertuples(), labels, scores)
    ]
