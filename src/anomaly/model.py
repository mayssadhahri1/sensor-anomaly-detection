"""Entraînement, évaluation et chargement du modèle de détection d'anomalies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_features, feature_columns

FALSE_ALARM_RATE = 0.01


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict:
        return {"precision": round(self.precision, 3), "recall": round(self.recall, 3), "f1": round(self.f1, 3)}


def make_pipeline(contamination: float) -> Pipeline:
    """Normalisation + Isolation Forest (non supervisé)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", IsolationForest(n_estimators=200, contamination=contamination, random_state=42)),
    ])


def time_split(df: pd.DataFrame, ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Découpage temporel : on entraîne sur le passé et on teste sur le futur."""
    cut = df["timestamp"].quantile(ratio)
    return df[df["timestamp"] <= cut], df[df["timestamp"] > cut]


def train(df: pd.DataFrame) -> tuple[Pipeline, Metrics]:
    """Entraîne le modèle et retourne ses métriques sur la période de test."""
    feats = build_features(df)
    train_df, test_df = time_split(feats)

    # On entraîne uniquement sur le fonctionnement normal (comme en usine)
    normal = train_df[train_df["is_anomaly"] == 0]
    # Comme on n'entraîne que sur des données normales, "contamination" représente
    # le taux de fausses alertes accepté : 1 % donne le meilleur F1 (voir README)
    pipe = make_pipeline(contamination=FALSE_ALARM_RATE)
    pipe.fit(normal[feature_columns()])

    y_true = test_df["is_anomaly"].to_numpy()
    y_pred = predict_labels(pipe, test_df)
    metrics = Metrics(
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
    )
    return pipe, metrics


def predict_labels(pipe: Pipeline, feats: pd.DataFrame) -> np.ndarray:
    """Retourne 1 pour une anomalie, 0 sinon (IsolationForest renvoie -1 / 1)."""
    raw = pipe.predict(feats[feature_columns()])
    return (raw == -1).astype(int)


def anomaly_scores(pipe: Pipeline, feats: pd.DataFrame) -> np.ndarray:
    """Score d'anomalie : plus il est élevé, plus la mesure est suspecte."""
    return -pipe.score_samples(feats[feature_columns()])


def save(pipe: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def load(path: Path) -> Pipeline:
    return joblib.load(path)
