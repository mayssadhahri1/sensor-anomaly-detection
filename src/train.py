"""Script d'entraînement : python -m src.train"""
import json
from pathlib import Path

from src.anomaly.data import load_or_create
from src.anomaly.model import save, train

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sensors.csv"
MODEL_PATH = ROOT / "models" / "isolation_forest.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"


def main() -> None:
    df = load_or_create(DATA_PATH)
    print(f"{len(df)} mesures chargées, {df['is_anomaly'].mean():.2%} d'anomalies")

    pipe, metrics = train(df)
    save(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics.as_dict(), indent=2))

    print("Modèle enregistré :", MODEL_PATH)
    print("Métriques sur la période de test :", metrics.as_dict())


if __name__ == "__main__":
    main()
