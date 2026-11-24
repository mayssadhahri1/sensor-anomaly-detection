import pandas as pd

from src.anomaly.data import SENSORS, generate_dataset
from src.anomaly.features import build_features, feature_columns
from src.anomaly.model import predict_labels, train


def small_dataset() -> pd.DataFrame:
    return generate_dataset(n_machines=2, n_minutes=4000, seed=7)


def test_dataset_contains_anomalies():
    df = small_dataset()
    assert set(SENSORS).issubset(df.columns)
    assert 0 < df["is_anomaly"].mean() < 0.2


def test_features_have_no_missing_values():
    feats = build_features(small_dataset())
    assert not feats[feature_columns()].isna().any().any()


def test_model_detects_injected_faults():
    pipe, metrics = train(small_dataset())
    assert metrics.recall > 0.5
    assert metrics.precision > 0.3

    labels = predict_labels(pipe, build_features(small_dataset()))
    assert set(labels).issubset({0, 1})
