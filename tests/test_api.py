import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from src.anomaly.data import generate_dataset
from src.anomaly.model import save, train


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Modèle temporaire pour ne pas dépendre d'un entraînement préalable
    path = tmp_path_factory.mktemp("models") / "model.joblib"
    pipe, _ = train(generate_dataset(n_machines=2, n_minutes=4000, seed=3))
    save(pipe, path)
    api_main.MODEL_PATH = path
    api_main._model = None
    return TestClient(api_main.app)


def reading(minute: int, temperature: float = 65.0) -> dict:
    return {
        "timestamp": f"2026-03-01T10:{minute:02d}:00",
        "machine_id": "M01",
        "temperature": temperature,
        "vibration": 2.0,
        "pressure": 5.5,
        "current": 12.0,
    }


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_normal_readings_are_not_flagged(client):
    res = client.post("/predict", json={"readings": [reading(m) for m in range(15)]})
    assert res.status_code == 200
    assert sum(p["is_anomaly"] for p in res.json()) <= 2


def test_overheat_is_flagged(client):
    readings = [reading(m) for m in range(14)] + [reading(14, temperature=95.0)]
    res = client.post("/predict", json={"readings": readings})
    assert res.json()[-1]["is_anomaly"] is True


def test_empty_request_is_rejected(client):
    assert client.post("/predict", json={"readings": []}).status_code == 422
