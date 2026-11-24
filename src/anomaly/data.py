"""Génération et chargement des données de capteurs industriels.

Les données sont simulées : chaque machine envoie une mesure par minute
(température, vibration, pression, courant). On injecte des pannes
connues pour pouvoir évaluer le modèle.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SENSORS = ["temperature", "vibration", "pressure", "current"]


def generate_machine(machine_id: str, n_minutes: int, rng: np.random.Generator) -> pd.DataFrame:
    """Simule le signal d'une machine avec un cycle journalier et du bruit."""
    t = np.arange(n_minutes)
    cycle = np.sin(2 * np.pi * t / 1440)  # cycle de 24 h

    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n_minutes, freq="min"),
        "machine_id": machine_id,
        "temperature": 65 + 4 * cycle + rng.normal(0, 0.8, n_minutes),
        "vibration": 2.0 + 0.3 * cycle + rng.normal(0, 0.15, n_minutes),
        "pressure": 5.5 + rng.normal(0, 0.1, n_minutes),
        "current": 12 + 1.5 * cycle + rng.normal(0, 0.4, n_minutes),
        "is_anomaly": 0,
    })

    # Injection de pannes : surchauffe, roulement usé, fuite de pression
    n_events = max(1, n_minutes // 2000)
    for _ in range(n_events):
        start = int(rng.integers(100, n_minutes - 120))
        length = int(rng.integers(20, 90))
        end = start + length
        kind = rng.choice(["overheat", "bearing", "leak"])
        if kind == "overheat":
            df.loc[start:end, "temperature"] += np.linspace(3, 15, end - start + 1)
            df.loc[start:end, "current"] += 2.5
        elif kind == "bearing":
            df.loc[start:end, "vibration"] += rng.normal(1.5, 0.4, end - start + 1)
        else:
            df.loc[start:end, "pressure"] -= np.linspace(0.3, 1.5, end - start + 1)
        df.loc[start:end, "is_anomaly"] = 1

    return df


def generate_dataset(n_machines: int = 5, n_minutes: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Crée le jeu de données complet pour plusieurs machines."""
    rng = np.random.default_rng(seed)
    frames = [generate_machine(f"M{i + 1:02d}", n_minutes, rng) for i in range(n_machines)]
    return pd.concat(frames, ignore_index=True)


def load_or_create(path: Path) -> pd.DataFrame:
    """Charge le CSV s'il existe, sinon le génère."""
    if path.exists():
        return pd.read_csv(path, parse_dates=["timestamp"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(path, index=False)
    return df
