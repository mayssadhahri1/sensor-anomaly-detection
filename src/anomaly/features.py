"""Construction des caractéristiques (features) à partir des séries temporelles."""
from __future__ import annotations

import pandas as pd

from .data import SENSORS

WINDOW = 15  # fenêtre glissante de 15 minutes


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute moyenne, écart-type et écart à la moyenne glissante pour chaque capteur.

    Le calcul est fait machine par machine pour ne pas mélanger les signaux.
    """
    df = df.sort_values(["machine_id", "timestamp"]).copy()
    grouped = df.groupby("machine_id")

    for col in SENSORS:
        roll = grouped[col].rolling(WINDOW, min_periods=1)
        df[f"{col}_mean"] = roll.mean().reset_index(level=0, drop=True)
        df[f"{col}_std"] = roll.std().reset_index(level=0, drop=True).fillna(0)
        df[f"{col}_delta"] = df[col] - df[f"{col}_mean"]

    return df


def feature_columns() -> list[str]:
    """Liste des colonnes utilisées par le modèle."""
    cols = list(SENSORS)
    for col in SENSORS:
        cols += [f"{col}_mean", f"{col}_std", f"{col}_delta"]
    return cols
