"""Tableau de bord Streamlit : streamlit run dashboard/app.py"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.anomaly.data import SENSORS, load_or_create  # noqa: E402
from src.anomaly.features import build_features  # noqa: E402
from src.anomaly.model import anomaly_scores, load, predict_labels  # noqa: E402

st.set_page_config(page_title="Supervision des machines", layout="wide")
st.title("Supervision des machines : détection d'anomalies")

model_path = ROOT / "models" / "isolation_forest.joblib"
if not model_path.exists():
    st.error("Aucun modèle trouvé. Lancez d'abord : python -m src.train")
    st.stop()


@st.cache_data
def prepare() -> pd.DataFrame:
    df = build_features(load_or_create(ROOT / "data" / "sensors.csv"))
    model = load(model_path)
    df["predicted"] = predict_labels(model, df)
    df["score"] = anomaly_scores(model, df)
    return df


df = prepare()

# Indicateurs globaux
metrics_file = ROOT / "models" / "metrics.json"
if metrics_file.exists():
    m = json.loads(metrics_file.read_text())
    c1, c2, c3 = st.columns(3)
    c1.metric("Précision", f"{m['precision']:.0%}")
    c2.metric("Rappel", f"{m['recall']:.0%}")
    c3.metric("F1-score", f"{m['f1']:.0%}")

machine = st.sidebar.selectbox("Machine", sorted(df["machine_id"].unique()))
sensor = st.sidebar.selectbox("Capteur", SENSORS)
view = df[df["machine_id"] == machine].set_index("timestamp")

st.subheader(f"{machine} : {sensor}")
st.line_chart(view[[sensor]])

st.subheader("Score d'anomalie")
st.area_chart(view[["score"]])

alerts = view[view["predicted"] == 1]
st.subheader(f"Alertes détectées : {len(alerts)}")
st.dataframe(alerts[SENSORS + ["score"]].sort_values("score", ascending=False).head(50))
