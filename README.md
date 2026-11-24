# Sensor Anomaly Detection

Détection d'anomalies sur des capteurs de machines industrielles (température, vibration, pression, courant) avec un pipeline de machine learning complet : données → features → modèle → API → tableau de bord.

![CI](https://github.com/mayssadhahri1/sensor-anomaly-detection/actions/workflows/ci.yml/badge.svg)

## Fonctionnalités

- **Données** : 5 machines, une mesure par minute pendant ~7 jours, avec des pannes simulées (surchauffe, roulement usé, fuite de pression).
- **Features** : moyenne, écart-type et écart à la moyenne sur une fenêtre glissante de 15 minutes, calculés machine par machine.
- **Modèle** : `StandardScaler` + `IsolationForest`, entraîné uniquement sur le fonctionnement normal, comme en usine où les pannes sont rares.
- **Évaluation** : découpage temporel (70 % passé pour l'entraînement, 30 % futur pour le test).
- **API REST** (FastAPI) : `POST /predict` renvoie pour chaque mesure un score et un indicateur d'anomalie.
- **Dashboard** (Streamlit) : courbes par capteur, score d'anomalie et liste des alertes.
- **Qualité** : tests `pytest`, image Docker, intégration continue GitHub Actions.

## Résultats

| Métrique | Valeur (période de test) |
|---|---|
| Précision | 0.70 |
| Rappel | 0.77 |
| F1-score | 0.73 |

Le seuil de fausses alertes (`FALSE_ALARM_RATE = 1 %`) a été choisi en comparant plusieurs valeurs :

| Taux | Précision | Rappel | F1 |
|---|---|---|---|
| 0,5 % | 0.56 | 0.17 | 0.26 |
| **1 %** | **0.70** | **0.77** | **0.73** |
| 2 % | 0.56 | 0.92 | 0.70 |

> Les données sont simulées : les résultats montrent la démarche, pas la performance sur une vraie usine.

## Architecture

```
src/anomaly/data.py      génération / chargement des données
src/anomaly/features.py  fenêtres glissantes par machine
src/anomaly/model.py     entraînement, évaluation, sauvegarde
src/train.py             script d'entraînement
api/main.py              API FastAPI
dashboard/app.py         tableau de bord Streamlit
tests/                   tests du pipeline et de l'API
```

## Lancer le projet

```bash
pip install -r requirements.txt
python -m src.train                    # génère les données et entraîne le modèle
uvicorn api.main:app --reload          # API sur http://localhost:8000/docs
streamlit run dashboard/app.py         # tableau de bord
pytest -q                              # tests
```

Avec Docker :

```bash
docker build -t sensor-anomaly-detection .
docker run -p 8000:8000 sensor-anomaly-detection
```

## Exemple d'appel

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "readings": [
    {"timestamp": "2026-03-01T10:00:00", "machine_id": "M01",
     "temperature": 95.0, "vibration": 2.1, "pressure": 5.4, "current": 14.2}
  ]
}'
```

## Pistes d'amélioration

- Tester un autoencodeur ou un modèle LSTM sur des séquences.
- Ajouter une alerte (email ou Slack) quand une machine dépasse un seuil.
- Brancher l'API sur un flux temps réel (MQTT ou Kafka).

## Auteure

Mayssa Dhahri, élève-ingénieure en génie logiciel à l'École Polytechnique de Sousse.
[LinkedIn](https://www.linkedin.com/in/mayssa-dhahri-38b1373bb/)
