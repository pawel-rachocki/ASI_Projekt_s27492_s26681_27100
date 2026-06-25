"""Flask API + GUI do serwowania modelu jakości wina (sekcja 5).

Endpointy:
    GET  /                 -> formularz HTML (11 cech) + wynik
    POST /predict          -> JSON: {"features": {...}}  ->  predykcja
    POST /predict/by_id    -> JSON: {"wine_id": N}        ->  cechy z Feast + predykcja
    GET  /health           -> status + źródło modelu
    GET  /monitoring/drift -> raport driftu (K-S + opcjonalnie Evidently)

Uruchomienie (dev):
    flask --app serving/app run --port 5000
Uruchomienie (prod):
    gunicorn serving.app:app -b 0.0.0.0:5000
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

# pozwól na import zarówno jako `serving.app`, jak i bezpośrednio `python serving/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serving.model_loader import load_model  # noqa: E402
from serving.monitoring import compute_drift, log_prediction  # noqa: E402

app = Flask(__name__)

# model ładowany raz przy starcie (Registry -> fallback lokalny mock)
MODEL = load_model()
FEATURES = MODEL.features

# domyślne wartości formularza (~mediany zbioru, dla wygody dema)
DEFAULTS = {
    "fixed acidity": 7.9,
    "volatile acidity": 0.52,
    "citric acid": 0.26,
    "residual sugar": 2.2,
    "chlorides": 0.079,
    "free sulfur dioxide": 14.0,
    "total sulfur dioxide": 38.0,
    "density": 0.9968,
    "pH": 3.31,
    "sulphates": 0.62,
    "alcohol": 10.2,
}


def _predict_from_features(feat: dict) -> dict:
    """Wspólna logika: walidacja, predykcja, logowanie."""
    missing = [f for f in FEATURES if f not in feat]
    if missing:
        raise ValueError(f"Brak cech: {missing}")
    try:
        row = {f: float(feat[f]) for f in FEATURES}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cechy muszą być liczbami: {exc}")

    X = pd.DataFrame([row])
    pred, proba = MODEL.predict(X)
    prediction = int(pred[0])
    probability = float(proba[0])
    label = "dobre" if prediction == 1 else "słabe"

    log_prediction(row, prediction, label, probability)
    return {
        "prediction": prediction,
        "label": label,
        "probability": round(probability, 4),
        "model_source": MODEL.source,
    }


@app.get("/")
def index():
    return render_template(
        "index.html", features=FEATURES, defaults=DEFAULTS, result=None, values=DEFAULTS
    )


@app.post("/")
def index_submit():
    values = {f: request.form.get(f, DEFAULTS.get(f)) for f in FEATURES}
    try:
        result = _predict_from_features(values)
        error = None
    except ValueError as exc:
        result, error = None, str(exc)
    return render_template(
        "index.html", features=FEATURES, defaults=DEFAULTS, result=result,
        values=values, error=error,
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    # body może być listą 11 wartości albo obiektem {"features": {...}} / {...}
    if isinstance(payload, list):
        feat = payload
    elif isinstance(payload, dict):
        feat = payload.get("features", payload)
    else:
        feat = {}
    if isinstance(feat, list):
        if len(feat) != len(FEATURES):
            return jsonify({"error": f"Oczekiwano {len(FEATURES)} wartości"}), 400
        feat = dict(zip(FEATURES, feat))
    try:
        return jsonify(_predict_from_features(feat))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/predict/by_id")
def predict_by_id():
    """Demo Feast: pobranie cech z online store po wine_id, potem predykcja."""
    payload = request.get_json(silent=True) or {}
    wine_id = payload.get("wine_id")
    if wine_id is None:
        return jsonify({"error": "Podaj 'wine_id'"}), 400
    try:
        from feast import FeatureStore

        store = FeatureStore(repo_path=str(Path(__file__).resolve().parents[1] / "feature_repo"))
        # Feast nie dopuszcza spacji w nazwach cech — mapujemy "fixed acidity" <-> "fixed_acidity"
        feast_name = {f: f.replace(" ", "_") for f in FEATURES}
        refs = [f"wine_features:{feast_name[f]}" for f in FEATURES]
        rows = store.get_online_features(
            features=refs, entity_rows=[{"wine_id": int(wine_id)}]
        ).to_dict()
        feat = {f: rows[feast_name[f]][0] for f in FEATURES}
    except Exception as exc:
        return jsonify({"error": f"Feast niedostępny lub brak wine_id: {exc}"}), 503
    result = _predict_from_features(feat)
    result["wine_id"] = int(wine_id)
    result["features"] = feat
    return jsonify(result)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_source": MODEL.source, "n_features": len(FEATURES)})


@app.get("/monitoring/drift")
def drift():
    return jsonify(compute_drift())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
