"""Odtworzenie modelu bazowego (mock) dla serwowania — identycznie jak notebook.

Dopóki nie istnieje MLflow Model Registry (Commit 2), serwowanie korzysta z tego
artefaktu jako "mocka" modelu produkcyjnego. Logika 1:1 z notebooks/01_baseline_eda.ipynb:
    drop_duplicates -> stratified split 80/20 (seed 42) -> StandardScaler -> LogisticRegression

Zapisuje:
    data/06_models/baseline_logreg.pkl          -> {"scaler","model","features","metrics"}
    serving/monitoring/reference_stats.parquet  -> rozkład cech treningowych (do detekcji driftu)

Uruchomienie:
    python serving/bootstrap_model.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = REPO_ROOT / "data" / "01_raw" / "winequality-red.csv"
MODELS_DIR = REPO_ROOT / "data" / "06_models"
MODEL_PATH = MODELS_DIR / "baseline_logreg.pkl"
REFERENCE_PATH = REPO_ROOT / "serving" / "monitoring" / "reference_stats.parquet"

RANDOM_STATE = 42

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def train_and_save() -> dict:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Brak danych: {RAW_CSV}. Uruchom najpierw: python serving/bootstrap_data.py"
        )

    df = pd.read_csv(RAW_CSV)
    df["target"] = (df["quality"] >= 6).astype(int)

    data = df.drop_duplicates().reset_index(drop=True)
    X = data[FEATURES]
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 3),
        "precision": round(float(precision_score(y_test, y_pred)), 3),
        "recall": round(float(recall_score(y_test, y_pred)), 3),
        "f1": round(float(f1_score(y_test, y_pred)), 3),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 3),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {"scaler": scaler, "model": model, "features": FEATURES, "metrics": metrics}
    joblib.dump(artifact, MODEL_PATH)
    print(f"[bootstrap_model] Zapisano model: {MODEL_PATH}")
    print(f"[bootstrap_model] Metryki (test): {metrics}")

    # referencja treningowa do detekcji driftu (surowe, nieskalowane cechy)
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    X_train.reset_index(drop=True).to_parquet(REFERENCE_PATH, index=False)
    print(f"[bootstrap_model] Zapisano referencję driftu: {REFERENCE_PATH}")

    return metrics


if __name__ == "__main__":
    train_and_save()
