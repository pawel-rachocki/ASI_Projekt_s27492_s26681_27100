"""Ładowanie modelu do serwowania.

Strategia (interfejs zespołu): najpierw spróbuj pobrać model z MLflow Model Registry
(stage Production) — to docelowe źródło, gdy powstanie Commit 2. Jeśli Registry jest
niedostępne, użyj lokalnego artefaktu bazowego (mock) z data/06_models/baseline_logreg.pkl.

Zwraca obiekt LoadedModel z polami: scaler, model, features, source.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_PATH = REPO_ROOT / "data" / "06_models" / "baseline_logreg.pkl"

# konfiguracja MLflow Registry (nadpisywalna zmiennymi środowiskowymi)
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "wine_quality")
MLFLOW_MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")  # None => Registry pomijany


@dataclass
class LoadedModel:
    scaler: object
    model: object
    features: List[str]
    source: str = field(default="unknown")

    def predict(self, X):
        """X: pandas.DataFrame z kolumnami == features (surowe, nieskalowane)."""
        X = X[self.features]
        Xs = self.scaler.transform(X)
        proba = self.model.predict_proba(Xs)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return pred, proba


def _try_load_from_registry() -> LoadedModel | None:
    """Próba pobrania modelu z MLflow Model Registry (Production). None gdy brak."""
    if not MLFLOW_TRACKING_URI:
        return None
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        uri = f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}"
        pipeline = mlflow.sklearn.load_model(uri)
        # Oczekujemy, że Commit 2 zapisze model jako sklearn Pipeline (scaler+model)
        # albo artefakt zgodny z interfejsem .predict_proba. Tu opakowujemy minimalnie.
        from sklearn.preprocessing import FunctionTransformer

        return LoadedModel(
            scaler=FunctionTransformer(),  # pipeline z Registry sam skaluje
            model=pipeline,
            features=_default_features(),
            source=f"mlflow:{uri}",
        )
    except Exception as exc:  # pragma: no cover - zależne od środowiska
        print(f"[model_loader] MLflow Registry niedostępne ({exc}); używam mocka lokalnego.")
        return None


def _default_features() -> List[str]:
    return [
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


def _load_from_local() -> LoadedModel:
    if not LOCAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Brak modelu: {LOCAL_MODEL_PATH}. Uruchom: python serving/bootstrap_model.py"
        )
    art = joblib.load(LOCAL_MODEL_PATH)
    return LoadedModel(
        scaler=art["scaler"],
        model=art["model"],
        features=art["features"],
        source=f"local:{LOCAL_MODEL_PATH.name}",
    )


def load_model() -> LoadedModel:
    """Zwraca załadowany model (Registry → fallback lokalny mock)."""
    loaded = _try_load_from_registry()
    if loaded is None:
        loaded = _load_from_local()
    print(f"[model_loader] Źródło modelu: {loaded.source}")
    return loaded
