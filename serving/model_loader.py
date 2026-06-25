"""Ładowanie modelu do serwowania.

Strategia: najpierw spróbuj pobrać model z MLflow Model Registry (stage Production).
Model z Registry może być:
  a) sklearn Pipeline z wbudowanym scalerem (wtedy FunctionTransformer jako pass-through), lub
  b) surowy klasyfikator (ExtraTrees, RF itp.) — wtedy scaler jest budowany z danych
     treningowych z data/05_model_input/X_train.csv, lub jako pass-through gdy brak danych.

Jeśli Registry jest niedostępne, załaduj lokalny artefakt z data/06_models/baseline_logreg.pkl
lub best_model.pkl.

Zwraca obiekt LoadedModel z polami: scaler, model, features, source.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_PATH = REPO_ROOT / "data" / "06_models" / "baseline_logreg.pkl"
BEST_MODEL_PATH  = REPO_ROOT / "data" / "06_models" / "best_model.pkl"
X_TRAIN_PATH     = REPO_ROOT / "data" / "05_model_input" / "X_train.csv"

# konfiguracja MLflow Registry (nadpisywalna zmiennymi środowiskowymi)
MLFLOW_MODEL_NAME  = os.getenv("MLFLOW_MODEL_NAME", "wine_quality")
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


def _build_scaler_from_train_data():
    """Buduje StandardScaler z danych treningowych (odtwarzając podział z preprocessing pipeline)."""
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    raw_csv_path = REPO_ROOT / "data" / "01_raw" / "winequality-red.csv"
    if raw_csv_path.exists():
        try:
            import pandas as pd
            from sklearn.model_selection import train_test_split
            
            df = pd.read_csv(raw_csv_path)
            cleaned = df.drop_duplicates().dropna().reset_index(drop=True)
            cleaned["target"] = (cleaned["quality"] >= 6).astype(int)
            
            X = cleaned[_default_features()]
            y = cleaned["target"]
            X_train, _, _, _ = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            scaler = StandardScaler()
            scaler.fit(X_train)
            return scaler
        except Exception as exc:
            print(f"[model_loader] Błąd budowania scalera z surowych danych: {exc}")
            
    # Fallback do danych przetworzonych
    if X_TRAIN_PATH.exists():
        try:
            import pandas as pd
            X_train = pd.read_csv(X_TRAIN_PATH)
            scaler = StandardScaler()
            scaler.fit(X_train[_default_features()])
            return scaler
        except Exception:
            pass
            
    return FunctionTransformer()  # pass-through — dane już były skalowane


def _is_sklearn_pipeline(obj) -> bool:
    """Sprawdza, czy obiekt jest sklearn Pipeline (zawiera scaler)."""
    try:
        from sklearn.pipeline import Pipeline
        return isinstance(obj, Pipeline)
    except ImportError:
        return False


def _try_load_from_registry() -> LoadedModel | None:
    """Próba pobrania modelu z MLflow Model Registry (Production). None gdy brak."""
    if not MLFLOW_TRACKING_URI:
        return None
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        uri = f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}"
        loaded_obj = mlflow.sklearn.load_model(uri)

        if _is_sklearn_pipeline(loaded_obj):
            # Pipeline z wbudowanym scalerem — pass-through scaler
            from sklearn.preprocessing import FunctionTransformer
            scaler = FunctionTransformer()
        else:
            # Surowy klasyfikator (ExtraTrees, RF itp.) — budujemy scaler z danych treningowych
            scaler = _build_scaler_from_train_data()

        return LoadedModel(
            scaler=scaler,
            model=loaded_obj,
            features=_default_features(),
            source=f"mlflow:{uri}",
        )
    except Exception as exc:  # pragma: no cover - zależne od środowiska
        print(f"[model_loader] MLflow Registry niedostępne ({exc}); używam modelu lokalnego.")
        return None


def _load_from_local() -> LoadedModel:
    """Ładuje lokalny model: best_model.pkl → baseline_logreg.pkl."""
    # Preferuj najlepszy model z eksperymentów jeśli istnieje
    if BEST_MODEL_PATH.exists():
        import pickle
        with open(BEST_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        scaler = _build_scaler_from_train_data()
        return LoadedModel(
            scaler=scaler,
            model=model,
            features=_default_features(),
            source=f"local:{BEST_MODEL_PATH.name}",
        )
    # Fallback do baseline (joblib artifact dict)
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
    """Zwraca załadowany model (Registry → best_model.pkl → baseline_logreg.pkl)."""
    loaded = _try_load_from_registry()
    if loaded is None:
        loaded = _load_from_local()
    print(f"[model_loader] Źródło modelu: {loaded.source}")
    return loaded
