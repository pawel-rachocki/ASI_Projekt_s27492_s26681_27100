import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from typing import Any


def train_baseline_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> Any:
    """Trenuje model LogisticRegression na danych treningowych i zwraca wytrenowany model.

    Parametry:
        C=1.0, max_iter=1000, random_state=42

    Śledzi eksperyment w MLflow (parametry, metryki na zbiorze treningowym, artefakt modelu).
    Jeśli MLflow jest niedostępny, trening przebiega bez śledzenia.

    Zwraca:
        Wytrenowany obiekt LogisticRegression.
    """
    params = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs", "random_state": 42}
    model = LogisticRegression(**params)
    y_flat = y_train.values.ravel()
    model.fit(X_train, y_flat)

    # ── MLflow tracking ──────────────────────────────────────────────────────
    try:
        import mlflow
        import mlflow.sklearn
        from pathlib import Path

        # Resolve project root so the sqlite URI is always relative to the repo
        _repo_root = Path(__file__).resolve().parents[4]
        _prev_dir = os.getcwd()
        os.chdir(_repo_root)

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("wine_quality_baseline")

        with mlflow.start_run(run_name="train_baseline_logreg"):
            # Params
            mlflow.log_param("C", params["C"])
            mlflow.log_param("max_iter", params["max_iter"])
            mlflow.log_param("solver", params["solver"])

            # Training-set metrics
            y_pred  = model.predict(X_train)
            y_proba = model.predict_proba(X_train)[:, 1]
            train_f1  = f1_score(y_flat, y_pred)
            train_auc = roc_auc_score(y_flat, y_proba)
            mlflow.log_metric("train_f1", round(float(train_f1), 4))
            mlflow.log_metric("train_roc_auc", round(float(train_auc), 4))

            # Model artifact
            mlflow.sklearn.log_model(sk_model=model, artifact_path="baseline_logreg")

        os.chdir(_prev_dir)
    except Exception as _mlflow_err:  # pragma: no cover
        # MLflow errors must never break the Kedro pipeline
        import warnings
        warnings.warn(f"[data_science] MLflow tracking skipped: {_mlflow_err}")

    return model
