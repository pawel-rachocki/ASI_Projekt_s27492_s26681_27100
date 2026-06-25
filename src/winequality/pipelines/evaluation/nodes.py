import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> dict:
    """Ewaluuje wytrenowany model na danych testowych i zwraca słownik metryk.

    Obliczane metryki:
        - accuracy
        - f1_score (binary, average='binary')
        - roc_auc

    Wyniki są wypisywane na stdout i zwracane jako słownik (który Kedro zapisuje do
    data/08_reporting/metrics.json przez dataset zdefiniowany w katalogu).
    """
    y_true = y_test.values.ravel()
    y_pred = model.predict(X_test)

    # roc_auc wymaga prawdopodobieństw lub scores, jeśli dostępne
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, average="binary")),
        "roc_auc": float(roc_auc_score(y_true, y_scores)),
    }

    print("=== Evaluation Metrics ===")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print("==========================")

    return metrics
