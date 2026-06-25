"""Logowanie każdej predykcji do pliku CSV (sekcja 5: monitoring).

Każdy wiersz: timestamp + 11 cech wejściowych + prediction + label + probability.
Plik: serving/logs/predictions.csv (tworzony przy pierwszym zapisie).
"""
from __future__ import annotations

import csv
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_PATH = LOG_DIR / "predictions.csv"

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

_HEADER = ["timestamp", *FEATURES, "prediction", "label", "probability"]
_lock = threading.Lock()


def log_prediction(
    features: Dict[str, float], prediction: int, label: str, probability: float
) -> None:
    """Dopisuje jedną predykcję do predictions.csv (thread-safe)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **{f: features.get(f) for f in FEATURES},
        "prediction": int(prediction),
        "label": label,
        "probability": round(float(probability), 4),
    }
    with _lock:
        is_new = not LOG_PATH.exists()
        with LOG_PATH.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_HEADER)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
