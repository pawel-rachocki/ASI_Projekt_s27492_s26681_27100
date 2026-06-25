"""Detekcja driftu danych (sekcja 5: monitoring).

Rdzeń: test Kolmogorowa-Smirnowa (scipy.stats.ks_2samp) per cecha — porównuje rozkład
cech z zalogowanych predykcji (serving/logs/predictions.csv) z rozkładem treningowym
(serving/monitoring/reference_stats.parquet). Cecha jest oznaczona jako "drift",
gdy p-value < progu (domyślnie 0.05).

Opcjonalnie: raport Evidently (DataDriftPreset) -> serving/logs/drift_report.html.
Evidently jest wrapowany w try/except, by ewentualna niekompatybilność wersji nie
psuła rdzennej analizy K-S.

Wynik zapisywany do serving/logs/drift_report.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
from scipy.stats import ks_2samp

BASE = Path(__file__).resolve().parents[1]
REFERENCE_PATH = BASE / "monitoring" / "reference_stats.parquet"
PREDICTIONS_PATH = BASE / "logs" / "predictions.csv"
DRIFT_JSON = BASE / "logs" / "drift_report.json"
DRIFT_HTML = BASE / "logs" / "drift_report.html"

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


def _evidently_report(reference: pd.DataFrame, current: pd.DataFrame) -> bool:
    """Generuje raport HTML Evidently. Zwraca True przy sukcesie."""
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference[FEATURES], current_data=current[FEATURES])
        DRIFT_HTML.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(DRIFT_HTML))
        return True
    except Exception as exc:  # pragma: no cover - zależne od wersji evidently
        print(f"[drift] Pominięto raport Evidently ({exc}).")
        return False


def compute_drift(p_threshold: float = 0.05, with_html: bool = True) -> Dict:
    """Liczy drift K-S per cecha i (opcjonalnie) raport Evidently."""
    if not REFERENCE_PATH.exists():
        return {
            "status": "no_reference",
            "message": "Brak reference_stats.parquet — uruchom serving/bootstrap_model.py.",
        }
    if not PREDICTIONS_PATH.exists():
        return {
            "status": "no_predictions",
            "message": "Brak predictions.csv — wykonaj kilka predykcji, potem sprawdź drift.",
        }

    reference = pd.read_parquet(REFERENCE_PATH)
    current = pd.read_csv(PREDICTIONS_PATH)

    per_feature = {}
    drifted = []
    for feat in FEATURES:
        if feat not in current.columns or current[feat].dropna().empty:
            continue
        stat, p_value = ks_2samp(reference[feat].dropna(), current[feat].dropna())
        is_drift = bool(p_value < p_threshold)
        per_feature[feat] = {
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "drift": is_drift,
        }
        if is_drift:
            drifted.append(feat)

    n_checked = len(per_feature)
    result = {
        "status": "ok",
        "n_samples_current": int(len(current)),
        "p_threshold": p_threshold,
        "n_features_checked": n_checked,
        "n_features_drifted": len(drifted),
        "drift_share": round(len(drifted) / n_checked, 3) if n_checked else 0.0,
        "dataset_drift": len(drifted) > n_checked / 2 if n_checked else False,
        "drifted_features": drifted,
        "per_feature": per_feature,
    }

    if with_html:
        result["html_report"] = _evidently_report(reference, current)

    DRIFT_JSON.parent.mkdir(parents=True, exist_ok=True)
    DRIFT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(compute_drift(), indent=2, ensure_ascii=False))
