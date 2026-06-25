"""Generuje parquet z cechami wina dla Feast (offline store).

Z data/01_raw/winequality-red.csv tworzy feature_repo/data/wine_features.parquet:
    wine_id (int), event_timestamp (stały), 11 cech (nazwy z podkreśleniami — Feast nie
    dopuszcza spacji).

Uruchomienie:
    cd feature_repo && python generate_features.py
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = REPO_ROOT / "data" / "01_raw" / "winequality-red.csv"
OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_PARQUET = OUT_DIR / "wine_features.parquet"

# mapowanie nazw kolumn CSV (ze spacjami) -> nazwy Feast (z podkreśleniami)
RENAME = {
    "fixed acidity": "fixed_acidity",
    "volatile acidity": "volatile_acidity",
    "citric acid": "citric_acid",
    "residual sugar": "residual_sugar",
    "chlorides": "chlorides",
    "free sulfur dioxide": "free_sulfur_dioxide",
    "total sulfur dioxide": "total_sulfur_dioxide",
    "density": "density",
    "pH": "pH",
    "sulphates": "sulphates",
    "alcohol": "alcohol",
}


def main() -> Path:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Brak danych: {RAW_CSV}. Uruchom: python serving/bootstrap_data.py"
        )
    df = pd.read_csv(RAW_CSV)[list(RENAME.keys())].rename(columns=RENAME)
    df = df.astype("float32")
    df.insert(0, "wine_id", range(len(df)))
    # Feast wymaga kolumny czasu; demo używa jednego znacznika dla całego zbioru
    df["event_timestamp"] = datetime(2024, 1, 1, tzinfo=timezone.utc)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"[generate_features] Zapisano {len(df)} wierszy: {OUT_PARQUET}")
    return OUT_PARQUET


if __name__ == "__main__":
    main()
