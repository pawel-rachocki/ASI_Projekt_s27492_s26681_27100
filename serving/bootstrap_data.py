"""Pobranie surowych danych Wine Quality (red) do data/01_raw/.

Dane w repo są śledzone przez DVC, ale remote zespołu to ścieżka Windows
(`D:\\dev\\ASI\\dvc-remote`), więc `dvc pull` nie zadziała na innej maszynie.
Ten skrypt pobiera identyczny (comma-separated) zbiór z Kaggle przez kagglehub.

Uruchomienie:
    python serving/bootstrap_data.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "01_raw"
TARGET_CSV = RAW_DIR / "winequality-red.csv"

KAGGLE_DATASET = "uciml/red-wine-quality-cortez-et-al-2009"
CSV_NAME = "winequality-red.csv"


def ensure_dataset() -> Path:
    """Zwraca ścieżkę do CSV; pobiera z Kaggle, jeśli go brak."""
    if TARGET_CSV.exists():
        print(f"[bootstrap_data] CSV już istnieje: {TARGET_CSV}")
        return TARGET_CSV

    print(f"[bootstrap_data] Pobieranie '{KAGGLE_DATASET}' przez kagglehub...")
    import kagglehub  # import tutaj, by skrypt nie wymagał kagglehub gdy CSV już jest

    path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    src = path / CSV_NAME
    if not src.exists():
        # fallback: znajdź dowolny pasujący plik
        matches = list(path.rglob(CSV_NAME))
        if not matches:
            raise FileNotFoundError(f"Nie znaleziono {CSV_NAME} w {path}")
        src = matches[0]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, TARGET_CSV)
    print(f"[bootstrap_data] Zapisano: {TARGET_CSV}")
    return TARGET_CSV


if __name__ == "__main__":
    csv = ensure_dataset()
    # szybka walidacja zgodności z kontraktem danych (1599 x 12)
    import pandas as pd

    df = pd.read_csv(csv)
    print(f"[bootstrap_data] Kształt: {df.shape} (oczekiwano (1599, 12))")
    assert df.shape == (1599, 12), "Niespodziewany kształt danych!"
    print("[bootstrap_data] OK")
