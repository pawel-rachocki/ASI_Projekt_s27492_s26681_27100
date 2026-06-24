"""Testy danych i logiki przygotowania targetu dla zbioru Wine Quality (red).

Uruchomienie:
    conda activate wine
    pytest -v
"""
from pathlib import Path

import pandas as pd
import pytest

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "01_raw" / "winequality-red.csv"

EXPECTED_FEATURES = [
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


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def test_data_file_exists():
    """Surowe dane istnieją (po `dvc pull` lub lokalnie)."""
    assert DATA_PATH.exists(), f"Brak pliku danych: {DATA_PATH}"


def test_shape(df):
    """Oczekiwany rozmiar zbioru: 1599 wierszy x 12 kolumn."""
    assert df.shape == (1599, 12)


def test_columns(df):
    """Kolumny zgodne z oczekiwanymi cechami + 'quality'."""
    assert list(df.columns) == EXPECTED_FEATURES + ["quality"]


def test_feature_count(df):
    """Liczba cech objaśniających wynosi 11."""
    features = [c for c in df.columns if c != "quality"]
    assert len(features) == 11


def test_no_missing_values(df):
    """Zbiór nie zawiera braków danych."""
    assert int(df.isnull().sum().sum()) == 0


def test_all_features_numeric(df):
    """Wszystkie cechy są numeryczne."""
    assert all(pd.api.types.is_numeric_dtype(df[c]) for c in EXPECTED_FEATURES)


def test_quality_in_valid_range(df):
    """Ocena 'quality' mieści się w skali 0-10."""
    assert df["quality"].between(0, 10).all()


def test_binarization_produces_two_classes(df):
    """Binaryzacja quality>=6 daje wyłącznie etykiety {0, 1}."""
    target = (df["quality"] >= 6).astype(int)
    assert set(target.unique()) == {0, 1}


def test_binarization_class_balance(df):
    """Balans klas po binaryzacji: 855 'dobrych' (1) i 744 'słabych' (0)."""
    target = (df["quality"] >= 6).astype(int)
    assert int(target.sum()) == 855
    assert int((target == 0).sum()) == 744
