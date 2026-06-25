"""Shared pytest fixtures for the wine quality project test suite.

Fixtures defined here are available to all test modules automatically.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Feature list (single source of truth for test suite)
# ---------------------------------------------------------------------------
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


@pytest.fixture
def sample_wine_features() -> dict:
    """Single wine sample with realistic physicochemical values."""
    return {
        "fixed acidity": 7.4,
        "volatile acidity": 0.70,
        "citric acid": 0.00,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Synthetic DataFrame with 50 rows (11 features + quality column).

    Values are drawn from uniform distributions that approximate the
    realistic ranges observed in the UCI Wine Quality (red) dataset.
    """
    rng = np.random.default_rng(seed=42)
    n = 50
    data = {
        "fixed acidity": rng.uniform(4.6, 15.9, n),
        "volatile acidity": rng.uniform(0.12, 1.58, n),
        "citric acid": rng.uniform(0.0, 1.0, n),
        "residual sugar": rng.uniform(1.0, 15.5, n),
        "chlorides": rng.uniform(0.012, 0.611, n),
        "free sulfur dioxide": rng.uniform(1.0, 72.0, n),
        "total sulfur dioxide": rng.uniform(6.0, 289.0, n),
        "density": rng.uniform(0.990, 1.004, n),
        "pH": rng.uniform(2.74, 4.01, n),
        "sulphates": rng.uniform(0.33, 2.0, n),
        "alcohol": rng.uniform(8.4, 14.9, n),
        "quality": rng.integers(3, 9, n),
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_model():
    """sklearn-like MagicMock model compatible with LoadedModel.predict interface.

    Returns:
        predict       -> np.array([1])
        predict_proba -> np.array([[0.3, 0.7]])
    """
    model = MagicMock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    return model
