"""Tests for preprocessing logic: binarisation, NaN handling, splits, scaling.

These tests operate on synthetic data only — no dependency on the raw CSV file.
They replicate (and extend) the transformations performed in the training pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Helpers
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


def _make_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Generate a small synthetic wine DataFrame."""
    rng = np.random.default_rng(seed)
    data = {f: rng.uniform(0.0, 10.0, n) for f in FEATURES}
    data["quality"] = rng.integers(3, 9, n)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Binary target creation
# ---------------------------------------------------------------------------

class TestBinaryTargetCreation:
    """quality >= 6 => 1 (good), quality < 6 => 0 (poor)."""

    def test_quality_6_maps_to_1(self):
        """Boundary value: quality exactly 6 must yield label 1."""
        df = pd.DataFrame({"quality": [6]})
        target = (df["quality"] >= 6).astype(int)
        assert target.iloc[0] == 1

    def test_quality_5_maps_to_0(self):
        """Value below threshold (5) must yield label 0."""
        df = pd.DataFrame({"quality": [5]})
        target = (df["quality"] >= 6).astype(int)
        assert target.iloc[0] == 0

    def test_binary_target_creation(self, sample_dataframe):
        """All labels produced are exactly {0, 1}."""
        target = (sample_dataframe["quality"] >= 6).astype(int)
        assert set(target.unique()).issubset({0, 1})

    def test_quality_above_threshold_all_ones(self):
        """Rows with quality 6-9 must all map to 1."""
        df = pd.DataFrame({"quality": [6, 7, 8, 9]})
        target = (df["quality"] >= 6).astype(int)
        assert target.tolist() == [1, 1, 1, 1]

    def test_quality_below_threshold_all_zeros(self):
        """Rows with quality 3-5 must all map to 0."""
        df = pd.DataFrame({"quality": [3, 4, 5]})
        target = (df["quality"] >= 6).astype(int)
        assert target.tolist() == [0, 0, 0]


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

class TestNaNHandling:
    """Verifies that fillna(mean) removes all missing values."""

    def test_no_nan_after_preprocessing(self):
        """Single injected NaN should be filled; resulting DataFrame has no NaN."""
        df = _make_df(n=20)
        # Inject NaN in two cells
        df.loc[0, "alcohol"] = float("nan")
        df.loc[5, "pH"] = float("nan")

        df_filled = df.fillna(df.mean(numeric_only=True))
        assert df_filled.isnull().sum().sum() == 0

    def test_fillna_uses_column_mean(self):
        """fillna(mean) replaces NaN with the column's mean, not a constant."""
        df = pd.DataFrame({"alcohol": [8.0, 10.0, 12.0, float("nan")]})
        expected_mean = 10.0  # mean of [8, 10, 12]
        df_filled = df.fillna(df.mean(numeric_only=True))
        assert df_filled.loc[3, "alcohol"] == pytest.approx(expected_mean)

    def test_no_nan_in_clean_dataframe(self, sample_dataframe):
        """Synthetic fixture contains no NaN — fillna should be a no-op."""
        filled = sample_dataframe.fillna(sample_dataframe.mean(numeric_only=True))
        assert filled.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

class TestTrainTestSplitProportions:
    """Verifies 80/20 split produces correct row counts."""

    def test_train_test_split_proportions(self, sample_dataframe):
        """80/20 split: train should have ~80 % rows, test ~20 %."""
        X = sample_dataframe[FEATURES]
        y = (sample_dataframe["quality"] >= 6).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        total = len(sample_dataframe)
        assert len(X_train) == pytest.approx(total * 0.8, abs=1)
        assert len(X_test) == pytest.approx(total * 0.2, abs=1)
        assert len(X_train) + len(X_test) == total

    def test_split_no_overlap(self, sample_dataframe):
        """Train and test index sets must be disjoint."""
        X = sample_dataframe[FEATURES]
        y = (sample_dataframe["quality"] >= 6).astype(int)

        X_train, X_test, _, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        assert set(X_train.index).isdisjoint(set(X_test.index))

    def test_split_reproducibility(self, sample_dataframe):
        """Same random_state must yield identical splits."""
        X = sample_dataframe[FEATURES]
        y = (sample_dataframe["quality"] >= 6).astype(int)

        X_train_a, X_test_a, _, _ = train_test_split(X, y, test_size=0.2, random_state=7)
        X_train_b, X_test_b, _, _ = train_test_split(X, y, test_size=0.2, random_state=7)

        pd.testing.assert_frame_equal(X_train_a.reset_index(drop=True),
                                       X_train_b.reset_index(drop=True))


# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------

class TestStandardScaler:
    """After fitting StandardScaler the transformed data should have mean ≈ 0, std ≈ 1."""

    def test_standard_scaler_mean_near_zero(self, sample_dataframe):
        """Column means of scaled data must be approximately 0."""
        X = sample_dataframe[FEATURES].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        col_means = np.abs(X_scaled.mean(axis=0))
        assert np.all(col_means < 1e-10), f"Non-zero means detected: {col_means}"

    def test_standard_scaler_std_near_one(self, sample_dataframe):
        """Column standard deviations of scaled data must be approximately 1."""
        X = sample_dataframe[FEATURES].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        col_stds = X_scaled.std(axis=0)
        assert np.allclose(col_stds, 1.0, atol=1e-10), f"Unexpected stds: {col_stds}"

    def test_scaler_stores_mean_and_scale(self, sample_dataframe):
        """StandardScaler must expose mean_ and scale_ after fitting."""
        X = sample_dataframe[FEATURES].values
        scaler = StandardScaler()
        scaler.fit(X)
        assert hasattr(scaler, "mean_") and len(scaler.mean_) == len(FEATURES)
        assert hasattr(scaler, "scale_") and len(scaler.scale_) == len(FEATURES)

    def test_transform_vs_fit_transform_equivalence(self, sample_dataframe):
        """fit_transform and fit+transform must produce identical results."""
        X = sample_dataframe[FEATURES].values
        scaler_a = StandardScaler()
        X_ft = scaler_a.fit_transform(X)

        scaler_b = StandardScaler()
        scaler_b.fit(X)
        X_t = scaler_b.transform(X)

        np.testing.assert_array_almost_equal(X_ft, X_t)
