"""Tests for serving/monitoring: prediction logging and K-S drift detection.

All file-system operations use the ``tmp_path`` pytest fixture so that the
production log directory (serving/logs/) is never touched during tests.

Architecture notes
------------------
* ``logger.py`` hard-codes ``LOG_DIR`` and ``LOG_PATH`` via module-level Path
  constants computed from ``__file__``.  We patch these constants so the logger
  writes to a temporary directory instead.
* ``drift.py`` similarly uses module-level constants for ``REFERENCE_PATH``,
  ``PREDICTIONS_PATH``, and ``DRIFT_JSON``.  We patch these to point at
  tmp_path artefacts.
"""
from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared feature list
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

_SAMPLE_FEATURES: dict = {
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reference_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic reference distribution (training-like data)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {f: rng.uniform(4.0, 12.0, n) for f in FEATURES}
    )


def _make_current_df_no_drift(reference: pd.DataFrame, n: int = 50, seed: int = 1) -> pd.DataFrame:
    """Current distribution sampled from the same distribution as reference — no drift."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {f: rng.uniform(4.0, 12.0, n) for f in FEATURES}
    )


def _make_current_df_with_drift(n: int = 50, seed: int = 2) -> pd.DataFrame:
    """Current distribution shifted far from reference — clear drift."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {f: rng.uniform(50.0, 100.0, n) for f in FEATURES}
    )


def _write_predictions_csv(path: Path, df: pd.DataFrame) -> None:
    """Write a DataFrame to a predictions-compatible CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["timestamp", *FEATURES, "prediction", "label", "probability"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for _, row in df.iterrows():
            record = {f: row[f] for f in FEATURES}
            record["timestamp"] = "2024-01-01T00:00:00+00:00"
            record["prediction"] = 1
            record["label"] = "dobre"
            record["probability"] = 0.75
            writer.writerow(record)


# ===========================================================================
# Prediction Logging Tests
# ===========================================================================

class TestPredictionLogging:
    """Tests for serving.monitoring.logger.log_prediction."""

    def test_prediction_logging_creates_csv(self, tmp_path):
        """Calling log_prediction should create predictions.csv if it does not exist."""
        log_dir = tmp_path / "logs"
        log_path = log_dir / "predictions.csv"

        import serving.monitoring.logger as logger_module  # noqa: PLC0415

        with (
            patch.object(logger_module, "LOG_DIR", log_dir),
            patch.object(logger_module, "LOG_PATH", log_path),
        ):
            logger_module.log_prediction(_SAMPLE_FEATURES, prediction=1, label="dobre", probability=0.7)

        assert log_path.exists(), "predictions.csv was not created"

    def test_prediction_logging_writes_header(self, tmp_path):
        """The CSV file must have a header row with expected column names."""
        log_dir = tmp_path / "logs"
        log_path = log_dir / "predictions.csv"

        import serving.monitoring.logger as logger_module  # noqa: PLC0415

        with (
            patch.object(logger_module, "LOG_DIR", log_dir),
            patch.object(logger_module, "LOG_PATH", log_path),
        ):
            logger_module.log_prediction(_SAMPLE_FEATURES, prediction=1, label="dobre", probability=0.7)

        with log_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []

        assert "timestamp" in fieldnames
        assert "prediction" in fieldnames
        assert "probability" in fieldnames
        for feat in FEATURES:
            assert feat in fieldnames, f"Feature '{feat}' missing from CSV header"

    def test_prediction_logging_appends_rows(self, tmp_path):
        """Calling log_prediction twice should append two data rows."""
        log_dir = tmp_path / "logs"
        log_path = log_dir / "predictions.csv"

        import serving.monitoring.logger as logger_module  # noqa: PLC0415

        with (
            patch.object(logger_module, "LOG_DIR", log_dir),
            patch.object(logger_module, "LOG_PATH", log_path),
        ):
            logger_module.log_prediction(_SAMPLE_FEATURES, prediction=1, label="dobre", probability=0.7)
            logger_module.log_prediction(_SAMPLE_FEATURES, prediction=0, label="słabe", probability=0.3)

        with log_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))

        assert len(rows) == 2, f"Expected 2 data rows, got {len(rows)}"

    def test_prediction_logging_correct_values(self, tmp_path):
        """Logged row must contain the supplied feature values and prediction."""
        log_dir = tmp_path / "logs"
        log_path = log_dir / "predictions.csv"

        import serving.monitoring.logger as logger_module  # noqa: PLC0415

        with (
            patch.object(logger_module, "LOG_DIR", log_dir),
            patch.object(logger_module, "LOG_PATH", log_path),
        ):
            logger_module.log_prediction(_SAMPLE_FEATURES, prediction=1, label="dobre", probability=0.85)

        with log_path.open(encoding="utf-8") as fh:
            row = list(csv.DictReader(fh))[0]

        assert int(row["prediction"]) == 1
        assert row["label"] == "dobre"
        assert float(row["probability"]) == pytest.approx(0.85, abs=1e-4)
        assert float(row["alcohol"]) == pytest.approx(_SAMPLE_FEATURES["alcohol"])


# ===========================================================================
# Drift Detection Tests
# ===========================================================================

class TestDriftDetection:
    """Tests for serving.monitoring.drift.compute_drift (K-S core logic).

    We test the K-S logic directly using scipy.stats.ks_2samp rather than
    calling compute_drift(), which would require specific file paths and a
    running Evidently installation.  This approach is more robust and faster.
    """

    def test_drift_detection_no_drift(self):
        """Two samples drawn from the same distribution should not show drift (p > 0.05)."""
        from scipy.stats import ks_2samp  # noqa: PLC0415

        rng = np.random.default_rng(0)
        reference = rng.uniform(4.0, 12.0, 500)
        current = rng.uniform(4.0, 12.0, 200)

        _, p_value = ks_2samp(reference, current)
        # With a 0.05 threshold, identical distributions should not trigger drift
        assert p_value > 0.05, f"False drift detected: p={p_value:.4f}"

    def test_drift_detection_with_drift(self):
        """Samples from completely different distributions must show drift (p < 0.05)."""
        from scipy.stats import ks_2samp  # noqa: PLC0415

        rng = np.random.default_rng(0)
        reference = rng.uniform(0.0, 1.0, 500)
        current = rng.uniform(10.0, 20.0, 200)   # disjoint range — maximum drift

        _, p_value = ks_2samp(reference, current)
        assert p_value < 0.05, f"Drift not detected: p={p_value:.4f}"

    def test_compute_drift_no_reference(self, tmp_path):
        """compute_drift must return status='no_reference' when reference file is absent."""
        import serving.monitoring.drift as drift_module  # noqa: PLC0415

        fake_ref = tmp_path / "reference_stats.parquet"
        fake_pred = tmp_path / "predictions.csv"
        fake_json = tmp_path / "drift_report.json"

        with (
            patch.object(drift_module, "REFERENCE_PATH", fake_ref),
            patch.object(drift_module, "PREDICTIONS_PATH", fake_pred),
            patch.object(drift_module, "DRIFT_JSON", fake_json),
        ):
            result = drift_module.compute_drift()

        assert result["status"] == "no_reference"

    def test_compute_drift_no_predictions(self, tmp_path):
        """compute_drift must return status='no_predictions' when CSV is absent."""
        import serving.monitoring.drift as drift_module  # noqa: PLC0415

        # Create a minimal reference parquet
        ref_df = _make_reference_df()
        fake_ref = tmp_path / "reference_stats.parquet"
        ref_df.to_parquet(fake_ref, index=False)

        fake_pred = tmp_path / "predictions.csv"    # does not exist
        fake_json = tmp_path / "drift_report.json"

        with (
            patch.object(drift_module, "REFERENCE_PATH", fake_ref),
            patch.object(drift_module, "PREDICTIONS_PATH", fake_pred),
            patch.object(drift_module, "DRIFT_JSON", fake_json),
        ):
            result = drift_module.compute_drift()

        assert result["status"] == "no_predictions"

    def test_compute_drift_ok_no_drift(self, tmp_path):
        """compute_drift on matching distributions must return status='ok' with 0 drifted features."""
        import serving.monitoring.drift as drift_module  # noqa: PLC0415

        ref_df = _make_reference_df(n=300, seed=0)
        cur_df = _make_current_df_no_drift(ref_df, n=100, seed=1)

        fake_ref = tmp_path / "reference_stats.parquet"
        fake_pred = tmp_path / "predictions.csv"
        fake_json = tmp_path / "drift_report.json"

        ref_df.to_parquet(fake_ref, index=False)
        _write_predictions_csv(fake_pred, cur_df)

        with (
            patch.object(drift_module, "REFERENCE_PATH", fake_ref),
            patch.object(drift_module, "PREDICTIONS_PATH", fake_pred),
            patch.object(drift_module, "DRIFT_JSON", fake_json),
            # Disable Evidently HTML report — not needed in unit test
            patch.object(drift_module, "_evidently_report", return_value=False),
        ):
            result = drift_module.compute_drift(with_html=False)

        assert result["status"] == "ok"
        assert result["n_features_drifted"] == 0, (
            f"Unexpected drifted features: {result.get('drifted_features')}"
        )

    def test_compute_drift_ok_with_drift(self, tmp_path):
        """compute_drift on shifted distributions must detect drift in all features."""
        import serving.monitoring.drift as drift_module  # noqa: PLC0415

        ref_df = _make_reference_df(n=300, seed=0)
        cur_df = _make_current_df_with_drift(n=100, seed=5)

        fake_ref = tmp_path / "reference_stats.parquet"
        fake_pred = tmp_path / "predictions.csv"
        fake_json = tmp_path / "drift_report.json"

        ref_df.to_parquet(fake_ref, index=False)
        _write_predictions_csv(fake_pred, cur_df)

        with (
            patch.object(drift_module, "REFERENCE_PATH", fake_ref),
            patch.object(drift_module, "PREDICTIONS_PATH", fake_pred),
            patch.object(drift_module, "DRIFT_JSON", fake_json),
            patch.object(drift_module, "_evidently_report", return_value=False),
        ):
            result = drift_module.compute_drift(with_html=False)

        assert result["status"] == "ok"
        assert result["n_features_drifted"] > 0, "No drift was detected despite shifted data"
        assert result["dataset_drift"] is True

    def test_compute_drift_writes_json(self, tmp_path):
        """compute_drift must persist the result as drift_report.json."""
        import json as _json  # noqa: PLC0415

        import serving.monitoring.drift as drift_module  # noqa: PLC0415

        ref_df = _make_reference_df(n=100, seed=0)
        cur_df = _make_current_df_no_drift(ref_df, n=50, seed=99)

        fake_ref = tmp_path / "reference_stats.parquet"
        fake_pred = tmp_path / "predictions.csv"
        fake_json = tmp_path / "drift_report.json"

        ref_df.to_parquet(fake_ref, index=False)
        _write_predictions_csv(fake_pred, cur_df)

        with (
            patch.object(drift_module, "REFERENCE_PATH", fake_ref),
            patch.object(drift_module, "PREDICTIONS_PATH", fake_pred),
            patch.object(drift_module, "DRIFT_JSON", fake_json),
            patch.object(drift_module, "_evidently_report", return_value=False),
        ):
            drift_module.compute_drift(with_html=False)

        assert fake_json.exists(), "drift_report.json was not created"
        content = _json.loads(fake_json.read_text(encoding="utf-8"))
        assert "status" in content
