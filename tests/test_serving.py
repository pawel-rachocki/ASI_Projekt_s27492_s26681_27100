"""Tests for the Flask serving API (serving/app.py).

Model loading is patched at module import time so that no actual model file is
required on disk.  The patch target is 'serving.model_loader.load_model',
which is what app.py calls at module level.

Strategy
--------
Because ``app.py`` executes ``MODEL = load_model()`` at import time, we must
ensure the patch is in place *before* the module is imported.  We achieve this
by using a session-scoped autouse fixture that patches the loader, forces a
reload of ``serving.app`` inside the patch context, then tears it down cleanly.
"""
from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Feature list (matches serving/model_loader.py _default_features())
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


def _build_mock_loaded_model() -> MagicMock:
    """Return a MagicMock that mimics serving.model_loader.LoadedModel."""
    loaded = MagicMock()
    loaded.features = FEATURES
    loaded.source = "mock:test"
    # predict() returns (pred_array, proba_array) – same as LoadedModel.predict
    loaded.predict.return_value = (np.array([1]), np.array([0.7]))
    return loaded


@pytest.fixture(scope="module")
def flask_app():
    """Provide a Flask test app with model loading fully mocked.

    We patch ``serving.model_loader.load_model`` and then force-reimport
    ``serving.app`` so that the module-level ``MODEL = load_model()`` call
    picks up our mock.
    """
    mock_loaded = _build_mock_loaded_model()

    # Remove previously cached modules to force a clean import
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("serving"):
            del sys.modules[mod_name]

    with patch("serving.model_loader.load_model", return_value=mock_loaded), \
         patch("serving.app.log_prediction"):
        import serving.app as app_module  # noqa: PLC0415
        app_module.app.config["TESTING"] = True
        yield app_module.app

    # Clean up after the module scope ends
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("serving"):
            del sys.modules[mod_name]


@pytest.fixture
def client(flask_app):
    """Flask test client derived from the mocked app."""
    with flask_app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        """GET /health must return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_key(self, client):
        """GET /health JSON body must contain a 'status' key."""
        response = client.get("/health")
        data = response.get_json()
        assert data is not None
        assert "status" in data

    def test_health_status_value(self, client):
        """GET /health 'status' value must equal 'ok'."""
        data = client.get("/health").get_json()
        assert data["status"] == "ok"

    def test_health_has_model_source(self, client):
        """GET /health must report the model source."""
        data = client.get("/health").get_json()
        assert "model_source" in data

    def test_health_has_n_features(self, client):
        """GET /health must report the number of features."""
        data = client.get("/health").get_json()
        assert "n_features" in data
        assert data["n_features"] == len(FEATURES)


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    def _valid_payload(self) -> dict:
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

    def test_predict_endpoint_valid_returns_200(self, client):
        """POST /predict with all 11 features must return HTTP 200."""
        response = client.post("/predict", json=self._valid_payload())
        assert response.status_code == 200

    def test_predict_endpoint_valid_has_prediction_key(self, client):
        """POST /predict response must contain a 'prediction' key."""
        response = client.post("/predict", json=self._valid_payload())
        data = response.get_json()
        assert data is not None
        assert "prediction" in data

    def test_predict_endpoint_valid_prediction_is_binary(self, client):
        """'prediction' value must be 0 or 1."""
        data = client.post("/predict", json=self._valid_payload()).get_json()
        assert data["prediction"] in (0, 1)

    def test_predict_endpoint_valid_has_probability(self, client):
        """POST /predict response must contain a 'probability' key."""
        data = client.post("/predict", json=self._valid_payload()).get_json()
        assert "probability" in data

    def test_predict_endpoint_missing_features(self, client):
        """POST /predict with empty JSON body must return 4xx or contain 'error' key."""
        response = client.post("/predict", json={})
        is_4xx = 400 <= response.status_code < 500
        data = response.get_json() or {}
        has_error_key = "error" in data
        assert is_4xx or has_error_key, (
            f"Expected 4xx or 'error' key; got status={response.status_code}, body={data}"
        )

    def test_predict_endpoint_list_payload(self, client):
        """POST /predict accepts a bare list of 11 numeric values."""
        values = [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
        response = client.post("/predict", json=values)
        assert response.status_code == 200
        data = response.get_json()
        assert "prediction" in data

    def test_predict_endpoint_wrong_list_length(self, client):
        """POST /predict with a list of wrong length must return 400."""
        response = client.post("/predict", json=[1.0, 2.0])
        assert response.status_code == 400

    def test_predict_endpoint_features_wrapper(self, client):
        """POST /predict accepts payload wrapped in {'features': {...}}."""
        response = client.post("/predict", json={"features": self._valid_payload()})
        assert response.status_code == 200
        data = response.get_json()
        assert "prediction" in data


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

class TestIndexPage:
    def test_index_page_returns_200(self, client):
        """GET / must return HTTP 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_index_page_is_html(self, client):
        """GET / must return an HTML content-type."""
        response = client.get("/")
        assert b"html" in response.content_type.lower().encode()
