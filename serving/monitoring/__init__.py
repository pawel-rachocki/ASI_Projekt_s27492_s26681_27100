"""Monitoring serwowania: logowanie predykcji + detekcja driftu danych."""
from .logger import log_prediction  # noqa: F401
from .drift import compute_drift  # noqa: F401
