#!/usr/bin/env bash
set -e

MODEL=/app/data/06_models/baseline_logreg.pkl

# Jeśli brak modelu (mock), spróbuj go odtworzyć. Wymaga danych w data/01_raw/.
# Najpewniej: zamontuj host ./data jako wolumin (patrz docker-compose.yml) — wtedy
# CSV i model wygenerowane lokalnie są już dostępne i ten krok zostaje pominięty.
if [ ! -f "$MODEL" ]; then
  echo "[entrypoint] Brak modelu — próba bootstrapu..."
  python /app/serving/bootstrap_data.py || echo "[entrypoint] bootstrap_data nieudany (brak danych/Kaggle creds)."
  python /app/serving/bootstrap_model.py || { echo "[entrypoint] Nie udało się zbudować modelu. Zamontuj ./data jako wolumin."; exit 1; }
fi

# (opcjonalnie) odśwież feature store, jeśli dane są dostępne
if [ -f /app/data/01_raw/winequality-red.csv ]; then
  ( cd /app/feature_repo && python generate_features.py && feast apply && \
    feast materialize 2020-01-01T00:00:00 "$(date -u +%Y-%m-%dT%H:%M:%S)" ) || echo "[entrypoint] Feast pominięty."
fi

echo "[entrypoint] Start gunicorn na :5000"
exec gunicorn serving.app:app -b 0.0.0.0:5000 --workers 2 --timeout 120
