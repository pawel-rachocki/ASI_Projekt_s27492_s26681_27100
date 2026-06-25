# Przewidywanie jakości wina

Projekt zaliczeniowy — przewidywanie jakości czerwonego wina (zbiór *Wine Quality – Red*).
Problem ujęty jako **klasyfikacja binarna**: `quality >= 6 → "dobre" (1)`, inaczej `"słabe" (0)`.

**Zespół:** s27492 · s26681 · 27100

> Pełny plan i podział prac: [`docs/PLAN_IMPLEMENTACJI.md`](docs/PLAN_IMPLEMENTACJI.md)
> Opis problemu i danych: [`docs/opis_problemu_i_danych.md`](docs/opis_problemu_i_danych.md)

---

## Stos technologiczny

`conda (Python 3.11)` · `pandas` · `scikit-learn` · `Jupyter` · `Kedro` · `MLflow` · `PyCaret` · `Optuna` · `DVC` · `Feast` · `Flask` · `Docker`

## Struktura repozytorium

```
data/            warstwy danych (wersjonowane przez DVC)
  01_raw/        surowe dane: winequality-red.csv
notebooks/       01_baseline_eda.ipynb — EDA + model bazowy (sekcja 2 wymagań)
src/winequality/ kod produkcyjny / pipeline Kedro
docs/            dokumentacja, plan, diagram architektury
tests/           testy (pytest)
environment.yml  definicja środowiska conda
```

---

## Uruchomienie (środowisko)

> Wymaga zainstalowanej **Minicondy/Anacondy**. Instalacja Minicondy (Windows):
> `winget install -e --id Anaconda.Miniconda3`

```bash
# 1. Utworzenie i aktywacja środowiska
conda env create -f environment.yml
conda activate wine

# 2. Pobranie danych wersjonowanych przez DVC
dvc pull          # po skonfigurowaniu remote przez zespół

# 3. Uruchomienie notebooka bazowego
jupyter lab notebooks/01_baseline_eda.ipynb
```

## Wersjonowanie danych (DVC)

- Kod żyje na **GitHub**; surowe dane są śledzone przez **DVC** (pliki `*.dvc` w Git, dane w remote DVC).
- Workflow zespołu:
  ```bash
  git pull        # najnowszy kod + wskaźniki .dvc
  dvc pull        # odtworzenie danych z remote
  ```

---

## Status sekcji wymagań

| Sekcja | Zakres | Kto |
|---|---|---|
| 1. Organizacja zespołu | repo, środowisko conda, struktura | s27492 |
| 2. Baseline (Jupyter) | EDA, preprocessing, model bazowy, ewaluacja (F1 0,745 / ROC-AUC 0,812) | s27492 |
| 3. Pipeline Kedro | ingest → preprocessing → train → eval | s26681 |
| 4. Udoskonalanie modelu | MLflow, AutoML, strojenie | s26681 |
| 5. Pipeline produkcyjny | Flask API/GUI, Docker, monitoring | 27100 |
| 6. MLOps (Opcja A) | DVC, Feast, MLflow Registry | s27492 / s26681 / 27100 |
| 7. Dokumentacja | README, diagram, instrukcja | cały zespół |
| 8. Prezentacja | slajdy + demo | 27100 |

---

## Uruchomienie — serwowanie modelu (sekcja 5, Commit 3 / s27100)

Pakiet produkcyjny: **Flask API + GUI**, **feature store (Feast)**, **monitoring** (logowanie
predykcji + detekcja driftu) i **konteneryzacja (Docker)**.

> **Uwaga:** MLflow Model Registry powstaje w Commit 2 (s26681) i jeszcze nie istnieje.
> Serwowanie ładuje więc **mock = model bazowy** (`data/06_models/baseline_logreg.pkl`).
> Kod ma już ścieżkę do Registry (`serving/model_loader.py`): gdy ustawisz
> `MLFLOW_TRACKING_URI`, model zostanie pobrany z `models:/wine_quality/Production`,
> a w razie niepowodzenia nastąpi automatyczny fallback do mocka.

### A. Lokalnie

```bash
# 1. środowisko (Python 3.11) i zależności serwowania
python -m venv .venv && source .venv/bin/activate        # lub: conda activate wine
pip install -r serving/requirements.txt

# 2. dane + mock modelu (DVC remote zespołu jest niedostępny poza Windows —
#    dane pobieramy z Kaggle przez kagglehub)
python serving/bootstrap_data.py     # -> data/01_raw/winequality-red.csv
python serving/bootstrap_model.py    # -> data/06_models/baseline_logreg.pkl (F1 ~0,745)

# 3. feature store (Feast)
cd feature_repo && python generate_features.py && feast apply && \
  feast materialize 2020-01-01T00:00:00 "$(date -u +%Y-%m-%dT%H:%M:%S)" && cd ..

# 4. uruchomienie API + GUI
gunicorn serving.app:app -b 0.0.0.0:5000     # lub: flask --app serving/app run --port 5000
```

GUI: **http://localhost:5000**

### B. Docker

```bash
# najpierw raz lokalnie kroki 1–2 powyżej (CSV + model w ./data, montowane jako wolumin)
docker compose -f docker/docker-compose.yml up --build
```

GUI: **http://localhost:5000**

### Endpointy API

| Metoda | Ścieżka | Opis |
|---|---|---|
| `GET` | `/` | formularz GUI (11 cech) → „dobre/słabe” + prawdopodobieństwo |
| `POST` | `/predict` | JSON `{"features": {…}}` lub lista 11 wartości → predykcja |
| `POST` | `/predict/by_id` | pobiera cechy z **Feast** po `wine_id`, potem predykcja |
| `GET` | `/health` | status + źródło modelu |
| `GET` | `/monitoring/drift` | raport driftu (K-S per cecha + opcjonalnie Evidently HTML) |

```bash
curl -X POST localhost:5000/predict -H 'Content-Type: application/json' \
  -d '{"features": {"fixed acidity":7.4,"volatile acidity":0.7,"citric acid":0.0,
       "residual sugar":1.9,"chlorides":0.076,"free sulfur dioxide":11,
       "total sulfur dioxide":34,"density":0.9978,"pH":3.51,"sulphates":0.56,"alcohol":9.4}}'
```

### Monitoring i drift

- Każda predykcja jest logowana do `serving/logs/predictions.csv` (timestamp + 11 cech + wynik).
- `GET /monitoring/drift` porównuje rozkład zalogowanych predykcji z rozkładem treningowym
  (test Kolmogorowa-Smirnowa, `p < 0,05` → drift) i zapisuje `serving/logs/drift_report.json`
  (oraz `drift_report.html`, jeśli Evidently jest dostępne).

### Architektura inferencji

Wybrano tryb **real-time** (pojedyncza predykcja na żądanie) — pasuje do interaktywnego GUI/API
i prostego demo; przetwarzanie wsadowe (batch) nie jest tu potrzebne.

### Prezentacja

Slajdy (Marp): [`docs/prezentacja.md`](docs/prezentacja.md) — render do PDF: `marp docs/prezentacja.md --pdf`.
