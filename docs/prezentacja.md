---
marp: true
title: Przewidywanie jakości wina — MLOps
author: s27492 · s26681 · 27100
paginate: true
theme: gaia
class: lead
---

# 🍷 Przewidywanie jakości wina

**Klasyfikacja binarna** właściwości fizykochemicznych czerwonego wina

Zespół: **s27492 · s26681 · 27100**

Dataset: *Wine Quality – Red* (UCI Machine Learning Repository)

Data: **czerwiec 2026**

---

## Slajd 2 — Problem

### Klasyfikacja binarna jakości wina

- Ocena sommeliera jest **kosztowna i subiektywna** — chcemy ją zastąpić modelem ML.
- Dane: **Wine Quality Red** (P. Cortez et al., UCI) — 1599 próbek, 11 cech fizykochemicznych.
- Ujęcie problemu:

  > `quality >= 6` → **„dobre"** (klasa 1)  
  > `quality < 6` → **„słabe"** (klasa 0)

- Balans klas: **53,5% dobre / 46,5% słabe** (lekko niezbalansowany).
- Metryki wiodące: **F1-score** i **ROC-AUC** — odporniejsze na niezbalansowanie niż sama accuracy.

---

## Slajd 3 — Architektura systemu

```
 ┌─────────────────────────────────────────────────────────────┐
 │                  Przepływ MLOps                             │
 │                                                             │
 │  CSV (DVC) ──► Kedro Pipelines ──► MLflow Tracking         │
 │                                        │                   │
 │  [ingest → preprocess → train → eval]  ▼                   │
 │                                  Model Registry             │
 │                                        │                   │
 │  Feast Feature Store ──────────────────►                   │
 │                                        ▼                   │
 │                                  Flask API + GUI            │
 │                                        │                   │
 │                                  Monitoring (K-S/Evidently) │
 │                                        │                   │
 │                                  Docker / docker-compose   │
 └─────────────────────────────────────────────────────────────┘
```

Ścieżka MLOps: **Opcja A** — DVC · Feast · MLflow Registry

---

## Slajd 4 — Sekcja 1+2: Setup + EDA Baseline

### Commit 1 — s27492

**Setup repozytorium:**
- Repo GitHub, środowisko **conda Python 3.11** (`environment.yml`)
- Struktura katalogów Kedro (`data/01_raw/` … `data/08_reporting/`)
- Wersjonowanie danych: **DVC** (`dvc add`, `dvc push`)

**EDA i model bazowy (`notebooks/01_baseline_eda.ipynb`):**
- Najsilniejsze korelacje z `quality`: `alcohol` (+), `sulphates` (+), `volatile acidity` (−)
- Usunięto **240 duplikatów**; brak braków danych
- Preprocessing: `StandardScaler`, stratyfikowany podział 80/20

**Wyniki modelu bazowego — Logistic Regression:**

| Metryka | Wartość |
|---|---|
| Accuracy | 0,735 |
| Precision | 0,761 |
| Recall | 0,729 |
| **F1** | **0,745** |
| **ROC-AUC** | **0,812** |

---

## Slajd 5 — Sekcja 3: Pipeline Kedro

### Commit 2 — s26681

4 modularne pipeline'y, spięte w `pipeline_registry.py`:

```
data_ingestion ──► data_preprocessing ──► data_science ──► evaluation
     │                    │                    │                │
load_raw_data    clean_data              train_and_        evaluate_
                 split_and_scale_data   register_best_    model
                                        model
                                        (PyCaret+Optuna)
```

| Pipeline | Kluczowe węzły | Wyjście Data Catalog |
|---|---|---|
| `data_ingestion` | `load_raw_data` | `data/01_raw/` |
| `data_preprocessing` | `clean_data`, `split_and_scale_data` | `data/05_model_input/` |
| `data_science` | `train_and_register_best_model` | `data/06_models/`, MLflow Registry |
| `evaluation` | `evaluate_model` | `data/08_reporting/`, MLflow metrics |

Uruchomienie: `kedro run` (wszystkie) lub `kedro run --pipeline data_science`

---

## Slajd 6 — Sekcja 4: MLflow + AutoML PyCaret

### Śledzenie eksperymentów

- **Tracking URI:** `sqlite:///mlflow.db` | UI: `mlflow ui --port 5001`
- Logowane: parametry, metryki (F1, ROC-AUC, accuracy), artefakty (model, wykresy, CSV)

### AutoML — 7 modeli sklearn (kryterium: **F1**, CV 5-fold)

| Model | F1 | ROC-AUC |
|---|---|---|
| **ExtraTrees** ⭐ | **0,811** | **0,879** |
| LightGBM | 0,806 | 0,853 |
| Random Forest | 0,805 | 0,868 |
| XGBoost | 0,793 | 0,851 |
| Gradient Boosting | 0,788 | 0,839 |
| Decision Tree | 0,762 | 0,734 |
| Logistic Regression | 0,761 | 0,813 |

Punkt odniesienia: F1 = **0,745** / ROC-AUC = **0,812** (Logistic Regression, baseline)

---

## Slajd 7 — Sekcja 4: Strojenie Optuna + porównanie modeli

### Optuna — Bayesian Optimization (TPE)

```python
tune_model(best, optimize="F1",
           search_library="optuna",
           search_algorithm="tpe",
           n_iter=50)
```

| Parametr | Wartość |
|---|---|
| Algorytm | TPE (Tree-structured Parzen Estimator) |
| Liczba prób | 50 |
| Metryka celu | F1 (maksymalizacja) |
| Cross-validation | 5-fold stratyfikowana |

### Porównanie metod strojenia

| Metoda | Próby | Adaptacyjna? | Wynik |
|---|---|---|---|
| Grid Search | ~200–1000 | ✗ | dobry |
| Random Search | 50 | ✗ | dobry |
| **Optuna (TPE)** | **50** | **✓** | **najlepszy** |

---

## Slajd 8 — Sekcja 5: Flask API + GUI

### Commit 3 — 27100

**Aplikacja Flask** (`serving/app.py`) — serwowanie real-time (pojedyncza predykcja):

| Metoda | Endpoint | Opis |
|---|---|---|
| `GET` | `/` | Formularz GUI — 11 pól numerycznych → „dobre/słabe" + p-stwo |
| `POST` | `/predict` | JSON `{"features": {...}}` → predykcja |
| `POST` | `/predict/by_id` | Cechy z Feast po `wine_id` → predykcja |
| `GET` | `/health` | Status serwisu + źródło modelu |
| `GET` | `/monitoring/drift` | Raport driftu (K-S per cecha + Evidently HTML) |

**Ładowanie modelu:** MLflow Registry → fallback lokalny mock (`baseline_logreg.pkl`)

```bash
curl -X POST localhost:5000/predict -H 'Content-Type: application/json' \
  -d '{"features": {"fixed acidity":7.4,"volatile acidity":0.7,...,"alcohol":9.4}}'
```

---

## Slajd 9 — Sekcja 5: Monitoring

### Logowanie predykcji

- Każda predykcja → `serving/logs/predictions.csv`
- Kolumny: `timestamp`, 11 cech wejściowych, `prediction` (0/1), `probability`

### Detekcja driftu danych

```
GET /monitoring/drift
```

1. Wczytuje `predictions.csv` (dane produkcyjne).
2. Porównuje rozkład każdej cechy z rozkładem treningowym.
3. Test **Kolmogorowa-Smirnowa** per cecha (`p < 0,05` → drift).
4. Opcjonalny raport HTML: **Evidently** (`drift_report.html`).
5. Wynik JSON: `{feature: "alcohol", drift: true, p_value: 0.023}`.

**Plik wyjściowy:** `serving/logs/drift_report.json`

---

## Slajd 10 — Sekcja 5: Docker

### Konteneryzacja

```
docker/
├── Dockerfile          # python:3.11-slim, gunicorn, port 5000
└── docker-compose.yml  # serwis wine-serving, woluminy data/ + logs/
```

**`docker-compose.yml` (kluczowe fragmenty):**
```yaml
services:
  wine-serving:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ../data:/app/data
      - ../serving/logs:/app/serving/logs
    environment:
      - MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

**Uruchomienie:**
```bash
# 1. Przygotowanie danych i modelu (raz)
python serving/bootstrap_data.py
python serving/bootstrap_model.py

# 2. Uruchomienie kontenera
docker compose -f docker/docker-compose.yml up --build
```
GUI: **http://localhost:5000**

---

## Slajd 11 — Sekcja 6: MLOps — DVC + Feast + MLflow Registry

### Trzy filary MLOps (Opcja A)

```
┌────────────────────────────────────────────────────────────┐
│  DVC           │  Feast              │  MLflow Registry    │
│  (dane)        │  (cechy)            │  (modele)           │
├────────────────┼─────────────────────┼─────────────────────┤
│ dvc add        │ feast apply         │ mlflow.register_    │
│ dvc push/pull  │ feast materialize   │   model(...)        │
│ *.dvc w Git    │ feature_repo/       │ Staging→Production  │
│ dane w remote  │ wine_features view  │ models:/wine.../    │
│                │ online store SQLite │   Production        │
└────────────────┴─────────────────────┴─────────────────────┘
```

**Przepływ end-to-end:**
1. `git pull && dvc pull` → odtworzenie danych
2. `kedro run` → pełny pipeline (preprocessing + trening + eval)
3. `mlflow ui` → podgląd eksperymentów
4. `feast materialize` → aktualizacja online store
5. `flask run` / `docker compose up` → serwowanie

---

## Slajd 12 — Demo: uruchomienie end-to-end

### Kroki demonstracyjne

```bash
# 0. Setup środowiska
conda activate wine

# 1. Dane
dvc pull                      # lub: python serving/bootstrap_data.py

# 2. Pipeline Kedro
kedro run

# 3. MLflow UI (podgląd eksperymentów)
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

# 4. Feast
cd feature_repo
python generate_features.py && feast apply
feast materialize 2020-01-01T00:00:00 "$(date -u +%Y-%m-%dT%H:%M:%S)"
cd ..

# 5. Flask API
flask --app serving/app run --port 5000
```

**Scenariusz demo (przeglądarka):**
1. `http://localhost:5000` — wypełnij formularz → **Przewiduj**
2. `curl -X POST .../predict` — JSON API
3. `GET /health` — źródło modelu
4. Kilka predykcji → pokaż `predictions.csv`
5. `GET /monitoring/drift` — raport K-S

---

## Slajd 13 — Wnioski + możliwe ulepszenia

### Osiągnięcia projektu

- ✅ Pełny łańcuch MLOps: **dane → cechy → model → API/GUI → monitoring → kontener**
- ✅ MLOps Opcja A: **DVC** (dane) + **Feast** (feature store) + **MLflow Registry** (modele)
- ✅ AutoML **PyCaret** + strojenie **Optuna** (Bayesian/TPE)
- ✅ Model bazowy: **F1 = 0,745 / ROC-AUC = 0,812** (Logistic Regression)
- ✅ Flask API + GUI + Docker + monitoring (K-S + Evidently)
- ✅ Testy jednostkowe (pytest) + pełna dokumentacja

### Możliwe ulepszenia

| Obszar | Propozycja |
|---|---|
| Model | Dodanie inżynierii cech (interakcje, cechy pochodne) |
| Niezbalansowanie | SMOTE lub `class_weight="balanced"` |
| AutoML | Rozszerzenie o AutoGluon / TPOT |
| Monitoring | Alerty automatyczne przy detekcji driftu |
| Infrastruktura | Wdrożenie w chmurze (GCP/AWS) zamiast lokalnego Dockera |
| CI/CD | GitHub Actions — automatyczne `kedro run` przy push |

---

### Dziękujemy za uwagę! 🍷

**Pytania?**

Kod: [github.com/…/ASI_Projekt_s27492_s26681_27100](https://github.com)  
Dokumentacja: `docs/` w repozytorium  
Demo: `http://localhost:5000`
