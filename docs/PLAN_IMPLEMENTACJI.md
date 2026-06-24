# Plan implementacji — Przewidywanie jakości wina

**Zespół:** s27492, s26681, 27100
**Dataset:** `docs/winequality-red.csv` (Wine Quality – Red, 1599 próbek, 11 cech + `quality`)
**Data planu:** 2026-06-24

---

## 1. Decyzje projektowe (ustalone)

| Decyzja | Wybór | Uzasadnienie |
|---|---|---|
| **Typ problemu ML** | Klasyfikacja binarna: `quality >= 6 → "dobre" (1)`, inaczej `"słabe" (0)` | Czytelne metryki (accuracy / F1 / ROC-AUC), proste i czytelne GUI, zbalansowany podział (~53% / 47%). |
| **Ścieżka MLOps (sekcja 6)** | **Opcja A – Repozytoria**: DVC + Feast + MLflow Model Registry | Pełny łańcuch wersjonowania danych → cech → modeli. |
| **Wdrożenie (sekcja 5)** | Lokalnie + **Docker** (`docker-compose`) | Spełnia wymóg infrastruktury, łatwe demo, zero kosztów chmury. |
| **AutoML** | PyCaret (główne), TPOT jako alternatywa | Szybkie porównanie wielu modeli na danych tabelarycznych. |
| **Strojenie hiperparametrów** | Optuna (Bayesian) + GridSearchCV (porównawczo) | Pokrywa wymóg „Grid/Random/Bayesian”. |
| **Serwowanie modelu** | Flask (API + proste GUI) | Zgodnie z wymaganiem prostego GUI Flask. |

> ⚠️ **Uwaga o wersji Pythona:** część bibliotek (PyCaret, Feast, AutoGluon) bywa niestabilna na Pythonie 3.13.
> **Rekomendacja:** wspólne środowisko **conda z Python 3.11** (plik `environment.yml` w repo).
> Conda > venv, bo: instaluje 3.11 jednym poleceniem (system ma 3.13), dostarcza prekompilowane binarki ML na Windows (LightGBM/XGBoost/Feast) i daje powtarzalne środowisko dla całego zespołu (`conda env create -f environment.yml`). To fundament „ustalonego środowiska pracy” z sekcji 1 wymagań.

---

## 2. Mapowanie wymagań → realizacja

| Sekcja wymagań | Co realizujemy |
|---|---|
| 1. Organizacja zespołu | Repo **GitHub** z historią, środowisko **conda (Python 3.11)**, `README`, struktura katalogów |
| 2. Baseline (Jupyter) | Notebook: EDA → preprocessing → model bazowy → ewaluacja |
| 3. Struktura + pipeline | Refaktoryzacja do modułów + pipeline **Kedro** (ingest → preprocessing → train → eval) |
| 4. Udoskonalanie modelu | **MLflow tracking**, **AutoML (PyCaret)**, inżynieria cech, strojenie (Optuna), porównanie modeli |
| 5. Pipeline produkcyjny | Pipeline inferencji + **Flask API/GUI** + **Docker** + monitoring (logowanie predykcji, drift) |
| 6. MLOps (Opcja A) | **DVC** (dane), **Feast** (feature store), **MLflow Model Registry** (modele) |
| 7. Dokumentacja | README, opis problemu/danych, architektura, diagram (draw.io), instrukcja uruchomienia |
| 8. Prezentacja | Slajdy 10–15 min + demo (Flask GUI / notebook) |

---

## 3. Architektura docelowa

```
winequality (repo)
│
├── data/                      # warstwy danych Kedro (DVC-tracked)
│   ├── 01_raw/                #   winequality-red.csv
│   ├── 02_intermediate/       #   po czyszczeniu
│   ├── 03_primary/            #   z target binarnym + podziałem train/test
│   ├── 05_model_input/        #   features gotowe do treningu
│   ├── 06_models/             #   wytrenowane modele (pkl)
│   └── 08_reporting/          #   metryki, wykresy
│
├── notebooks/
│   └── 01_baseline_eda.ipynb  # SEKCJA 2 (baseline)
│
├── src/winequality/           # kod produkcyjny (refaktoryzacja)
│   ├── pipelines/
│   │   ├── data_ingestion/
│   │   ├── data_preprocessing/
│   │   ├── training/
│   │   └── evaluation/
│   ├── pipeline_registry.py
│   └── settings.py
│
├── feature_repo/              # Feast (feature store)
├── serving/                   # Flask API + GUI + monitoring
│   ├── app.py
│   ├── templates/
│   ├── monitoring/            # logowanie predykcji + drift
│   └── logs/
│
├── conf/                      # konfiguracja Kedro (catalog, parameters)
├── tests/                     # testy (pytest)
├── docker/                    # Dockerfile(y) + docker-compose.yml
├── .dvc/  + data/*.dvc        # DVC
├── mlruns/ lub mlflow.db      # MLflow tracking + registry
├── docs/                      # dokumentacja + diagram architektury
└── README.md
```

### Przepływ danych (high-level)

```
CSV (raw) ──DVC──► Kedro: ingestion ──► preprocessing ──► Feast feature store
                                                  │
                                                  ▼
                          Kedro: training ──► MLflow (tracking + Model Registry)
                                                  │
                                                  ▼
                          Flask API/GUI ◄── model z Registry ──► predykcja
                                                  │
                                                  ▼
                          Monitoring: logowanie predykcji + detekcja driftu
```

---

## 4. Podział pracy — 3 równe pakiety = 3 commity

Każdy z trzech autorów odpowiada za jeden pakiet o porównywalnym nakładzie.
Każdy pakiet to **jeden główny commit** (autor może rozbić go na kilka mniejszych pod tym samym kontem — kluczowy jest równy, samodzielny zakres).
Pakiety są ułożone tak, by **minimalizować konflikty** (różne katalogi) i mieć jasny interfejs między sobą.

### 🟦 Commit 1 — autor **s27492**: „Dane i baseline” (fundament)

**Cel:** uruchomić repo, środowisko i kompletny notebook bazowy + wersjonowanie danych.

**Zadania:**
1. **Setup repozytorium i środowiska** (sekcja 1)
   - Repo na **GitHub** (`git init` → remote → pierwszy push), `README.md` (szkielet), `.gitignore`
   - Środowisko **conda**: `environment.yml` (Python 3.11) + instrukcja `conda env create -f environment.yml`
   - Struktura katalogów (`data/`, `notebooks/`, `src/`, `docs/`, `tests/`)
   - Umieszczenie `winequality-red.csv` w `data/01_raw/`
2. **Notebook bazowy** `notebooks/01_baseline_eda.ipynb` (sekcja 2)
   - EDA: rozkłady cech, korelacje (heatmapa), wartości odstające, balans klas
   - Definicja targetu binarnego (`quality >= 6`)
   - Preprocessing: braki/duplikaty, skalowanie (StandardScaler), podział train/test
   - Model bazowy: Logistic Regression
   - Ewaluacja: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
   - Notebook udostępniony online (GitHub/Colab) — link w README
3. **DVC — wersjonowanie danych** (sekcja 6, Opcja A — część 1)
   - `dvc init` w repo GitHub; **pliki-wskaźniki `*.dvc` + `.dvc/config` commitowane do Git**, surowe dane ignorowane przez Git
   - `dvc add data/01_raw/winequality-red.csv`
   - Remote DVC = osobny magazyn na dane (GitHub **nie** przechowuje plików DVC): **Google Drive** (`dvc remote add`) lub lokalny katalog; `dvc push`
   - Workflow dla zespołu: `git pull` → `dvc pull` (odtworzenie danych). Opis w README
4. **Dokumentacja** (sekcja 7 — część)
   - Opis problemu ML i opis danych (sekcja README + `docs/`)
   - Podsumowanie wniosków z EDA

**Pliki/obszary:** `README.md`, `environment.yml`, `.gitignore`, `notebooks/`, `data/`, `.dvc/`, `*.dvc`, `docs/opis_problemu_i_danych.md`
**Interfejs dla zespołu:** uzgodniony target binarny i lista cech; surowe dane w `data/01_raw/`.

---

### 🟩 Commit 2 — autor **s26681**: „Pipeline Kedro + eksperymenty + rejestr modeli”

**Cel:** zrefaktoryzować baseline do pipeline'u Kedro, dodać śledzenie eksperymentów, AutoML, strojenie i rejestr modeli.

**Zadania:**
1. **Projekt i pipeline Kedro** (sekcja 3)
   - `kedro new`, konfiguracja `conf/base/catalog.yml` i `parameters.yml`
   - Pipeline'y: `data_ingestion`, `data_preprocessing` (refaktoryzacja kodu z notebooka), `training`, `evaluation`
   - Spięcie w `pipeline_registry.py`; uruchomienie `kedro run`
2. **Śledzenie eksperymentów — MLflow** (sekcja 4)
   - Integracja MLflow z Kedro (`kedro-mlflow` lub ręcznie): logowanie parametrów, metryk, artefaktów
3. **AutoML + inżynieria cech + strojenie** (sekcja 4)
   - AutoML: PyCaret `compare_models()` (porównanie ≥ 4 modeli: LogReg, RandomForest, XGBoost, LightGBM)
   - Inżynieria cech: selekcja (feature importance) + transformacje (np. interakcje, log)
   - Strojenie hiperparametrów: Optuna (Bayesian) + GridSearchCV porównawczo
   - Tabela porównawcza modeli (metryki) w `data/08_reporting/`
4. **MLflow Model Registry** (sekcja 6, Opcja A — część 3)
   - Rejestracja najlepszego modelu, etapy (`Staging`/`Production`), wersjonowanie
5. **Dokumentacja** (sekcja 7 — część)
   - Diagram architektury (draw.io) → `docs/architektura.png`
   - Opis pipeline'u Kedro i eksperymentów

**Pliki/obszary:** `src/winequality/pipelines/`, `conf/`, `pipeline_registry.py`, `data/05_model_input`–`08_reporting`, `mlruns/`, `docs/architektura.*`
**Interfejs dla zespołu:** najlepszy model w MLflow Model Registry (nazwa + stage `Production`) — punkt wejścia dla serwowania.

---

### 🟥 Commit 3 — autor **27100**: „Feature store, serwowanie, monitoring i wdrożenie”

**Cel:** udostępnić model przez API/GUI, dodać feature store, monitoring i konteneryzację.

**Zadania:**
1. **Feast — feature store** (sekcja 6, Opcja A — część 2)
   - `feature_repo/`: definicje encji i feature view dla cech wina
   - `feast apply`, materializacja; pobieranie cech do inferencji
2. **Flask API + proste GUI** (sekcja 5)
   - `serving/app.py`: endpoint `/predict` (JSON) + endpoint `/health`
   - GUI: formularz HTML (11 cech) → wynik „dobre/słabe” + prawdopodobieństwo
   - Ładowanie modelu z **MLflow Model Registry** (stage `Production`)
   - Pipeline inferencji (real-time, pojedyncza predykcja)
3. **Monitoring** (sekcja 5)
   - Logowanie każdej predykcji (wejście + wynik + timestamp) do pliku/CSV
   - Detekcja driftu danych (np. Evidently lub test K-S względem rozkładu treningowego)
4. **Konteneryzacja / wdrożenie** (sekcja 5)
   - `docker/Dockerfile` + `docker-compose.yml` (serwis Flask + ewentualnie MLflow)
   - Instrukcja `docker compose up`
5. **Dokumentacja + prezentacja** (sekcje 7 i 8)
   - Sekcja „Uruchomienie” w README (lokalnie + Docker)
   - Slajdy prezentacji (10–15 min) + scenariusz demo

**Pliki/obszary:** `feature_repo/`, `serving/`, `docker/`, `docs/prezentacja.*`, sekcja README „Uruchomienie”
**Interfejs dla zespołu:** działające demo (GUI + API) korzystające z modelu z Registry.

---

## 5. Bilans nakładu pracy (równość udziału)

| Autor | Pakiet | Główne sekcje wymagań | Szac. udział |
|---|---|---|---|
| **s27492** | Dane i baseline | 1, 2, 6(DVC), 7(część) | ~33% |
| **s26681** | Kedro + eksperymenty + rejestr | 3, 4, 6(Registry), 7(diagram) | ~33% |
| **27100** | Feature store + serwowanie + wdrożenie | 5, 6(Feast), 7(uruchomienie), 8 | ~33% |

Każdy autor: realizuje pełną „pionową” część systemu, dotyka dokumentacji i ma samodzielny, weryfikowalny commit.

---

## 6. Kolejność i zależności

```
s27492 (dane, target, baseline, DVC)
        │  surowe dane + ustalony target
        ▼
s26681 (Kedro pipeline, MLflow, AutoML, Registry)
        │  najlepszy model w Model Registry
        ▼
27100  (Feast, Flask API/GUI, monitoring, Docker, prezentacja)
```

**Praca równoległa (by nie blokować się nawzajem):**
- s26681 i 27100 mogą startować wcześnie na **mocku** (np. tymczasowy `model.pkl` z baseline), a podmienić na finalny model z Registry przy integracji.
- Wspólne ustalenia na starcie (na pierwszym spotkaniu):
  1. Definicja targetu i lista cech (kontrakt danych).
  2. Nazwy: model w Registry, ścieżki w `catalog.yml`, port Flask.
  3. Środowisko conda: Python 3.11 + `environment.yml`.

---

## 7. Kamienie milowe

1. **M1 – Fundament:** repo + środowisko + notebook baseline + DVC (commit s27492).
2. **M2 – Pipeline:** Kedro + MLflow + AutoML + strojenie + Registry (commit s26681).
3. **M3 – Produkcja:** Feast + Flask + monitoring + Docker + prezentacja (commit 27100).
4. **M4 – Integracja i prezentacja:** wspólne testy end-to-end, finalne README, próba demo.

---

## 8. Stos technologiczny (skrót)

`conda (Python 3.11)` · `pandas` · `scikit-learn` · `Jupyter` · `Kedro` · `MLflow` · `PyCaret`/`TPOT` · `Optuna` · `DVC` · `Feast` · `Flask` · `Evidently` (drift) · `Docker` / `docker-compose` · `pytest`

---

## 9. Ryzyka i mitygacje

| Ryzyko | Mitygacja |
|---|---|
| Niekompatybilność bibliotek z Python 3.13 | Wspólne środowisko conda 3.11; zamrożenie wersji w `environment.yml`. |
| Feast trudny w konfiguracji lokalnej | Użyć lokalnego `file`/`sqlite` provider; minimalny feature view. |
| Blokowanie się autorów (zależności sekwencyjne) | Praca na mocku modelu, integracja na końcu (M4). |
| Słaba jakość modelu (niezbalansowanie) | F1/ROC-AUC zamiast samej accuracy; ewentualnie `class_weight`/SMOTE. |
| Konflikty w Git | Rozdzielne katalogi per autor (patrz „Pliki/obszary”). |
