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

```text
data/            Warstwy danych (wersjonowane przez DVC)
  01_raw/        Surowe dane: winequality-red.csv
  02_intermediate/ Dane po wstępnym czyszczeniu
  05_model_input/  Dane podzielone i przeskalowane wejściowe do modeli
  06_models/     Zapisane pliki modeli (*.pkl)
  08_reporting/  Raporty i metryki ewaluacji (json, csv)
notebooks/       01_baseline_eda.ipynb — EDA + model bazowy
src/winequality/ Kod źródłowy / rurociągi Kedro
conf/            Konfiguracja Kedro (katalogi danych, parametry)
feature_repo/    Konfiguracja i definicje Feast Feature Store
scripts/         Skrypty pomocnicze (np. run_experiments.py)
serving/         Kod produkcyjny aplikacji Flask (API, monitoring, szablony GUI)
docker/          Konfiguracja konteneryzacji Docker (Dockerfile, docker-compose)
tests/           Testy jednostkowe (pytest)
environment.yml  Definicja środowiska conda
```

---

## Status sekcji wymagań

| Sekcja | Zakres | Kto | Status |
|---|---|---|---|
| 1. Organizacja zespołu | repo, środowisko conda, struktura | s27492 | Zaimplementowane |
| 2. Baseline (Jupyter) | EDA, preprocessing, model bazowy, ewaluacja (F1 0,745 / ROC-AUC 0,812) | s27492 | Zaimplementowane |
| 3. Pipeline Kedro | ingest → preprocessing → train → eval | s26681 | Zaimplementowane |
| 4. Udoskonalanie modelu | MLflow, AutoML, strojenie | s26681 | Zaimplementowane |
| 5. Pipeline produkcyjny | Flask API/GUI, Docker, monitoring | 27100 | Zaimplementowane |
| 6. MLOps (Opcja A) | DVC, Feast, MLflow Registry | s27492 / s26681 / 27100 | Zaimplementowane |
| 7. Dokumentacja | README, diagram, instrukcja | cały zespół | Zaimplementowane |
| 8. Prezentacja | slajdy + demo | 27100 | Zaimplementowane |

---

## 1. Środowisko i instalacja

Projekt wymaga środowiska **Python 3.11** z odpowiednimi bibliotekami. Wszystkie wymagane zależności (w tym specyficzne wersje dla Kedro, Feast, MLflow, PyCaret i Optuna) są zdefiniowane w pliku `environment.yml`.

```bash
# 1. Utworzenie środowiska conda na podstawie pliku environment.yml
conda env create -f environment.yml

# 2. Aktywacja środowiska
conda activate wine
```

---

## 2. Wersjonowanie danych i bootstrapping

### A. Wersjonowanie danych (DVC)
- Kod źródłowy jest śledzony przez system **Git** i hostowany na GitHubie.
- Duże pliki danych i modeli są śledzone przez **DVC** (pliki `*.dvc` w Git, a rzeczywiste pliki w zdalnym repozytorium/remote DVC).
- Standardowy przepływ odtworzenia danych z DVC:
  ```bash
  git pull        # Pobranie najnowszego kodu i wskaźników .dvc
  dvc pull        # Pobranie rzeczywistych danych z zdalnego repozytorium
  ```

### B. Pobieranie danych (Kaggle Bootstrapping)
Ponieważ oficjalny remote DVC zespołu jest skonfigurowany pod lokalną ścieżkę w systemie Windows i może być niedostępny z innych lokalizacji/środowisk, przygotowano alternatywny skrypt pobierający dane surowe bezpośrednio z Kaggle za pomocą biblioteki `kagglehub`:

```bash
# Pobranie danych surowych bezpośrednio z Kaggle do data/01_raw/winequality-red.csv
python serving/bootstrap_data.py

# Odtworzenie modelu bazowego i statystyk referencyjnych driftu
# Generuje: data/06_models/baseline_logreg.pkl oraz serving/monitoring/reference_stats.parquet
python serving/bootstrap_model.py
```

---

## 3. Kedro Pipelines

Kod produkcyjny przetwarzania danych i budowania modelu bazowego jest zorganizowany w postaci rurociągów Kedro. Dostępne są 3 rurociągi połączone w domyślny pipeline (`__default__`):
1. **data_preprocessing**: Ingest danych surowych (`wine_raw`), czyszczenie, binaryzacja zmiennej celu (`wine_cleaned`) oraz podział na zbiór treningowy/testowy wraz ze skalowaniem (`X_train`, `X_test`, `y_train`, `y_test`).
2. **data_science**: Trenowanie bazowego modelu regresji logistycznej na danych treningowych i zapis do `baseline_logreg.pkl`.
3. **evaluation**: Ewaluacja modelu na zbiorze testowym i eksport metryk do `metrics.json`.

Aby uruchomić rurociągi:

```bash
# Uruchomienie pełnego domyślnego rurociągu (wszystkie 3 etapy w sekwencji)
kedro run

# Uruchomienie wybranego rurociągu osobno
kedro run --pipeline=data_preprocessing
kedro run --pipeline=data_science
kedro run --pipeline=evaluation
```

### Szybkie uruchomienie (bez pełnego środowiska conda)

Pełne `environment.yml` zawiera ciężkie zależności AutoML (PyCaret, Feast, Optuna,
XGBoost, LightGBM), których rozwiązanie potrafi trwać bardzo długo lub kończyć się
konfliktami. Sam rurociąg Kedro ich nie potrzebuje. Jeśli chcesz tylko uruchomić
pipeline, wystarczy lekkie środowisko (zweryfikowane, instalacja w kilka sekund):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install kedro kedro-datasets pandas scikit-learn mlflow
pip install -e .
kedro run
```

> Uwaga: katalog `conf/local/` jest wymagany przez Kedro (domyślny run env to `local`).
> Jest on śledzony przez `.gitkeep`, więc istnieje od razu po sklonowaniu repozytorium.

---

## 4. Eksperymenty AutoML, Optuna i MLflow Model Registry

Do automatycznego doboru modeli, optymalizacji hiperparametrów i rejestracji modeli służy skrypt `scripts/run_experiments.py`. 

### Jak działa proces:
1. **AutoML**: Porównuje modele (LogisticRegression, RandomForest, ExtraTrees, DecisionTree, GradientBoosting, a także XGBoost i LightGBM, jeśli są dostępne) przy użyciu PyCaret lub natywnego fallbacku w scikit-learn.
2. **Strojenie Optuna**: Najlepszy model pod kątem metryki F1 jest optymalizowany za pomocą Optuny (próbkowanie bayesowskie TPE, 30 prób, 5-krotna stratyfikowana walidacja krzyżowa).
3. **Śledzenie MLflow**: Parametry prób, metryki modeli i artefakty są logowane do lokalnej bazy SQLite (`sqlite:///mlflow.db`).
4. **Model Registry**: Najlepszy nastrojony model jest rejestrowany w rejestrze MLflow pod nazwą `wine_quality` i automatycznie promowany do etapu `Production`.
5. **Kopia lokalna**: Model jest również zapisywany na dysku jako `data/06_models/best_model.pkl`.

```bash
# Uruchomienie eksperymentów i automatycznej rejestracji modelu
python scripts/run_experiments.py
```

### Uruchomienie interfejsu MLflow UI
Aby przeglądać zarejestrowane eksperymenty oraz zarządzać wersjami modeli w rejestrze, należy uruchomić serwer MLflow wskazując lokalną bazę SQLite i port `5001`:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```
Interfejs jest dostępny pod adresem: **http://localhost:5001**

---

## 5. Feast Feature Store

Projekt wykorzystuje Feast do zarządzania cechami w czasie rzeczywistym. Definicje cech znajdują się w katalogu `feature_repo/`.

```bash
# 1. Wygenerowanie pliku Parquet z cechami (offline store)
python feature_repo/generate_features.py

# 2. Rejestracja definicji Feast (tworzy registry.db i online_store.db w feature_repo/)
cd feature_repo
feast apply

# 3. Materializacja cech do bazy online (SQLite) za dany okres
feast materialize 2020-01-01T00:00:00 "$(date -u +%Y-%m-%dT%H:%M:%S)"
cd ..
```
*Uwaga: Baza online store Feast jest wykorzystywana przez endpoint `/predict/by_id` do pobierania cech fizykochemicznych wina na podstawie identyfikatora `wine_id` przed wykonaniem predykcji.*

---

## 6. Serwowanie modelu (Flask API + GUI)

Aplikacja serwująca (Flask + Gunicorn) obsługuje automatyczne ładowanie modeli w następującej kolejności (chain of fallbacks):
1. **Model Registry (MLflow)**: Jeśli zmienna środowiskowa `MLFLOW_TRACKING_URI` jest ustawiona, aplikacja próbuje pobrać model oznaczony jako `Production` z rejestru `wine_quality`.
2. **Lokalny model nastrojony**: W przypadku braku dostępu do MLflow, próbuje załadować `data/06_models/best_model.pkl`.
3. **Model bazowy (baseline)**: W ostateczności ładuje mock `data/06_models/baseline_logreg.pkl`.

### A. Uruchomienie lokalne (Local)

Przed uruchomieniem aplikacji serwującej upewnij się, że zależności zostały zainstalowane (np. poprzez aktywne środowisko conda `wine` lub instalację dependencies serwowania):
```bash
pip install -r serving/requirements.txt
```

Skonfiguruj zmienną środowiskową `MLFLOW_TRACKING_URI` wskazującą na bazę SQLite, aby włączyć pobieranie modelu z rejestru:
```bash
# Linux/macOS
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Windows (CMD)
set MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Windows (PowerShell)
$env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
```

Uruchomienie serwera aplikacji:
```bash
# Wersja deweloperska (Flask)
flask --app serving/app run --port 5000

# Wersja produkcyjna (Gunicorn)
gunicorn serving.app:app -b 0.0.0.0:5000
```
Interfejs GUI dostępny pod adresem: **http://localhost:5000**

### B. Uruchomienie w kontenerze Docker

Serwer aplikacji wraz z wbudowanym Feast i monitoringiem można uruchomić za pomocą Docker Compose.

1. Upewnij się, że wygenerowałeś dane i model lokalnie (Bootstrap krok 2), ponieważ katalog `./data` jest montowany jako wolumin.
2. Jeśli chcesz, aby kontener pobierał model bezpośrednio z serwera MLflow uruchomionego na hoście, odkomentuj linię ze zmienną `MLFLOW_TRACKING_URI` w `docker/docker-compose.yml`:
   ```yaml
   environment:
     - MLFLOW_TRACKING_URI=http://host.docker.internal:5001
   ```
3. Uruchomienie kontenera:
   ```bash
   docker compose -f docker/docker-compose.yml up --build
   ```
Interfejs GUI dostępny pod adresem: **http://localhost:5000**

### C. Architektura inferencji

Wybrano tryb **real-time** (pojedyncza predykcja na żążądanie) — pasuje do interaktywnego GUI/API i prostego demo; przetwarzanie wsadowe (batch) nie jest tu potrzebne.

---

## 7. Interfejs API i monitoring driftu

### Dostępne endpointy

| Metoda | Ścieżka | Opis |
|---|---|---|
| `GET` | `/` | Formularz interaktywny GUI (wprowadzanie 11 cech ręcznie) |
| `POST` | `/predict` | Predykcja na podstawie przekazanego JSONa (lista wartości lub słownik `{"features": {}}`) |
| `POST` | `/predict/by_id` | Pobiera cechy z Feature Store (Feast) po `wine_id` i wykonuje predykcję |
| `GET` | `/health` | Zwraca status zdrowia aplikacji oraz źródło aktualnie załadowanego modelu |
| `GET` | `/monitoring/drift` | Raport z analizy dryfu danych (test Kolmogorowa-Smirnowa oraz opcjonalnie HTML z Evidently) |

#### Przykład zapytania do `/predict`
```bash
curl -X POST localhost:5000/predict -H 'Content-Type: application/json' \
  -d '{"features": {"fixed acidity":7.4,"volatile acidity":0.7,"citric acid":0.0,
       "residual sugar":1.9,"chlorides":0.076,"free sulfur dioxide":11,
       "total sulfur dioxide":34,"density":0.9978,"pH":3.51,"sulphates":0.56,"alcohol":9.4}}'
```

#### Przykład zapytania do `/predict/by_id`
```bash
curl -X POST localhost:5000/predict/by_id -H 'Content-Type: application/json' \
  -d '{"wine_id": 10}'
```

### Monitoring i Drift
- Każde zapytanie predykcyjne loguje timestamp, wejściowe cechy oraz wynik do pliku `serving/logs/predictions.csv`.
- Wywołanie `GET /monitoring/drift` porównuje rozkład cech z logu predykcji z rozkładem treningowym (`serving/monitoring/reference_stats.parquet`) za pomocą testu Kolmogorowa-Smirnowa. Jeśli p-value dla którejś cechy spadnie poniżej 0.05, wykrywany jest drift. Raport zapisuje się w `serving/logs/drift_report.json` oraz `drift_report.html` (jeśli Evidently jest zainstalowane).

---

## Uruchamianie testów jednostkowych

Testy jednostkowe sprawdzające poprawność przetwarzania danych, modeli i serwowania uruchamia się za pomocą pytest:
```bash
pytest
```

---

## Prezentacja

Slajdy (Marp): [`docs/prezentacja.md`](docs/prezentacja.md) — render do PDF: `marp docs/prezentacja.md --pdf`.
