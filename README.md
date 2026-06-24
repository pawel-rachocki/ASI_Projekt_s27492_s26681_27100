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
