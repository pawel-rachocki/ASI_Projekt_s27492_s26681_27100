---
marp: true
title: Przewidywanie jakości wina
author: s27492 · s26681 · 27100
paginate: true
theme: gaia
class: lead
---

# 🍷 Przewidywanie jakości wina

**Klasyfikacja binarna** właściwości fizykochemicznych czerwonego wina

Zespół: **s27492 · s26681 · 27100**

Dataset: *Wine Quality – Red* (UCI, 1599 próbek, 11 cech)

---

## Problem

- Ocena jakości wina przez sommeliera jest **kosztowna i subiektywna**.
- Cel: przewidzieć jakość na podstawie **11 cech fizykochemicznych**.
- Ujęcie: **klasyfikacja binarna**

  `quality >= 6 → dobre (1)`  ·  `quality < 6 → słabe (0)`

- Balans klas: **53,5% dobre / 46,5% słabe** (lekko niezbalansowany).
- Metryki wiodące: **F1** i **ROC-AUC** (lepsze przy niezbalansowaniu niż sama accuracy).

---

## Dane

- 1599 próbek, 11 cech numerycznych + `quality`.
- Najsilniejsze korelacje z jakością:
  - `alcohol` (+), `sulphates` (+), `volatile acidity` (−).
- Preprocessing: usunięcie **240 duplikatów**, `StandardScaler`, stratyfikowany podział 80/20.
- Wersjonowanie danych: **DVC** (`*.dvc` w Git, dane w remote).

---

## Architektura systemu

```
 CSV (raw) ──DVC──► baseline (notebook, LogReg)
                        │  model.pkl (mock / Registry)
                        ▼
 Feast (feature store) ─► Flask API + GUI ─► predykcja
                        │
                        ▼
 Monitoring: logowanie predykcji + detekcja driftu (K-S / Evidently)
                        │
                        ▼
                 Docker / docker-compose
```

Ścieżka MLOps: **Opcja A** — DVC (dane) · Feast (cechy) · MLflow Registry (modele).

---

## Pipeline ML

1. **Ingest** → surowe dane (`data/01_raw`, DVC).
2. **Preprocessing** → duplikaty, target binarny, skalowanie, split.
3. **Training** → model bazowy + (docelowo) AutoML + strojenie (Optuna).
4. **Evaluation** → accuracy / precision / recall / F1 / ROC-AUC.
5. **Serving** → Flask API/GUI, real-time inference.

---

## Wyniki — model bazowy (Logistic Regression)

| Metryka | Wartość |
|---|---|
| accuracy | 0,735 |
| precision | 0,761 |
| recall | 0,729 |
| **F1** | **0,745** |
| **ROC-AUC** | **0,812** |

Punkt odniesienia (mock) dla serwowania; etap udoskonalania (Commit 2) ma go poprawić.

---

## Wdrożenie — serwowanie (Commit 3)

- **Flask API + GUI** — formularz 11 cech → „dobre/słabe” + prawdopodobieństwo.
- Endpointy: `/predict`, `/predict/by_id` (Feast), `/health`, `/monitoring/drift`.
- **Feature store (Feast)** — encja `wine`, feature view `wine_features`, online store SQLite.
- Ładowanie modelu: **MLflow Registry → fallback lokalny mock** (`model_loader.py`).
- Tryb inferencji: **real-time** (pojedyncza predykcja).

---

## Monitoring

- **Logowanie predykcji** → `serving/logs/predictions.csv` (timestamp + cechy + wynik).
- **Detekcja driftu**:
  - rdzeń: test **Kolmogorowa-Smirnowa** per cecha (`p < 0,05` → drift),
  - opcjonalnie raport **Evidently** (HTML dashboard).
- Endpoint `GET /monitoring/drift` → podsumowanie JSON na żywo.

---

## Konteneryzacja

- `docker/Dockerfile` — `python:3.11-slim`, gunicorn.
- `docker/docker-compose.yml` — serwis `wine-serving`, port 5000, wolumin `data/` + `logs/`.
- Uruchomienie:
  ```bash
  docker compose -f docker/docker-compose.yml up --build
  ```
- Zero kosztów chmury, łatwe demo.

---

## Demo (scenariusz)

1. `http://localhost:5000` — wypełnij formularz, kliknij **Przewiduj**.
2. Pokaż JSON: `curl -X POST .../predict`.
3. `GET /health` — źródło modelu.
4. Wykonaj kilka predykcji → pokaż `predictions.csv`.
5. `GET /monitoring/drift` — raport K-S.
6. (opcjonalnie) `POST /predict/by_id` — cechy z Feast.

---

## Stos technologiczny

`Python 3.11` · `pandas` · `scikit-learn` · `Jupyter` · `Kedro` · `MLflow`
`PyCaret` · `Optuna` · **`DVC`** · **`Feast`** · **`Flask`** · **`Evidently`** · **`Docker`**

---

## Podsumowanie

- Pełny łańcuch: **dane → cechy → model → API/GUI → monitoring → kontener**.
- MLOps Opcja A: DVC + Feast + MLflow Registry.
- Model bazowy: **F1 0,745 / ROC-AUC 0,812** (do poprawy przez AutoML + strojenie).
- Wdrożenie lokalne + Docker, gotowe demo.

### Dziękujemy za uwagę! 🍷
