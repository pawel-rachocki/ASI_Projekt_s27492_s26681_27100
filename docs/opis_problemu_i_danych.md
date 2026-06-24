# Opis problemu i danych

## 1. Problem biznesowy / ML

Celem projektu jest **przewidywanie jakości czerwonego wina** na podstawie jego właściwości
fizykochemicznych (bez potrzeby kosztownej oceny eksperckiej *sommeliera*).

Oryginalny atrybut `quality` to ocena ekspercka w skali 0–10 (w danych występują wartości 3–8).
Ze względu na czytelność, użyteczność praktyczną i silnie niezbalansowane klasy w wersji
wieloklasowej, problem ujmujemy jako **klasyfikację binarną**:

| Klasa | Definicja | Etykieta |
|---|---|---|
| Wino **dobre** | `quality >= 6` | `1` |
| Wino **słabe** | `quality < 6` | `0` |

**Metryki ewaluacji:** accuracy, precision, recall, **F1-score**, **ROC-AUC**, macierz pomyłek.
(F1/ROC-AUC traktujemy jako wiodące, bo lepiej oddają jakość przy lekkim niezbalansowaniu.)

## 2. Źródło danych

- Plik: `data/01_raw/winequality-red.csv`
- Zbiór: *Wine Quality – Red* (P. Cortez i in., UCI Machine Learning Repository)
- Wersjonowanie: **DVC** (dane poza Gitem, w Git tylko wskaźniki `*.dvc`)

## 3. Charakterystyka zbioru

- **Liczba próbek:** 1599
- **Liczba cech:** 11 (wszystkie numeryczne, ciągłe) + 1 kolumna celu (`quality`)
- **Braki danych:** 0
- **Duplikaty wierszy:** 240 (do rozważenia usunięcie w preprocessingu)

### Balans klas (po binaryzacji)

| Klasa | Liczność | Udział |
|---|---|---|
| `1` – dobre (`quality >= 6`) | 855 | 53,5% |
| `0` – słabe (`quality < 6`)  | 744 | 46,5% |

Zbiór jest **lekko niezbalansowany** — akceptowalny bez specjalnych technik, ale w razie
potrzeby można zastosować `class_weight` lub SMOTE.

### Rozkład oryginalnej oceny `quality`

| quality | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| liczność | 10 | 53 | 681 | 638 | 199 | 18 |

## 4. Cechy (zmienne objaśniające)

| # | Cecha | Opis |
|---|---|---|
| 1 | fixed acidity | kwasowość stała (kwas winowy) |
| 2 | volatile acidity | kwasowość lotna (kwas octowy) — wysoka pogarsza smak |
| 3 | citric acid | kwas cytrynowy — świeżość |
| 4 | residual sugar | cukier resztkowy |
| 5 | chlorides | zawartość soli |
| 6 | free sulfur dioxide | wolny SO₂ — ochrona przed utlenianiem |
| 7 | total sulfur dioxide | całkowity SO₂ |
| 8 | density | gęstość |
| 9 | pH | kwasowość w skali pH |
| 10 | sulphates | siarczany — dodatek konserwujący |
| 11 | alcohol | zawartość alkoholu (często silnie skorelowana z jakością) |

**Zmienna celu:** `quality` → binaryzowana do `target ∈ {0, 1}`.

## 5. Wnioski z EDA i wynik modelu bazowego

- Najsilniejsze korelacje z jakością: `alcohol` (+), `sulphates` (+), `volatile acidity` (−) — zgodnie z oczekiwaniami.
- Usunięto 240 duplikatów; pozostałe cechy bez braków.
- Zastosowano `StandardScaler` (cechy w różnych skalach) + stratyfikowany podział 80/20.

### Model bazowy (Logistic Regression, zbiór testowy)

| Metryka | Wartość |
|---|---|
| accuracy | 0,735 |
| precision | 0,761 |
| recall | 0,729 |
| **F1** | **0,745** |
| **ROC-AUC** | **0,812** |

Artefakt zapisany w `data/06_models/baseline_logreg.pkl` (scaler + model + lista cech + metryki) —
służy jako „mock” dla zespołu do czasu powstania finalnego modelu w MLflow Model Registry.
To punkt odniesienia, który etap udoskonalania (s26681: AutoML + strojenie) ma poprawić.
