"""
run_experiments.py
==================
Skrypt eksperymentalny dla klasyfikacji jakości wina.

Obejmuje:
- Porównanie modeli AutoML (PyCaret lub fallback do scikit-learn)
- Strojenie hiperparametrów przy użyciu Optuna (optymalizacja bayesowska, 30 prób)
- Śledzenie eksperymentów w MLflow (logowanie parametrów, metryk, artefaktów)
- Rejestrację modelu w MLflow Model Registry (oznaczenie jako wine_quality / Production)

Uruchomienie:
    python scripts/run_experiments.py
"""
from __future__ import annotations

import os
import sys
import pickle
import warnings
import logging
from pathlib import Path

# ── Rozwiązywanie ścieżek projektu i zmiana katalogu roboczego ────────────────
# Określamy ścieżkę do katalogu skryptu oraz głównego katalogu projektu (root),
# a następnie zmieniamy katalog roboczy, aby MLflow prawidłowo odnalazł lokalną bazę sqlite.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.chdir(PROJECT_ROOT)

# Dodajemy katalog źródłowy src do ścieżki wyszukiwania modułów Pythona
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Ignorujemy ostrzeżenia (warnings) i ustawiamy podstawową konfigurację logowania
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Importy standardowych bibliotek ML i MLOps ─────────────────────────────────
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
from optuna.samplers import TPESampler

# Wyłączamy zbędne logi gadatliwości biblioteki Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Definicje ścieżek do katalogów danych, modeli i raportów ───────────────────
DATA_DIR = PROJECT_ROOT / "data" / "05_model_input"
MODELS_DIR = PROJECT_ROOT / "data" / "06_models"
REPORTING_DIR = PROJECT_ROOT / "data" / "08_reporting"

# Tworzymy katalogi, jeśli jeszcze nie istnieją
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTING_DIR.mkdir(parents=True, exist_ok=True)

# Konfiguracja tracking URI dla bazy danych MLflow oraz nazwy rejestrów
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "wine_quality_experiments"
MODEL_REGISTRY_NAME = "wine_quality"

# ── Opcjonalne ładowanie ciężkich zależności (PyCaret, XGBoost, LightGBM) ──────
# Sprawdzamy dostępność bibliotek do celów AutoML.
PYCARET_AVAILABLE = False
try:
    from pycaret.classification import (
        setup as pycaret_setup,
        compare_models as pycaret_compare,
        pull as pycaret_pull,
        get_config,
    )
    PYCARET_AVAILABLE = True
    print("[INFO] PyCaret jest dostępny — zostanie użyty do porównania AutoML.")
except ImportError:
    print("[INFO] PyCaret nie jest dostępny — użyty zostanie fallback do scikit-learn.")

# Sprawdzamy dostępność XGBoost
XGBOOST_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

# Sprawdzamy dostępność LightGBM
LIGHTGBM_AVAILABLE = False
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    pass

print(f"[INFO] XGBoost : {'dostępny' if XGBOOST_AVAILABLE else 'niedostępny'}")
print(f"[INFO] LightGBM: {'dostępny' if LIGHTGBM_AVAILABLE else 'niedostępny'}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. ŁADOWANIE DANYCH
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """
    Wczytuje podzielone i przeskalowane zbiory danych z warstwy data/05_model_input/.
    W przypadku braku plików X_train.csv, skrypt awaryjnie pobiera dane surowe i generuje podziały.
    """
    x_train_path = DATA_DIR / "X_train.csv"
    if x_train_path.exists():
        X_train = pd.read_csv(DATA_DIR / "X_train.csv")
        X_test  = pd.read_csv(DATA_DIR / "X_test.csv")
        y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
        y_test  = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()
        print(f"[INFO] Załadowano istniejący podział danych — treningowe: {X_train.shape}, testowe: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    # ── Ścieżka awaryjna: generowanie podziałów z surowego pliku CSV ─────────
    print("[WARN] Plik X_train.csv nie został znaleziony. Generowanie podziałów z surowych danych...")
    raw_path = PROJECT_ROOT / "data" / "01_raw" / "winequality-red.csv"
    df = pd.read_csv(raw_path, sep=None, engine="python")
    if "quality" not in df.columns:
        df = pd.read_csv(raw_path, sep=",")

    # Binaryzacja zmiennej celu: wina o jakości >= 6 oznaczamy jako 1 (dobre), resztę jako 0 (słabe)
    df["quality_binary"] = (df["quality"] >= 6).astype(int)
    df.drop(columns=["quality"], inplace=True)

    X = df.drop(columns=["quality_binary"])
    y = df["quality_binary"]

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Podział stratyfikowany (80% trening, 20% test, seed 42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    # Skalowanie cech z zachowaniem informacji o kolumnach
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Zapisujemy wygenerowane zbiory do katalogu 05_model_input
    X_train_sc.to_csv(DATA_DIR / "X_train.csv", index=False)
    X_test_sc.to_csv(DATA_DIR / "X_test.csv", index=False)
    pd.Series(y_train.values, name="quality_binary").to_csv(DATA_DIR / "y_train.csv", index=False)
    pd.Series(y_test.values, name="quality_binary").to_csv(DATA_DIR / "y_test.csv", index=False)

    print(f"[INFO] Wygenerowano podział danych — treningowe: {X_train_sc.shape}, testowe: {X_test_sc.shape}")
    return X_train_sc, X_test_sc, y_train.reset_index(drop=True), y_test.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. KONFIGURACJA MLFLOW
# ─────────────────────────────────────────────────────────────────────────────

def setup_mlflow():
    """Konfiguruje ścieżkę śledzenia i nazwę eksperymentu w bibliotece MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"[INFO] MLflow tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"[INFO] MLflow eksperyment  : {EXPERIMENT_NAME}")


# ─────────────────────────────────────────────────────────────────────────────
# 3A. AUTOML — Ścieżka PyCaret
# ─────────────────────────────────────────────────────────────────────────────

def run_pycaret_comparison(X_train, y_train):
    """Uruchamia porównanie AutoML przy użyciu biblioteki PyCaret i loguje wyniki do MLflow."""
    print("\n[KROK] Uruchamianie AutoML przez PyCaret...")

    train_df = X_train.copy()
    train_df["target"] = y_train.values

    # Inicjalizacja konfiguracji PyCaret
    pycaret_setup(
        data=train_df,
        target="target",
        session_id=42,
        verbose=False,
        log_experiment=False,
    )

    # Definiujemy listę modeli do porównania
    include = ["lr", "rf", "et", "dt"]
    if XGBOOST_AVAILABLE:
        include.append("xgboost")
    if LIGHTGBM_AVAILABLE:
        include.append("lightgbm")

    # Porównujemy modele sortując według metryki F1
    best = pycaret_compare(include=include, sort="F1", n_select=1, verbose=False)
    results_df = pycaret_pull()
    # Zapisujemy tabelę porównawczą do pliku CSV
    results_df.to_csv(REPORTING_DIR / "model_comparison.csv", index=False)
    print(f"[INFO] Tabela porównawcza zapisana → {REPORTING_DIR / 'model_comparison.csv'}")

    # Rejestrujemy parametry i metryki każdego z modeli w osobnym przebiegu (run) MLflow
    for _, row in results_df.iterrows():
        mname  = row.get("Model", "unknown")
        f1_val  = row.get("F1", 0.0)
        auc_val = row.get("AUC", 0.0)
        with mlflow.start_run(run_name=f"automl_{str(mname).replace(' ', '_')}"):
            mlflow.log_param("model_type", mname)
            mlflow.log_param("source", "pycaret_compare")
            mlflow.log_metric("f1", float(f1_val) if pd.notna(f1_val) else 0.0)
            mlflow.log_metric("roc_auc", float(auc_val) if pd.notna(auc_val) else 0.0)

    print(f"[INFO] Porównanie PyCaret zakończone. Najlepszy model: {type(best).__name__}")
    return best, results_df


# ─────────────────────────────────────────────────────────────────────────────
# 3B. AUTOML — Ścieżka scikit-learn (Fallback)
# ─────────────────────────────────────────────────────────────────────────────

def run_sklearn_comparison(X_train, X_test, y_train, y_test):
    """Ręczne porównanie modeli scikit-learn i logowanie każdego przebiegu do MLflow."""
    print("\n[KROK] Uruchamianie porównania modeli scikit-learn (fallback)...")

    # Walidacja krzyżowa (5-krotny StratifiedKFold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Inicjalizacja modeli kandydujących z domyślnymi parametrami
    candidates = {
        "LogisticRegression": LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs"),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=100, random_state=42, eval_metric="logloss",
            use_label_encoder=False, verbosity=0,
        )
    if LIGHTGBM_AVAILABLE:
        candidates["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)

    results = []
    best_model = None
    best_f1 = -1.0

    # Iteracja po wszystkich modelach w celu ich oceny w walidacji krzyżowej
    for name, clf in candidates.items():
        print(f"    Ocena modelu {name}...")
        # Wyznaczamy średni wynik F1 i ROC-AUC z walidacji krzyżowej
        f1_cv  = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1").mean()
        auc_cv = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc").mean()

        # Logowanie wyników do MLflow
        with mlflow.start_run(run_name=f"automl_{name}"):
            mlflow.log_param("model_type", name)
            mlflow.log_param("source", "sklearn_compare")
            mlflow.log_metric("f1", round(float(f1_cv), 4))
            mlflow.log_metric("roc_auc", round(float(auc_cv), 4))

        results.append({
            "model": name,
            "f1": round(float(f1_cv), 4),
            "roc_auc": round(float(auc_cv), 4),
        })

        # Wybór najlepszego modelu na podstawie metryki F1
        if f1_cv > best_f1:
            best_f1 = f1_cv
            best_model = (name, clf)

    # Tworzenie i zapisywanie tabeli porównawczej
    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    results_df.to_csv(REPORTING_DIR / "model_comparison.csv", index=False)
    print(f"[INFO] Tabela porównawcza zapisana → {REPORTING_DIR / 'model_comparison.csv'}")
    print(results_df.to_string(index=False))

    best_name, best_clf = best_model
    print(f"\n[INFO] Najlepszy model: {best_name} (Średni CV F1={best_f1:.4f})")
    
    # Dopasowanie wybranego najlepszego modelu do całego zbioru treningowego
    best_clf.fit(X_train, y_train)
    return best_clf, best_name, results_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. OPTYMALIZACJA OPTUNA
# ─────────────────────────────────────────────────────────────────────────────

def _build_trial_model(trial, model_class_name):
    """Tworzy model z hiperparametrami zaproponowanymi przez Optuna w danej próbie (trial)."""
    if model_class_name in ("LogisticRegression",):
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 100.0, log=True),
            max_iter=trial.suggest_int("max_iter", 200, 2000, step=200),
            solver="lbfgs",
            random_state=42,
        )
    elif model_class_name in ("RandomForestClassifier", "RandomForest"):
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            random_state=42,
        )
    elif model_class_name in ("ExtraTreesClassifier", "ExtraTrees"):
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            random_state=42,
        )
    elif model_class_name in ("GradientBoostingClassifier", "GradientBoosting"):
        return GradientBoostingClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            random_state=42,
        )
    elif model_class_name in ("XGBClassifier", "XGBoost") and XGBOOST_AVAILABLE:
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 400, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
            random_state=42,
        )
    elif model_class_name in ("LGBMClassifier", "LightGBM") and LIGHTGBM_AVAILABLE:
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 400, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            verbose=-1,
            random_state=42,
        )
    else:
        # Ogólny fallback do Random Forest
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            random_state=42,
        )


def run_optuna_tuning(best_model, model_name: str, X_train, X_test, y_train, y_test, n_trials: int = 30):
    """Optymalizacja hiperparametrów za pomocą Optuna (próbkowanie TPE - proces bayesowski)."""
    print(f"\n[KROK] Strojenie Optuna dla {model_name} — {n_trials} prób (TPE)...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_class_name = type(best_model).__name__

    # Funkcja celu Optuny oceniająca model na bazie średniej F1 w walidacji krzyżowej
    def objective(trial):
        clf = _build_trial_model(trial, model_class_name)
        return cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1").mean()

    # Tworzymy badanie (study) maksymalizujące F1
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_cv_f1  = study.best_value
    print(f"[INFO] Optuna najlepsza średnia F1 z walidacji krzyżowej: {best_cv_f1:.4f}")
    print(f"[INFO] Optuna optymalne parametry: {best_params}")

    # Replay parametrów: budujemy końcowy, nastrojony model
    class _MockTrial:
        """Służy do ponownego odtworzenia modelu na podstawie słownika parametrów."""
        def __init__(self, params):
            self._p = params
        def suggest_float(self, name, *a, **kw):
            return self._p[name]
        def suggest_int(self, name, *a, **kw):
            return int(self._p[name])

    tuned_clf = _build_trial_model(_MockTrial(best_params), model_class_name)
    tuned_clf.fit(X_train, y_train)

    # Ewaluacja na wydzielonym zbiorze testowym
    y_pred  = tuned_clf.predict(X_test)
    y_proba = tuned_clf.predict_proba(X_test)[:, 1]
    test_f1  = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_proba)
    print(f"[INFO] Ewaluacja na zbiorze testowym — F1: {test_f1:.4f}  ROC-AUC: {test_auc:.4f}")

    return tuned_clf, best_params, best_cv_f1, test_f1, test_auc


# ─────────────────────────────────────────────────────────────────────────────
# 5. LOGOWANIE MODELU DO MLFLOW I REJESTRACJA W REJESTRZE MODELI
# ─────────────────────────────────────────────────────────────────────────────

def log_and_register_model(tuned_model, model_name, best_params, cv_f1, test_f1, test_auc):
    """Zapisuje nastrojony model do bazy MLflow i promuje go do fazy Production w Model Registry."""
    print("\n[KROK] Logowanie i rejestracja modelu w MLflow Model Registry...")

    with mlflow.start_run(run_name=f"tuned_{model_name}_optuna") as run:
        run_id = run.info.run_id

        # Logujemy parametry optymalizacji i hiperparametry modelu
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("tuning_method", "optuna_tpe")
        mlflow.log_param("n_trials", 30)
        for k, v in best_params.items():
            mlflow.log_param(k, v)

        # Logujemy końcowe metryki walidacji oraz metryki testowe
        mlflow.log_metric("cv_f1", round(float(cv_f1), 4))
        mlflow.log_metric("test_f1", round(float(test_f1), 4))
        mlflow.log_metric("test_roc_auc", round(float(test_auc), 4))

        # Zapisujemy model do MLflow i jednocześnie rejestrujemy go w Model Registry
        mlflow.sklearn.log_model(
            sk_model=tuned_model,
            artifact_path="model",
            registered_model_name=MODEL_REGISTRY_NAME,
        )
        print(f"[INFO] MLflow run ID: {run_id}")

    # ── Przeniesienie zarejestrowanej wersji modelu do etapu Production ────────
    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_REGISTRY_NAME)
    version_num = "unknown"
    if versions:
        # Pobieramy najwyższy numer wersji modelu
        latest = max(versions, key=lambda v: int(v.version))
        version_num = latest.version
        try:
            # Przenosimy model do etapu "Production" i archiwizujemy poprzednie wersje
            client.transition_model_version_stage(
                name=MODEL_REGISTRY_NAME,
                version=version_num,
                stage="Production",
                archive_existing_versions=True,
            )
            print(f"[INFO] Model '{MODEL_REGISTRY_NAME}' wersja v{version_num} ustawiony jako Production")
        except Exception as e:
            # Fallback dla starszych lub specyficznych konfiguracji MLflow za pomocą aliasów
            print(f"[WARN] Zmiana etapu się nie powiodła ({e}); próba ustawienia aliasu...")
            try:
                client.set_registered_model_alias(MODEL_REGISTRY_NAME, "Production", version_num)
                print(f"[INFO] Alias 'Production' ustawiony dla modelu '{MODEL_REGISTRY_NAME}' v{version_num}")
            except Exception as e2:
                print(f"[WARN] Ustawienie aliasu również się nie powiodło: {e2}")
    else:
        print("[WARN] Nie znaleziono żadnych wersji modelu w rejestrze po rejestracji.")

    return run_id, version_num


# ─────────────────────────────────────────────────────────────────────────────
# 6. ZAPIS NAJLEPSZEGO MODELU NA DYSK LOKALNY
# ─────────────────────────────────────────────────────────────────────────────

def save_best_model(model):
    """Zapisuje binarny plik modelu na dysk w celu szybkiego wczytania jako fallback."""
    path = MODELS_DIR / "best_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] Najlepszy model zapisany na dysk lokalny → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FUNKCJA GŁÓWNA (MAIN)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Klasyfikacja Jakości Wina — AutoML + Optuna + Eksperymenty MLflow")
    print("=" * 70)

    # Ładujemy dane treningowe i testowe
    X_train, X_test, y_train, y_test = load_data()
    # Konfigurujemy tracking MLflow
    setup_mlflow()

    # Wybór ścieżki AutoML (PyCaret vs ręczne porównanie scikit-learn)
    if PYCARET_AVAILABLE:
        best_model, comparison_df = run_pycaret_comparison(X_train, y_train)
        model_name = type(best_model).__name__
    else:
        best_model, model_name, comparison_df = run_sklearn_comparison(
            X_train, X_test, y_train, y_test
        )

    print(f"\n[INFO] Dalsza optymalizacja dla wybranego modelu: {model_name}")

    # Uruchamiamy optymalizację hiperparametrów przez Optunę
    tuned_model, best_params, cv_f1, test_f1, test_auc = run_optuna_tuning(
        best_model, model_name, X_train, X_test, y_train, y_test, n_trials=30
    )

    # Zapisujemy najlepszy model na dysku lokalnym (dla serwowania bez rejestru)
    save_best_model(tuned_model)

    # Logujemy model do MLflow i promujemy wersję w rejestrze modeli jako Production
    run_id, version_num = log_and_register_model(
        tuned_model, model_name, best_params, cv_f1, test_f1, test_auc
    )

    # Wyświetlamy końcowe zestawienie eksperymentu na ekranie
    print("\n" + "=" * 70)
    print("  PODSUMOWANIE EKSPERYMENTU")
    print("=" * 70)
    print(f"  Silnik AutoML  : {'PyCaret' if PYCARET_AVAILABLE else 'scikit-learn fallback'}")
    print(f"  Najlepszy model: {model_name}")
    print(f"  Średnia CV F1  : {cv_f1:.4f}")
    print(f"  Test F1        : {test_f1:.4f}")
    print(f"  Test ROC-AUC   : {test_auc:.4f}")
    print(f"  MLflow Run ID  : {run_id}")
    print(f"  Nazwa rejestru : {MODEL_REGISTRY_NAME}")
    print(f"  Wersja rejestru: {version_num}")
    print(f"  Status         : Production")
    print("=" * 70)

    return {
        "automl_source": "PyCaret" if PYCARET_AVAILABLE else "sklearn_fallback",
        "best_model": model_name,
        "cv_f1": cv_f1,
        "test_f1": test_f1,
        "test_roc_auc": test_auc,
        "run_id": run_id,
        "registry_name": MODEL_REGISTRY_NAME,
        "registry_version": version_num,
    }


if __name__ == "__main__":
    main()
