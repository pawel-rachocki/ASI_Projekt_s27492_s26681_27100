import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict


def clean_data(data: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Usuwa duplikaty, uzupełnia braki i tworzy target binarny na podstawie progu (threshold).

    quality >= threshold → 1, else → 0.
    Kolumna 'quality' jest następnie usuwana.
    """
    cleaned = data.drop_duplicates().reset_index(drop=True)
    cleaned = cleaned.dropna()  # handle missing values
    cleaned["target"] = (cleaned["quality"] >= threshold).astype(int)
    cleaned = cleaned.drop("quality", axis=1)
    return cleaned


def split_and_scale_data(
    data: pd.DataFrame, parameters: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Dzieli dane na zbiory treningowe/testowe i skaluje cechy za pomocą StandardScaler.

    Zwraca cztery obiekty: X_train, X_test, y_train, y_test jako oddzielne DataFrames/Series.
    """
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=parameters.get("test_size", 0.2),
        random_state=parameters.get("random_state", 42),
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    y_train_df = y_train.reset_index(drop=True).rename("target")
    y_test_df = y_test.reset_index(drop=True).rename("target")

    return X_train_scaled, X_test_scaled, y_train_df.to_frame(), y_test_df.to_frame()
