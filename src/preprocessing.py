"""Reusable preprocessing utilities for the liver cirrhosis ML project."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(file_path: str) -> pd.DataFrame:
    """Load a CSV dataset and validate that it is not empty."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError("The dataset is empty.")

    return data


def split_features_target(
    data: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate input features from the target column."""
    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {list(data.columns)}"
        )

    features = data.drop(columns=[target_column])
    target = data[target_column]

    return features, target


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing for numeric and categorical columns."""
    numeric_columns = features.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = features.select_dtypes(
        exclude=["number", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
