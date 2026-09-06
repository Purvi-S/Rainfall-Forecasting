"""
Preprocessing utilities: build a scikit-learn ColumnTransformer that imputes
missing values, one-hot-encodes categoricals and scales numeric features.

Two variants are exposed:
  * ``build_preprocessor(scale=True)``  -- scaled numerics, needed by SVM.
  * ``build_preprocessor(scale=False)`` -- tree models don't need scaling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "RainTomorrow"


def split_features_target(df: pd.DataFrame):
    """Return (X, y) where y is a 0/1 integer array for RainTomorrow=Yes."""
    y = (df[TARGET].astype(str).str.strip().str.lower() == "yes").astype(int)
    X = df.drop(columns=[TARGET])
    return X, y


def column_types(X: pd.DataFrame):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame, scale: bool = True) -> ColumnTransformer:
    """Construct the preprocessing ColumnTransformer for the given frame."""
    numeric, categorical = column_types(X)

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ]
    )
