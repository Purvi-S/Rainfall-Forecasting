"""
Load the weather dataset.

If ``data/weather.csv`` does not exist, a synthetic dataset is generated
automatically so the pipeline always runs. To use the real-world data,
download the Kaggle "Rain in Australia" dataset (weatherAUS.csv) and save it
as ``data/weather.csv`` -- the columns are already compatible.
"""
from __future__ import annotations

import os
import pandas as pd

from . import generate_synthetic_data as synth

TARGET = "RainTomorrow"


def load_data(path: str = "data/weather.csv", rows: int = 50_000) -> pd.DataFrame:
    """Load the dataset from ``path``, generating synthetic data if absent."""
    if not os.path.exists(path):
        print(f"[data_loader] '{path}' not found -- generating synthetic data.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df = synth.generate(rows)
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
        print(f"[data_loader] Loaded {len(df):,} rows from '{path}'.")

    # Drop rows with no target label (real dataset has some).
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    d = load_data()
    print(d.head())
    print("\nShape:", d.shape)
