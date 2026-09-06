"""
Generate a realistic synthetic daily-weather dataset for the rainfall
forecasting pipeline.

The columns mirror the well-known Kaggle "Rain in Australia" (weatherAUS)
dataset so that the exact same pipeline runs unchanged on the real data.
The target is ``RainTomorrow`` (yes/no): will it rain the next day?

The generator builds genuine (but noisy) relationships between the features
and the target -- e.g. high afternoon humidity, low pressure and rain today
all raise tomorrow's rain probability -- so the models learn real signal and
report *realistic* accuracy rather than a trivially separable 99%.

Usage:
    python -m src.generate_synthetic_data --rows 50000 --out data/weather.csv
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

WIND_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def generate(n_rows: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic weather DataFrame with ``n_rows`` daily records."""
    rng = np.random.default_rng(seed)

    # --- base atmospheric features -------------------------------------------
    min_temp = rng.normal(12, 6, n_rows).round(1)
    max_temp = (min_temp + rng.normal(9, 3, n_rows)).round(1)

    humidity_9am = np.clip(rng.normal(68, 18, n_rows), 0, 100).round(0)
    # afternoon humidity is correlated with the morning value
    humidity_3pm = np.clip(humidity_9am - rng.normal(15, 12, n_rows), 0, 100).round(0)

    pressure_9am = rng.normal(1017, 7, n_rows).round(1)
    pressure_3pm = (pressure_9am - rng.normal(1.5, 2, n_rows)).round(1)

    wind_speed = np.clip(rng.gamma(shape=2.2, scale=9, size=n_rows), 0, 130).round(0)
    cloud_3pm = np.clip(rng.normal(4.5, 2.6, n_rows), 0, 8).round(0)

    temp_9am = (min_temp + rng.normal(3, 2, n_rows)).round(1)
    temp_3pm = (max_temp - rng.normal(2, 2, n_rows)).round(1)

    rainfall_today_mm = np.clip(rng.gamma(shape=0.4, scale=6, size=n_rows), 0, 200).round(1)
    rain_today = (rainfall_today_mm > 1.0).astype(int)

    wind_dir_3pm = rng.choice(WIND_DIRS, size=n_rows)

    # --- latent "rain tomorrow" score ----------------------------------------
    # Standardise the drivers, weight them, add noise, then threshold.
    def z(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-9)

    score = (
        0.9 * z(humidity_3pm)
        + 0.5 * z(humidity_9am)
        - 0.8 * z(pressure_3pm)
        + 0.6 * z(cloud_3pm)
        + 0.7 * z(rainfall_today_mm)
        + 0.3 * z(wind_speed)
        - 0.3 * z(max_temp)
        + rng.normal(0, 1.3, n_rows)          # irreducible noise -> realistic accuracy
    )

    # Threshold chosen so ~22% of days have rain tomorrow (class imbalance,
    # like the real dataset).
    threshold = np.quantile(score, 0.78)
    rain_tomorrow = (score > threshold).astype(int)

    df = pd.DataFrame(
        {
            "MinTemp": min_temp,
            "MaxTemp": max_temp,
            "Rainfall": rainfall_today_mm,
            "WindGustSpeed": wind_speed,
            "Humidity9am": humidity_9am,
            "Humidity3pm": humidity_3pm,
            "Pressure9am": pressure_9am,
            "Pressure3pm": pressure_3pm,
            "Cloud3pm": cloud_3pm,
            "Temp9am": temp_9am,
            "Temp3pm": temp_3pm,
            "WindDir3pm": wind_dir_3pm,
            "RainToday": np.where(rain_today == 1, "Yes", "No"),
            "RainTomorrow": np.where(rain_tomorrow == 1, "Yes", "No"),
        }
    )

    # --- inject realistic missing values (~3% per numeric column) ------------
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        mask = rng.random(n_rows) < 0.03
        df.loc[mask, col] = np.nan

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic weather data.")
    parser.add_argument("--rows", type=int, default=50_000, help="Number of rows.")
    parser.add_argument("--out", type=str, default="data/weather.csv", help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    rate = (df["RainTomorrow"] == "Yes").mean()
    print(f"Wrote {len(df):,} rows to {args.out}  (rain-tomorrow rate: {rate:.1%})")


if __name__ == "__main__":
    main()
