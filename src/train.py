"""
Train and compare three classifiers for next-day rainfall prediction:
Random Forest, SVM (RBF) and XGBoost.

For each model it reports accuracy, precision, recall, F1 and ROC-AUC on a
held-out test set, then saves the best model (by ROC-AUC) and a metrics
report to disk.

Usage:
    python -m src.train
    python -m src.train --data data/weather.csv --test-size 0.2
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .data_loader import load_data
from .preprocess import build_preprocessor, split_features_target


def build_models(X):
    """Return {name: (pipeline, needs_scaling)} for each classifier."""
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=42, class_weight="balanced"
    )
    svm = SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, class_weight="balanced", random_state=42)
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )

    return {
        "RandomForest": (Pipeline([("prep", build_preprocessor(X, scale=False)), ("clf", rf)]), False),
        "SVM": (Pipeline([("prep", build_preprocessor(X, scale=True)), ("clf", svm)]), True),
        "XGBoost": (Pipeline([("prep", build_preprocessor(X, scale=False)), ("clf", xgb)]), False),
    }


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train rainfall forecasting models.")
    parser.add_argument("--data", default="data/weather.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    df = load_data(args.data)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}  "
          f"Positive rate: {y.mean():.1%}\n")

    results = {}
    fitted = {}
    for name, (pipe, _) in build_models(X_train).items():
        print(f"Training {name} ...")
        pipe.fit(X_train, y_train)
        results[name] = evaluate(pipe, X_test, y_test)
        fitted[name] = pipe
        m = results[name]
        print(f"  acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  "
              f"rec={m['recall']:.3f}  f1={m['f1']:.3f}  auc={m['roc_auc']:.3f}")

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_path = os.path.join(args.models_dir, "best_model.joblib")
    joblib.dump(fitted[best_name], best_path)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "positive_rate": round(float(y.mean()), 4),
        "test_size": args.test_size,
        "results": results,
        "best_model": best_name,
        "best_model_path": best_path,
    }
    report_path = os.path.join(args.reports_dir, "metrics.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nBest model: {best_name} (ROC-AUC {results[best_name]['roc_auc']:.3f})")
    print(f"Saved model  -> {best_path}")
    print(f"Saved report -> {report_path}")


if __name__ == "__main__":
    main()
