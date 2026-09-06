"""
Evaluate the saved best model and write plots to the reports/ directory:
  * confusion_matrix.png
  * roc_curve.png
  * feature_importance.png  (when the model exposes importances)

Usage:
    python -m src.evaluate
"""
from __future__ import annotations

import argparse
import os

import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)
from sklearn.model_selection import train_test_split

from .data_loader import load_data
from .preprocess import split_features_target


def _feature_names(pipeline) -> list[str]:
    """Recover output feature names from the fitted ColumnTransformer."""
    try:
        return list(pipeline.named_steps["prep"].get_feature_names_out())
    except Exception:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the saved model.")
    parser.add_argument("--data", default="data/weather.csv")
    parser.add_argument("--model", default="models/best_model.joblib")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    os.makedirs(args.reports_dir, exist_ok=True)
    model = joblib.load(args.model)

    df = load_data(args.data)
    X, y = split_features_target(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds, target_names=["No Rain", "Rain"]))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds, display_labels=["No Rain", "Rain"], cmap="Blues", ax=ax
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(args.reports_dir, "confusion_matrix.png"), dpi=130)
    plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    fig.tight_layout()
    fig.savefig(os.path.join(args.reports_dir, "roc_curve.png"), dpi=130)
    plt.close(fig)

    # Feature importance (tree models only)
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        names = _feature_names(model)
        importances = clf.feature_importances_
        if names and len(names) == len(importances):
            order = np.argsort(importances)[::-1][:15]
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh([names[i] for i in order][::-1], importances[order][::-1], color="#4a7db5")
            ax.set_title("Top 15 Feature Importances")
            ax.set_xlabel("Importance")
            fig.tight_layout()
            fig.savefig(os.path.join(args.reports_dir, "feature_importance.png"), dpi=130)
            plt.close(fig)

    print(f"Plots written to {args.reports_dir}/")


if __name__ == "__main__":
    main()
