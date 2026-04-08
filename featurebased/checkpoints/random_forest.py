#!/usr/bin/env python3
"""
train_rf.py — Train a Random Forest baseline for polygon pair matching.

The RF operates on the **feature difference** between two polygons
(poly1_features - poly2_features), producing a single 14-dim input vector
per pair.  This is in contrast to the MLP, which receives both feature
vectors separately.

Hyperparameters were selected via prior grid search and are fixed here
for reproducibility.

Outputs:
    output/rf_model.joblib                — trained model checkpoint
    output/rf_classification_report.txt   — precision / recall / f1
    output/rf_confusion_matrix.npy        — confusion matrix as numpy array

Usage:
    python train_rf.py --city berlin
    python train_rf.py --city berlin --data-dir /path/to/data --output-dir output
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import argparse
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

SEED = 42
TEST_SIZE = 0.25

# Best hyperparameters (determined via grid search)
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 50,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": False,
}

FEATURE_NAMES = [
    "area",
    "length",
    "width",
    "height",
    "area-to-length ratio",
    "centroid x",
    "centroid y",
    "elongation",
    "sine angle",
    "cosine angle",
    "degree of convexity",
    "circularity",
    "number of nodes",
    "polygon roughness",
]


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train a Random Forest baseline on polygon pair features."
    )
    parser.add_argument("--city", type=str, default="berlin",
                        help="City name used to locate the data files (default: berlin)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing the .npy data files (default: data)")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save model and results (default: output)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ────────────────────────────────────────────────────────
    # Shape of features: (2, n_samples, 14) — two polygons per pair
    # Shape of labels:   (n_samples,)       — binary (0 = no match, 1 = match)
    print(f"Loading data for city: {args.city}")

    features = np.load(data_dir / f"{args.city}_X_pairs_dataset.npy")
    labels = np.load(data_dir / f"{args.city}_y_pairs_dataset.npy")

    print(f"  Feature array shape : {features.shape}")
    print(f"  Label array shape   : {labels.shape}")

    # ── 2. Feature engineering ──────────────────────────────────────────────
    # The RF receives the element-wise difference between the two polygon
    # feature vectors as input.  This encodes how much each geometric
    # property differs between the two polygons in a pair.
    X = features[0, :, :] - features[1, :, :]
    y = labels

    print(f"  Input matrix shape  : {X.shape}  ({len(FEATURE_NAMES)} features)")
    print(f"  Class distribution  : {dict(zip(*np.unique(y, return_counts=True)))}")

    # ── 3. Train / test split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )
    print(f"\n  Train samples: {len(X_train):,}")
    print(f"  Test  samples: {len(X_test):,}")

    # ── 4. Train Random Forest ──────────────────────────────────────────────
    print(f"\n  Training Random Forest with params:")
    for k, v in RF_PARAMS.items():
        print(f"    {k:>20s} = {v}")

    model = RandomForestClassifier(
        **RF_PARAMS,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── 5. Evaluate ─────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print("  Classification Report")
    print(f"{'=' * 60}")
    print(report)
    print("  Confusion Matrix:")
    print(cm)

    # ── 6. Save outputs ────────────────────────────────────────────────────
    model_path = output_dir / "rf_model.joblib"
    report_path = output_dir / "rf_classification_report.txt"
    cm_path = output_dir / "rf_confusion_matrix.npy"

    joblib.dump(model, model_path)
    np.save(cm_path, cm)
    with open(report_path, "w") as f:
        f.write(f"Random Forest — {args.city}\n")
        f.write(f"Seed: {SEED} | Test size: {TEST_SIZE}\n")
        f.write(f"Hyperparameters: {RF_PARAMS}\n\n")
        f.write(report)
        f.write(f"\nConfusion Matrix:\n{cm}\n")

    print(f"\n  Model saved  → {model_path}")
    print(f"  Report saved → {report_path}")
    print(f"  CM saved     → {cm_path}")

    # ── 7. Print Gini feature importances (built-in) ───────────────────────
    print(f"\n{'=' * 60}")
    print("  Built-in Gini Feature Importances")
    print(f"{'=' * 60}")
    gini_imp = model.feature_importances_
    sorted_idx = np.argsort(gini_imp)[::-1]
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"    {rank:2d}. {FEATURE_NAMES[idx]:>22s}  {gini_imp[idx]:.4f}")


if __name__ == "__main__":
    main()
