#!/usr/bin/env python3
"""
feature_importance.py — Permutation Feature Importance for MLP and Random Forest.

Computes permutation importance (accuracy drop) for each of the 14 geometric
features by independently shuffling that feature in both polygon inputs.

Outputs:
    output/feature_importance_scores.csv   — tabular importance values
    output/feature_importance.pdf           — grouped bar chart (Figure 9)

Usage:
    python feature_importance.py --city berlin
    python feature_importance.py --city berlin --skip-rf
    python feature_importance.py --city berlin --plot-only  # reuse saved CSV
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import joblib

# Non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.collections import PatchCollection       # noqa: E402

from featurebased.PolygonMLP import PolygonPairClassifier

# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

SEED       = 42
BATCH_SIZE = 2048
THRESHOLD  = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Paths – adjust to your repo layout
MODEL_DIR   = Path("featurebased/checkpoints")
DATA_DIR    = Path("data")
OUTPUT_DIR  = MODEL_DIR / Path("output")

CONFIG_PATH  = MODEL_DIR / "mlp_config.yaml"
WEIGHTS_PATH = MODEL_DIR / "mlp_model.pt"
RF_PATH      = MODEL_DIR / "rf_model.joblib"       # optional

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
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def set_all_seeds(seed: int = SEED):
    """Fix every source of randomness."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_test_data(city: str):
    """Return (poly1, poly2, y_true) as numpy arrays."""
    X = np.load(DATA_DIR / f"{city}_X_pairs_dataset.npy")
    y = np.load(DATA_DIR / f"{city}_y_pairs_dataset.npy")
    return X[0], X[1], np.array(y)


def load_mlp_model() -> torch.nn.Module:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f).get("model", {})
    model = PolygonPairClassifier(**cfg)
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    return model.eval().to(DEVICE)


# ═════════════════════════════════════════════════════════════════════════════
# Prediction wrappers
# ═════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def predict_mlp(model, poly1, poly2):
    """Return binary predictions from the MLP."""
    preds = []
    for i in range(0, len(poly1), BATCH_SIZE):
        p1 = torch.tensor(poly1[i:i + BATCH_SIZE], dtype=torch.float32, device=DEVICE)
        p2 = torch.tensor(poly2[i:i + BATCH_SIZE], dtype=torch.float32, device=DEVICE)
        logits = model(p1, p2)
        preds.append((torch.sigmoid(logits).squeeze() > THRESHOLD).cpu().numpy())
    return np.concatenate(preds)


def predict_rf(rf_model, poly1, poly2):
    """Return binary predictions from the Random Forest."""
    X = np.concatenate([poly1, poly2], axis=1)
    return rf_model.predict(X)


# ═════════════════════════════════════════════════════════════════════════════
# Permutation importance (generic)
# ═════════════════════════════════════════════════════════════════════════════

def permutation_importance(predict_fn, poly1, poly2, y_true,
                           n_repeats: int = 1, seed: int = SEED):
    """
    Compute permutation feature importance via accuracy drop.

    For each feature i the column is independently shuffled in both
    poly1 and poly2.  A fixed RNG state per feature ensures exact
    reproducibility.

    Args:
        predict_fn: callable(poly1, poly2) → binary predictions
        n_repeats:  number of shuffle repeats (results are averaged)

    Returns:
        baseline_acc, importance_array (n_features,)
    """
    from sklearn.metrics import accuracy_score

    baseline_acc = accuracy_score(y_true, predict_fn(poly1, poly2))
    n_features = poly1.shape[1]
    importances = np.zeros(n_features)

    for i in range(n_features):
        drops = []
        for r in range(n_repeats):
            # Deterministic seed per (feature, repeat)
            rng = np.random.RandomState(seed + i * 1000 + r)

            p1 = copy.deepcopy(poly1)
            p2 = copy.deepcopy(poly2)
            rng.shuffle(p1[:, i])
            rng.shuffle(p2[:, i])

            acc_perm = accuracy_score(y_true, predict_fn(p1, p2))
            drops.append(baseline_acc - acc_perm)

        importances[i] = np.mean(drops)
        print(f"  Feature {i:2d} ({FEATURE_NAMES[i]:>22s}): "
              f"Δ Accuracy = {importances[i]:+.4f}")

    return baseline_acc, importances


# ═════════════════════════════════════════════════════════════════════════════
# Plotting (Figure 9)
# ═════════════════════════════════════════════════════════════════════════════

def plot_importance(df: pd.DataFrame, path: str):
    """Grouped horizontal bar chart from a DataFrame with columns
    'Feature', 'RF', 'MLP'."""

    has_rf = "RF" in df.columns and df["RF"].notna().any()

    # Sort by average importance
    if has_rf:
        df["_avg"] = (df["RF"] + df["MLP"]) / 2
    else:
        df["_avg"] = df["MLP"]
    df = df.sort_values("_avg", ascending=True).drop(columns="_avg")

    fig, ax = plt.subplots(figsize=(14, 9))
    y = np.arange(len(df))
    bar_h = 0.35 if has_rf else 0.6

    colors_mlp = plt.cm.RdPu(np.linspace(0.45, 0.85, len(df)))
    bars_mlp = ax.barh(
        y + (bar_h / 2 if has_rf else 0),
        df["MLP"].values, bar_h,
        color=colors_mlp, label="MLP",
    )

    if has_rf:
        colors_rf = plt.cm.Blues(np.linspace(0.45, 0.85, len(df)))
        bars_rf = ax.barh(
            y - bar_h / 2,
            df["RF"].values, bar_h,
            color=colors_rf, label="Random Forest",
        )

    # Value labels
    x_max = df[["MLP"] + (["RF"] if has_rf else [])].max().max()
    for idx, row in df.reset_index(drop=True).iterrows():
        ax.text(row["MLP"] + x_max * 0.01,
                idx + (bar_h / 2 if has_rf else 0),
                f'{row["MLP"]:.2f} %', va="center", fontsize=11)
        if has_rf:
            ax.text(row["RF"] + x_max * 0.01,
                    idx - bar_h / 2,
                    f'{row["RF"]:.2f} %', va="center", fontsize=11)

    ax.set_yticks(y)
    ax.set_yticklabels(df["Feature"].values, fontsize=13)
    ax.set_xlabel("Accuracy Drop (Permutation Importance) [%]", fontsize=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Permutation Feature Importance")
    parser.add_argument("--city", type=str, default="berlin")
    parser.add_argument("--skip-rf", action="store_true",
                        help="Skip Random Forest (e.g. if model file unavailable)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Re-generate plot from existing CSV")
    parser.add_argument("--n-repeats", type=int, default=1,
                        help="Shuffle repeats per feature (default: 1)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path  = OUTPUT_DIR / "feature_importance_scores.csv"
    plot_path = OUTPUT_DIR / "feature_importance.pdf"

    # ── Plot-only mode ──────────────────────────────────────────────────────
    if args.plot_only:
        df = pd.read_csv(csv_path)
        print(f"Loaded scores from {csv_path}")
        print(df.to_string(index=False))
        plot_importance(df, str(plot_path))
        return

    # ── Full computation ────────────────────────────────────────────────────
    set_all_seeds(SEED)
    print(f"Device : {DEVICE}")
    print(f"Seed   : {SEED}")
    print(f"City   : {args.city}\n")

    poly1, poly2, y_true = load_test_data(args.city)
    print(f"Test samples: {len(y_true):,}")

    results = {"Feature": FEATURE_NAMES}

    # ── MLP ─────────────────────────────────────────────────────────────────
    print("\n── MLP Permutation Importance ─────────────────────────────")
    model = load_mlp_model()
    mlp_base, mlp_imp = permutation_importance(
        lambda p1, p2: predict_mlp(model, p1, p2),
        poly1, poly2, y_true,
        n_repeats=args.n_repeats,
    )
    print(f"\n  Baseline Accuracy: {mlp_base:.4f}")
    results["MLP"] = mlp_imp * 100      # → percent

    # ── RF (optional) ───────────────────────────────────────────────────────
    if not args.skip_rf and RF_PATH.exists():
        print("\n── Random Forest Permutation Importance ───────────────────")
        rf = joblib.load(RF_PATH)
        rf_base, rf_imp = permutation_importance(
            lambda p1, p2: predict_rf(rf, p1, p2),
            poly1, poly2, y_true,
            n_repeats=args.n_repeats,
        )
        print(f"\n  Baseline Accuracy: {rf_base:.4f}")
        results["RF"] = rf_imp * 100
    else:
        print("\n  Skipping RF (--skip-rf or model not found).")

    # ── Save table ──────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"  Scores saved → {csv_path}")
    print(f"{'=' * 60}")
    print(df.to_string(index=False))

    # ── Plot ────────────────────────────────────────────────────────────────
    plot_importance(df, str(plot_path))


if __name__ == "__main__":
    main()
