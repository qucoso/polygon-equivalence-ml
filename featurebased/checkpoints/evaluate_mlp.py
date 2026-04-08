#!/usr/bin/env python3
"""
evaluate_mlp.py — Invariance & F1 evaluation for the feature-based MLP polygon classifier.

Supports two evaluation modes:

  1. **Invariance evaluation** (default, uses Berlin dataset):
     Randomly samples one polygon and evaluates translation, rotation, and
     scale invariance, producing publication-ready figures.

  2. **City F1 evaluation** (--city <name> --f1):
     Loads the pre-scaled paired dataset for the given city from
         data/<city>_X_pairs_dataset.npy
         data/<city>_y_pairs_dataset.npy
     and computes the F1 score (plus precision, recall, accuracy) on the
     full dataset.

Output figures are written to:
    featurebased/checkpoints/output/<city>/MLP_translation.png
    featurebased/checkpoints/output/<city>/MLP_rotation.png
    featurebased/checkpoints/output/<city>/MLP_scale.png

Usage examples:
    # Invariance evaluation on Berlin (default)
    python evaluate_mlp.py [--polygon-idx 123]

    # Invariance evaluation on Kampala
    python evaluate_mlp.py --city kampala [--polygon-idx 42]

    # F1 score on the full Kampala paired dataset
    python evaluate_mlp.py --city kampala --f1

    # Both invariance plots AND F1 score for Kampala
    python evaluate_mlp.py --city kampala --f1 --polygon-idx 42
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import time
import argparse
from pathlib import Path
from itertools import product

import numpy as np
import geopandas as gpd
import joblib
import yaml
import torch

from shapely import Polygon, MultiPolygon
from shapely.affinity import rotate, scale, translate
from shapely.geometry import box
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Non-interactive backend — must precede pyplot import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import matplotlib.cm as cm                               # noqa: E402
import matplotlib.colors as mcolors                      # noqa: E402
import matplotlib.patches as mpatches                    # noqa: E402
import matplotlib.ticker as ticker                       # noqa: E402
from matplotlib.patches import Polygon as MplPolygon     # noqa: E402
from matplotlib.patches import Wedge                     # noqa: E402
from matplotlib.collections import PatchCollection       # noqa: E402
from matplotlib_scalebar.scalebar import ScaleBar        # noqa: E402

import data.helper_main as hf
from featurebased.PolygonMLP import PolygonPairClassifier


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5          # Fixed sigmoid decision boundary

# -- Paths -------------------------------------------------------------------
MODEL_DIR    = Path("featurebased/checkpoints")
DATA_DIR     = Path("data")
BASE_OUTPUT  = MODEL_DIR / Path("output")
SCALER_PATH  = DATA_DIR / "scaler.joblib"
CONFIG_PATH  = MODEL_DIR / "mlp_config.yaml"
WEIGHTS_PATH = MODEL_DIR / "mlp_model.pt"

# -- Inference ----------------------------------------------------------------
BATCH_SIZE = 4096        # MLP is lightweight; large batches are fine

# -- Translation grid ---------------------------------------------------------
TRANSLATION_CELLS = 50

# -- Rotation -----------------------------------------------------------------
ROTATION_STEP   = 5      # degrees
ROTATION_ANGLES = np.arange(0, 360, ROTATION_STEP)

# -- Scale --------------------------------------------------------------------
SCALE_STEP  = 0.1
SCALE_RANGE = np.arange(0.2, 1.8 + SCALE_STEP, SCALE_STEP)
SCALE_X, SCALE_Y = np.meshgrid(SCALE_RANGE, SCALE_RANGE)

# -- Figures ------------------------------------------------------------------
FIGURE_DPI = 300
CMAP_NAME  = "RdYlGn"

# -- City dataset file templates ----------------------------------------------
CITY_X_TEMPLATE = "data/{city}_X_pairs_dataset.npy"
CITY_Y_TEMPLATE = "data/{city}_y_pairs_dataset.npy"

# -- Default city geometry files ----------------------------------------------
CITY_GEOM_TEMPLATE = "data/all_geoms_{city}.joblib"


# ═══════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_polygon(geom) -> Polygon:
    """Return the largest polygon from a MultiPolygon, or the polygon itself."""
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return geom


def extract_features_batch(polygons: list, scaler) -> np.ndarray:
    """Compute and scale the 14-dim feature vector for each polygon.

    Delegates to hf.process_geometry() — the SAME extraction function used
    during training — so the feature order matches the fitted scaler exactly:

        0  area              7  elong (PCA)
        1  length            8  sin_angle (PCA)
        2  width  (bbox)     9  cos_angle (PCA)
        3  height (bbox)    10  convex_ratio
        4  area / length    11  circularity
        5  centroid_x       12  n_nodes
        6  centroid_y       13  polygon_roughness
    """
    raw = np.stack([hf.process_geometry(p) for p in polygons])

    # Guard against NaN from degenerate polygons
    nan_mask = np.isnan(raw)
    if nan_mask.any():
        print(f"  WARNING: {nan_mask.any(axis=1).sum()} polygons produced NaN "
              f"features — filled with 0.")
        raw = np.nan_to_num(raw, nan=0.0)

    return scaler.transform(raw).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def mlp_predict_scores(model: torch.nn.Module,
                       feat_orig: np.ndarray,
                       feat_variants: np.ndarray,
                       device: torch.device = DEVICE,
                       batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Sigmoid match probability for (original, variant) feature pairs.

    Args:
        feat_orig:     (14,) scaled feature vector of the reference polygon.
        feat_variants: (N, 14) scaled feature vectors of variant polygons.

    Returns:
        (N,) numpy array of match probabilities in [0, 1].
    """
    model.eval()
    n = len(feat_variants)
    orig_rep = np.tile(feat_orig, (n, 1))      # repeat original for pairing

    scores = []
    for i in range(0, n, batch_size):
        p1 = torch.from_numpy(orig_rep[i:i + batch_size]).to(device)
        p2 = torch.from_numpy(feat_variants[i:i + batch_size]).to(device)
        logits = model(p1, p2)
        scores.append(torch.sigmoid(logits).squeeze(-1).cpu().numpy())
    return np.concatenate(scores)


@torch.inference_mode()
def mlp_predict_pairs(model: torch.nn.Module,
                      poly1_feats: np.ndarray,
                      poly2_feats: np.ndarray,
                      device: torch.device = DEVICE,
                      batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Predict match probabilities for pre-scaled polygon feature pairs.

    Args:
        model:       Trained PolygonPairClassifier.
        poly1_feats: (N, D) array — features for the first polygon in each pair,
                     already scaled.
        poly2_feats: (N, D) array — features for the second polygon in each pair,
                     already scaled.
        device:      Torch device.
        batch_size:  Inference batch size.

    Returns:
        (N,) numpy array of sigmoid match probabilities in [0, 1].
    """
    model.eval()
    n = poly1_feats.shape[0]

    scores = []
    for i in range(0, n, batch_size):
        p1 = torch.from_numpy(poly1_feats[i:i + batch_size].astype(np.float32)).to(device)
        p2 = torch.from_numpy(poly2_feats[i:i + batch_size].astype(np.float32)).to(device)
        logits = model(p1, p2)
        scores.append(torch.sigmoid(logits).squeeze(-1).cpu().numpy())
    return np.concatenate(scores)



# ═══════════════════════════════════════════════════════════════════════════════
# F1-score evaluation on city paired dataset
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_city_f1(model: torch.nn.Module,
                     city: str,
                     device: torch.device = DEVICE,
                     threshold: float = THRESHOLD) -> dict:
    """Load the pre-scaled paired dataset for *city* and compute F1 metrics.

    Expected files (already scaled, ready for inference):
        data/<city>_X_pairs_dataset.npy   — shape (2, N, D)  where X[0] = poly1, X[1] = poly2
        data/<city>_y_pairs_dataset.npy   — shape (N,)

    Args:
        model:     Trained PolygonPairClassifier.
        city:      City name (lowercase), e.g. "kampala".
        device:    Torch device.
        threshold: Decision boundary for converting probabilities to labels.

    Returns:
        Dictionary with keys: f1, precision, recall, accuracy, n_samples,
        n_positive, n_negative.
    """
    x_path = Path(CITY_X_TEMPLATE.format(city=city))
    y_path = Path(CITY_Y_TEMPLATE.format(city=city))

    if not x_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {x_path}. "
            f"Expected pre-scaled paired features for city '{city}'.")
    if not y_path.exists():
        raise FileNotFoundError(
            f"Label file not found: {y_path}. "
            f"Expected binary labels for city '{city}'.")

    X = np.load(str(x_path))
    y = np.load(str(y_path))

    # X is expected to have shape (2, N, D) where X[0] = poly1, X[1] = poly2
    if X.ndim == 3 and X.shape[0] == 2:
        poly1_feats = X[0]
        poly2_feats = X[1]
    else:
        raise ValueError(
            f"Unexpected X shape {X.shape}. Expected (2, N, D) where "
            f"X[0] contains poly1 features and X[1] contains poly2 features.")

    print(f"  Feature matrix : {x_path}  shape={X.shape}")
    print(f"  Label vector   : {y_path}  shape={y.shape}")
    print(f"  Pair count     : {poly1_feats.shape[0]:,}")
    print(f"  Feature dim    : {poly1_feats.shape[1]}")
    print(f"  Positive pairs : {int(y.sum()):,}  "
          f"({y.sum() / len(y) * 100:.1f} %)")
    print(f"  Negative pairs : {int((1 - y).sum()):,}  "
          f"({(1 - y).sum() / len(y) * 100:.1f} %)")

    # Run inference with separate poly1 / poly2 arrays
    probs = mlp_predict_pairs(model, poly1_feats, poly2_feats, device)
    preds = (probs >= threshold).astype(int)

    # Compute metrics
    metrics = {
        "f1":         f1_score(y, preds),
        "precision":  precision_score(y, preds, zero_division=0),
        "recall":     recall_score(y, preds, zero_division=0),
        "accuracy":   accuracy_score(y, preds),
        "n_samples":  len(y),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
    }
    return metrics



# ═══════════════════════════════════════════════════════════════════════════════
# Invariance computations
# ═══════════════════════════════════════════════════════════════════════════════

def compute_translation_scores(model, scaler, geom_orig, shifted_gdf,
                               orig_crs, device=DEVICE):
    """Match probability: original vs. every translated polygon."""
    shifted_native = shifted_gdf.to_crs(orig_crs)
    variants = [ensure_polygon(g) for g in shifted_native.geometry]

    feat_orig = extract_features_batch([geom_orig], scaler)[0]
    feat_vars = extract_features_batch(variants, scaler)
    return mlp_predict_scores(model, feat_orig, feat_vars, device)


def compute_rotation_scores(model, scaler, geom_orig, device=DEVICE,
                            angles=ROTATION_ANGLES):
    """Match probability: original vs. every rotated polygon."""
    variants = [rotate(geom_orig, float(a), origin="centroid") for a in angles]

    feat_orig = extract_features_batch([geom_orig], scaler)[0]
    feat_vars = extract_features_batch(variants, scaler)
    return mlp_predict_scores(model, feat_orig, feat_vars, device)


def compute_scale_scores(model, scaler, geom_orig, device=DEVICE):
    """Match probability: original vs. every (x, y)-scaled polygon."""
    xf, yf = SCALE_X.ravel(), SCALE_Y.ravel()
    variants = [scale(geom_orig, xfact=float(x), yfact=float(y), origin="center")
                for x, y in zip(xf, yf)]

    feat_orig = extract_features_batch([geom_orig], scaler)[0]
    feat_vars = extract_features_batch(variants, scaler)
    return mlp_predict_scores(model, feat_orig, feat_vars, device)


# ═══════════════════════════════════════════════════════════════════════════════
# Color-scale norm (threshold at 50 % of the colour range)
# ═══════════════════════════════════════════════════════════════════════════════

def make_threshold_norm(threshold: float = THRESHOLD) -> mcolors.TwoSlopeNorm:
    """Diverging norm: vmin = 2·t − 1 | vcenter = t (50 %) | vmax = 1.0.

    For the default threshold 0.5 this yields [0.0, 0.5, 1.0].
    """
    vmin = max(2.0 * threshold - 1.0, 0.0)
    return mcolors.TwoSlopeNorm(vcenter=threshold, vmin=vmin, vmax=1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting helpers (all PNG, no interactive display)
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig, path: str):
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_translation(name, scores, geom_proj, shifted_gdf, x_step, y_step,
                     norm, path):
    """Translation-invariance heatmap with reference polygon overlay."""
    gdf = shifted_gdf.copy()
    gdf["score"] = scores
    gdf["bbox"] = gdf.geometry.apply(
        lambda g: box(g.centroid.x - x_step / 2, g.centroid.y - y_step / 2,
                      g.centroid.x + x_step / 2, g.centroid.y + y_step / 2))

    cmap = plt.colormaps.get_cmap(CMAP_NAME)
    fig, ax = plt.subplots(figsize=(10, 10))

    patches, colours = [], []
    for _, row in gdf.iterrows():
        patches.append(MplPolygon(list(row["bbox"].exterior.coords), closed=True))
        colours.append(row["score"])

    pc = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor="none")
    pc.set_array(np.array(colours))
    pc.set_zorder(0)
    ax.add_collection(pc)

    # Reference polygon (filled + outline)
    ax.add_patch(MplPolygon(list(geom_proj.exterior.coords), closed=True,
                            facecolor="black", alpha=0.2, zorder=5))
    ax.add_patch(MplPolygon(list(geom_proj.exterior.coords), closed=True,
                            facecolor="none", edgecolor="black", lw=2, zorder=6))

    bounds = np.array([g.bounds for g in gdf["bbox"]])
    ax.set_xlim(bounds[:, 0].min(), bounds[:, 2].max())
    ax.set_ylim(bounds[:, 1].min(), bounds[:, 3].max())
    ax.set_aspect("equal")
    ax.tick_params(length=0, labelbottom=False, labelleft=False)

    grid_len = ax.get_xticks()[1] - ax.get_xticks()[0]
    ax.add_artist(ScaleBar(
        1, units="m", location="lower right", frameon=False,
        height_fraction=0.01, border_pad=1, fixed_value=grid_len,
        color="black", font_properties={"size": 18}, scale_loc="top"))

    cbar = plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Match Probability", fontsize=14)
    ax.grid(True, color="white", lw=0.8, alpha=0.5, zorder=10)
    ax.set_title(name, fontsize=20, fontweight="bold", pad=15)
    _save(fig, path)


def plot_rotation(name, scores, norm, path,
                  angles=ROTATION_ANGLES, step=ROTATION_STEP,
                  tolerance_deg=30.0):
    """Rotation-invariance polar wedge diagram."""
    cmap = plt.colormaps.get_cmap(CMAP_NAME)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"aspect": "equal"})
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.axis("off")

    for a, s in zip(angles, scores):
        ax.add_patch(Wedge((0, 0), 1, a, a + step, width=1,
                           facecolor=cmap(norm(s)), edgecolor="none"))

    if tolerance_deg is not None:
        ax.add_patch(Wedge((0, 0), 1, 360 - tolerance_deg, tolerance_deg,
                           width=1, fill=False, edgecolor="black",
                           linestyle="--", lw=2))

    for a in (0, 90, 180, 270):
        r = np.deg2rad(a)
        ax.text(1.2 * np.cos(r), 1.2 * np.sin(r), f"{a}\u00b0",
                ha="center", va="center", fontsize=18)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Match Probability", fontsize=13)
    ax.set_title(name, fontsize=20, fontweight="bold", pad=20)
    _save(fig, path)


def plot_scale(name, scores, norm, path,
               tolerance_rect=(0.8, 0.8, 0.4, 0.4)):
    """Scale-invariance 2-D heatmap (x-factor vs. y-factor)."""
    grid = scores.reshape(SCALE_X.shape)
    fig, ax = plt.subplots(figsize=(10, 9))

    mesh = ax.pcolormesh(SCALE_X, SCALE_Y, grid, shading="auto",
                         cmap=CMAP_NAME, norm=norm)

    cbar = plt.colorbar(mesh, ax=ax)
    cbar.locator = ticker.MaxNLocator(nbins=5); cbar.update_ticks()
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("Match Probability", fontsize=16)

    if tolerance_rect:
        rx, ry, rw, rh = tolerance_rect
        ax.add_patch(mpatches.Rectangle(
            (rx, ry), rw, rh, lw=2, edgecolor="black",
            facecolor="none", linestyle="--"))

    ax.plot(1.0, 1.0, "+k", markersize=15, markeredgewidth=2, zorder=10)

    ticks = np.arange(0.5, 1.8, 0.5)
    ax.set_xticks(ticks); ax.set_xticklabels([f"{v:.1f}" for v in ticks], fontsize=18)
    ax.set_yticks(ticks); ax.set_yticklabels([f"{v:.1f}" for v in ticks], fontsize=18)
    ax.set_xlim(SCALE_RANGE.min(), SCALE_RANGE.max())
    ax.set_ylim(SCALE_RANGE.min(), SCALE_RANGE.max())
    ax.set_aspect("equal")
    ax.set_xlabel("Scale Factor X", fontsize=16)
    ax.set_ylabel("Scale Factor Y", fontsize=16)
    ax.grid(True, color="white", lw=0.8, alpha=0.5, zorder=5)
    ax.set_title(name, fontsize=20, fontweight="bold", pad=15)
    _save(fig, path)


# ═══════════════════════════════════════════════════════════════════════════════
# Console formatting
# ═══════════════════════════════════════════════════════════════════════════════

def _header(text: str):
    print(f"\n{'=' * 72}\n  {text}\n{'=' * 72}")


def _stats(name: str, scores: np.ndarray, extras: dict = None):
    print(f"  [{name}]  range=[{scores.min():.4f}, {scores.max():.4f}]  "
          f"mean={scores.mean():.4f}  std={scores.std():.4f}")
    for k, v in (extras or {}).items():
        print(f"    {k}: {v}")


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading helper
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(device: torch.device = DEVICE):
    """Load the MLP model and return (model, config_dict)."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    model_cfg = config.get("model", config)

    model = PolygonPairClassifier(**model_cfg)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Architecture : PolygonPairClassifier")
    print(f"  Parameters   : {total:,}  (~{total * 4 / 1024**2:.2f} MB)")

    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    model.eval().to(device)
    print(f"  Weights      : {WEIGHTS_PATH}")
    return model, model_cfg


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Feature-based MLP invariance & F1 evaluation")
    parser.add_argument("--polygon-idx", type=int, default=None,
                        help="Polygon index for invariance tests (random if omitted)")
    parser.add_argument("--city", type=str, default="berlin",
                        help="City name (lowercase). Determines geometry file "
                             "and paired-dataset paths. Default: berlin")
    parser.add_argument("--f1", action="store_true",
                        help="Compute F1 score on the full city paired dataset "
                             "(data/<city>_X_pairs_dataset.npy)")
    parser.add_argument("--skip-invariance", action="store_true",
                        help="Skip invariance plots (useful when only --f1 is needed)")
    args = parser.parse_args()

    city = args.city.lower()

    t_start = time.time()

    # City-specific output directory
    output_dir = BASE_OUTPUT / city
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device     : {DEVICE}")
    print(f"Threshold  : {THRESHOLD}  (fixed sigmoid boundary)")
    print(f"City       : {city}")
    print(f"Output dir : {output_dir.resolve()}")

    # ── 1. Load model & scaler ──────────────────────────────────────────────
    _header("LOADING MODEL")
    model, _ = load_model(DEVICE)

    scaler = hf.CustomFeatureScaler.load(str(SCALER_PATH))
    print(f"  Scaler       : {SCALER_PATH}")

    # ── 2. F1 evaluation on city paired dataset (if requested) ──────────────
    if args.f1:
        _header(f"F1 EVALUATION — {city.upper()}")
        t0 = time.time()
        metrics = evaluate_city_f1(model, city, DEVICE, THRESHOLD)
        elapsed = time.time() - t0

        print(f"\n  ┌─────────────────────────────────────────┐")
        print(f"  │  City       : {city:<27s}│")
        print(f"  │  Samples    : {metrics['n_samples']:<27,}│")
        print(f"  │  Positive   : {metrics['n_positive']:<27,}│")
        print(f"  │  Negative   : {metrics['n_negative']:<27,}│")
        print(f"  │  Threshold  : {THRESHOLD:<27.2f}│")
        print(f"  ├─────────────────────────────────────────┤")
        print(f"  │  F1 Score   : {metrics['f1']:<27.4f}│")
        print(f"  │  Precision  : {metrics['precision']:<27.4f}│")
        print(f"  │  Recall     : {metrics['recall']:<27.4f}│")
        print(f"  │  Accuracy   : {metrics['accuracy']:<27.4f}│")
        print(f"  └─────────────────────────────────────────┘")
        print(f"  Time: {elapsed:.1f} s")

    # ── 3. Invariance evaluation (unless skipped) ───────────────────────────
    if args.skip_invariance:
        if not args.f1:
            print("\n  Nothing to do: --skip-invariance without --f1.")
    else:
        # ── 3a. Load city geometry data ─────────────────────────────────────
        _header("REFERENCE POLYGON")

        geom_path = Path(CITY_GEOM_TEMPLATE.format(city=city))
        if not geom_path.exists():
            raise FileNotFoundError(
                f"Geometry file not found: {geom_path}. "
                f"Expected a joblib file with polygon geometries for '{city}'.")

        city_data = joblib.load(geom_path)
        orig_crs  = city_data.crs
        print(f"  Geometry file: {geom_path}")
        print(f"  Dataset CRS  : {orig_crs}")
        print(f"  Total polys  : {len(city_data)}")

        idx = args.polygon_idx if args.polygon_idx is not None \
            else np.random.randint(0, len(city_data))
        geom = ensure_polygon(city_data.iloc[idx].geometry)
        print(f"  Selected idx : {idx}")
        print(f"  Vertices     : {len(geom.exterior.coords)}")
        print(f"  Area         : {geom.area:.6f}")

        # Projected version for translation grid (metres)
        geom_proj = (gpd.GeoDataFrame(geometry=[geom], crs=orig_crs)
                     .to_crs(25832).geometry.iloc[0])

        # ── 3b. Build translation grid ──────────────────────────────────────
        minx, miny, maxx, maxy = geom_proj.bounds
        half = round(max(maxx - minx, maxy - miny) / 2) * 2
        xr = np.linspace(-half, half + 1, TRANSLATION_CELLS)
        yr = np.linspace(-half, half + 1, TRANSLATION_CELLS)
        xs, ys = xr[1] - xr[0], yr[1] - yr[0]

        shifted = [translate(geom_proj, xoff=dx, yoff=dy)
                   for dx, dy in product(xr, yr)]
        shifted.append(geom_proj)  # include the original
        shifted_gdf = gpd.GeoDataFrame(geometry=shifted, crs=25832)
        print(f"  Grid         : {TRANSLATION_CELLS}x{TRANSLATION_CELLS} "
              f"(step {xs:.1f} m)")

        norm = make_threshold_norm()

        # ── 3c. Translation invariance ──────────────────────────────────────
        _header("TRANSLATION INVARIANCE")
        t0 = time.time()
        tr = compute_translation_scores(model, scaler, geom,
                                        shifted_gdf, orig_crs, DEVICE)
        print(f"  Time: {time.time() - t0:.1f} s")
        _stats("MLP", tr)
        p = output_dir / "MLP_translation.png"
        plot_translation("MLP", tr, geom_proj, shifted_gdf, xs, ys, norm, str(p))
        print(f"  -> {p}")

        # ── 3d. Rotation invariance ─────────────────────────────────────────
        _header("ROTATION INVARIANCE")
        t0 = time.time()
        rot = compute_rotation_scores(model, scaler, geom, DEVICE)
        print(f"  Time: {time.time() - t0:.1f} s")
        _stats("MLP", rot, {
            "Score at 0 deg": f"{rot[0]:.4f}",
            "Worst angle":
                f"{ROTATION_ANGLES[rot.argmin()]} deg ({rot.min():.4f})"})
        p = output_dir / "MLP_rotation.png"
        plot_rotation("MLP", rot, norm, str(p))
        print(f"  -> {p}")

        # ── 3e. Scale invariance ────────────────────────────────────────────
        _header("SCALE INVARIANCE")
        t0 = time.time()
        sc = compute_scale_scores(model, scaler, geom, DEVICE)
        print(f"  Time: {time.time() - t0:.1f} s")
        grid = sc.reshape(SCALE_X.shape)
        ci   = int(np.argmin(np.abs(SCALE_RANGE - 1.0)))
        diag = np.diag(grid)
        wi   = int(diag.argmin())
        _stats("MLP", sc, {
            "Score at (1,1)": f"{grid[ci, ci]:.4f}",
            "Worst uniform":
                f"{SCALE_RANGE[wi]:.1f}x ({diag[wi]:.4f})"})
        p = output_dir / "MLP_scale.png"
        plot_scale("MLP", sc, norm, str(p))
        print(f"  -> {p}")

    # ── Done ────────────────────────────────────────────────────────────────
    _header("FINISHED")
    elapsed_total = time.time() - t_start
    print(f"  Total time : {elapsed_total:.1f} s")
    print(f"  Output dir : {output_dir.resolve()}")


if __name__ == "__main__":
    main()
