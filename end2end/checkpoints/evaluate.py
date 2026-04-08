#!/usr/bin/env python3
"""
evaluate.py — Reproducible evaluation pipeline for polygon encoder models.

Modes (mutually exclusive):
  --compute-thresholds   Compute optimal F1 thresholds on a city dataset
                         and save them to thresholds.json.
  --evaluate-f1          Compute F1/Precision/Recall on a (possibly new) city
                         dataset using previously saved thresholds.
  --plot-invariance      Generate translation / rotation / scale figures
                         using previously saved thresholds.

Common options:
  --city <name>          City name (must have all_geoms_<city>.joblib and
                         <city>_idx_parameter.joblib in data/).
  --polygon-idx <int>    Index of the reference polygon for invariance tests.

Usage examples:
    # 1) Compute new thresholds on Berlin data
    python evaluate.py --compute-thresholds --city berlin

    # 2) Evaluate F1 on a new city with existing thresholds
    python evaluate.py --evaluate-f1 --city munich

    # 3) Generate invariance figures for all models
    python evaluate.py --plot-invariance --city berlin --polygon-idx 42

    # 4) Combine: compute thresholds AND plot invariance
    python evaluate.py --compute-thresholds --plot-invariance --city berlin
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from itertools import product
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
import yaml

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

from shapely.affinity import rotate, scale, translate
from shapely import Polygon, MultiPolygon
from shapely.geometry import box
from tqdm import tqdm

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

# Local project modules
import data.helper_main as hf
from end2end.helper.architectures.graph import GraphPolygonEncoder
from end2end.helper.architectures.perceiver import PolygonPerceiver
from end2end.helper.helper_architecture import CyclicRelativePosEncoding


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Paths -------------------------------------------------------------------
MODEL_DIR       = Path("end2end/checkpoints")
DATA_DIR        = Path("data")
OUTPUT_DIR      = MODEL_DIR / Path("output")
THRESHOLDS_FILE = DATA_DIR / "thresholds.json"

# -- Inference ----------------------------------------------------------------
BATCH_SIZE  = 256
NUM_WORKERS = 0

# -- Translation grid ---------------------------------------------------------
TRANSLATION_CELLS = 50

# -- Rotation -----------------------------------------------------------------
ROTATION_STEP   = 5  # degrees
ROTATION_ANGLES = np.arange(0, 360, ROTATION_STEP)

# -- Scale --------------------------------------------------------------------
SCALE_STEP  = 0.1
SCALE_RANGE = np.arange(0.2, 1.8 + SCALE_STEP, SCALE_STEP)
SCALE_X, SCALE_Y = np.meshgrid(SCALE_RANGE, SCALE_RANGE)

# -- Figures ------------------------------------------------------------------
FIGURE_DPI = 300
CMAP_NAME  = "RdYlGn"

# ============================================================================
# Set Random Seeds
# ============================================================================

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ═══════════════════════════════════════════════════════════════════════════════
# Model registry
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelSpec:
    """Holds paths, config, encoder, and evaluation threshold for one model."""
    name: str
    model_type: str                                            # "graph" | "sequence"
    config_path: str
    weights_path: Optional[str]          = None
    config: Dict[str, Any]               = field(default_factory=dict, repr=False)
    encoder: Optional[torch.nn.Module]   = field(default=None, repr=False)
    threshold: Optional[float]           = field(default=None)

    @property
    def has_weights(self) -> bool:
        return self.weights_path is not None and os.path.isfile(self.weights_path)

    @property
    def has_config(self) -> bool:
        return os.path.isfile(self.config_path)


MODEL_REGISTRY: List[ModelSpec] = [
    ModelSpec("GATv2",          "graph",    str(MODEL_DIR / "gatv2_config.yaml"),
                                             str(MODEL_DIR / "gatv2_model.pt")),
    ModelSpec("GINE",           "graph",    str(MODEL_DIR / "gine_config.yaml"),
                                             str(MODEL_DIR / "gine_model.pt")),
    ModelSpec("MessagePassing", "graph",    str(MODEL_DIR / "mp_config.yaml"),
                                             str(MODEL_DIR / "mp_model.pt")),
    ModelSpec("MPCA",            "graph",    str(MODEL_DIR / "mpca_config.yaml"),
                                              str(MODEL_DIR / "mpca_model.pt")),
    ModelSpec("Perceiver",       "sequence", str(MODEL_DIR / "perceiver_config.yaml"),
                                              str(MODEL_DIR / "perceiver_model.pt")),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Core utilities
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def build_encoder(config: dict, model_type: str) -> torch.nn.Module:
    if model_type == "graph":
        return GraphPolygonEncoder(**config.get("graph_encoder", config))
    if model_type == "sequence":
        return PolygonPerceiver(**config.get("perceiver_encoder", config))
    raise ValueError(f"Unknown model_type: {model_type}")


def load_weights(encoder: torch.nn.Module, path: str) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    encoder.load_state_dict(ckpt)
    return encoder


def model_summary(encoder: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in encoder.parameters())
    train = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    return {"total": total, "trainable": train,
            "size_mb": round(total * 4 / 1024**2, 2)}


def get_pos_encoder(spec: ModelSpec) -> CyclicRelativePosEncoding:
    """Create the CyclicRelativePosEncoding that matches a model's config."""
    key = "graph_encoder" if spec.model_type == "graph" else "perceiver_encoder"
    sub = spec.config.get(key, spec.config)
    return CyclicRelativePosEncoding(d_pos_enc=sub.get("d_pos_enc", 16))


def ensure_polygon(geom) -> Polygon:
    """Return the largest polygon from a MultiPolygon, or the polygon itself."""
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return geom


def make_threshold_norm(threshold: float) -> mcolors.Normalize:
    """Diverging colour norm centred on the optimal threshold.

    Colour-scale mapping (RdYlGn):
        0 %   ->  vmin = 2*threshold - 1   (red)
        50 %  ->  vcenter = threshold       (yellow)
        100 % ->  vmax = 1.0                (green)
    """
    vmin = 2.0 * threshold - 1.0
    if vmin >= threshold or threshold >= 1.0:
        return mcolors.Normalize(vmin=max(vmin, -1.0), vmax=1.0)
    return mcolors.TwoSlopeNorm(vcenter=threshold, vmin=max(vmin, -1.0), vmax=1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Graph helpers
# ═══════════════════════════════════════════════════════════════════════════════

def polygon_to_graph(coords: torch.Tensor,
                     pos_encoder: CyclicRelativePosEncoding = None) -> Data:
    """Convert Nx2 coordinate tensor to a PyG Data object with ring topology."""
    n = coords.size(0)
    src = torch.arange(n, dtype=torch.long)
    edge_index = to_undirected(
        torch.stack([src, torch.roll(src, -1)], dim=0), num_nodes=n
    )
    data = Data(pos=coords, edge_index=edge_index)
    if pos_encoder is not None:
        with torch.no_grad():
            data.cyclic_pe = pos_encoder(coords.unsqueeze(0)).squeeze(0)
    return data


def prepare_graph_batch(tensors: List[torch.Tensor], device: torch.device,
                        pos_encoder=None) -> Batch:
    """Build a batched PyG Batch from a list of coordinate tensors."""
    return Batch.from_data_list(
        [polygon_to_graph(t, pos_encoder) for t in tensors]
    ).to(device)


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence helpers
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_sequence_batch(tensors: List[torch.Tensor], device: torch.device,
                           pos_encoder=None):
    """Pad, mask, and compute cyclic PE for a list of coordinate tensors."""
    lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long)
    padded = pad_sequence(tensors, batch_first=True).to(device)
    mask = (torch.arange(padded.shape[1], device=device)[None, :]
            >= lengths[:, None].to(device))
    pe_padded = None
    if pos_encoder is not None:
        with torch.no_grad():
            pes = [pos_encoder(t.unsqueeze(0)).squeeze(0) for t in tensors]
        pe_padded = pad_sequence(pes, batch_first=True).to(device)
    return padded, pe_padded, mask


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset & collation
# ═══════════════════════════════════════════════════════════════════════════════

class PairDataset(Dataset):
    """Benchmark dataset of polygon pairs with match/non-match labels."""

    def __init__(self, data_path: str, idx_path: str,
                 model_mode: str = "graph", pos_encoder=None):
        self.all_data   = joblib.load(data_path)
        self.idx_list   = joblib.load(idx_path)
        self.model_mode = model_mode
        self.pos_encoder = pos_encoder

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        row = self.idx_list.iloc[idx]
        p1 = self.all_data.iloc[row.idx_pair_1].geometry
        p2 = self.all_data.iloc[row.idx_pair_2].geometry
        target = row.method == "positiv"

        if row.manipulation:
            p1 = ensure_polygon(hf.apply_manipulation(p1, row.parameter[0]))
            p2 = ensure_polygon(hf.apply_manipulation(p2, row.parameter[1]))

        c1 = torch.tensor(np.array(p1.exterior.coords), dtype=torch.float32)
        c2 = torch.tensor(np.array(p2.exterior.coords), dtype=torch.float32)

        if self.model_mode == "graph":
            c1 = polygon_to_graph(c1, pos_encoder=self.pos_encoder)
            c2 = polygon_to_graph(c2, pos_encoder=self.pos_encoder)

        return c1, c2, torch.tensor(target, dtype=torch.long)


def collate_graph_pairs(batch):
    """Collate function for graph-based models (PyG Batch)."""
    g1, g2, tgt = zip(*batch)
    return {"poly1": Batch.from_data_list(g1),
            "poly2": Batch.from_data_list(g2),
            "target": torch.stack(tgt)}


class SequencePairCollator:
    """Collate for sequence models; pre-computes PE in DataLoader workers."""

    def __init__(self, pos_encoder=None):
        self.pos_encoder = pos_encoder

    def __call__(self, batch):
        p1_list, p2_list, tgt = zip(*batch)

        len1 = torch.tensor([p.size(0) for p in p1_list], dtype=torch.long)
        len2 = torch.tensor([p.size(0) for p in p2_list], dtype=torch.long)

        p1_pad = pad_sequence(p1_list, batch_first=True, padding_value=0.0)
        p2_pad = pad_sequence(p2_list, batch_first=True, padding_value=0.0)

        m1 = torch.arange(p1_pad.size(1)).unsqueeze(0) >= len1.unsqueeze(1)
        m2 = torch.arange(p2_pad.size(1)).unsqueeze(0) >= len2.unsqueeze(1)

        out = {"poly1": p1_pad, "poly2": p2_pad,
               "poly1_mask": m1, "poly2_mask": m2,
               "target": torch.tensor(tgt, dtype=torch.long)}

        if self.pos_encoder is not None:
            with torch.no_grad():
                pe1 = [self.pos_encoder(p.unsqueeze(0)).squeeze(0) for p in p1_list]
                pe2 = [self.pos_encoder(p.unsqueeze(0)).squeeze(0) for p in p2_list]
            out["poly1_pe"] = pad_sequence(pe1, batch_first=True)
            out["poly2_pe"] = pad_sequence(pe2, batch_first=True)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# Inference helpers
# ═══════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def compute_all_similarities(encoder, model_type, dataloader, device,
                             pos_encoder=None):
    """Cosine similarity for every pair in *dataloader*."""
    encoder.eval()
    all_tgt, all_sim = [], []

    for batch in tqdm(dataloader, desc="  Inference", unit="batch"):
        if model_type == "graph":
            emb1 = encoder(batch["poly1"].to(device))
            emb2 = encoder(batch["poly2"].to(device))
        else:
            p1 = batch["poly1"].to(device)
            p2 = batch["poly2"].to(device)
            m1 = batch["poly1_mask"].to(device)
            m2 = batch["poly2_mask"].to(device)

            if "poly1_pe" in batch:
                pe1 = batch["poly1_pe"].to(device)
                pe2 = batch["poly2_pe"].to(device)
            elif pos_encoder is not None:
                l1 = (~m1).sum(1)
                l2 = (~m2).sum(1)
                pe1 = pad_sequence(
                    [pos_encoder(p1[i, :l1[i]].unsqueeze(0)).squeeze(0)
                     for i in range(p1.size(0))], batch_first=True).to(device)
                pe2 = pad_sequence(
                    [pos_encoder(p2[i, :l2[i]].unsqueeze(0)).squeeze(0)
                     for i in range(p2.size(0))], batch_first=True).to(device)
            else:
                pe1 = pe2 = None

            emb1 = encoder(p1, pe1, m1)
            emb2 = encoder(p2, pe2, m2)

        sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1)
        all_sim.append(sim.cpu())
        all_tgt.append(batch["target"])

    return torch.cat(all_tgt).numpy(), torch.cat(all_sim).numpy()


def compute_optimal_f1(y_true, similarities):
    """Optimal F1 threshold via the precision-recall curve."""
    prec, rec, thr = precision_recall_curve(y_true, similarities)
    denom = np.where(prec + rec == 0, 1.0, prec + rec)
    f1s = 2 * prec * rec / denom
    best = int(np.argmax(f1s))
    t = float(thr[best])
    yp = (similarities >= t).astype(int)
    return {"threshold": t, "f1": float(f1s[best]),
            "precision": float(precision_score(y_true, yp)),
            "recall":    float(recall_score(y_true, yp))}


# ── Embedding helpers for a single original polygon vs. many variants ────────

def _embed_original(encoder, model_type, geom_wgs84, device, pos_encoder):
    """Return the embedding of the unmodified reference polygon."""
    org = torch.tensor(np.array(geom_wgs84.exterior.coords), dtype=torch.float32)
    if model_type == "graph":
        batch = Batch.from_data_list(
            [polygon_to_graph(org, pos_encoder)]).to(device)
        return encoder(batch).cpu()
    seq = org.unsqueeze(0).to(device)
    mask = torch.zeros(seq.shape[:2], dtype=torch.bool, device=device)
    pe = pos_encoder(org.unsqueeze(0)).to(device) if pos_encoder else None
    return encoder(seq, pe, mask).cpu()


def _embed_variants(encoder, model_type, tensors, device, pos_encoder,
                    batch_size=BATCH_SIZE):
    """Embed a list of polygon-coordinate tensors in batches."""
    embs = []
    for i in range(0, len(tensors), batch_size):
        chunk = tensors[i:i + batch_size]
        if model_type == "graph":
            embs.append(
                encoder(prepare_graph_batch(chunk, device, pos_encoder)).cpu())
        else:
            p, pe, m = prepare_sequence_batch(chunk, device, pos_encoder)
            embs.append(encoder(p, pe, m).cpu())
    return torch.cat(embs, dim=0)


@torch.inference_mode()
def compute_translation_scores(encoder, model_type, geom_wgs84, shifted_gdf,
                               device, pos_encoder=None):
    """Cosine similarity: original vs. every translated polygon."""
    encoder.eval()
    wgs = shifted_gdf.to_crs(4326)
    tensors = [torch.tensor(np.array(g.exterior.coords), dtype=torch.float32)
               for g in wgs.geometry]
    org_emb = _embed_original(encoder, model_type, geom_wgs84, device, pos_encoder)
    var_emb = _embed_variants(encoder, model_type, tensors, device, pos_encoder)
    return torch.nn.functional.cosine_similarity(var_emb, org_emb, dim=1).numpy()


@torch.inference_mode()
def compute_rotation_scores(encoder, model_type, geom_wgs84, device,
                            pos_encoder=None, angles=ROTATION_ANGLES):
    """Cosine similarity: original vs. every rotated polygon."""
    encoder.eval()
    polys = [rotate(geom_wgs84, float(a), origin="centroid") for a in angles]
    tensors = [torch.tensor(np.array(p.exterior.coords), dtype=torch.float32)
               for p in polys]
    org_emb = _embed_original(encoder, model_type, geom_wgs84, device, pos_encoder)
    var_emb = _embed_variants(encoder, model_type, tensors, device, pos_encoder)
    return torch.nn.functional.cosine_similarity(var_emb, org_emb, dim=1).numpy()


@torch.inference_mode()
def compute_scale_scores(encoder, model_type, geom_wgs84, device,
                         pos_encoder=None):
    """Cosine similarity: original vs. every (x, y)-scaled polygon."""
    encoder.eval()
    xf, yf = SCALE_X.ravel(), SCALE_Y.ravel()
    polys = [scale(geom_wgs84, xfact=float(x), yfact=float(y), origin="center")
             for x, y in zip(xf, yf)]
    tensors = [torch.tensor(np.array(p.exterior.coords), dtype=torch.float32)
               for p in polys]
    org_emb = _embed_original(encoder, model_type, geom_wgs84, device, pos_encoder)
    var_emb = _embed_variants(encoder, model_type, tensors, device, pos_encoder)
    return torch.nn.functional.cosine_similarity(var_emb, org_emb, dim=1).numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting (all figures saved as PNG — no interactive display)
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig, path):
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_translation(name, scores, geom_proj, shifted_gdf, x_step, y_step,
                     norm, path):
    """Translation-invariance heatmap."""
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
    cbar.set_label("Cosine Similarity", fontsize=14)
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
    cbar.set_label("Cosine Similarity", fontsize=13)
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
    cbar.set_label("Cosine Similarity", fontsize=16)

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


def _stats(name, scores, extras=None):
    print(f"  [{name}]  range=[{scores.min():.4f}, {scores.max():.4f}]  "
          f"mean={scores.mean():.4f}  std={scores.std():.4f}")
    for k, v in (extras or {}).items():
        print(f"    {k}: {v}")


# ═══════════════════════════════════════════════════════════════════════════════
# Threshold persistence
# ═══════════════════════════════════════════════════════════════════════════════

def save_thresholds(results: Dict[str, dict], path: Path):
    """Persist per-model threshold + metrics to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  Thresholds written to {path}")


def load_thresholds(path: Path) -> Dict[str, dict]:
    """Load previously computed thresholds from JSON."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Run --compute-thresholds first.")
    with open(path, "r") as fh:
        data = json.load(fh)
    print(f"  Thresholds loaded from {path}")
    return data


def apply_thresholds_to_specs(loaded: Dict[str, ModelSpec],
                              eval_results: Dict[str, dict]):
    """Assign saved thresholds to each loaded ModelSpec; drop models without one."""
    for name, spec in list(loaded.items()):
        if name in eval_results:
            spec.threshold = eval_results[name]["threshold"]
            print(f"    {name}: threshold={spec.threshold:.6f}  "
                  f"F1={eval_results[name]['f1']:.4f}")
        else:
            print(f"    {name}: no threshold in JSON — removed.")
            del loaded[name]


# ═══════════════════════════════════════════════════════════════════════════════
# CLI argument parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible evaluation pipeline for polygon encoder models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute new thresholds on Berlin data
  python evaluate.py --compute-thresholds --city berlin

  # Evaluate F1 on a new city using existing thresholds
  python evaluate.py --evaluate-f1 --city munich

  # Generate invariance figures only
  python evaluate.py --plot-invariance --city berlin --polygon-idx 42

  # Compute thresholds AND generate figures in one run
  python evaluate.py --compute-thresholds --plot-invariance --city berlin
        """,
    )

    # ── Mode flags (can be combined) ────────────────────────────────────────
    parser.add_argument(
        "--compute-thresholds",
        action="store_true",
        default=False,
        help="Compute optimal F1 thresholds on the city dataset and save "
             "them to thresholds.json.",
    )
    parser.add_argument(
        "--evaluate-f1",
        action="store_true",
        default=False,
        help="Compute F1/Precision/Recall on the city dataset using "
             "previously saved (or just computed) thresholds.",
    )
    parser.add_argument(
        "--plot-invariance",
        action="store_true",
        default=False,
        help="Generate translation / rotation / scale invariance figures.",
    )

    # ── Common options ──────────────────────────────────────────────────────
    parser.add_argument(
        "--city",
        type=str,
        default="berlin",
        help="City name. Expects data/all_geoms_<city>.joblib and "
             "data/<city>_idx_parameter.joblib  (default: berlin).",
    )
    parser.add_argument(
        "--polygon-idx",
        type=int,
        default=None,
        help="Index of the reference polygon for invariance tests. "
             "Random if omitted.",
    )

    args = parser.parse_args()

    # If the user passes no mode flag at all, show help and exit.
    if not (args.compute_thresholds or args.evaluate_f1 or args.plot_invariance):
        parser.print_help()
        parser.exit(
            1, "\nError: specify at least one of --compute-thresholds, "
               "--evaluate-f1, --plot-invariance.\n")

    return args


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline building blocks
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_models() -> Dict[str, ModelSpec]:
    """Load configs + weights for every model in the registry."""
    _header("LOADING MODELS")
    loaded: Dict[str, ModelSpec] = {}

    for spec in MODEL_REGISTRY:
        print(f"\n  {spec.name} ({spec.model_type})")
        if not spec.has_config:
            print(f"    Config missing: {spec.config_path} — skipped.")
            continue

        spec.config  = load_config(spec.config_path)
        spec.encoder = build_encoder(spec.config, spec.model_type)
        s = model_summary(spec.encoder)
        print(f"    Params : {s['total']:,} ({s['trainable']:,} trainable, "
              f"~{s['size_mb']:.2f} MB)")

        if not spec.has_weights:
            print(f"    Weights missing — skipped.")
            continue

        spec.encoder = load_weights(spec.encoder, spec.weights_path)
        spec.encoder.eval().to(DEVICE)
        print(f"    Weights loaded  ->  eval mode on {DEVICE}")
        loaded[spec.name] = spec

    print(f"\n  Models ready: {len(loaded)}/{len(MODEL_REGISTRY)}")
    return loaded


def run_compute_thresholds(loaded: Dict[str, ModelSpec],
                           city_data: Path, city_idx: Path) -> Dict[str, dict]:
    """Mode 1: compute optimal F1 thresholds on the given city dataset."""
    _header("COMPUTING OPTIMAL F1 THRESHOLDS")
    eval_results: Dict[str, dict] = {}

    for name, spec in loaded.items():
        print(f"\n  {name}")
        pe = get_pos_encoder(spec)

        ds = PairDataset(
            str(city_data), str(city_idx),
            model_mode=spec.model_type,
            pos_encoder=pe if spec.model_type == "graph" else None,
        )
        collate = (collate_graph_pairs if spec.model_type == "graph"
                   else SequencePairCollator(pos_encoder=pe))
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate, num_workers=NUM_WORKERS,
                        pin_memory=(DEVICE.type == "cuda"),
                        persistent_workers=(NUM_WORKERS > 0))

        print(f"    Pairs: {len(ds)}")
        t0 = time.time()
        y_true, sims = compute_all_similarities(
            spec.encoder, spec.model_type, dl, DEVICE,
            pos_encoder=pe if spec.model_type == "sequence" else None)
        dt = time.time() - t0
        print(f"    Time : {dt:.1f} s  ({len(ds) / dt:.0f} pairs/s)")

        m = compute_optimal_f1(y_true, sims)
        eval_results[name] = m
        spec.threshold = m["threshold"]

        print(f"    Threshold : {m['threshold']:.6f}")
        print(f"    F1        : {m['f1']:.4f}")
        print(f"    Precision : {m['precision']:.4f}")
        print(f"    Recall    : {m['recall']:.4f}")
        del ds, dl

    # Save to disk
    save_thresholds(eval_results, THRESHOLDS_FILE)

    if eval_results:
        best = max(eval_results, key=lambda k: eval_results[k]["f1"])
        print(f"  Best model: {best}  "
              f"(F1 = {eval_results[best]['f1'] * 100:.2f} %)")

    return eval_results


def run_evaluate_f1(loaded: Dict[str, ModelSpec],
                    city_data: Path, city_idx: Path):
    """Mode 2: evaluate F1 on a (possibly new) city using saved thresholds."""
    _header("F1 EVALUATION WITH SAVED THRESHOLDS")

    for name, spec in loaded.items():
        if spec.threshold is None:
            print(f"  {name}: no threshold available — skipped.")
            continue

        print(f"\n  {name}  (threshold={spec.threshold:.6f})")
        pe = get_pos_encoder(spec)

        ds = PairDataset(
            str(city_data), str(city_idx),
            model_mode=spec.model_type,
            pos_encoder=pe if spec.model_type == "graph" else None,
        )
        collate = (collate_graph_pairs if spec.model_type == "graph"
                   else SequencePairCollator(pos_encoder=pe))
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate, num_workers=NUM_WORKERS,
                        pin_memory=(DEVICE.type == "cuda"),
                        persistent_workers=(NUM_WORKERS > 0))

        print(f"    Pairs: {len(ds)}")
        t0 = time.time()
        y_true, sims = compute_all_similarities(
            spec.encoder, spec.model_type, dl, DEVICE,
            pos_encoder=pe if spec.model_type == "sequence" else None)
        dt = time.time() - t0
        print(f"    Time : {dt:.1f} s  ({len(ds) / dt:.0f} pairs/s)")

        # Metrics with the fixed threshold
        y_pred = (sims >= spec.threshold).astype(int)
        prec = float(precision_score(y_true, y_pred))
        rec  = float(recall_score(y_true, y_pred))
        denom = prec + rec if (prec + rec) > 0 else 1.0
        f1 = 2 * prec * rec / denom

        print(f"    F1        : {f1:.4f}")
        print(f"    Precision : {prec:.4f}")
        print(f"    Recall    : {rec:.4f}")

        # Also report what the optimal threshold would be on THIS dataset
        opt = compute_optimal_f1(y_true, sims)
        print(f"    (Optimal on this data: threshold={opt['threshold']:.6f}  "
              f"F1={opt['f1']:.4f})")

        del ds, dl


def run_plot_invariance(loaded: Dict[str, ModelSpec],
                        city_data: Path, polygon_idx: Optional[int]):
    """Mode 3: translation / rotation / scale invariance figures."""

    # ── Load reference polygon ──────────────────────────────────────────────
    _header("REFERENCE POLYGON")
    all_geoms = joblib.load(city_data)
    orig_crs  = all_geoms.crs
    print(f"  Dataset CRS  : {orig_crs}")
    print(f"  Total polys  : {len(all_geoms)}")

    idx = (polygon_idx if polygon_idx is not None
           else np.random.randint(0, len(all_geoms)))
    geom_raw = ensure_polygon(all_geoms.iloc[idx].geometry)
    print(f"  Selected idx : {idx}")
    print(f"  Vertices     : {len(geom_raw.exterior.coords)}")

    # CRS variants
    if orig_crs is None:
        print("  Warning: no CRS — assuming EPSG:4326.")
        geom_wgs  = geom_raw
        geom_proj = (gpd.GeoDataFrame(geometry=[geom_raw], crs=4326)
                         .to_crs(25832).geometry.iloc[0])
    elif orig_crs.to_epsg() == 4326:
        geom_wgs  = geom_raw
        geom_proj = (gpd.GeoDataFrame(geometry=[geom_raw], crs=4326)
                         .to_crs(25832).geometry.iloc[0])
    else:
        geom_proj = (gpd.GeoDataFrame(geometry=[geom_raw], crs=orig_crs)
                         .to_crs(25832).geometry.iloc[0])
        geom_wgs  = (gpd.GeoDataFrame(geometry=[geom_raw], crs=orig_crs)
                         .to_crs(4326).geometry.iloc[0])

    print(f"  Area (proj)  : {geom_proj.area:.1f} m²")

    # ── Translation grid (metres, EPSG:25832) ──────────────────────────────
    minx, miny, maxx, maxy = geom_proj.bounds
    half = round(max(maxx - minx, maxy - miny) / 2) * 2
    xr = np.linspace(-half, half + 1, TRANSLATION_CELLS)
    yr = np.linspace(-half, half + 1, TRANSLATION_CELLS)
    xs, ys = xr[1] - xr[0], yr[1] - yr[0]

    shifted = [translate(geom_proj, xoff=dx, yoff=dy)
               for dx, dy in product(xr, yr)]
    shifted.append(geom_proj)
    shifted_gdf = gpd.GeoDataFrame(geometry=shifted, crs=25832)
    print(f"  Grid         : {TRANSLATION_CELLS}x{TRANSLATION_CELLS} "
          f"(step {xs:.1f} m)")

    # ── Translation ─────────────────────────────────────────────────────────
    _header("TRANSLATION INVARIANCE")
    for name, spec in loaded.items():
        if spec.threshold is None:
            print(f"  {name}: no threshold — skipped."); continue
        pe = get_pos_encoder(spec)
        print(f"\n  {name} ... ", end="", flush=True)
        t0 = time.time()
        sc = compute_translation_scores(spec.encoder, spec.model_type,
                                        geom_wgs, shifted_gdf, DEVICE,
                                        pos_encoder=pe)
        print(f"{time.time() - t0:.1f} s")
        _stats(name, sc)
        norm = make_threshold_norm(spec.threshold)
        p = OUTPUT_DIR / f"{name}_translation.png"
        plot_translation(name, sc, geom_proj, shifted_gdf, xs, ys, norm, str(p))
        print(f"    -> {p}")

    # ── Rotation ────────────────────────────────────────────────────────────
    _header("ROTATION INVARIANCE")
    for name, spec in loaded.items():
        if spec.threshold is None:
            print(f"  {name}: no threshold — skipped."); continue
        pe = get_pos_encoder(spec)
        print(f"\n  {name} ... ", end="", flush=True)
        t0 = time.time()
        sc = compute_rotation_scores(spec.encoder, spec.model_type,
                                     geom_wgs, DEVICE, pos_encoder=pe)
        print(f"{time.time() - t0:.1f} s")
        _stats(name, sc, {
            "Score at 0 deg": f"{sc[0]:.4f}",
            "Worst angle":
                f"{ROTATION_ANGLES[sc.argmin()]} deg ({sc.min():.4f})"})
        norm = make_threshold_norm(spec.threshold)
        p = OUTPUT_DIR / f"{name}_rotation.png"
        plot_rotation(name, sc, norm, str(p))
        print(f"    -> {p}")

    # ── Scale ───────────────────────────────────────────────────────────────
    _header("SCALE INVARIANCE")
    for name, spec in loaded.items():
        if spec.threshold is None:
            print(f"  {name}: no threshold — skipped."); continue
        pe = get_pos_encoder(spec)
        print(f"\n  {name} ... ", end="", flush=True)
        t0 = time.time()
        sc = compute_scale_scores(spec.encoder, spec.model_type,
                                  geom_wgs, DEVICE, pos_encoder=pe)
        print(f"{time.time() - t0:.1f} s")
        grid = sc.reshape(SCALE_X.shape)
        ci = int(np.argmin(np.abs(SCALE_RANGE - 1.0)))
        diag = np.diag(grid)
        wi = int(diag.argmin())
        _stats(name, sc, {
            "Score at (1,1)": f"{grid[ci, ci]:.4f}",
            "Worst uniform":
                f"{SCALE_RANGE[wi]:.1f}x ({diag[wi]:.4f})"})
        norm = make_threshold_norm(spec.threshold)
        p = OUTPUT_DIR / f"{name}_scale.png"
        plot_scale(name, sc, norm, str(p))
        print(f"    -> {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()

    # ── Resolve city paths ──────────────────────────────────────────────────
    city      = args.city
    city_data = DATA_DIR / f"all_geoms_{city}.joblib"
    city_idx  = DATA_DIR / f"{city}_idx_parameter.joblib"

    print(f"  City         : {city}")
    print(f"  Data file    : {city_data}")
    print(f"  Index file   : {city_idx}")
    print(f"  Device       : {DEVICE}")
    print(f"  PyTorch      : {torch.__version__}")
    print(f"  Output dir   : {OUTPUT_DIR.resolve()}")

    modes = []
    if args.compute_thresholds: modes.append("compute-thresholds")
    if args.evaluate_f1:        modes.append("evaluate-f1")
    if args.plot_invariance:    modes.append("plot-invariance")
    print(f"  Modes        : {', '.join(modes)}")

    # ── Load all models once ────────────────────────────────────────────────
    loaded = load_all_models()
    if not loaded:
        print("  No models available. Exiting.")
        return

    # ── Threshold handling ──────────────────────────────────────────────────
    eval_results: Dict[str, dict] = {}

    if args.compute_thresholds:
        # Mode 1: compute fresh thresholds on this city
        eval_results = run_compute_thresholds(loaded, city_data, city_idx)
    else:
        # Modes 2 & 3 need existing thresholds
        _header("LOADING SAVED THRESHOLDS")
        try:
            eval_results = load_thresholds(THRESHOLDS_FILE)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            return
        apply_thresholds_to_specs(loaded, eval_results)
        if not loaded:
            print("  No models with valid thresholds. Exiting.")
            return

    # ── Mode 2: evaluate F1 on (possibly new) city ─────────────────────────
    if args.evaluate_f1:
        run_evaluate_f1(loaded, city_data, city_idx)

    # ── Mode 3: invariance figures ──────────────────────────────────────────
    if args.plot_invariance:
        run_plot_invariance(loaded, city_data, args.polygon_idx)

    # ── Done ────────────────────────────────────────────────────────────────
    _header("FINISHED")
    print(f"  Total time : {time.time() - t_start:.1f} s")
    print(f"  Figures in : {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
