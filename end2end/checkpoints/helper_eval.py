# Standard library
from typing import Optional, Dict, Any

# Third-party: Core
import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
import yaml
from tqdm import tqdm

# Third-party: PyTorch
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# Third-party: PyTorch Geometric
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected

# Third-party: Scikit-learn
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

# Third-party: Matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.patches import Polygon as MplPolygon, Circle, Wedge
from matplotlib.collections import PatchCollection
from matplotlib_scalebar.scalebar import ScaleBar

# Third-party: Shapely
from shapely.affinity import rotate, scale
from shapely import Polygon, MultiPolygon
from shapely.geometry import box

# Local
import data.helper_main as hf
from end2end.helper.architectures.graph import GraphPolygonEncoder
from end2end.helper.architectures.perceiver import PolygonPerceiver
from end2end.helper.helper_architecture import CyclicRelativePosEncoding


SCALE_STEP = 0.1
SCALE_RANGE = np.arange(0.2, 1.8 + SCALE_STEP, SCALE_STEP)
SCALE_X, SCALE_Y = np.meshgrid(SCALE_RANGE, SCALE_RANGE)

ROTATION_STEP = 5  # degrees
ROTATION_ANGLES = np.arange(0, 360, ROTATION_STEP)

def load_config(config_path: str) -> dict:
    """Load a YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def build_encoder(config: dict, model_type: str) -> torch.nn.Module:
    """
    Instantiate the encoder architecture from a config dict.
    
    For graph models, expects a top-level key 'graph_encoder'.
    For sequence models, expects a top-level key 'perceiver_encoder'.
    """
    if model_type == "graph":
        encoder_config = config.get("graph_encoder", config)
        return GraphPolygonEncoder(**encoder_config)
    elif model_type == "sequence":
        encoder_config = config.get("perceiver_encoder", config)
        return PolygonPerceiver(**encoder_config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def load_weights(encoder: torch.nn.Module, weights_path: str) -> torch.nn.Module:
    """Load pre-trained weights into an encoder."""
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    
    # Handle case where checkpoint is wrapped (e.g., {"model_state_dict": ...})
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    
    encoder.load_state_dict(checkpoint)
    return encoder


def get_model_summary(encoder: torch.nn.Module) -> dict:
    """Compute model statistics."""
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    size_mb = total_params * 4 / (1024 ** 2)  # float32
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": round(size_mb, 2),
    }

class end2endSet(Dataset):
    def __init__(self, all_data_path, idx_list_path, model_mode="graph", pos_encoder=None):
        self.all_data = joblib.load(all_data_path)
        self.idx_list = joblib.load(idx_list_path)
        self.model_mode = model_mode
        self.pos_encoder = pos_encoder  # ← NEU


    def __len__(self):
        return len(self.idx_list)
    def __getitem__(self, idx):
        p1 = self.all_data.iloc[self.idx_list.iloc[idx].idx_pair_1].geometry
        p2 = self.all_data.iloc[self.idx_list.iloc[idx].idx_pair_2].geometry
        target = self.idx_list.iloc[idx].method == "positiv"
        param = self.idx_list.iloc[idx].parameter

        if self.idx_list.iloc[idx].manipulation:
            p1 = hf.apply_manipulation(p1, param[0])
            p2 = hf.apply_manipulation(p2, param[1])

            p1 = ensure_polygon(p1)
            p2 = ensure_polygon(p2)

        p1 = torch.tensor(p1.exterior.coords, dtype=torch.float32)
        p2 = torch.tensor(p2.exterior.coords, dtype=torch.float32)

        if self.model_mode == "graph":
            p1 = polygon_to_graph(p1, pos_encoder=self.pos_encoder) 
            p2 = polygon_to_graph(p2, pos_encoder=self.pos_encoder) 

        return p1, p2, torch.tensor(target, dtype=torch.long)


def collate_graph_pairs(batch: list) -> dict:
    graphs1, graphs2, target = zip(*batch)

    batch1 = Batch.from_data_list(graphs1)
    batch2 = Batch.from_data_list(graphs2)

    return {
        "poly1": batch1,
        "poly2": batch2,
        "target": torch.stack(target)
    }

def collate_pair_sequence(batch):
    poly1, poly2, target = zip(*batch)

    # Lengths für Masken berechnen
    poly1_lengths = torch.tensor([poly.size(0) for poly in poly1], dtype=torch.long)
    poly2_lengths = torch.tensor([poly.size(0) for poly in poly2], dtype=torch.long)

    # Padding anwenden
    poly1_padded = pad_sequence(poly1, batch_first=True, padding_value=0.0)
    poly2_padded = pad_sequence(poly2, batch_first=True, padding_value=0.0)

    # Masken erstellen (True für Padding-Positionen)
    max_len1 = poly1_padded.size(1)
    max_len2 = poly2_padded.size(1)

    idx_range1 = torch.arange(max_len1).unsqueeze(0)
    idx_range2 = torch.arange(max_len2).unsqueeze(0)

    poly1_mask = (idx_range1 >= poly1_lengths.unsqueeze(1))
    poly2_mask = (idx_range2 >= poly2_lengths.unsqueeze(1))

    return {
        "poly1": poly1_padded,
        "poly2": poly2_padded,
        "poly1_mask": poly1_mask,
        "poly2_mask": poly2_mask,
        "target": torch.tensor(target, dtype=torch.long)
    }

@torch.no_grad()
def compute_all_similarities(
    encoder: torch.nn.Module,
    model_type: str,
    dataloader: DataLoader,
    device: torch.device,
    pos_encoder=None,
) -> tuple:
    """
    Run inference on all pairs and return (all_targets, all_similarities).
    """
    encoder.eval()
    all_targets = []
    all_sims = []

    for batch in tqdm(dataloader, desc="Computing similarities", unit="batch"):
        targets = batch["target"]

        if model_type == "graph":
            poly1 = batch["poly1"].to(device)
            poly2 = batch["poly2"].to(device)
            emb1 = encoder(poly1)
            emb2 = encoder(poly2)
        else:
            p1 = batch["poly1"].to(device)
            p2 = batch["poly2"].to(device)
            m1 = batch["poly1_mask"].to(device)
            m2 = batch["poly2_mask"].to(device)

            # Perceiver needs cyclic_pe — compute on the fly if not in batch
            if "poly1_pe" in batch:
                pe1 = batch["poly1_pe"].to(device)
                pe2 = batch["poly2_pe"].to(device)
            elif pos_encoder is not None:
                # Compute per-sample, then pad
                pe1_list, pe2_list = [], []
                # We need unpadded lengths to compute PE correctly
                lengths1 = (~m1).sum(dim=1)
                lengths2 = (~m2).sum(dim=1)
                for i in range(p1.size(0)):
                    l1 = lengths1[i].item()
                    l2 = lengths2[i].item()
                    pe1_list.append(pos_encoder(p1[i, :l1].unsqueeze(0)).squeeze(0))
                    pe2_list.append(pos_encoder(p2[i, :l2].unsqueeze(0)).squeeze(0))
                pe1 = pad_sequence(pe1_list, batch_first=True).to(device)
                pe2 = pad_sequence(pe2_list, batch_first=True).to(device)
            else:
                pe1, pe2 = None, None

            emb1 = encoder(p1, pe1, m1)
            emb2 = encoder(p2, pe2, m2)

        sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1)
        all_sims.append(sim.cpu())
        all_targets.append(targets)

    return torch.cat(all_targets).numpy(), torch.cat(all_sims).numpy()


def compute_optimal_f1(y_true: np.ndarray, similarities: np.ndarray):
    """
    Find optimal F1 threshold from precision-recall curve.
    Returns dict with threshold, f1, precision, recall.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, similarities)
    denom = np.where(precision + recall == 0, 1.0, precision + recall)
    f1_scores = (2 * precision * recall) / denom

    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    # Compute precision/recall at optimal threshold
    y_pred = (similarities >= best_threshold).astype(int)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)

    return {
        "threshold": best_threshold,
        "f1": best_f1,
        "precision": prec,
        "recall": rec,
    }


def polygon_to_graph(coords_tensor: torch.Tensor, pos_encoder: CyclicRelativePosEncoding = None) -> Data:
    """Convert polygon coordinates to a PyG graph with ring topology."""
    num_nodes = coords_tensor.size(0)
    source = torch.arange(num_nodes, dtype=torch.long)
    target = torch.roll(source, -1)
    edge_index = to_undirected(
        torch.stack([source, target], dim=0), num_nodes=num_nodes
    )
    data = Data(pos=coords_tensor, edge_index=edge_index)

    if pos_encoder is not None:
        with torch.no_grad():
            cyclic_pe = pos_encoder(coords_tensor.unsqueeze(0)).squeeze(0)
        data.cyclic_pe = cyclic_pe

    return data


def get_pos_encoder_for_model(spec) -> CyclicRelativePosEncoding:
    """
    Extract d_pos_enc from a model's config and create the matching
    CyclicRelativePosEncoding. Works for both graph and sequence models.
    """
    config = spec.config
    if spec.model_type == "graph":
        encoder_config = config.get("graph_encoder", config)
    else:
        encoder_config = config.get("perceiver_encoder", config)

    d_pos_enc = encoder_config.get("d_pos_enc", 16)
    return CyclicRelativePosEncoding(d_pos_enc=d_pos_enc)


def prepare_batch_graph(
    polygon_tensors: list,
    device: torch.device,
    pos_encoder: CyclicRelativePosEncoding = None,
) -> Batch:
    """Build a batched PyG graph from a list of polygon coordinate tensors."""
    data_list = [polygon_to_graph(poly, pos_encoder=pos_encoder) for poly in polygon_tensors]
    return Batch.from_data_list(data_list).to(device)


def prepare_batch_sequence(
    polygon_tensors: list,
    device: torch.device,
    pos_encoder: CyclicRelativePosEncoding = None,
):
    """
    Pad polygon sequences, create attention mask, and compute cyclic PE.
    
    Returns:
        padded: [B, max_len, 2] - padded coordinate sequences
        cyclic_pe_padded: [B, max_len, d_pos_enc] - padded positional encodings
        mask: [B, max_len] - True for padding positions
    """
    lengths = torch.tensor([p.shape[0] for p in polygon_tensors], dtype=torch.long)
    padded = pad_sequence(polygon_tensors, batch_first=True).to(device)
    max_len = padded.shape[1]
    mask = (torch.arange(max_len, device=device)[None, :] >= lengths[:, None])

    # Compute cyclic positional encodings per polygon, then pad
    if pos_encoder is not None:
        cyclic_pes = []
        with torch.no_grad():
            for poly in polygon_tensors:
                pe = pos_encoder(poly.unsqueeze(0)).squeeze(0)  # [seq_len, d_pos_enc]
                cyclic_pes.append(pe)
        cyclic_pe_padded = pad_sequence(cyclic_pes, batch_first=True).to(device)
    else:
        cyclic_pe_padded = None

    return padded, cyclic_pe_padded, mask


@torch.no_grad()
def compute_translation_scores(
    encoder: torch.nn.Module,
    model_type: str,
    geom_wgs84,
    all_geoms_shifted_gdf: gpd.GeoDataFrame,
    device: torch.device,
    batch_size: int = 512,
    pos_encoder: CyclicRelativePosEncoding = None,
) -> np.ndarray:
    """
    Compute cosine similarity between the original polygon embedding
    and embeddings of all translated polygons.
    """
    encoder.eval()

    # Extract WGS84 coordinates for all shifted polygons
    shifted_wgs84 = all_geoms_shifted_gdf.to_crs(4326)
    polygon_tensors = [
        torch.tensor(np.array(poly.exterior.coords), dtype=torch.float32)
        for poly in shifted_wgs84.geometry
    ]

    # --- Original polygon embedding ---
    org_coords = torch.tensor(
        np.array(geom_wgs84.exterior.coords), dtype=torch.float32
    )

    if model_type == "graph":
        org_batch = Batch.from_data_list(
            [polygon_to_graph(org_coords, pos_encoder=pos_encoder)]
        ).to(device)
        org_emb = encoder(org_batch).cpu()
    else:
        # Perceiver: forward(x, cyclic_pe, mask)
        org_seq = org_coords.unsqueeze(0).to(device)  # [1, seq_len, 2]
        org_mask = torch.zeros(org_seq.shape[:2], dtype=torch.bool, device=device)
        if pos_encoder is not None:
            org_pe = pos_encoder(org_coords.unsqueeze(0)).to(device)  # [1, seq_len, d_pos_enc]
        else:
            org_pe = None
        org_emb = encoder(org_seq, org_pe, org_mask).cpu()

    # --- Shifted polygon embeddings (in batches) ---
    all_embs = []
    for i in range(0, len(polygon_tensors), batch_size):
        batch_tensors = polygon_tensors[i : i + batch_size]

        if model_type == "graph":
            batch_data = prepare_batch_graph(batch_tensors, device, pos_encoder=pos_encoder)
            emb = encoder(batch_data).cpu()
        else:
            padded, cyclic_pe_padded, mask = prepare_batch_sequence(
                batch_tensors, device, pos_encoder=pos_encoder
            )
            emb = encoder(padded, cyclic_pe_padded, mask).cpu()

        all_embs.append(emb)

    all_embs = torch.cat(all_embs, dim=0)

    scores = torch.nn.functional.cosine_similarity(all_embs, org_emb, dim=1)
    return scores.numpy()

def ensure_polygon(geom) -> Polygon:
    """If geom is a MultiPolygon, return the largest part by area."""
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return geom


def plot_translation_invariance(
    model_name: str,
    scores: np.ndarray,
    geom_proj,
    all_geoms_shifted_gdf: gpd.GeoDataFrame,
    x_step: float,
    y_step: float,
    vmin: float = None,
    vmax: float = None,
    figsize: tuple = (10, 10),
):
    """
    Plot translation invariance heatmap for a single model.
    """
    gdf = all_geoms_shifted_gdf.copy()
    gdf["score"] = scores

    # Fixed-size bounding boxes for grid cells
    def fixed_size_bbox(geom_):
        cx, cy = geom_.centroid.x, geom_.centroid.y
        return box(cx - x_step / 2, cy - y_step / 2, cx + x_step / 2, cy + y_step / 2)

    gdf["bbox"] = gdf["geometry"].apply(fixed_size_bbox)

    # Color normalization
    if vmin is None:
        vmin = scores.min()
    if vmax is None:
        vmax = scores.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps.get_cmap("RdYlGn")

    fig, ax = plt.subplots(figsize=figsize)

    # Grid cell patches
    patches = []
    colors_ = []
    for _, row in gdf.iterrows():
        polygon_patch = MplPolygon(list(row["bbox"].exterior.coords), closed=True)
        patches.append(polygon_patch)
        colors_.append(row["score"])

    p = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor="none")
    p.set_array(np.array(colors_))
    p.set_alpha(1)
    p.set_zorder(0)
    ax.add_collection(p)

    # Reference polygon (filled + outline)
    ax.add_patch(MplPolygon(
        list(geom_proj.exterior.coords), closed=True,
        facecolor="black", edgecolor=None, alpha=0.2, zorder=5
    ))
    ax.add_patch(MplPolygon(
        list(geom_proj.exterior.coords), closed=True,
        facecolor="none", edgecolor="black", linewidth=2, zorder=6
    ))

    # Axis setup
    all_bounds = np.array([geom.bounds for geom in gdf["bbox"]])
    ax.set_xlim(all_bounds[:, 0].min(), all_bounds[:, 2].max())
    ax.set_ylim(all_bounds[:, 1].min(), all_bounds[:, 3].max())

    len_grid = ax.get_xticks()[1] - ax.get_xticks()[0]
    ax.set_aspect("equal")
    ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)

    # Scale bar
    scalebar = ScaleBar(
        1, units="m", location="lower right", frameon=False,
        box_alpha=0.5, height_fraction=0.01, border_pad=1,
        fixed_value=len_grid, color="black",
        font_properties={"size": 18}, scale_loc="top",
    )
    ax.add_artist(scalebar)

    # Colorbar
    cbar = plt.colorbar(p, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Cosine Similarity", fontsize=14)

    ax.grid(True, which="major", color="white", linestyle="-",
            linewidth=0.8, alpha=0.5, zorder=10)

    ax.set_title(f"{model_name}", fontsize=20, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.show()
    plt.close(fig)

    # Print summary statistics
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
    print(f"  Median: {np.median(scores):.4f}")


@torch.no_grad()
def compute_rotation_scores(
    encoder: torch.nn.Module,
    model_type: str,
    geom_wgs84,
    device: torch.device,
    angles: np.ndarray = ROTATION_ANGLES,
    pos_encoder: CyclicRelativePosEncoding = None,
) -> np.ndarray:
    """
    Compute cosine similarity between the original polygon embedding
    and embeddings of the polygon rotated at each angle.
    """
    encoder.eval()

    # Generate rotated polygons (in WGS84 space)
    rotated_polys = [rotate(geom_wgs84, angle=a, origin="centroid") for a in angles]

    polygon_tensors = [
        torch.tensor(np.array(poly.exterior.coords), dtype=torch.float32)
        for poly in rotated_polys
    ]

    # --- Original polygon embedding ---
    org_coords = torch.tensor(
        np.array(geom_wgs84.exterior.coords), dtype=torch.float32
    )

    if model_type == "graph":
        org_batch = Batch.from_data_list(
            [polygon_to_graph(org_coords, pos_encoder=pos_encoder)]
        ).to(device)
        org_emb = encoder(org_batch).cpu()
    else:
        org_seq = org_coords.unsqueeze(0).to(device)
        org_mask = torch.zeros(org_seq.shape[:2], dtype=torch.bool, device=device)
        if pos_encoder is not None:
            org_pe = pos_encoder(org_coords.unsqueeze(0)).to(device)
        else:
            org_pe = None
        org_emb = encoder(org_seq, org_pe, org_mask).cpu()

    # --- Rotated polygon embeddings ---
    if model_type == "graph":
        batch_data = prepare_batch_graph(polygon_tensors, device, pos_encoder=pos_encoder)
        all_embs = encoder(batch_data).cpu()
    else:
        padded, cyclic_pe_padded, mask = prepare_batch_sequence(
            polygon_tensors, device, pos_encoder=pos_encoder
        )
        all_embs = encoder(padded, cyclic_pe_padded, mask).cpu()

    scores = torch.nn.functional.cosine_similarity(all_embs, org_emb, dim=1)
    return scores.numpy()


def plot_rotation_invariance(
    model_name: str,
    scores: np.ndarray,
    angles: np.ndarray = ROTATION_ANGLES,
    step_width: float = ROTATION_STEP,
    vmin: float = None,
    vmax: float = None,
    figsize: tuple = (8, 8),
    tolerance_deg: float = 30.0,
):
    """
    Plot rotation invariance as a polar wedge diagram for a single model.
    
    Args:
        tolerance_deg: Half-width of the "acceptable" rotation range 
                       (dashed wedge centered at 0°). Set to None to disable.
    """
    if vmin is None:
        vmin = scores.min()
    if vmax is None:
        vmax = scores.max()

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.colormaps.get_cmap("RdYlGn")

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"aspect": "equal"})
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")

    # Colored wedge segments
    for angle, score in zip(angles, scores):
        color = colormap(norm(score))
        wedge = Wedge(
            center=(0, 0), r=1,
            theta1=angle, theta2=angle + step_width,
            width=1, facecolor=color, edgecolor="none",
        )
        ax.add_patch(wedge)

    # Tolerance wedge (dashed outline)
    if tolerance_deg is not None:
        wedge_outline = Wedge(
            center=(0, 0), r=1,
            theta1=360 - tolerance_deg, theta2=tolerance_deg,
            width=1, fill=False,
            edgecolor="black", linestyle="--", linewidth=2,
        )
        ax.add_patch(wedge_outline)

    # Angle labels
    for angle in [0, 90, 180, 270]:
        rad = np.deg2rad(angle)
        x = 1.2 * np.cos(rad)
        y = 1.2 * np.sin(rad)
        ax.text(x, y, f"{angle}°", ha="center", va="center", fontsize=18)

    # Colorbar
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Cosine Similarity", fontsize=13)

    ax.set_title(f"{model_name}", fontsize=20, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.show()
    plt.close(fig)

    # Summary statistics
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
    print(f"  Score at 0°: {scores[0]:.4f}")
    # Find worst rotation angle
    worst_idx = scores.argmin()
    print(f"  Worst angle: {angles[worst_idx]}° (score={scores[worst_idx]:.4f})")


@torch.no_grad()
def compute_scale_scores(
    encoder: torch.nn.Module,
    model_type: str,
    geom_wgs84,
    device: torch.device,
    scale_x: np.ndarray = SCALE_X,
    scale_y: np.ndarray = SCALE_Y,
    batch_size: int = 512,
    pos_encoder: CyclicRelativePosEncoding = None,
) -> np.ndarray:
    """
    Compute cosine similarity between the original polygon embedding
    and embeddings of the polygon scaled by all (xfact, yfact) combinations.
    """
    encoder.eval()

    # Generate scaled polygons
    xf_flat = scale_x.reshape(-1)
    yf_flat = scale_y.reshape(-1)

    scaled_polys = [
        scale(geom_wgs84, xfact=float(xf), yfact=float(yf), origin="center")
        for xf, yf in zip(xf_flat, yf_flat)
    ]

    polygon_tensors = [
        torch.tensor(np.array(poly.exterior.coords), dtype=torch.float32)
        for poly in scaled_polys
    ]

    # --- Original polygon embedding ---
    org_coords = torch.tensor(
        np.array(geom_wgs84.exterior.coords), dtype=torch.float32
    )

    if model_type == "graph":
        org_batch = Batch.from_data_list(
            [polygon_to_graph(org_coords, pos_encoder=pos_encoder)]
        ).to(device)
        org_emb = encoder(org_batch).cpu()
    else:
        org_seq = org_coords.unsqueeze(0).to(device)
        org_mask = torch.zeros(org_seq.shape[:2], dtype=torch.bool, device=device)
        if pos_encoder is not None:
            org_pe = pos_encoder(org_coords.unsqueeze(0)).to(device)
        else:
            org_pe = None
        org_emb = encoder(org_seq, org_pe, org_mask).cpu()

    # --- Scaled polygon embeddings (in batches) ---
    all_embs = []
    for i in range(0, len(polygon_tensors), batch_size):
        batch_tensors = polygon_tensors[i : i + batch_size]

        if model_type == "graph":
            batch_data = prepare_batch_graph(batch_tensors, device, pos_encoder=pos_encoder)
            emb = encoder(batch_data).cpu()
        else:
            padded, cyclic_pe_padded, mask = prepare_batch_sequence(
                batch_tensors, device, pos_encoder=pos_encoder
            )
            emb = encoder(padded, cyclic_pe_padded, mask).cpu()

        all_embs.append(emb)

    all_embs = torch.cat(all_embs, dim=0)

    scores = torch.nn.functional.cosine_similarity(all_embs, org_emb, dim=1)
    return scores.numpy()


def plot_scale_invariance(
    model_name: str,
    scores: np.ndarray,
    scale_x: np.ndarray = SCALE_X,
    scale_y: np.ndarray = SCALE_Y,
    vmin: float = None,
    vmax: float = None,
    figsize: tuple = (10, 9),
    tolerance_rect: tuple = (0.8, 0.8, 0.4, 0.4),
):
    """
    Plot scale invariance as a 2D heatmap (x-scale vs y-scale).

    Args:
        tolerance_rect: (x, y, width, height) for the "acceptable" scale range
                        rectangle. Set to None to disable.
    """
    if vmin is None:
        vmin = scores.min()
    if vmax is None:
        vmax = scores.max()

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    prob_grid = scores.reshape(scale_x.shape)

    fig, ax = plt.subplots(figsize=figsize)

    mesh = ax.pcolormesh(
        scale_x, scale_y, prob_grid,
        shading="auto", cmap="RdYlGn", norm=norm,
    )

    # Colorbar
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.locator = ticker.MaxNLocator(nbins=5)
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("Cosine Similarity", fontsize=16)

    # Tolerance rectangle
    if tolerance_rect is not None:
        rx, ry, rw, rh = tolerance_rect
        rect = mpatches.Rectangle(
            (rx, ry), rw, rh,
            linewidth=2, edgecolor="black",
            facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)

    # Mark the identity point (1.0, 1.0)
    ax.plot(1.0, 1.0, marker="+", color="black", markersize=15, markeredgewidth=2, zorder=10)

    # Axis formatting
    tick_values = np.arange(0.5, 1.8, 0.5)
    ax.set_xticks(tick_values)
    ax.set_xticklabels([f"{v:.1f}" for v in tick_values], fontsize=18)
    ax.set_yticks(tick_values)
    ax.set_yticklabels([f"{v:.1f}" for v in tick_values], fontsize=18)

    ax.set_xlim(SCALE_RANGE.min(), SCALE_RANGE.max())
    ax.set_ylim(SCALE_RANGE.min(), SCALE_RANGE.max())
    ax.set_aspect("equal")

    ax.set_xlabel("Scale Factor X", fontsize=16)
    ax.set_ylabel("Scale Factor Y", fontsize=16)

    ax.grid(True, which="major", color="white", linestyle="-",
            linewidth=0.8, alpha=0.5, zorder=5)
    ax.set_axisbelow(False)

    ax.set_title(f"{model_name}", fontsize=20, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.show()
    plt.close(fig)

    # Summary statistics
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")

    # Score at identity (1.0, 1.0)
    center_idx = np.argmin(np.abs(SCALE_RANGE - 1.0))
    identity_score = prob_grid[center_idx, center_idx]
    print(f"  Score at (1.0, 1.0): {identity_score:.4f}")

    # Score along uniform scaling diagonal
    diag_scores = np.diag(prob_grid)
    diag_factors = SCALE_RANGE[:len(diag_scores)]
    worst_diag_idx = diag_scores.argmin()
    print(f"  Worst uniform scale: {diag_factors[worst_diag_idx]:.1f}x "
          f"(score={diag_scores[worst_diag_idx]:.4f})")