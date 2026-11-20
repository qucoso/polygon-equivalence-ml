"""Shared constants and helpers for the ablation experiments (Figures 11–13, Table 6).

Paths can be overridden via environment variables (defaults point into the repo):
  PE_DATA_DIR      folder with the berlin_* data files      (default /data/data_files)
  PE_SUBSETS_DIR   subset .npz files written by build_subsets (default experiments/subsets)
  PE_RUNS_DIR      training-run output (checkpoints, configs)  (default experiments/runs)
  PE_RESULTS_DIR   metrics CSVs and plots                     (default experiments/results)
  PE_CACHE_DIR     polygon cache for end-to-end evaluation    (default $PE_RUNS_DIR/cache)
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
CITY = "berlin"
SEEDS = (0, 1, 2)

TASKS = ("intersection", "cluster", "modified", "random", "same_center_dif_shape")
TRAIN_SETS = TASKS + ("ref",)
EVAL_SETS = TASKS + ("ref", "hard", "easy", "uniform")

E2E_STRATEGY = {
    "modified": "modify",
    "same_center_dif_shape": "same_center_different_shape",
    "random": "random_other",
    "cluster": "cluster",
    "intersection": "intersecting",
}

FEATURE_NAMES = [
    "area", "length", "width", "height", "area_per_length", "centroid_x", "centroid_y",
    "elongation", "sin_angle", "cos_angle", "convex_ratio", "circularity", "n_nodes",
    "polygon_roughness",
]


def env_path(name, default):
    return Path(os.environ.get(name, default)).expanduser()


def data_dir():
    return env_path("PE_DATA_DIR", "/data/data_files")


def subsets_dir():
    return env_path("PE_SUBSETS_DIR", REPO / "experiments/subsets")


def runs_dir():
    return env_path("PE_RUNS_DIR", REPO / "experiments/runs")


def results_dir():
    return env_path("PE_RESULTS_DIR", REPO / "experiments/results")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def load_subset(name):
    z = np.load(subsets_dir() / f"{name}.npz")
    return {k: z[k] for k in z.files}


def binary_metrics(y_true, score, threshold):
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(score) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    auc = float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "f1": f1, "precision": precision, "recall": recall, "tnr": tnr,
        "accuracy": (tp + tn) / len(y_true), "balanced_accuracy": (recall + tnr) / 2,
        "auc": auc, "threshold": float(threshold), "n": int(len(y_true)),
    }


def cache_dir():
    return env_path("PE_CACHE_DIR", runs_dir() / "cache")
