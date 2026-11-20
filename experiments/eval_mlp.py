"""Evaluate every MLP run on the validation split of all nine evaluation sets.

Writes results/mlp_metrics.csv, consumed by plot_results.py to produce Figure 11
(cross-task matrix, E1) and Figure 13 (hard/easy/uniform distributions, E3)."""
import argparse
import re

import numpy as np
import pandas as pd
import torch
import yaml

from experiments.common import (CITY, EVAL_SETS, REPO, TASKS, binary_metrics, data_dir,
                                load_subset, results_dir, runs_dir, subsets_dir)
from featurebased.PolygonMLP import PolygonPairClassifier

BATCH = 4096


def find_runs(pretrained):
    runs = []
    for path in sorted((runs_dir() / "mlp").glob("*_s*")):
        match = re.fullmatch(r"(.+)_s(\d+)", path.name)
        if match and (path / "model.pt").exists():
            runs.append((match.group(1), int(match.group(2)), path / "model.pt", path / "config.yaml"))
    if pretrained:
        runs.append(("pretrained_full", 0, pretrained, REPO / "featurebased/checkpoints/mlp_config.yaml"))
    return runs


def load_model(weights, config_path):
    with open(config_path) as f:
        model = PolygonPairClassifier(**yaml.safe_load(f)["model"])
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval()


@torch.inference_mode()
def predict_proba(model, p1, p2):
    out = []
    for i in range(0, len(p1), BATCH):
        logits = model(torch.from_numpy(p1[i:i + BATCH]), torch.from_numpy(p2[i:i + BATCH]))
        out.append(torch.sigmoid(logits).view(-1).numpy())
    return np.concatenate(out)


def load_eval_sets():
    X = np.load(data_dir() / f"{CITY}_X_pairs_dataset.npy", mmap_mode="r")
    y_all = np.load(data_dir() / f"{CITY}_y_pairs_dataset.npy")
    row_task = np.load(subsets_dir() / "row_task.npy")
    sets = {}
    for name in EVAL_SETS:
        idx = np.sort(load_subset(name)["val_idx"])
        block = np.asarray(X[:, idx], dtype=np.float32)
        sets[name] = {"p1": block[0], "p2": block[1], "y": y_all[idx], "task": row_task[idx]}
    return sets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default=None, help="optional published MLP checkpoint")
    args = parser.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    sets = load_eval_sets()
    rows = []
    for train_set, seed, weights, config in find_runs(args.pretrained):
        model = load_model(weights, config)
        for eval_name, data in sets.items():
            prob = predict_proba(model, data["p1"], data["p2"])
            row = {"model": train_set, "seed": seed, "eval_set": eval_name,
                   **binary_metrics(data["y"], prob, 0.5)}
            for code, task in enumerate(TASKS):
                mask = data["task"] == code
                row[f"tnr_{task}"] = float((prob[mask] < 0.5).mean()) if mask.any() else float("nan")
            rows.append(row)
        print("evaluated", train_set, seed, flush=True)

    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "mlp_metrics.csv", index=False)
    print("wrote", out / "mlp_metrics.csv")


if __name__ == "__main__":
    main()
