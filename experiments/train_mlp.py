"""Train one MLP run on a subset (task or reference) with the fixed published hyperparameters.

Training runs for the paper's ablation studies (Figures 11–13): one run per negative-sample
task (and the reference mix "ref") x three seeds; evaluated by eval_mlp.py / pfi_mlp.py.
"""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from experiments.common import (CITY, REPO, TRAIN_SETS, data_dir, load_subset, runs_dir,
                                seed_everything, write_json)
from featurebased.PolygonMLP import PolygonPairClassifier
from featurebased.dataset import PolygonPairDataset
from featurebased.trainer import MLPTrainer


def make_loader(X, y, idx, batch_size, shuffle, num_workers):
    idx = np.sort(idx)
    dataset = PolygonPairDataset(y[idx], X=np.asarray(X[:, idx]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      persistent_workers=num_workers > 0, pin_memory=torch.cuda.is_available())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=TRAIN_SETS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", default=str(REPO / "featurebased/checkpoints/mlp_config.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None,
                        help="default: patience from the config; set >= epochs for a fixed step budget")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config["tqdm_loader"] = False
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    if args.patience is not None:
        config["early_stopping"]["patience"] = args.patience

    out = Path(args.out_dir) if args.out_dir else runs_dir() / "mlp" / f"{args.dataset}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = np.load(data_dir() / f"{CITY}_X_pairs_dataset.npy", mmap_mode="r")
    y = np.load(data_dir() / f"{CITY}_y_pairs_dataset.npy")
    subset = load_subset(args.dataset)
    batch_size = config["dataset"]["batch_size"]
    train_loader = make_loader(X, y, subset["train_idx"], batch_size, True, args.num_workers)
    val_loader = make_loader(X, y, subset["val_idx"], batch_size, False, args.num_workers)

    model = PolygonPairClassifier(**config["model"])
    trainer = MLPTrainer(model, config, train_loader, val_loader, logger=None, device=device)

    patience = config["early_stopping"]["patience"]
    best, stale, history = float("inf"), 0, []
    start = time.time()
    for epoch in range(1, config["num_epochs"] + 1):
        row = {"epoch": epoch, **trainer.train_epoch(), **trainer.evaluate()}
        history.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
              flush=True)
        if row["val_loss"] < best:
            best, stale = row["val_loss"], 0
            torch.save(trainer.model.state_dict(), out / "model.pt")
        else:
            stale += 1
            if stale >= patience:
                break

    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(copy.deepcopy(config), f)
    write_json(history, out / "history.json")
    write_json({"dataset": args.dataset, "seed": args.seed, "best_val_loss": best,
                "epochs_run": len(history), "steps_per_epoch": len(train_loader),
                "optimizer_steps": len(history) * len(train_loader),
                "n_train": int(len(subset["train_idx"])), "n_val": int(len(subset["val_idx"])),
                "seconds": time.time() - start}, out / "run.json")


if __name__ == "__main__":
    main()
