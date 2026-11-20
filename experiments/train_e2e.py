"""Train one Perceiver run with a single negative task (or the reference mix).

Used for the paper's ablation studies:
  --dataset <task>           Figures 11 & 13 (cross-task generalisation / distribution shifts)
  --dataset ref --variant …  Table 6 (positional-encoding ablation, PE_VARIANTS below)
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import yaml

from experiments.common import (CITY, E2E_STRATEGY, REPO, TRAIN_SETS, data_dir, runs_dir,
                                seed_everything, write_json)

sys.path.insert(0, str(REPO / "end2end"))

from end2end.helper.architectures.perceiver import PolygonPerceiver  # noqa: E402
from end2end.helper.dataset import get_dataloader  # noqa: E402
from end2end.helper.trainer import EmbeddingTrainer  # noqa: E402

RANDOM_POOL_SIZE = 52586

# Positional-encoding study (Table 6): standard training (published config, no random-pair
# pool limit). Variants reported in the paper; "concat_linear" is the published configuration.
PE_VARIANTS = {
    "concat_linear": {"pos_fusion": "concat",    "pe_mode": "linear", "d_pos_enc": 16},
    "add_linear64":  {"pos_fusion": "add_fixed", "pe_mode": "linear", "d_pos_enc": 64},
    "none":          {"pos_fusion": "none",      "pe_mode": "linear", "d_pos_enc": 16},
}


def build_config(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    d = data_dir()
    config["seed"] = args.seed
    config["tqdm_loader"] = False
    config["calculate_accuracy"] = False
    config["calculate_threshold"] = True
    config["num_epochs"] = args.epochs if args.epochs is not None else config["num_epochs"]
    config["early_stopping"]["patience"] = (
        args.patience if args.patience is not None else config["early_stopping"]["patience"])
    config["dataset"].update({
        "parquet_path": str(d / f"{CITY}_polygons.parquet"),
        "hard_candidates_path": str(d / f"{CITY}_hard_negative_candidates.json"),
        "intersection_path": str(d / f"{CITY}_intersections_pairs.csv"),
        "num_workers": args.num_workers,
        "random_pool_size": RANDOM_POOL_SIZE,
    })
    if args.num_workers == 0:
        config["dataset"]["prefetch_factor"] = None
    if args.variant:
        config["perceiver_encoder"].update(PE_VARIANTS[args.variant])
        config["dataset"]["random_pool_size"] = None
    if args.dataset != "ref":
        strategies = {name: 0.0 for name in E2E_STRATEGY.values()}
        strategies[E2E_STRATEGY[args.dataset]] = 1.0
        config["dataset"]["negative_strategies"] = strategies
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=TRAIN_SETS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", default=str(REPO / "end2end/checkpoints/perceiver_config.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None,
                        help="default: patience from the config; set >= epochs for a fixed step budget")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--variant", default=None, choices=tuple(PE_VARIANTS),
                        help="positional-encoding study, Table 6 (requires --dataset ref)")
    args = parser.parse_args()
    if args.variant and args.dataset != "ref":
        parser.error("--variant is only defined for --dataset ref")

    config = build_config(args)
    if args.out_dir:
        out = Path(args.out_dir)
    elif args.variant:
        out = runs_dir() / "pe_study" / f"{args.variant}_s{args.seed}"
    else:
        out = runs_dir() / "e2e" / f"{args.dataset}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloader(config, model_mode="sequence")
    config["dataset"]["train_len"] = len(train_loader)
    encoder = PolygonPerceiver(**config["perceiver_encoder"]).to(device)
    trainer = EmbeddingTrainer(encoder, config=config, logger=_Log(), acc_at_k_loader=None,
                               model_mode="sequence", device=device)

    best, stale, history = -float("inf"), 0, []
    start = time.time()
    for epoch in range(1, config["num_epochs"] + 1):
        loss = trainer.train_epoch(train_loader)
        val = trainer.evaluate(dataloader=val_loader)
        row = {"epoch": epoch, "train_loss": loss, **val}
        history.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
              flush=True)
        score = val.get("f1_at_best_threshold", 0.0)
        if score > best:
            best, stale = score, 0
            torch.save(trainer.encoder.state_dict(), out / "encoder.pt")
        else:
            stale += 1
            if stale >= config["early_stopping"]["patience"]:
                break

    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(copy.deepcopy(config), f)
    write_json(history, out / "history.json")
    write_json({"dataset": args.dataset, "seed": args.seed, "best_val_f1": best,
                "epochs_run": len(history), "steps_per_epoch": len(train_loader),
                "optimizer_steps": len(history) * len(train_loader),
                "n_parameters": sum(p.numel() for p in encoder.parameters()),
                "seconds": time.time() - start}, out / "run.json")


class _Log:
    def info(self, *_):
        pass


if __name__ == "__main__":
    main()
