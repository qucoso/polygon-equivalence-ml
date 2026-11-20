"""Permutation feature importance for every MLP run on every evaluation set.

Each feature column is shuffled independently in both polygon inputs (as in the published
analysis). Importance is the mean drop in accuracy (threshold 0.5) and in AUC over repeats.

Writes results/mlp_pfi.csv, consumed by plot_results.py to produce Figure 12
(permutation feature importance of the MLP by training data, evaluated on the reference mix).
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.common import FEATURE_NAMES, results_dir
from experiments.eval_mlp import find_runs, load_eval_sets, load_model, predict_proba


def importance(model, data, n_repeats, seed):
    y = data["y"]
    base_prob = predict_proba(model, data["p1"], data["p2"])
    base_acc = float(((base_prob >= 0.5) == y).mean())
    base_auc = float(roc_auc_score(y, base_prob))
    rows = []
    for feature, name in enumerate(FEATURE_NAMES):
        acc_drop, auc_drop = [], []
        for repeat in range(n_repeats):
            rng = np.random.RandomState(seed + feature * 1000 + repeat)
            p1, p2 = data["p1"].copy(), data["p2"].copy()
            rng.shuffle(p1[:, feature])
            rng.shuffle(p2[:, feature])
            prob = predict_proba(model, p1, p2)
            acc_drop.append(base_acc - float(((prob >= 0.5) == y).mean()))
            auc_drop.append(base_auc - float(roc_auc_score(y, prob)))
        rows.append({"feature": name, "delta_accuracy": np.mean(acc_drop),
                     "delta_auc": np.mean(auc_drop), "baseline_accuracy": base_acc,
                     "baseline_auc": base_auc})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--pretrained", default=None)
    args = parser.parse_args()

    sets = load_eval_sets()
    records = []
    for train_set, seed, weights, config in find_runs(args.pretrained):
        model = load_model(weights, config)
        for eval_name, data in sets.items():
            for row in importance(model, data, args.repeats, seed=42):
                records.append({"model": train_set, "seed": seed, "eval_set": eval_name, **row})
        print("pfi done", train_set, seed, flush=True)

    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out / "mlp_pfi.csv", index=False)
    print("wrote", out / "mlp_pfi.csv")


if __name__ == "__main__":
    main()
