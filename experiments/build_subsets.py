"""Build the balanced index subsets shared by the MLP and the Perceiver.

Every dataset has exactly 50 % positives. Negatives follow the task proportions of the
dataset. Task and reference datasets are split 90/10 (train/val) separately per class,
the E3 distributions (hard/easy/uniform) are evaluation-only and contain val-sized sets.

These subsets are the training/evaluation data of the paper's ablation studies
(Figure 11: task x task, Figure 12: PFI on the reference mix, Figure 13: hard/easy/uniform).
"""
import argparse

import joblib
import numpy as np
import pandas as pd

from experiments.common import (CITY, EVAL_SETS, TASKS, TRAIN_SETS, data_dir,
                                subsets_dir, write_json)

E3_DISTRIBUTIONS = {
    "hard": {"intersection": 0.35, "cluster": 0.30, "same_center_dif_shape": 0.25,
             "modified": 0.05, "random": 0.05},
    "easy": {"random": 0.45, "modified": 0.45, "cluster": 0.04,
             "same_center_dif_shape": 0.03, "intersection": 0.03},
    "uniform": {t: 0.2 for t in TASKS},
}


def allocate(total, proportions):
    names = list(proportions)
    raw = np.array([proportions[n] for n in names]) * total
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts))[: total - counts.sum()]:
        counts[i] += 1
    return dict(zip(names, counts.tolist()))


def draw(rng, pools, counts):
    parts = []
    for task, n in counts.items():
        if n > len(pools[task]):
            raise ValueError(f"{task}: need {n} rows, only {len(pools[task])} available")
        parts.append(rng.choice(pools[task], size=n, replace=False))
    return np.concatenate(parts)


def mirror_partner(shuffle_idx, n_source):
    source = shuffle_idx % n_source
    order = np.argsort(source, kind="stable")
    partner = np.empty(len(source), dtype=np.int64)
    partner[order[0::2]] = order[1::2]
    partner[order[1::2]] = order[0::2]
    return partner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    d, out = data_dir(), subsets_dir()
    out.mkdir(parents=True, exist_ok=True)

    table = joblib.load(d / f"{CITY}_idx_parameter_final.joblib")
    y = np.load(d / f"{CITY}_y_pairs_dataset.npy")
    n_rows = np.load(d / f"{CITY}_X_pairs_dataset.npy", mmap_mode="r").shape[1]
    method = table["method"].to_numpy()
    assert len(table) == n_rows == len(y)
    assert ((method == "positiv") == (y == 1)).all()

    pools = {t: np.flatnonzero(method == t) for t in TASKS}
    pos_pool = np.flatnonzero(method == "positiv")
    sizes = {t: len(p) for t, p in pools.items()}
    n_neg = min(sizes.values())
    n_val = int(round(args.val_fraction * n_neg))
    print("task sizes:", sizes, "| negatives per dataset:", n_neg, "| val per class:", n_val)

    mix = {t: sizes[t] / sum(sizes.values()) for t in TASKS}
    proportions = {t: {k: float(k == t) for k in TASKS} for t in TASKS}
    proportions["ref"] = mix
    proportions.update(E3_DISTRIBUTIONS)

    rng = np.random.default_rng(args.seed)
    manifest = {"seed": args.seed, "n_neg_train_sets": n_neg, "n_val_per_class": n_val,
                "task_sizes": sizes, "proportions": proportions, "datasets": {}}

    for name in EVAL_SETS:
        eval_only = name not in TRAIN_SETS
        n_class = n_val if eval_only else n_neg
        counts = allocate(n_class, proportions[name])
        neg = draw(rng, pools, counts)
        pos = rng.choice(pos_pool, size=n_class, replace=False)
        rng.shuffle(neg)
        rng.shuffle(pos)
        if eval_only:
            val = np.concatenate([neg, pos])
            train = np.empty(0, dtype=np.int64)
        else:
            val = np.concatenate([neg[:n_val], pos[:n_val]])
            train = np.concatenate([neg[n_val:], pos[n_val:]])
        rng.shuffle(val)
        rng.shuffle(train)
        assert len(np.unique(np.concatenate([train, val]))) == len(train) + len(val)
        assert y[val].mean() == 0.5 and (len(train) == 0 or y[train].mean() == 0.5)
        np.savez(out / f"{name}.npz", train_idx=train.astype(np.int64), val_idx=val.astype(np.int64))
        manifest["datasets"][name] = {
            "n_train": int(len(train)), "n_val": int(len(val)), "negative_counts": counts,
            "val_negative_counts": {t: int((method[val] == t).sum()) for t in TASKS},
        }
        print(f"{name:>24}: train={len(train):>7} val={len(val):>6}")

    codes = np.array([TASKS.index(m) if m in TASKS else -1 for m in method], dtype=np.int8)
    np.save(out / "row_task.npy", codes)
    write_json(manifest, out / "manifest.json")

    shuffle_idx = np.load(d / f"{CITY}_shuffle_idx.npy")
    partner = mirror_partner(shuffle_idx, n_rows // 2)
    records = []
    for train_name in TRAIN_SETS:
        train = np.load(out / f"{train_name}.npz")["train_idx"]
        in_train = np.zeros(n_rows, dtype=bool)
        in_train[train] = True
        for eval_name in EVAL_SETS:
            val = np.load(out / f"{eval_name}.npz")["val_idx"]
            exact = in_train[val]
            either = exact | in_train[partner[val]]
            records.append({"train_set": train_name, "eval_set": eval_name,
                            "val_rows_in_train": exact.mean(),
                            "val_rows_or_mirror_in_train": either.mean()})
    pd.DataFrame(records).to_csv(out / "overlap.csv", index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
