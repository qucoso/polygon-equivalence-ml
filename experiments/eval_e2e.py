"""Evaluate every Perceiver run on the validation split of all nine evaluation sets.

Pair geometries (including the stored manipulations) are rebuilt once from the pair table and
cached. Each model embeds both cached polygon lists once; pair scores are cosine similarities. The
decision threshold of a model is the best-F1 threshold on the validation set of its own
training dataset and is then applied unchanged to all other sets. AUC is threshold-free.

Writes results/e2e_metrics.csv (+ e2e_scores.npz), consumed by plot_results.py to produce
Figure 11 (cross-task matrix, E1) and Figure 13 (hard/easy/uniform distributions, E3).
"""
import argparse
import pickle
import re
import sys
from multiprocessing import Pool

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from shapely import MultiPolygon
from sklearn.metrics import precision_recall_curve
from torch.nn.utils.rnn import pad_sequence

from experiments.common import (CITY, EVAL_SETS, REPO, TASKS, binary_metrics, cache_dir,
                                data_dir, load_subset, results_dir, runs_dir, subsets_dir)

sys.path.insert(0, str(REPO / "end2end"))

import data.helper_main as hf  # noqa: E402
from end2end.helper.architectures.perceiver import PolygonPerceiver  # noqa: E402
from end2end.helper.helper_architecture import CyclicRelativePosEncoding  # noqa: E402

_GEOMS = None
_TABLE = None


def _largest_polygon(geom):
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return geom


def _row_polygons(row_id):
    row = _TABLE.iloc[row_id]
    g1 = _GEOMS.iloc[row.idx_pair_1].geometry
    g2 = _GEOMS.iloc[row.idx_pair_2].geometry
    if row.manipulation:
        g1 = hf.apply_manipulation(g1, row.parameter[0])
        g2 = hf.apply_manipulation(g2, row.parameter[1])
    coords = [np.asarray(_largest_polygon(g).exterior.coords, dtype=np.float32) for g in (g1, g2)]
    return coords[0], coords[1]


def build_cache(workers):
    global _GEOMS, _TABLE
    path = cache_dir() / "eval_polygons.pkl"
    if path.exists():
        return path
    _TABLE = joblib.load(data_dir() / f"{CITY}_idx_parameter_final.joblib")
    _GEOMS = joblib.load(data_dir() / f"all_geoms_{CITY}.joblib")
    rows = np.unique(np.concatenate([load_subset(n)["val_idx"] for n in EVAL_SETS]))
    with Pool(workers) as pool:
        pairs = pool.map(_row_polygons, rows.tolist(), chunksize=256)
    cache_dir().mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"rows": rows, "poly1": [p[0] for p in pairs], "poly2": [p[1] for p in pairs]}, f)
    return path


@torch.inference_mode()
def embed(encoder, polygons, pos_encoder, device, batch_size):
    order = np.argsort([len(p) for p in polygons], kind="stable")
    out = None
    for start in range(0, len(order), batch_size):
        ids = order[start:start + batch_size]
        tensors = [torch.from_numpy(polygons[i]) for i in ids]
        lengths = torch.tensor([len(t) for t in tensors])
        padded = pad_sequence(tensors, batch_first=True)
        pe = pad_sequence([pos_encoder(t.unsqueeze(0)).squeeze(0) for t in tensors], batch_first=True)
        mask = torch.arange(padded.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)
        emb = encoder(padded.to(device), pe.to(device), mask.to(device)).float().cpu()
        if out is None:
            out = torch.empty(len(polygons), emb.size(1))
        out[torch.from_numpy(ids)] = emb
    return out


def find_runs(pretrained):
    runs = []
    for path in sorted((runs_dir() / "e2e").glob("*_s*")):
        match = re.fullmatch(r"(.+)_s(\d+)", path.name)
        if match and (path / "encoder.pt").exists():
            runs.append((match.group(1), int(match.group(2)), path / "encoder.pt", path / "config.yaml"))
    if pretrained:
        runs.append(("pretrained_full", 0, pretrained, REPO / "end2end/checkpoints/perceiver_config.yaml"))
    return runs


def match_positions(rows, idx, name, cache_path):
    """Map subset row ids to cache positions and fail loudly if the cache is stale."""
    pos = np.searchsorted(rows, idx)
    if pos.size and (pos[-1] >= len(rows) or not np.array_equal(rows[pos], idx)):
        raise RuntimeError(
            f"Evaluation cache {cache_path} does not match the current subsets (set '{name}'). "
            f"Delete it and rebuild the cache."
        )
    return pos


def best_f1_threshold(y, sim):
    precision, recall, thresholds = precision_recall_curve(y, sim)
    f1 = 2 * precision * recall / np.where(precision + recall == 0, 1.0, precision + recall)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", default=None, help="optional published Perceiver checkpoint")
    parser.add_argument("--build-cache-only", action="store_true")
    args = parser.parse_args()

    cache_path = build_cache(args.workers)
    if args.build_cache_only:
        print("cache ready:", cache_path)
        return
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    y_all = np.load(data_dir() / f"{CITY}_y_pairs_dataset.npy")
    row_task = np.load(subsets_dir() / "row_task.npy")
    sets = {}
    for name in EVAL_SETS:
        idx = np.sort(load_subset(name)["val_idx"])
        pos = match_positions(cache["rows"], idx, name, cache_path)
        sets[name] = {"pos": pos, "y": y_all[idx], "task": row_task[idx]}

    device = torch.device(args.device)
    rows, scores_out = [], {}
    for train_set, seed, weights, config_path in find_runs(args.pretrained):
        with open(config_path) as f:
            enc_cfg = yaml.safe_load(f)["perceiver_encoder"]
        encoder = PolygonPerceiver(**enc_cfg)
        encoder.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        encoder = encoder.to(device).eval()
        pos_encoder = CyclicRelativePosEncoding(d_pos_enc=enc_cfg.get("d_pos_enc", 16),
                                                 mode=enc_cfg.get("pe_mode", "linear"))

        e1 = embed(encoder, cache["poly1"], pos_encoder, device, args.batch_size)
        e2 = embed(encoder, cache["poly2"], pos_encoder, device, args.batch_size)
        sim = torch.nn.functional.cosine_similarity(e1, e2, dim=1).numpy()

        own = "ref" if train_set == "pretrained_full" else train_set
        threshold = best_f1_threshold(sets[own]["y"], sim[sets[own]["pos"]])
        for eval_name, data in sets.items():
            s = sim[data["pos"]]
            row = {"model": train_set, "seed": seed, "eval_set": eval_name,
                   **binary_metrics(data["y"], s, threshold)}
            for code, task in enumerate(TASKS):
                mask = data["task"] == code
                row[f"tnr_{task}"] = float((s[mask] < threshold).mean()) if mask.any() else float("nan")
            rows.append(row)
            scores_out[f"{train_set}_s{seed}/{eval_name}"] = s
        print("evaluated", train_set, seed, f"threshold={threshold:.4f}", flush=True)

    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "e2e_metrics.csv", index=False)
    np.savez_compressed(out / "e2e_scores.npz", **scores_out)
    print("wrote", out / "e2e_metrics.csv")


if __name__ == "__main__":
    main()
