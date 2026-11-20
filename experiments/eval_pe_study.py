"""Evaluate the positional-encoding study exactly like the published Perceiver evaluation.

Same data and metric as `end2end/checkpoints/evaluate.py --compute-thresholds`: all pairs of
`berlin_idx_parameter.joblib` (stored manipulations applied, closed exterior rings), cosine
similarity of the embeddings, and the threshold that maximises F1 on this set. The published
checkpoint is evaluated as a control and must reproduce data/thresholds.json (F1 = 0.9377).

Pair geometries are built once and cached as flat coordinate arrays; every model then embeds all
polygons in length-sorted batches on the GPU.

Writes results/pe_study_metrics.csv and results/pe_study_summary.csv with the F1 scores (and
Delta-F1 vs. the published "concat_linear" variant) reported in Table 6.
"""
import argparse
import re
from multiprocessing import Pool

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from shapely import MultiPolygon
from sklearn.metrics import precision_recall_curve

import data.helper_main as hf
from end2end.helper.architectures.perceiver import PolygonPerceiver
from end2end.helper.helper_architecture import CyclicRelativePosEncoding
from experiments.common import CITY, REPO, binary_metrics, cache_dir, data_dir, results_dir, runs_dir

REFERENCE = "concat_linear"
PUBLISHED_F1 = 0.9377148406950562
CHUNK = 4000

_TABLE = None
_GEOMS = None


def _coords(geom):
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)
    return np.asarray(geom.exterior.coords, dtype=np.float32)


def _build_chunk(bounds):
    start, stop = bounds
    coords, lengths, labels = [], [], []
    for row in _TABLE.iloc[start:stop].itertuples(index=False):
        g1 = _GEOMS.iloc[row.idx_pair_1].geometry
        g2 = _GEOMS.iloc[row.idx_pair_2].geometry
        if row.manipulation:
            g1 = hf.apply_manipulation(g1, row.parameter[0])
            g2 = hf.apply_manipulation(g2, row.parameter[1])
        for geom in (g1, g2):
            c = _coords(geom)
            coords.append(c)
            lengths.append(len(c))
        labels.append(row.method == "positiv")
    return np.concatenate(coords), np.asarray(lengths, np.int64), np.asarray(labels, np.int8)


def load_pairs(sample, workers):
    global _TABLE, _GEOMS
    name = "pe_study_pairs.npz" if sample is None else f"pe_study_pairs_sample{sample}.npz"
    path = cache_dir() / name
    if path.exists():
        z = np.load(path)
        return z["coords"], z["offsets"], z["lengths"], z["labels"]

    table_path = data_dir() / f"{CITY}_idx_parameter.joblib"
    if not table_path.exists():
        raise FileNotFoundError(f"{table_path} is required (set PE_DATA_DIR, see experiments/common.py)")
    _TABLE = joblib.load(table_path)
    if sample is not None:
        _TABLE = _TABLE.sample(n=sample, random_state=0)
    _GEOMS = joblib.load(data_dir() / f"all_geoms_{CITY}.joblib")
    bounds = [(i, min(i + CHUNK, len(_TABLE))) for i in range(0, len(_TABLE), CHUNK)]
    with Pool(workers) as pool:
        parts = pool.map(_build_chunk, bounds, chunksize=1)
    coords = np.concatenate([p[0] for p in parts])
    lengths = np.concatenate([p[1] for p in parts])
    labels = np.concatenate([p[2] for p in parts])
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    cache_dir().mkdir(parents=True, exist_ok=True)
    np.savez(path, coords=coords, offsets=offsets, lengths=lengths, labels=labels)
    print(f"cached {len(labels)} pairs ({len(coords)} vertices) in {path}", flush=True)
    return coords, offsets, lengths, labels


@torch.inference_mode()
def embed_all(encoder, pos_encoder, coords, offsets, lengths, device, batch_size):
    coords_t = torch.from_numpy(coords)
    order = np.argsort(lengths, kind="stable")
    out = None
    for start in range(0, len(order), batch_size):
        ids = order[start:start + batch_size]
        n_max = int(lengths[ids].max())
        steps = np.arange(n_max)[None]
        valid = steps < lengths[ids][:, None]
        index = np.where(valid, offsets[ids][:, None] + steps, 0)
        valid_t = torch.from_numpy(valid)
        x = (coords_t[torch.from_numpy(index)] * valid_t.unsqueeze(-1)).to(device)
        mask = (~valid_t).to(device)
        emb = encoder(x, pos_encoder.forward_padded(x, mask), mask).float().cpu().numpy()
        if out is None:
            out = np.empty((len(lengths), emb.shape[1]), dtype=np.float32)
        out[ids] = emb
    return out


def cosine_pairs(emb):
    a, b = emb[0::2], emb[1::2]
    return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-8)


def best_f1_threshold(y, sim):
    precision, recall, thresholds = precision_recall_curve(y, sim)
    f1 = 2 * precision * recall / np.where(precision + recall == 0, 1.0, precision + recall)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def find_runs(include_published):
    runs = []
    for path in sorted((runs_dir() / "pe_study").glob("*_s*")):
        match = re.fullmatch(r"(.+)_s(\d+)", path.name)
        if match and (path / "encoder.pt").exists():
            runs.append((match.group(1), int(match.group(2)), path / "encoder.pt", path / "config.yaml"))
    if include_published:
        runs.append(("published", 0, REPO / "end2end/checkpoints/perceiver_model.pt",
                     REPO / "end2end/checkpoints/perceiver_config.yaml"))
    return runs


def summarise(metrics):
    rows = []
    ref = metrics[metrics.variant == REFERENCE].f1
    for variant, group in metrics.groupby("variant", sort=False):
        f1 = group.f1
        row = {"variant": variant, "n_seeds": len(f1), "f1_mean": f1.mean(),
               "f1_std": f1.std(ddof=1) if len(f1) > 1 else np.nan,
               "precision_mean": group.precision.mean(), "recall_mean": group.recall.mean(),
               "auc_mean": group.auc.mean()}
        if variant not in (REFERENCE, "published") and len(f1) > 1 and len(ref) > 1:
            row["delta_f1_vs_" + REFERENCE] = f1.mean() - ref.mean()
            row["delta_f1_se"] = np.sqrt(f1.var(ddof=1) / len(f1) + ref.var(ddof=1) / len(ref))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8, help="processes for building the pair cache")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample", type=int, default=None,
                        help="only for tests: random subset of pairs instead of the full table")
    parser.add_argument("--no-published", action="store_true", help="skip the published checkpoint")
    args = parser.parse_args()

    coords, offsets, lengths, labels = load_pairs(args.sample, args.workers)
    device = torch.device(args.device)
    rows = []
    for variant, seed, weights, config_path in find_runs(not args.no_published):
        with open(config_path) as f:
            enc_cfg = yaml.safe_load(f)["perceiver_encoder"]
        encoder = PolygonPerceiver(**enc_cfg)
        encoder.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        encoder = encoder.to(device).eval()
        pos_encoder = CyclicRelativePosEncoding(d_pos_enc=enc_cfg.get("d_pos_enc", 16),
                                                mode=enc_cfg.get("pe_mode", "linear")).to(device)
        sim = cosine_pairs(embed_all(encoder, pos_encoder, coords, offsets, lengths, device, args.batch_size))
        threshold = best_f1_threshold(labels, sim)
        rows.append({"variant": variant, "seed": seed, **binary_metrics(labels, sim, threshold)})
        print(f"evaluated {variant} seed={seed} f1={rows[-1]['f1']:.4f} threshold={threshold:.6f}", flush=True)

    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.sample is None else f"_sample{args.sample}"
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / f"pe_study_metrics{suffix}.csv", index=False)
    summary = summarise(metrics)
    summary.to_csv(out / f"pe_study_summary{suffix}.csv", index=False)

    pd.set_option("display.width", 200)
    print()
    print(summary.round(4).to_string(index=False))
    published = metrics[metrics.variant == "published"]
    if args.sample is None and len(published):
        gap = abs(float(published.f1.iloc[0]) - PUBLISHED_F1)
        print(f"\ncontrol: published F1 {published.f1.iloc[0]:.4f} vs data/thresholds.json "
              f"{PUBLISHED_F1:.4f} -> {'OK' if gap < 1e-3 else 'MISMATCH, check the evaluation'}")


if __name__ == "__main__":
    main()
