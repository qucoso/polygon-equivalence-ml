"""
Training Dataset Creation Pipeline

Creates positive and negative polygon pairs for polygon matching,
computes geometric features, scales them, and produces the final training arrays.
"""

import os
import json
import random
import argparse
from itertools import combinations

import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
from sklearn.cluster import KMeans

import data.helper_main as hf
from data.helper_main import CustomFeatureScaler

# ============================================================================
# Configuration
# ============================================================================

RANDOM_SEED = 42
NUM_CLUSTERS = 1000

FEATURES = [
    "area", "length", "width", "height", "area_per_length",
    "centroid_x", "centroid_y", "elong", "sin_angle", "cos_angle",
    "convex_ratio", "circularity", "n_nodes", "polygon_roughness",
]

METHOD_LIST = [
    "original", "simplify_1", "simplify_2", "simplify_3",
    "buffer_smooth", "chaikin_smooth", "taubin_smooth",
]

PAIR_COMBINATIONS = (
    [("original", m) for m in METHOD_LIST]
    + [(m, "original") for m in METHOD_LIST]
    + [("original", "original")] * 8
)

FACTORS = {
    "intersecting":          {"factor": 10, "number": 0},
    "cluster":               {"factor": 20, "number": 0},
    "modified":              {"factor": 40, "number": 0},
    "random":                {"factor": 10, "number": 0},
    "same_center_dif_shape": {"factor": 20, "number": 0},
}


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create training dataset for polygon matching.",
    )
    parser.add_argument(
        "--city",
        type=str,
        required=True,
        help="City name used for input/output file naming (e.g. 'berlin').",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory for all input and output files (default: 'data').",
    )
    return parser.parse_args()


# ============================================================================
# Helper: build path inside data directory
# ============================================================================

def data_path(data_dir: str, filename: str) -> str:
    return os.path.join(data_dir, filename)


# ============================================================================
# Pipeline steps
# ============================================================================

def load_polygons(city: str, data_dir: str) -> gpd.GeoDataFrame:
    path = data_path(data_dir, f"all_geoms_{city}.joblib")
    gdf = joblib.load(path)
    print(f"{len(gdf)} polygons loaded from {path}")
    return gdf


def add_spatial_clusters(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf["x"] = gdf.geometry.centroid.x
    gdf["y"] = gdf.geometry.centroid.y
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=RANDOM_SEED).fit(
        gdf[["x", "y"]]
    )
    gdf["cluster"] = kmeans.labels_
    return gdf


def find_intersection_pairs(gdf: gpd.GeoDataFrame) -> list[tuple[int, int]]:
    sindex = gdf.sindex
    pairs = []
    for i, geom in enumerate(gdf.geometry):
        for j in sindex.intersection(geom.bounds):
            if i < j and geom.intersects(gdf.geometry.iloc[j]):
                pairs.append((i, int(j)))
    pairs = [tuple(sorted(p)) for p in pairs]
    print(f"{len(pairs)} intersection pairs found.")
    return pairs


def compute_factor_numbers(
    n_polygons: int,
    intersection_pairs: list[tuple[int, int]],
) -> tuple[int, list[tuple[int, int]]]:
    """Compute how many pairs each negative category should contain."""
    total = n_polygons * len(PAIR_COMBINATIONS)

    cap = int(total * FACTORS["intersecting"]["factor"] / 100)
    if cap < len(intersection_pairs):
        intersection_pairs = random.sample(intersection_pairs, cap)

    FACTORS["intersecting"]["number"] = min(cap, len(intersection_pairs))

    intersect_mod_ratio = (
        FACTORS["intersecting"]["factor"] + FACTORS["modified"]["factor"]
    ) / 100
    FACTORS["modified"]["number"] = (
        int(intersect_mod_ratio * total) - FACTORS["intersecting"]["number"]
    )
    FACTORS["cluster"]["number"] = int(
        FACTORS["cluster"]["factor"] / 100 * total
    )
    FACTORS["random"]["number"] = int(
        FACTORS["random"]["factor"] / 100 * total
    )
    FACTORS["same_center_dif_shape"]["number"] = int(
        FACTORS["same_center_dif_shape"]["factor"] / 100 * total
    )

    # ---- summary ----------------------------------------------------------
    print(f"Positive examples: {total:,}")
    print(f"Negative examples: {total:,}")
    for key, fd in FACTORS.items():
        pct = fd["number"] / total * 100
        print(f"  {key}: {fd['number']:,} ({pct:.0f}%)")
    print(f"Total training dataset size: {2 * total:,}")

    return total, intersection_pairs


def build_cluster_pairs(
    gdf: gpd.GeoDataFrame,
    existing: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    dict_cluster = (
        gdf.groupby("cluster")
        .apply(lambda x: x.index.tolist(), include_groups=False)
        .to_dict()
    )
    all_possible = []
    for indices in dict_cluster.values():
        if len(indices) >= 2:
            all_possible.extend(combinations(sorted(indices), 2))

    unique = [p for p in all_possible if p not in existing]
    random.shuffle(unique)
    pairs = unique[: FACTORS["cluster"]["number"]]
    existing.update(pairs)
    print(f"{len(pairs)} cluster pairs created.")
    return pairs, existing


def build_random_and_samecentre_pairs(
    n: int, existing: set[tuple[int, int]],
) -> tuple[list, list, set]:
    random_pairs, existing = hf.generate_unique_random_pairs(
        n, existing, FACTORS["random"]["number"]
    )
    print(f"{len(random_pairs)} random pairs created.")

    sc_pairs, existing = hf.generate_unique_random_pairs(
        n, existing, FACTORS["same_center_dif_shape"]["number"]
    )
    print(f"{len(sc_pairs)} same-center-different-shape pairs created.")
    return random_pairs, sc_pairs, existing


def build_modified_pairs(n: int) -> list[tuple[int, int]]:
    indices = np.random.randint(0, n - 1, FACTORS["modified"]["number"])
    pairs = [(int(i), int(i)) for i in indices]
    print(f"{len(pairs)} modified pairs created.")
    return pairs


def assemble_pair_dataframe(
    n: int,
    geometries: gpd.GeoSeries,
    intersection_pairs: list,
    cluster_pairs: list,
    random_pairs: list,
    modify_pairs: list,
    sc_pairs: list,
) -> pd.DataFrame:
    """Combine every pair category + positive pairs into one DataFrame."""

    negative_categories = {
        "intersection":          intersection_pairs,
        "cluster":               cluster_pairs,
        "random":                random_pairs,
        "modified":              modify_pairs,
        "same_center_dif_shape": sc_pairs,
    }

    parts: list[pd.DataFrame] = []
    for name, pairs in negative_categories.items():
        df = pd.DataFrame(pairs, columns=["idx_pair_1", "idx_pair_2"])
        df["manipulation"] = name in ("modified", "same_center_dif_shape")
        df["method"] = name
        df["parameter"] = None
        parts.append(df)

    # positive pairs --------------------------------------------------------
    positive_df = pd.DataFrame(
        [
            (i, i, True, "positiv", combo)
            for i in range(n)
            for combo in PAIR_COMBINATIONS
        ],
        columns=["idx_pair_1", "idx_pair_2", "manipulation", "method", "parameter"],
    )
    positive_df["parameter"] = positive_df.apply(
        lambda row: hf.prepare_simplify_parameter(
            geometries.iloc[row["idx_pair_1"]], row["parameter"]
        ),
        axis=1,
    )
    parts.append(positive_df)

    return pd.concat(parts, ignore_index=True)


def assign_samecentre_parameters(
    df: pd.DataFrame, gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    xmin, ymin, xmax, ymax = gdf.total_bounds
    mask = df["method"] == "same_center_dif_shape"
    n_sc = mask.sum()

    xs = np.random.uniform(xmin, xmax, n_sc)
    ys = np.random.uniform(ymin, ymax, n_sc)

    params = pd.Series(
        [
            [
                {"new_centroid": {"x": x, "y": y}},
                {"new_centroid": {"x": x, "y": y}},
            ]
            for x, y in zip(xs, ys)
        ]
    )
    df.loc[mask, "parameter"] = params.values
    return df


def assign_modified_parameters(
    df: pd.DataFrame, geometries: gpd.GeoSeries,
) -> pd.DataFrame:
    mask = df["method"] == "modified"
    mod_ids = df.loc[mask, "idx_pair_1"]
    bounds_list = [geometries.iloc[i].bounds for i in mod_ids]

    aug_results = []
    for b in bounds_list:
        param_list = [{"original": {}}, hf.get_augmentation_parameter(b)]
        random.shuffle(param_list)
        aug_results.append(param_list)

    df.loc[mask, "parameter"] = pd.Series(aug_results).values
    return df


def compute_features(
    gdf: gpd.GeoDataFrame,
    df: pd.DataFrame,
) -> np.ndarray:
    results = hf.process_geometry_chunks(
        all_geometries=gdf,
        idx_parameter=df,
        chunk_size=500_000,
        n_features=len(FEATURES),
    )
    print(f"Feature array shape: {results.shape}")
    return results


def scale_features(
    results: np.ndarray, data_dir: str
) -> np.ndarray:
    scaler_path = data_path(data_dir, f"scaler.joblib")

    if os.path.exists(scaler_path):
        scaler = CustomFeatureScaler.load(scaler_path)
        X_scaled = scaler.transform(np.concatenate(results, axis=0))
    else:
        scaler = CustomFeatureScaler()
        X_scaled = scaler.fit_transform(
            np.concatenate(results, axis=0), FEATURES
        )
        scaler.save(scaler_path)
        print(f"Scaler saved to {scaler_path}")

    print(
        pd.DataFrame(X_scaled, columns=FEATURES)
        .describe()
        .round(3)
        .to_string()
    )
    return X_scaled


def double_and_shuffle(
    X_pairs: np.ndarray,
    y_pairs: np.ndarray,
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Mirror pair order (A,B) → (B,A), concatenate, shuffle."""

    df2 = df.copy()
    df2[["idx_pair_1", "idx_pair_2"]] = df2[["idx_pair_2", "idx_pair_1"]].values
    df2["parameter"] = df2["parameter"].apply(
        lambda x: x[::-1] if x else x
    )
    df_full = pd.concat([df, df2], axis=0).reset_index(drop=True)

    shuffle_idx = np.random.permutation(len(df_full))
    df_full = df_full.iloc[shuffle_idx].reset_index(drop=True)

    X_full = (
        np.concatenate([X_pairs, X_pairs[[1, 0], :, :]], axis=1)[:, shuffle_idx]
        .astype(np.float32)
    )
    y_full = (
        np.concatenate([y_pairs, y_pairs], axis=0)[shuffle_idx]
        .astype(np.int8)
    )

    print(f"Final dataset shapes: X={X_full.shape}, y={y_full.shape}")
    return X_full, y_full, df_full, shuffle_idx


def save_outputs(
        data_dir: str,
        city: str,
        df: pd.DataFrame,
        df_final: pd.DataFrame,
        results: np.ndarray,
        shuffle_idx: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
    joblib.dump(df,       data_path(data_dir, f"{city}_idx_parameter.joblib"))
    np.save(              data_path(data_dir, f"{city}_unscaled_features.npy"), results)
    np.save(              data_path(data_dir, f"{city}_shuffle_idx.npy"), shuffle_idx)
    joblib.dump(df_final, data_path(data_dir, f"{city}_idx_parameter_final.joblib"))
    np.save(              data_path(data_dir, f"{city}_X_pairs_dataset.npy"), X)
    np.save(              data_path(data_dir, f"{city}_y_pairs_dataset.npy"), y)
    print(f"All outputs saved to {data_dir}/")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = parse_args()
    city = args.city.lower()
    data_dir = args.data_dir

    # reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # load & cluster --------------------------------------------------------
    gdf = load_polygons(city, data_dir)
    n = len(gdf)
    gdf = add_spatial_clusters(gdf)

    # negative pairs --------------------------------------------------------
    intersection_pairs = find_intersection_pairs(gdf)
    all_existing = set(intersection_pairs)

    total_examples, intersection_pairs = compute_factor_numbers(
        n, intersection_pairs
    )
    all_existing = set(intersection_pairs)

    with open(data_path(data_dir, f"data_distri_{city}.json"), "w") as f:
        json.dump(FACTORS, f)

    cluster_pairs, all_existing = build_cluster_pairs(gdf, all_existing)
    random_pairs, sc_pairs, all_existing = build_random_and_samecentre_pairs(
        n, all_existing
    )
    modify_pairs = build_modified_pairs(n)

    # assemble pair table ---------------------------------------------------
    geometries = gdf.geometry
    idx_parameter = assemble_pair_dataframe(
        n, geometries,
        intersection_pairs, cluster_pairs, random_pairs,
        modify_pairs, sc_pairs,
    )
    idx_parameter = assign_samecentre_parameters(idx_parameter, gdf)
    idx_parameter = assign_modified_parameters(idx_parameter, geometries)
    idx_parameter.sort_values(
        by=["idx_pair_1", "idx_pair_2"], ascending=True, inplace=True,
    )
    print("Pair index built.")

    # features & scaling ----------------------------------------------------
    results_array = compute_features(gdf, idx_parameter)
    X_scaled = scale_features(results_array, data_dir)

    X_pairs = X_scaled.reshape(2, -1, len(FEATURES))
    y_pairs = np.where(idx_parameter["method"] == "positiv", 1, 0)

    # double & shuffle ------------------------------------------------------
    X_final, y_final, df_final, shuffle_idx = double_and_shuffle(
        X_pairs, y_pairs, idx_parameter,
    )

    # persist ---------------------------------------------------------------
    save_outputs(
        data_dir, city, idx_parameter, df_final,
        results_array, shuffle_idx, X_final, y_final,
    )
    print("Training dataset creation complete.")


if __name__ == "__main__":
    main()