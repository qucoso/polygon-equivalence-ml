"""
End-to-End Preprocessing Pipeline

1. Spatial clustering of building footprints via KMeans.
2. Generation of polygon variation parquet file.
3. Hard-negative candidate index (cluster-based).
4. Intersection pair extraction.
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from tqdm import tqdm

import data.helper_main as hf

# ============================================================================
# Configuration
# ============================================================================

RANDOM_SEED = 42
NUM_CLUSTERS = 1000
N_NEIGHBORS = 100

VARIATION_LIST = [
    "original", "simplify_0", "simplify_1", "simplify_2",
    "buffer_smooth", "chaikin_smooth", "taubin_smooth",
]


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end preprocessing pipeline for polygon matching.",
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

def load_polygons(city: str, data_dir: str) -> pd.DataFrame:
    gdf = joblib.load(f"{data_dir}/all_geoms_{city}.joblib")
    gdf["x"] = gdf.to_crs(epsg=3857).geometry.centroid.x
    gdf["y"] = gdf.to_crs(epsg=3857).geometry.centroid.y
    print(f"{len(gdf)} polygons loaded for '{city}'.")
    return gdf


def add_spatial_clusters(gdf):
    coords = gdf[["x", "y"]]
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=RANDOM_SEED).fit(coords)
    gdf["cluster"] = kmeans.labels_
    gdf = gdf.sort_values(by="cluster", ascending=True)
    print(f"Clustered {len(gdf)} polygons into {NUM_CLUSTERS} clusters.")
    return gdf


def generate_polygon_variations(gdf, parquet_path: str) -> None:
    polygon_ids = gdf.osm_id.astype(str).values
    geometries = gdf.geometry.values

    data_for_table = []
    print("Preparing polygon variations...")

    for poly_id, geom in tqdm(zip(polygon_ids, geometries), total=len(polygon_ids)):
        variation_params = hf.prepare_simplify_parameter(geom, VARIATION_LIST)

        for i, param in enumerate(variation_params):
            manipulated = hf.apply_manipulation(geom, parameter=param)

            if manipulated is None or manipulated.is_empty or not manipulated.exterior:
                coords_list = []
            else:
                coords_list = np.asarray(
                    manipulated.exterior.coords, dtype=np.float32
                ).tolist()

            data_for_table.append({
                "polygon_id": poly_id,
                "variation": f"v{i}",
                "coordinates": coords_list,
            })

    print("Converting to Arrow table...")
    table = pa.Table.from_pandas(pd.DataFrame(data_for_table), preserve_index=False)

    pq.write_table(
        table,
        parquet_path,
        row_group_size=len(VARIATION_LIST) * 100,
        use_dictionary=["polygon_id", "variation"],
        compression="snappy",
    )
    print(f"Parquet file written to {parquet_path}")


def build_hard_negative_index(gdf, output_path: str) -> None:
    hard_negative_candidates = {}
    for _, group in gdf.groupby("cluster"):
        ids = group["osm_id"].tolist()
        for osm_id in ids:
            hard_negative_candidates[osm_id] = [
                other for other in ids if other != osm_id
            ]

    with open(output_path, "w") as f:
        json.dump(hard_negative_candidates, f)

    print(f"Hard-negative candidate index saved to {output_path}")


def extract_intersection_pairs(gdf, output_path: str) -> None:
    sindex = gdf.sindex
    intersection_pairs = []

    for i, geom in enumerate(gdf.geometry):
        candidates = list(sindex.intersection(geom.bounds))
        for j in candidates:
            if i < j and geom.intersects(gdf.geometry.iloc[j]):
                intersection_pairs.append((i, int(j)))

    intersection_pairs = [tuple(sorted(p)) for p in intersection_pairs]
    print(f"{len(intersection_pairs)} intersection pairs found.")

    intersections_df = pd.DataFrame(
        [
            (gdf.iloc[i].osm_id, gdf.iloc[j].osm_id)
            for i, j in intersection_pairs
        ],
        columns=["id1", "id2"],
    )
    intersections_df.to_csv(output_path, index=False)
    print(f"Intersection pairs saved to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = parse_args()
    city = args.city.lower()
    data_dir = args.data_dir

    np.random.seed(RANDOM_SEED)

    # derived output paths
    parquet_path = data_path(data_dir, f"{city}_polygons.parquet")
    hard_neg_path = data_path(data_dir, f"{city}_hard_negative_candidates.json")
    intersections_path = data_path(data_dir, f"{city}_intersections_pairs.csv")

    # pipeline
    gdf = load_polygons(city, data_dir)
    gdf = add_spatial_clusters(gdf)
    generate_polygon_variations(gdf, parquet_path)
    build_hard_negative_index(gdf, hard_neg_path)
    extract_intersection_pairs(gdf, intersections_path)

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()