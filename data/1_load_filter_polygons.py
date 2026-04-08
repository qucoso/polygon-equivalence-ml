"""
Data Preprocessing: Load OSM building footprints, filter, and export.
"""

import glob
import os
import argparse

import geopandas as gpd
import joblib
import pandas as pd

# ============================================================================
# Configuration
# ============================================================================

MIN_NODES = 10

# ============================================================================
# Functions
# ============================================================================

def get_largest_polygon(geom):
    """For MultiPolygons, keep only the largest polygon by area."""
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda p: p.area)
    elif geom.geom_type == "Polygon":
        return geom
    return None


def process_city(city: str = "berlin"):
    """Load, preprocess and export polygon data for the given city."""
    input_dir = f"{city}"
    output_path = f"all_geoms_{city}.joblib"

    shp_files = glob.glob(os.path.join(input_dir, "*.shp"))
    area_gdfs = [gpd.read_file(f) for f in shp_files if "_a_" in f]
    allnOSMPoly = pd.concat(area_gdfs, ignore_index=True)

    allnOSMPoly["geometry"] = allnOSMPoly["geometry"].apply(get_largest_polygon)
    allnOSMPoly["area"] = allnOSMPoly.geometry.to_crs(epsg=3857).area
    allnOSMPoly["n_nodes"] = allnOSMPoly.geometry.apply(lambda g: len(g.exterior.coords))

    allnOSMPoly = (
        allnOSMPoly[allnOSMPoly["fclass"] != "building"]
        .query(f"n_nodes > {MIN_NODES}")
        .drop_duplicates(subset="geometry")
        .reset_index(drop=True)
    )

    joblib.dump(allnOSMPoly, output_path)
    print(f"{len(allnOSMPoly)} polygons saved to {output_path}")

    return allnOSMPoly


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load, filter and export polygon data for a city")
    parser.add_argument("--city", default="berlin", help="City name for input folder and output file prefix")
    args = parser.parse_args()

    process_city(args.city)

