"""
Helper functions for building footprint polygon matching.

Provides geometry manipulation, feature extraction, custom feature scaling,
and parallel processing utilities for training dataset creation.
"""

import gc
import math
import random

import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.affinity import translate, rotate, scale
from shapely.geometry import Polygon, MultiPolygon, LinearRing
from shapely.geometry.polygon import orient
from shapelysmooth import taubin_smooth
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


# ============================================================================
# Smoothing functions
# ============================================================================

def chaikin(coords, iterations=1):
    """Apply Chaikin's corner-cutting algorithm to a coordinate sequence."""
    for _ in range(iterations):
        new_coords = []
        n = len(coords)
        for i in range(n - 1):
            p0 = coords[i]
            p1 = coords[i + 1]
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            new_coords.extend([q, r])
        new_coords.append(new_coords[0])
        coords = new_coords
    return coords


def chaikin_smooth(polygon, iterations=1):
    """Apply Chaikin smoothing to a Polygon or MultiPolygon."""
    def smooth_ring(ring):
        coords = list(ring.coords)
        if len(coords) < 4:
            return ring
        return LinearRing(chaikin(coords, iterations))

    if isinstance(polygon, Polygon):
        exterior = smooth_ring(polygon.exterior)
        interiors = [smooth_ring(r) for r in polygon.interiors]
        return orient(Polygon(exterior, interiors))
    elif isinstance(polygon, MultiPolygon):
        return MultiPolygon([chaikin_smooth(p, iterations) for p in polygon.geoms])
    else:
        raise TypeError("Input must be a Shapely Polygon or MultiPolygon")


def buffer_smooth(polygon, radius=0.0005):
    """Smooth a polygon via positive then negative buffering."""
    return polygon.buffer(radius).buffer(-radius)


# ============================================================================
# CRS transformation
# ============================================================================

def transform_polygon_crs(polygon, source_crs="EPSG:3857", target_crs="EPSG:4326"):
    """Reproject a polygon from source_crs to target_crs."""
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    transformed_coords = [
        transformer.transform(x, y) for x, y in polygon.exterior.coords
    ]
    return Polygon(transformed_coords)


def combine_bounds(geometries):
    """Compute the combined bounding box of multiple geometries."""
    minx_list, miny_list, maxx_list, maxy_list = [], [], [], []
    for geom in geometries:
        minx, miny, maxx, maxy = geom.bounds
        minx_list.append(minx)
        miny_list.append(miny)
        maxx_list.append(maxx)
        maxy_list.append(maxy)
    return (min(minx_list), min(miny_list), max(maxx_list), max(maxy_list))


# ============================================================================
# Simplification and tolerance computation
# ============================================================================

def calculate_tolerances(bbox, factors=None):
    """Compute simplification tolerances relative to the bounding box size."""
    if factors is None:
        factors = [0.005, 0.01, 0.1]
    try:
        width = abs(bbox[2] - bbox[0])
        height = abs(bbox[3] - bbox[1])
        avg_size = (width + height) / 2
        return [avg_size * f for f in factors]
    except Exception as e:
        print(f"Error computing tolerances for bbox {bbox}: {e}")
        return [0.0] * len(factors)


def simplify_one_geometry(geom, factors=None):
    """Return simplified versions of a geometry at multiple tolerance levels."""
    if factors is None:
        factors = [0.005, 0.01, 0.1]
    try:
        if geom.is_empty:
            return [geom] * len(factors)
        tolerances = calculate_tolerances(geom.bounds, factors)
        return [geom.simplify(tol, preserve_topology=True) for tol in tolerances]
    except Exception as e:
        print(f"Error during simplification: {e}")
        return [Polygon()] * len(factors)


# ============================================================================
# Parameter preparation for positive pairs
# ============================================================================

def prepare_simplify_parameter(geom, parameter_list):
    """Build manipulation parameter dicts for a list of method names."""
    result = []
    factors = calculate_tolerances(geom.bounds)

    for param in parameter_list:
        if param.startswith("simplify"):
            try:
                idx = int(param.replace("simplify_", ""))
                tol = factors[idx] if 0 <= idx < len(factors) else factors[-1]
            except ValueError:
                tol = factors[0]
            result.append({"simplify": {"tolerance": tol}})
        elif param == "buffer_smooth":
            radius = calculate_tolerances(bbox=geom.bounds, factors=[0.2])[-1]
            result.append({"buffer_smooth": {"radius": radius}})
        elif param == "taubin_smooth":
            result.append({"taubin_smooth": {"factor": 0.3, "mu": -0.3, "steps": 4}})
        elif param == "chaikin_smooth":
            result.append({"chaikin_smooth": {"iterations": random.randint(1, 3)}})
        else:
            result.append({"original": {}})

    random.shuffle(result)
    return result


def prepare_simplify_parameter_single(bounds, param):
    """Build a single manipulation parameter dict from a method name."""
    if param.startswith("simplify"):
        try:
            idx = int(param.replace("simplify_", ""))
            factors = calculate_tolerances(bounds)
            tol = factors[idx] if 0 <= idx < len(factors) else factors[-1]
        except ValueError:
            factors = calculate_tolerances(bounds)
            tol = factors[0]
        return {"simplify": {"tolerance": tol}}
    elif param == "buffer_smooth":
        radius = calculate_tolerances(bounds, factors=[0.2])[-1]
        return {"buffer_smooth": {"radius": radius}}
    elif param == "taubin_smooth":
        return {"taubin_smooth": {"factor": 0.3, "mu": -0.3, "steps": 4}}
    elif param == "chaikin_smooth":
        return {"chaikin_smooth": {"iterations": random.randint(1, 3)}}
    else:
        return {"original": {}}


# ============================================================================
# Augmentation parameters for negative (modified) pairs
# ============================================================================

def random_interval():
    """Sample a random scale factor from two disjoint intervals."""
    intervals = [(0.2, 0.5), (3, 4)]
    return random.uniform(*random.choice(intervals))


def biased_random_uniform(min_val, max_val, bias_power=2.0):
    """Sample from a power-biased uniform distribution (biased towards min_val)."""
    u = random.random() ** bias_power
    return min_val + (max_val - min_val) * u


def get_augmentation_parameter(bounds, method_forced=False):
    """Generate a random augmentation (scale, rotate, or translate) parameter dict."""
    methods = ["scale", "rotate", "translate"]
    method = random.choice(methods) if not method_forced else method_forced

    if method == "scale":
        case_equal = np.random.rand() < 0.5
        rand_angle = random.uniform(0, 2 * math.pi)
        rand_distance = random_interval()
        return {method: {
            "xfact": rand_distance if case_equal else math.cos(rand_angle) * rand_distance,
            "yfact": rand_distance if case_equal else math.sin(rand_angle) * rand_distance,
            "origin": "center",
        }}
    elif method == "rotate":
        rand_angle = random.uniform(15, 345)
        return {method: {"angle": rand_angle, "origin": "center"}}
    elif method == "translate":
        rand_angle = random.uniform(0, 2 * math.pi)
        min_val, max_val = calculate_tolerances(bbox=bounds, factors=[0.3, 6])
        rand_distance = biased_random_uniform(min_val, max_val, bias_power=4.0)
        return {method: {
            "xoff": math.cos(rand_angle) * rand_distance,
            "yoff": math.sin(rand_angle) * rand_distance,
        }}
    else:
        return {"original": {}}


# ============================================================================
# Geometry manipulation dispatch
# ============================================================================

MODIFIED_FUNCTIONS = {
    "translate": translate,
    "rotate": rotate,
    "scale": scale,
    "original": lambda geom, **kwargs: geom,
    "new_centroid": lambda geom, **kwargs: translate(
        geom, xoff=kwargs["x"] - geom.centroid.x, yoff=kwargs["y"] - geom.centroid.y
    ),
    "taubin_smooth": taubin_smooth,
    "buffer_smooth": buffer_smooth,
    "chaikin_smooth": chaikin_smooth,
    "simplify": lambda geom, **kwargs: geom.simplify(kwargs.get("tolerance", 0.01)),
}


def apply_manipulation(geom, parameter):
    """Apply a single manipulation to a geometry based on a parameter dict."""
    if parameter is None:
        return geom
    if isinstance(parameter, dict):
        param_type = next(iter(parameter))
        func = MODIFIED_FUNCTIONS.get(param_type)
        if func:
            return func(geom, **parameter[param_type])
        else:
            raise ValueError(f"Unknown manipulation function: {param_type}")
    raise ValueError(f"Unknown parameter type: {type(parameter)}, {parameter}")


# ============================================================================
# Random pair generation
# ============================================================================

def generate_unique_random_pairs(n, all_existing_pairs, needed_count):
    """Generate unique random index pairs that do not exist in all_existing_pairs."""
    result = []
    max_tries = needed_count * 20
    tries = 0

    while len(result) < needed_count and tries < max_tries:
        batch_size = min(needed_count - len(result), 1000) * 2

        idx1 = np.random.randint(0, n - 1, batch_size)
        idx2 = np.random.randint(0, n - 1, batch_size)

        valid = idx1 != idx2
        idx1, idx2 = idx1[valid], idx2[valid]
        pairs = np.array([np.minimum(idx1, idx2), np.maximum(idx1, idx2)]).T

        for i, j in pairs:
            pair = (int(i), int(j))
            if pair not in all_existing_pairs:
                all_existing_pairs.add(pair)
                result.append(pair)
                if len(result) >= needed_count:
                    break

        tries += batch_size

    if len(result) < needed_count:
        print(f"Warning: could only generate {len(result)} of {needed_count} unique pairs.")

    return result, all_existing_pairs


# ============================================================================
# Geometric feature extraction
# ============================================================================

def extract_geometric_features(polygon):
    """Extract per-edge geometric features (dx, dy, segment length, cos angle)."""
    next_points = np.roll(polygon, shift=-1, axis=0)
    delta = next_points - polygon
    dx, dy = delta[:, 0], delta[:, 1]
    segment_length = np.linalg.norm(delta, axis=1)

    prev_points = np.roll(polygon, shift=1, axis=0)
    prev_delta = polygon - prev_points
    prev_length = np.linalg.norm(prev_delta, axis=1)

    dot_product = np.sum(prev_delta * delta, axis=1)
    cos_angle = dot_product / (prev_length * segment_length + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.stack([dx, dy, segment_length, cos_angle], axis=1, dtype=np.float32)


def polygon_pca_fast(polygon):
    """Compute PCA eigenvalues, eigenvectors, and mean of polygon exterior coords."""
    if not isinstance(polygon, Polygon) or polygon.is_empty:
        return None, None, None
    try:
        coords = np.asarray(polygon.exterior.coords)
        if coords.shape[0] < 3:
            return None, None, None
        mean = coords.mean(axis=0)
        coords_centered = coords - mean
        cov = np.cov(coords_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order].T
        return eigenvalues, eigenvectors, mean
    except Exception:
        return None, None, None


def extract_pca_features_fast(polygon):
    """Extract elongation, sin and cos of principal axis angle via PCA."""
    eigenvalues, eigenvectors, _ = polygon_pca_fast(polygon)
    if eigenvalues is None or eigenvectors is None:
        return np.nan, np.nan, np.nan
    lambda1, lambda2 = eigenvalues
    elong = np.sqrt(lambda2) / np.sqrt(lambda1) if lambda1 > 0 else np.nan
    vec = eigenvectors[0]
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0:
        return elong, np.nan, np.nan
    sin_angle = vec[1] / vec_norm
    cos_angle = vec[0] / vec_norm
    return elong, sin_angle, cos_angle


def extract_shape_features_fast(geom):
    """Extract convex_ratio, circularity, node count, and polygon roughness."""
    if geom is None or geom.is_empty:
        return [np.nan] * 4
    try:
        convex = geom.convex_hull
        polygon_roughness = geom.length / convex.length
        convex_area = convex.area
        convex_ratio = geom.area / convex_area if convex_area > 0 else np.nan
        perimeter = geom.length
        circularity = (4 * np.pi * geom.area) / (perimeter ** 2) if perimeter > 0 else np.nan
        n_nodes = len(geom.exterior.coords)
        return [convex_ratio, circularity, n_nodes, polygon_roughness]
    except Exception:
        return [np.nan] * 4


def process_geometry(geom, n_features=14):
    """Extract all 14 geometric features from a single polygon."""
    if geom is None or geom.is_empty:
        return np.full(n_features, np.nan)
    try:
        minx, miny, maxx, maxy = geom.bounds
        width = maxx - minx
        height = maxy - miny
        area = geom.area
        length = geom.length
        centroid = geom.centroid
        centroid_x = centroid.x if centroid else np.nan
        centroid_y = centroid.y if centroid else np.nan
        area_per_length = area / length if length > 0 else np.nan

        elong, sin_angle, cos_angle = extract_pca_features_fast(geom)
        shape_features = extract_shape_features_fast(geom)

        features = [
            area, length, width, height, area_per_length,
            centroid_x, centroid_y, elong, sin_angle, cos_angle,
        ] + shape_features

        return np.array(features)
    except Exception:
        return np.full(n_features, np.nan)


# ============================================================================
# Parallel chunk processing
# ============================================================================

def process_geometry_chunks(
    all_geometries,
    idx_parameter,
    chunk_size=1_000_000,
    n_jobs=-1,
    n_features=14,
):
    """Process geometry pairs in chunks with parallel feature extraction."""
    n_rows = len(idx_parameter)
    n_chunks = (n_rows + chunk_size - 1) // chunk_size
    results_array = np.zeros((2, n_rows, n_features), dtype=np.float32)

    def _worker(geom, param, num_param):
        if param:
            param = param[num_param]
        manipulated_geom = apply_manipulation(geom, param)
        return process_geometry(manipulated_geom, n_features=n_features)

    for chunk_idx in tqdm(range(n_chunks), desc="Processing geometry chunks"):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, n_rows)

        params_chunk = idx_parameter.iloc[start_idx:end_idx]
        pair_1_indices = params_chunk["idx_pair_1"].values
        pair_2_indices = params_chunk["idx_pair_2"].values
        parameters = params_chunk["parameter"].values

        geoms_1 = all_geometries.iloc[pair_1_indices]["geometry"].values
        geoms_2 = all_geometries.iloc[pair_2_indices]["geometry"].values

        tasks = [
            joblib.delayed(_worker)(geom, param, 0)
            for geom, param in zip(geoms_1, parameters)
        ]
        tasks.extend(
            joblib.delayed(_worker)(geom, param, 1)
            for geom, param in zip(geoms_2, parameters)
        )

        if not tasks:
            continue

        chunk_results = joblib.Parallel(n_jobs=n_jobs)(tasks)

        current_chunk_size = end_idx - start_idx
        results_array[0, start_idx:end_idx, :] = np.vstack(chunk_results[:current_chunk_size])
        results_array[1, start_idx:end_idx, :] = np.vstack(chunk_results[current_chunk_size:])

        del params_chunk, geoms_1, geoms_2, tasks, chunk_results
        gc.collect()

    return results_array


# ============================================================================
# Custom feature scaler
# ============================================================================

class CustomFeatureScaler:
    """
    Feature scaler that applies group-specific scaling strategies:
    - Unbounded features: log1p + MinMax
    - Ratio features: MinMax
    - Bounded features (already in [0,1]): no scaling
    - Geographic features: MinMax
    """

    def __init__(self):
        self.unbounded_features = ["area", "length", "width", "height", "n_nodes"]
        self.ratio_features = ["area_per_length", "elong", "polygon_roughness"]
        self.bounded_features = [
            "sin_angle", "cos_angle", "convex_ratio", "circularity",
            "centroid_x", "centroid_y",
        ]
        self.geo_features = []

        self.unbounded_scaler = MinMaxScaler()
        self.ratio_scaler = MinMaxScaler()
        self.bounded_scaler = None
        self.geo_scaler = MinMaxScaler()

        self.feature_names = None
        self.feature_indices = {}

    def _build_indices(self, feature_names):
        """Map feature group names to column indices."""
        self.feature_names = feature_names
        for group_name, feature_list in [
            ("unbounded", self.unbounded_features),
            ("ratio", self.ratio_features),
            ("bounded", self.bounded_features),
            ("geo", self.geo_features),
        ]:
            self.feature_indices[group_name] = [
                i for i, name in enumerate(feature_names) if name in feature_list
            ]

    def fit(self, X, feature_names):
        """Fit scalers on training data."""
        self._build_indices(feature_names)

        if self.feature_indices["unbounded"]:
            X_log = np.log1p(X[:, self.feature_indices["unbounded"]])
            self.unbounded_scaler.fit(X_log)
        if self.feature_indices["ratio"]:
            self.ratio_scaler.fit(X[:, self.feature_indices["ratio"]])
        if self.feature_indices["geo"]:
            self.geo_scaler.fit(X[:, self.feature_indices["geo"]])

        return self

    def transform(self, X):
        """Transform features using fitted scalers."""
        if self.feature_names is None:
            raise ValueError("Scaler has not been fitted. Call .fit() first.")

        X_scaled = X.copy()

        if self.feature_indices["unbounded"]:
            X_log = np.log1p(X[:, self.feature_indices["unbounded"]])
            X_scaled[:, self.feature_indices["unbounded"]] = self.unbounded_scaler.transform(X_log)
        if self.feature_indices["ratio"]:
            X_scaled[:, self.feature_indices["ratio"]] = self.ratio_scaler.transform(
                X[:, self.feature_indices["ratio"]]
            )
        if self.feature_indices["geo"]:
            X_scaled[:, self.feature_indices["geo"]] = self.geo_scaler.transform(
                X[:, self.feature_indices["geo"]]
            )

        return X_scaled

    def fit_transform(self, X, feature_names):
        """Fit and transform in one step."""
        self.fit(X, feature_names)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        """Reverse the scaling transformation."""
        if self.feature_names is None:
            raise ValueError("Scaler has not been fitted. Call .fit() first.")

        X_original = X_scaled.copy()

        if self.feature_indices["unbounded"]:
            X_inv = self.unbounded_scaler.inverse_transform(
                X_scaled[:, self.feature_indices["unbounded"]]
            )
            X_original[:, self.feature_indices["unbounded"]] = np.expm1(X_inv)
        if self.feature_indices["ratio"]:
            X_original[:, self.feature_indices["ratio"]] = self.ratio_scaler.inverse_transform(
                X_scaled[:, self.feature_indices["ratio"]]
            )
        if self.feature_indices["geo"]:
            X_original[:, self.feature_indices["geo"]] = self.geo_scaler.inverse_transform(
                X_scaled[:, self.feature_indices["geo"]]
            )

        return X_original

    def save(self, filepath):
        """Persist the fitted scaler to disk."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        """Load a fitted scaler from disk."""
        return joblib.load(filepath)
