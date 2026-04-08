# Reproduction Guide

This document provides all instructions necessary to reproduce every result reported in the paper. Two reproduction paths are available:

| **Path** | **Hardware** | **Time** | **What it covers** |
|:---|:---|:---|:---|
| **A. Checkpoint-based** (recommended) | Any machine, no GPU needed | Minutes | Evaluate pre-trained models, reproduce all figures and tables |
| **B. Full retraining** | HPC with GPU (strongly recommended) | Hours–days | Retrain all models from scratch using the documented search spaces |

---

## Prerequisites

| **Requirement** | **Version** | **Notes** |
|:---|:---|:---|
| Python | ≥ 3.12 | See `pyproject.toml` for the exact constraint |
| [uv](https://github.com/astral-sh/uv) | Latest | Fast, deterministic dependency management |
| CUDA | Any recent version | **Only** required for Path B (full retraining) |

---

## 1. Environment Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository (if not already done)
git clone https://github.com/qucoso/polygon-equivalence-ml.git
cd polygon-equivalence-ml

# Create the virtual environment and install all dependencies
uv sync
```

> All package versions are locked via `uv.lock`, ensuring identical environments across machines.

All commands below assume you are in the **repository root** and the virtual environment is active. If using `uv`, prefix commands with `uv run` or activate the environment with `source .venv/bin/activate`.

---

## 2. Quick Start — One-Command Demo

Run the reproducibility demo script to verify the setup and see an exemplary evaluation:

```bash
bash run_reproducibility.sh
```

This script:
1. Sets up the virtual environment via `uv sync`
2. Evaluates the **End-2-End** models using pre-computed thresholds from `data/thresholds.json`
3. Generates **invariance plots** (translation, rotation, scale) for the Perceiver model
4. Prints F1, Precision, and Recall metrics to the console

The script uses pre-computed optimal thresholds (determined on the Berlin dataset) so that **no raw data files or GPU are needed**. Output figures are saved to `end2end/checkpoints/output/`.

> **Note:** The invariance plots require `data/all_geoms_<city-name>.joblib`. If this file is not available, the script will evaluate using thresholds only. See [§ Data Creation Pipeline](#7-data-creation-pipeline) for how to generate this file.

---

## 3. Pre-Computed Thresholds

Optimal F1 thresholds were computed on the Berlin dataset and are stored in `data/thresholds.json`:

| **Model** | **Threshold** | **F1** | **Precision** | **Recall** |
|:---|:---|:---|:---|:---|
| Perceiver | 0.9989 | 0.9377 | 0.9153 | 0.9613 |
| MPCA | 0.9946 | 0.9290 | 0.8956 | 0.9649 |
| MessagePassing | 0.9995 | 0.8465 | 0.8367 | 0.8565 |
| GINE | 0.9998 | 0.8371 | 0.8214 | 0.8535 |
| GATv2 | 0.9997 | 0.8265 | 0.7795 | 0.8794 |

These thresholds are loaded automatically by the evaluation scripts when using the `--evaluate-f1` or `--plot-invariance` modes.

---

## 4. Pre-Trained Checkpoints

All pre-trained model checkpoints are included in the repository:

| **Model** | **Config** | **Weights** | **Size** |
|:---|:---|:---|:---|
| Feature-based MLP | `featurebased/checkpoints/mlp_config.yaml` | `featurebased/checkpoints/mlp_model.pt` | 317 KB |
| Perceiver | `end2end/checkpoints/perceiver_config.yaml` | `end2end/checkpoints/perceiver_model.pt` | 1.0 MB |
| GINE | `end2end/checkpoints/gine_config.yaml` | `end2end/checkpoints/gine_model.pt` | 1.7 MB |
| GATv2 | `end2end/checkpoints/gatv2_config.yaml` | `end2end/checkpoints/gatv2_model.pt` | 403 KB |
| MP | `end2end/checkpoints/mp_config.yaml` | `end2end/checkpoints/mp_model.pt` | 737 KB |
| MPCA | `end2end/checkpoints/mpca_config.yaml` | `end2end/checkpoints/mpca_model.pt` | 831 KB |

---

## 5. Reproducing Individual Figures and Tables

Each subsection below explains how to reproduce a specific figure or table from the paper. All commands are run from the **repository root**.

### Table 3 — Location Encoding Comparison

**What:** Comparison of different location encoding strategies for the end-to-end models.

**How:** Modify the encoding configuration in the respective YAML config file (e.g., `end2end/checkpoints/perceiver_config.yaml`), then evaluate:

```bash
# 1. Edit the config to set the desired encoding type:
#    perceiver_encoder:
#      loc_encoding_type: "multiscale_learnable"   # or: "sinusoidal", "none", etc.
#      loc_encoding_min_freq: 1000.0
#      loc_encoding_max_freq: 5600.0

# 2. Retrain the model with the modified config
uv run python end2end/main.py --model sequence --config_path end2end/checkpoints/perceiver_config.yaml

# 3. Evaluate
uv run python end2end/checkpoints/evaluate.py --compute-thresholds --city berlin
```

> **Note:** This requires the Berlin dataset files. See [§ Data Creation Pipeline](#7-data-creation-pipeline).

---

### Table 4 — Hyperparameter Optimization Results

**What:** Best hyperparameters found via two-phase Optuna optimization for each architecture.

**How:** Run Optuna-based hyperparameter search. **This is computationally expensive and HPC access with GPU is strongly recommended.**

```bash
# End-to-end models (Perceiver)
uv run python end2end/main.py --model sequence --optuna --n_trials 100

# End-to-end models (Graph: GINE, GATv2, MP, MPCA)
uv run python end2end/main.py --model graph --optuna --n_trials 100

# Feature-based MLP
uv run python featurebased/main.py --optuna --n_trials 50
```

The search spaces used are documented in [README.md § Hyperparameter Search Spaces](README.md#hyperparameter-search-spaces). Results are logged via MLflow and can be viewed with `mlflow ui`.

---

### Table 5 — Generalization to Cities Worldwide

**What:** F1 scores when applying the Berlin-trained models to polygons from other cities.

**How:**

```bash
# 1. Download OSM data for the target city from https://download.geofabrik.de/

# 2. Run the data pipeline for the new city
#    (update input paths in each script to point to the downloaded shapefiles)
uv run python data/1_load_filter_polygons.py        # → data/all_geoms_<city>.joblib
uv run python data/2_create_dataset_features.py      # → data/<city>_X/y_pairs_dataset.npy
uv run python data/3_create_dataset_end2end.py       # → data/polygons.parquet + index files

# 3. Evaluate end-to-end models using pre-computed Berlin thresholds
uv run python end2end/checkpoints/evaluate.py --evaluate-f1 --city <city_name>

# 4. Evaluate feature-based MLP
uv run python featurebased/checkpoints/evaluate_mlp.py --city <city_name> --f1
```

> The pre-computed thresholds from `data/thresholds.json` (derived on Berlin) are used automatically. This matches the paper's methodology of training on Berlin and evaluating on other cities.

---

### Figures 7 & 8 — Feature-Based MLP Invariance

**What:** Translation, rotation, and scale invariance evaluation of the feature-based MLP (and optional RF baseline).

**How:**

```bash
# Generate invariance plots (translation, rotation, scale)
uv run python featurebased/checkpoints/evaluate_mlp.py --city berlin

# With a specific polygon index for reproducible plots
uv run python featurebased/checkpoints/evaluate_mlp.py --city berlin --polygon-idx 42

# Additionally compute F1 score on the full paired dataset
uv run python featurebased/checkpoints/evaluate_mlp.py --city berlin --f1
```

**Output:** Figures are saved to `featurebased/checkpoints/output/berlin/`:
- `MLP_translation.png`
- `MLP_rotation.png`
- `MLP_scale.png`

**Required data files:**
- `data/all_geoms_berlin.joblib` (for invariance plots)
- `data/berlin_X_pairs_dataset.npy` and `data/berlin_y_pairs_dataset.npy` (for F1 evaluation)
- `data/scaler.joblib` (fitted feature scaler)

---

### Figure 9 — Feature Importance

**What:** Permutation feature importance for the MLP (and optionally the Random Forest baseline).

**How:**

```bash
# MLP only (RF checkpoint not included due to size)
uv run python featurebased/checkpoints/feature_importance.py --city berlin --skip-rf

# Both MLP and RF (requires training the RF first)
uv run python featurebased/checkpoints/random_forest.py --city berlin
uv run python featurebased/checkpoints/feature_importance.py --city berlin
```

**Output:**
- `featurebased/checkpoints/output/feature_importance_scores.csv`
- `featurebased/checkpoints/output/feature_importance.pdf`

**Required data files:**
- `data/berlin_X_pairs_dataset.npy` and `data/berlin_y_pairs_dataset.npy`

---

### Figure 10 — End-to-End Model Invariance

**What:** Translation, rotation, and scale invariance for Perceiver and GNN models.

**How:**

```bash
# Generate invariance plots for all registered models
uv run python end2end/checkpoints/evaluate.py --plot-invariance --city berlin

# With a specific polygon index
uv run python end2end/checkpoints/evaluate.py --plot-invariance --city berlin --polygon-idx 42

# Compute thresholds AND generate plots in one run
uv run python end2end/checkpoints/evaluate.py --compute-thresholds --plot-invariance --city berlin
```

**Output:** Figures are saved to `end2end/checkpoints/output/`:
- `<ModelName>_translation.png`
- `<ModelName>_rotation.png`
- `<ModelName>_scale.png`

**Required data files:**
- `data/all_geoms_<city>.joblib` (for invariance plots)
- `data/<city>_idx_parameter.joblib` (for threshold computation and F1 evaluation)

> **Note:** To control which models are evaluated, edit the `MODEL_REGISTRY` list in `end2end/checkpoints/evaluate.py`. By default, only the Perceiver is active. Uncomment the other model entries to include GATv2, GINE, MP, and MPCA.

---

## 6. Full Retraining from Scratch (HPC Required)

> **⚠ Computational cost:** Full retraining requires GPU access and significant compute time. We strongly recommend using an HPC cluster. The end-to-end models were trained on NVIDIA A100 GPUs.

### Step A — Create the Training Data

```bash
# 1. Download Berlin OSM shapefiles from https://download.geofabrik.de/
#    Place them in a local directory.

# 2. Update input paths in each data script, then run:
uv run python data/1_load_filter_polygons.py
uv run python data/2_create_dataset_features.py
uv run python data/3_create_dataset_end2end.py
```

See [§ Data Creation Pipeline](#7-data-creation-pipeline) for details on each step.

### Step B — Train the Feature-Based MLP

```bash
# Single training run with the best config
cd featurebased
uv run python main.py

# Or: Hyperparameter search with Optuna
uv run python main.py --optuna --n_trials 50
```

The training config is loaded from `checkpoints/mlp_config.yaml`. Update the data paths (`X_path`, `y_path`) to point to your locally generated files.

### Step C — Train the End-to-End Models

```bash
# Perceiver (sequence-based)
uv run python end2end/main.py --model sequence --config_path end2end/checkpoints/perceiver_config.yaml

# GNN (graph-based: trains GINE/GATv2/MP/MPCA depending on config)
uv run python end2end/main.py --model graph --config_path end2end/checkpoints/gine_config.yaml

# Hyperparameter optimization via Optuna
uv run python end2end/main.py --model sequence --optuna --n_trials 100
```

Update the data paths (`parquet_path`, `hard_candidates_path`, `intersection_path`) in the YAML config files to point to your locally generated data.

### Step D — Compute Thresholds and Evaluate

```bash
# Compute optimal F1 thresholds on Berlin
uv run python end2end/checkpoints/evaluate.py --compute-thresholds --city berlin

# Evaluate F1 on a new city using the Berlin thresholds
uv run python end2end/checkpoints/evaluate.py --evaluate-f1 --city <city>
```

---

## 7. Data Creation Pipeline

The three-step pipeline transforms raw OpenStreetMap polygons into training-ready datasets. For raw data, download building footprint shapefiles from [Geofabrik](https://download.geofabrik.de/).

### Step 1 — Load & Filter Polygons

**Script:** `data/1_load_filter_polygons.py`

Loads raw OSM building footprint shapefiles, applies quality filters, and exports a cleaned GeoDataFrame.

| **Operation** | **Description** |
|:---|:---|
| Read shapefiles | Reads all area shapefiles (`*_a_*.shp`) from the input directory |
| Resolve MultiPolygons | Retains the largest polygon by area |
| Filter classes | Removes generic `building` class entries |
| Filter vertices | Removes polygons with fewer than 10 vertices |
| Deduplicate | Removes duplicate geometries |

**Output:** `all_geoms_{city}.joblib` — Cleaned GeoDataFrame of building footprints.

### Step 2 — Feature-Based Dataset

**Script:** `data/2_create_dataset_features.py`

Creates the training dataset for the feature-based MLP:

1. Clusters building footprints spatially using KMeans on projected centroids
2. Generates **positive pairs** via cartographic generalizations (Douglas-Peucker, morphological smoothing, Chaikin's corner cutting, Taubin spectral smoothing)
3. Generates **hard-negative pairs** across five categories (modified, same center, cluster, random, intersecting)
4. Extracts 14 engineered geometric features per polygon
5. Applies group-specific feature scaling (Log+MinMax for unbounded features, MinMax for ratios)
6. Doubles the dataset by swapping pair order and shuffles

**Output:**
| File | Description |
|:---|:---|
| `{city}_X_pairs_dataset.npy` | Feature matrix of polygon pairs, shape `(2, N, 14)` |
| `{city}_y_pairs_dataset.npy` | Binary labels (1 = equivalent, 0 = non-equivalent) |
| `scaler.joblib` | Fitted feature scaler for inference |

### Step 3 — End-to-End Dataset

**Script:** `data/3_create_dataset_end2end.py`

Prepares data for the end-to-end models (Perceiver and GNNs):

1. Clusters building footprints spatially
2. Generates polygon variations using the same generalization methods as Step 2
3. Writes all polygon variations as coordinate sequences to Parquet
4. Builds a cluster-based hard-negative candidate index

**Output:**
| File | Description |
|:---|:---|
| `polygons.parquet` | All polygon coordinate sequences with metadata |
| `hard_negative_candidates.json` | Cluster-based candidate mapping for hard-negative mining |
| `intersections_pairs.csv` | Pre-computed intersection pairs for negative sampling |

### Data Availability

The processed feature-based dataset (Step 2 output) is publicly available on Kaggle:

**[Geometric Uncertainty Dataset — OSM Polygons](https://www.kaggle.com/datasets/qucoso/geometric-uncertainty-dataset-osm-polygons)**

> The end-to-end dataset (Step 3 output) must be generated locally, as raw coordinate sequences are too large for static hosting.

---

## 8. Configuration Reference

Both model families are configured via YAML files stored alongside their checkpoints. Key configuration sections:

**End-to-end models** (`end2end/checkpoints/<model>_config.yaml`):
```yaml
# Training
hyperparameter:
  lr: 0.0002            # Learning rate
  weight_decay: 0.00003 # L2 regularization
miner:
  triplet_margin: 1.0   # Margin for triplet loss

# Architecture (example: Perceiver)
perceiver_encoder:
  d_model: 64           # Hidden layer dimensionality
  num_heads: 16         # Number of attention heads
  num_latents: 8        # Number of latent vectors
  loc_encoding_type: "multiscale_learnable"

# Data paths (update these to your local paths)
dataset:
  parquet_path: "../../data/polygons.parquet"
  hard_candidates_path: "../../data/hard_negative_candidates.json"
  intersection_path: "../../data/intersections_pairs.csv"
```

**Feature-based MLP** (`featurebased/checkpoints/mlp_config.yaml`):
```yaml
model:
  shape_feat_dim: 14
  hidden_layers: [128, 256, 96, 128]
  dropout_rate: 0.1
  sinusoidal_mode: "multiscale_learnable"

# Data paths (update these to your local paths)
dataset:
  X_path: '../../data/berlin_X_pairs_dataset.npy'
  y_path: '../../data/berlin_y_pairs_dataset.npy'
```
