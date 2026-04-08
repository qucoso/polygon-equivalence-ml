# Deep Learning for Geometric Polygon Equivalence

Source code and experimental framework for the paper:

> **Polygon Equivalence Learning under Geometric Uncertainty: A Comparison of Three Neural Approaches.**

> **All results reported in the paper are fully reproducible.**
> Pre-trained checkpoints and pre-computed thresholds are included in this repository.
> See **[REPRODUCE.md](REPRODUCE.md)** for step-by-step instructions — no GPU required.

---

## Abstract

Geometric uncertainty refers to the fact that the precise geometry of a geographic entity may vary according to various parameters (e.g. the map scale and the data acquisition process). Determining the equivalence of two polygons under geometric uncertainty is a challenging task, as manual approaches — such as setting fixed thresholds — do not scale beyond specific scenarios. To address this gap, this work explored neural approaches for polygon equivalence learning. Three neural approaches were compared: (1) a feature-based approach (i.e. hand-crafted feature vectors combined with machine learning classifiers), (2) graph-based models (i.e. GINE, GATv2, and a custom Message Passing architecture), and (3) a sequence-based approach (i.e. a Perceiver IO architecture to process vertex sequences). Results show high classification accuracy and high generalisation capabilities across different spatial regions for the three approaches. The feature-based approach achieved the highest accuracy with an F1 score of 99.8%. Among the end-to-end models, the sequence-based Perceiver approach proved to be superior to the graph-based models (max. 84.7%), achieving an F1 score of 94.9%. Overall, these results suggest that neural methods have the potential to reliably determine polygon equivalence, providing a good basis for future work in Geo-AI applications.

---

## Results

| **Model Architecture** | **Type** | **Input** | **F1 Score** |
|:---|:---|:---|:---|
| Feature-based MLP | Feature-based | Engineered features (14-d) | **99.8 %** |
| Perceiver | Sequence-based | Raw coordinates | **93.8 %** |
| MPCA | Graph-based (custom) | Raw coordinates | 92.9 % |
| MessagePassing | Graph-based | Raw coordinates | 84.6 % |
| GINE | Graph-based | Raw coordinates | 83.7 % |
| GATv2 | Graph-based | Raw coordinates | 82.6 % |

While the feature-based approach offers a pragmatic and extremely accurate solution, the Perceiver model demonstrates the potential of end-to-end approaches to learn generalisable representations directly from raw geodata.

---

## Paper-to-Code Mapping

The following table maps every figure and table in the paper to the corresponding script and reproduction command. For detailed step-by-step instructions, see [REPRODUCE.md](REPRODUCE.md).

| **Paper Element** | **Description** | **Script** | **Command** |
|:---|:---|:---|:---|
| Table 3 | Location Encoding Comparison | `end2end/main.py`, `featurebased/main.py` | By retraining with adjusted location encoding parameters in the YAML file, the F1 scores can be calculated using `evaluate<_mlp>.py` |
| Table 4 | Hyperparameter Optimization Results | `end2end/main.py`, `featurebased/main.py` | Full retraining with Optuna (see [REPRODUCE.md § Full Retraining](REPRODUCE.md#full-retraining-from-scratch-hpc-required)) |
| Table 5 | Generalization to Cities Worldwide | `data/1_load_filter_polygons.py` → `evaluate.py` | Run data pipeline for new city, then `python end2end/checkpoints/evaluate.py --evaluate-f1 --city <name>` |
| Figure 7 | MLP Translation/Rotation/Scale Invariance | `featurebased/checkpoints/evaluate_mlp.py` | `python featurebased/checkpoints/evaluate_mlp.py --city <name>` |
| Figure 8 | MLP Invariance Variances | `featurebased/checkpoints/evaluate_mlp.py` | Same as Figure 7 |
| Figure 9 | Feature Importance (MLP & RF) | `featurebased/checkpoints/feature_importance.py` | `python featurebased/checkpoints/feature_importance.py --city <name> --skip-rf` |
| Figure 10 | End-to-End Invariance (Perceiver, GNNs) | `end2end/checkpoints/evaluate.py` | `python end2end/checkpoints/evaluate.py --plot-invariance --city <name>` |

---

## Repository Structure

```text
.
├── README.md                              # Project overview & paper-to-code mapping (this file)
├── REPRODUCE.md                           # Full reproduction guide
├── pyproject.toml                         # Dependencies & project metadata
├── uv.lock                                # Locked dependency versions
├── run_reproducibility.sh                 # One-command reproducibility demo
│
├── data/                                  # Data preparation pipeline (Steps 1–3)
│   ├── __init__.py
│   ├── helper_main.py                     # Shared utilities: feature extraction, scaling, augmentation
│   ├── 1_load_filter_polygons.py          # Step 1: Load & filter raw OSM building footprints
│   ├── 2_create_dataset_features.py       # Step 2: Create feature-based dataset (14-d features)
│   ├── 3_create_dataset_end2end.py        # Step 3: Create end-to-end dataset (Parquet + index)
│   └── thresholds.json                    # Pre-computed optimal F1 thresholds per model
│
├── featurebased/                          # Feature-based model (MLP)
│   ├── __init__.py
│   ├── main.py                            # Training script (single run or Optuna)
│   ├── dataset.py                         # PyTorch dataset for feature pairs
│   ├── PolygonMLP.py                      # PolygonPairClassifier architecture
│   ├── trainer.py                         # Training loop
│   └── checkpoints/
│       ├── mlp_config.yaml                # Best hyperparameters
│       ├── mlp_model.pt                   # Pre-trained weights
│       ├── evaluate_mlp.py                # Invariance & F1 evaluation (Figures 7, 8)
│       ├── feature_importance.py          # Permutation importance (Figure 9)
│       └── random_forest.py               # RF baseline training script
│
├── end2end/                               # End-to-end models (Perceiver & GNNs)
│   ├── __init__.py
│   ├── main.py                            # Training script (single run or Optuna)
│   ├── helper/
│   │   ├── architectures/
│   │   │   ├── graph.py                   # MP, MPCA, GINE, GATv2 architectures
│   │   │   └── perceiver.py               # Perceiver architecture
│   │   ├── dataset.py                     # PyTorch dataset for coordinate sequences
│   │   ├── helper_architecture.py         # CyclicRelativePosEncoding & utilities
│   │   ├── polygonaugmenter.py            # Online augmentation (scale, rotate, translate)
│   │   └── trainer.py                     # Metric learning training loop
│   └── checkpoints/
│       ├── perceiver_config.yaml          # Perceiver best hyperparameters
│       ├── perceiver_model.pt             # Perceiver pre-trained weights
│       ├── gine_config.yaml / .pt         # GINE checkpoint
│       ├── gatv2_config.yaml / .pt        # GATv2 checkpoint
│       ├── mp_config.yaml / .pt           # MessagePassing checkpoint
│       ├── mpca_config.yaml / .pt         # MPCA checkpoint
│       ├── evaluate.py                    # End-to-end evaluation pipeline (Figure 10)
│       └── helper_eval.py                 # Evaluation utilities
```

---

## Hyperparameter Search Spaces

Hyperparameter optimization was conducted in **two sequential phases** for every architecture:

1. **Phase 1 — Architecture Search** (100 Optuna trials): Structural hyperparameters were tuned while training hyperparameters were held at sensible defaults.
2. **Phase 2 — Training Search** (50 Optuna trials): Training hyperparameters were optimized with the best architecture configuration fixed from Phase 1.

### GATv2 / MP / GINE — Phase 1 (Architecture)

| **Hyperparameter** | **Search Space** | **Sampling** |
|:---|:---|:---|
| Dim Hidden Layers | {32, 64, 96} | Categorical |
| Dim Embedding | {32, 64, 128} | Categorical |
| Dim Location Encodings | {8, 16, 32} | Categorical |
| # MP-Layers | [2, 10] | Integer, uniform |
| Dropout Rate | [0.0, 0.3] | Continuous, uniform |
| # Heads *(GATv2 only)* | {2, 4, 8, 16} | Categorical |

### GATv2 / MP / GINE — Phase 2 (Training)

| **Hyperparameter** | **Search Space** | **Sampling** |
|:---|:---|:---|
| Learning Rate | [10⁻⁶, 10⁻³] | Log-uniform |
| Weight Decay | [10⁻⁶, 10⁻³] | Log-uniform |
| Margin | [0.2, 1.0], step 0.1 | Discrete uniform |

### Perceiver / MPCA — Phase 1 (Architecture)

| **Hyperparameter** | **Search Space** | **Sampling** | **Applies to** |
|:---|:---|:---|:---|
| Dim Hidden Layers | {32, 64} | Categorical | Both |
| Dim Positional Encoding | {4, 6, 8, 16} | Categorical | Both |
| Dim Location Encodings | {8, 16, 32} | Categorical | Both |
| Dim Embedding | {32, 64, 128} | Categorical | Both |
| # Heads | {2, 4, 8, 16} | Categorical | Both |
| # Latents | [2, 16] | Integer, uniform | Both |
| Latent Dropout Rate | [0.0, 0.4], step 0.1 | Discrete uniform | Both |
| # Self-Attention Layers | [2, 4] | Integer, uniform | Perceiver only |
| # Pool Queries | {8, 16, 32, 64} | Categorical | Perceiver only |
| # Attention Layers (Cross + Self) | [1, 3] | Integer, uniform | MPCA only |
| # MP-Layers | [2, 10] | Integer, uniform | MPCA only |

### Perceiver / MPCA — Phase 2 (Training)

| **Hyperparameter** | **Search Space** | **Sampling** |
|:---|:---|:---|
| Learning Rate | [10⁻⁶, 10⁻³] | Log-uniform |
| Weight Decay | [10⁻⁶, 10⁻³] | Log-uniform |
| Margin | [0.2, 1.0], step 0.1 | Discrete uniform |

### Feature-Based MLP

| **Hyperparameter** | **Search Space** | **Sampling** |
|:---|:---|:---|
| # Hidden Layers | [2, 5] | Integer, uniform |
| Units per Layer | [32, 256], step 32 | Integer, uniform (per layer) |
| # Fourier Frequencies | [4, 10], step 2 | Integer, uniform |
| Dropout Rate | [0.1, 0.5], step 0.05 | Discrete uniform |
| Learning Rate | [10⁻⁵, 10⁻³] | Log-uniform |
| Weight Decay | [10⁻⁷, 10⁻⁴] | Log-uniform |

---

## Data Pipeline

The `data/` directory contains a three-step pipeline that transforms raw OpenStreetMap building footprints into training-ready datasets.

> **Execution order:** Step 1 → Step 2 / Step 3 (Steps 2 and 3 are independent of each other).

| **Step** | **Script** | **Output** | **Consumed by** |
|:---|:---|:---|:---|
| 1. Load & Filter | `1_load_filter_polygons.py` | Cleaned GeoDataFrame (`.joblib`) | Steps 2, 3 |
| 2. Feature Dataset | `2_create_dataset_features.py` | Feature matrix + labels (`.npy`) | Feature-based MLP |
| 3. End-to-End Dataset | `3_create_dataset_end2end.py` | Coordinate sequences (`.parquet`) + candidate index (`.json`) | Perceiver, GNNs |

Raw OSM polygon shapefiles can be downloaded per region from [Geofabrik](https://download.geofabrik.de/).

### Positive Pairs (Equivalents)

Equivalent pairs are generated using cartographic generalization to simulate geometric uncertainty:

| **Method** | **Description** |
|:---|:---|
| Douglas-Peucker Simplification | Applied at 0.5 %, 1.0 %, and 10.0 % tolerance |
| Morphological Smoothing | Buffer operations (positive followed by negative) to round corners |
| Chaikin's Corner Cutting | Recursive smoothing, significantly increasing vertex count |
| Taubin Spectral Smoothing | Removes high-frequency noise while preserving volume |

### Negative Pairs (Non-Equivalents)

| **Category** | **Share** | **Strategy** |
|:---|:---|:---|
| Modified | 40 % | Augmented perturbations (translation, rotation, scaling) |
| Same Center, Different Shape | 20 % | Distinct polygons centered at the same coordinate |
| Cluster | 20 % | Polygons from the immediate spatial neighborhood |
| Random | 10 % | Randomly sampled polygon pairs |
| Intersecting | 10 % | Physically overlapping polygons representing different objects |

### Engineered Features (14-d)

| **Feature** | **Scaling** |
|:---|:---|
| Area | Log + MinMax |
| Perimeter | Log + MinMax |
| Width | Log + MinMax |
| Height | Log + MinMax |
| Area-to-Length Ratio | MinMax |
| Centroid X / Y | MinMax |
| Elongation (PCA) | MinMax |
| Sine / Cosine Angle (PCA) | MinMax |
| Convexity Ratio | MinMax |
| Circularity | MinMax |
| Node Count | Log + MinMax |
| Polygon Roughness | MinMax |

> **Scaling strategy:** Log + MinMax is applied to unbounded features, MinMax to ratios, and identity to naturally bounded features. See `data/helper_main.py` for implementation details.

### Data Availability

The processed feature-based dataset is publicly available:

**[Kaggle — Geometric Uncertainty Dataset (OSM Polygons)](https://www.kaggle.com/datasets/qucoso/geometric-uncertainty-dataset-osm-polygons)**

This dataset contains the pre-computed feature vectors and labels and can be used directly with the feature-based model without running the data pipeline.

> **Note:** The end-to-end dataset (Parquet + candidate index) must be generated locally via Step 3, as raw coordinate sequences are too large for static hosting.

---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, deterministic dependency management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Recreate the exact environment
uv sync
```

All dependencies and their exact versions are pinned in `pyproject.toml` and `uv.lock`.

- **Python:** ≥ 3.12 (see `pyproject.toml`)
- **CUDA:** Required only for full retraining; checkpoint-based reproduction runs on CPU

---

## License

This project is licensed under the MIT License. See `pyproject.toml` for details.