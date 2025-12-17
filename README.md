# Deep Learning for Geometric Polygon Equivalence

This repository contains the source code and experimental framework for the Paper: **"Polygon Equivalence Learning under Geometric Uncertainty: A Comparison of three Neural Approaches."**

## 📌 Abstract

Identifying semantically equivalent but geometrically different geodata polygons is a key challenge in GIS and spatial data integration. Variations in detail, resolution, or data source often prevent reliable equivalence determination using strict geometric predicates.

This work develops and evaluates automated methods to overcome these representational variances using Deep Learning. Based on a synthetically created dataset of approximately **1.9 million polygon pairs** derived from OpenStreetMap (OSM), this repository implements and compares three distinct representation approaches:

1.  **Feature-based MLP (Reference):** A Multi-Layer Perceptron operating on engineered geometric features.
2.  **Graph-based Models:** End-to-end learning using Graph Neural Networks (MP, GINE, GATv2).
3.  **Sequence-based (Perceiver):** A novel approach using a Perceiver IO architecture with cross-attention on raw coordinate sequences.

A core contribution of the end-to-end models is the implementation of **robust, multiscale sinusoidal location encodings**, ensuring sensitivity to translations, rotations, and scalings without manual feature extraction.

## 💾 Dataset Generation

The models are trained and evaluated on a large-scale synthetic dataset containing approximately **1.9 million polygon pairs**, derived from real-world OpenStreetMap (OSM) polygons.

### Positive Pairs (Equivalents)
To simulate "geometric uncertainty," we generated equivalent pairs using **cartographic generalization**. Following the principle of *Trivial Vertex Invariance* (Mai, Jiang et al., 2022), these pairs retain semantic equality despite changes in vertex count or minor topological variations. The dataset includes six specific generalization types:

1.  **Douglas-Peucker Simplification:** Applied with tolerances of 0.5%, 1.0%, and 10.0% (iterative vertex elimination).
2.  **Morphological Smoothing:** Buffer operations (positive followed by negative) to round corners.
3.  **Chaikin's Corner Cutting:** Recursive smoothing, significantly increasing vertex count.
4.  **Taubin Spectral Smoothing:** Removes high-frequency noise while preserving volume.

### Negative Pairs (Non-Equivalents)
To prevent the models from learning simple distance thresholds, "Hard Negatives" were generated using specific sampling strategies. The distribution of negative pairs during training is as follows:

*   **Modify (44%):** Slight perturbations of non-equivalent polygons.
*   **Same Center, Different Shape (20%):** Distinct polygons centered at the same coordinate to penalize pure location-based matching.
*   **Cluster (20%):** Polygons selected from the immediate neighborhood.
*   **Random Other (10%):** Randomly selected polygons from the dataset.
*   **Intersecting (6%):** Polygons that physically overlap (high IoU) but represent different objects.

### Data Availability & Pipeline
The data handling differs by approach:

*   **Feature-based (MLP):** Uses pre-calculated feature vectors stored in static files. The complete processed dataset is available here: **[Link to Kaggle Dataset](https://www.kaggle.com/datasets/qucoso/geometric-uncertainty-dataset-osm-polygons)**.
*   **End-to-End (Perceiver/GNN):** Uses a raw `polygons.parquet` file containing the base polygons and their generalized versions. During training, the pipeline performs **on-the-fly augmentation**, generating the specific positive and negative pairings dynamically based on the probabilities listed above.

## 📂 Repository Structure

The repository is divided into two main pipelines: `featurebased` (using pre-calculated shape features) and `end2end` (learning directly from raw coordinates).

```text
.
├── end2end/                 # End-to-End Deep Learning Models (Graph & Sequence)
│   ├── config.yaml          # Hyperparameters and data paths for E2E models
│   ├── main.py              # Entry point for training Perceiver and GNNs
│   ├── helper/
│   │   ├── dataset.py       # PyTorch Dataset for raw polygon pairs (Parquet)
│   │   ├── helper_architecture.py  # Model architectures (Perceiver, GNN, LocEncodings)
│   │   ├── polygonaugmenter.py     # On-the-fly geometric augmentations
│   │   └── trainer.py       # Training loop, Metric Learning losses, and Evaluation
│   └── ...
├── featurebased/            # Feature-based Reference Model (MLP)
│   ├── config.yaml          # Hyperparameters for the MLP
│   ├── main.py              # Entry point for training the MLP
│   ├── dataset.py           # Dataset loader for pre-calculated feature vectors (.npy)
│   ├── trainer.py           # Training loop for binary classification
│   └── ...
└── ...
```

## 🚀 Getting Started

### Prerequisites

The project is built using Python 3.9+ and PyTorch. Key dependencies include:

*   `torch` & `torchvision`
*   `torch_geometric` (for Graph models)
*   `mlflow` (for experiment tracking)
*   `optuna` (for hyperparameter optimization)
*   `pandas`, `numpy`, `pyarrow` (for data handling)

To install the dependencies:

```bash
pip install torch torch-geometric mlflow optuna pandas pyarrow tqdm scikit-learn
```

### Data Preparation

The models rely on specific data formats:
*   **End-to-End:** Requires a `.parquet` file containing polygon coordinates and IDs, and a `.csv` defining intersection pairs.
*   **Feature-based:** Requires `.npy` files containing pre-calculated feature vectors (`X`) and labels (`y`).

*Note: Please update the `parquet_path`, `intersection_path`, `X_path`, and `y_path` in the respective `config.yaml` files to point to your local data.*

## 🧠 Methodologies & Usage

### 1. End-to-End Approaches (Perceiver & GNN)

This pipeline learns embeddings directly from raw polygon coordinates. It utilizes **Metric Learning** (Triplet Loss, Multi-Similarity Loss) to cluster semantically equivalent polygons.

**Key Features:**
*   **Sinusoidal Multi-Scale Location Encoding:** Maps coordinates to high-dimensional Fourier features to capture fine-grained geometric details.
*   **Augmentation:** On-the-fly rotation, scaling, and translation during training.

**Training:**

To train the Sequence-based **Perceiver** model:
```bash
cd end2end
python main.py --model sequence
```

To train the **Graph-based** model:
```bash
cd end2end
python main.py --model graph
```

**Hyperparameter Optimization:**
To run an Optuna study for hyperparameter tuning:
```bash
python main.py --model sequence --optuna --n_trials 50
```

### 2. Feature-based Approach (MLP)

This serves as the pragmatic reference model. It takes a vector of engineered shape features (e.g., area, perimeter, compactness, Hu moments) and classifies pairs as equivalent or not.

**Training:**

```bash
cd featurebased
python main.py
```

## 📊 Results

The evaluation results from the Master's thesis demonstrate the efficacy of both approaches:

| Model Architecture | Input Data | F1 Score | Notes |
| :--- | :--- | :--- | :--- |
| **Feature-based MLP** | Engineered Features | **99.8%** | Highest accuracy, highly efficient. |
| **Perceiver (Sequence)** | Raw Coordinates | **94.9%** | Best end-to-end model; learns generalizable embeddings. |
| **Graph Models (GNN)** | Raw Coordinates | ~84.7% | Lower performance compared to sequence-based approach. |

While the Feature-based approach offers a pragmatic and extremely accurate solution, the success of the Perceiver model underscores the potential of end-to-end approaches to learn generalizable representations directly from raw geodata, forming a promising basis for future Geo-AI applications.

## 🛠 Configuration

Both pipelines utilize `config.yaml` files for easy experimentation.

**Example `end2end/config.yaml` snippet:**
```yaml
miner:
  miner_type: "triplet"       # Triplet or Multi-Similarity
  triplet_margin: 0.5

perceiver_encoder:
  num_latents: 12             # Latent bottlenecks
  loc_encoding_type: "multiscale_learnable"
  loc_encoding_min_freq: 1000.0
```