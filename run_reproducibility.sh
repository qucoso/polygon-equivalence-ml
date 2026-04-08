#!/usr/bin/env bash
# =============================================================================
# run_reproducibility.sh
#
# Reproducibility demo for the paper:
#   "Polygon Equivalence Learning under Geometric Uncertainty:
#    A Comparison of Three Neural Approaches"
#
# This script demonstrates that all paper results can be reproduced from the
# provided checkpoints. It performs the following steps:
#
#   1. Creates a virtual environment and installs all pinned dependencies
#   2. Evaluates the Perceiver model using pre-computed thresholds
#      (no raw data or GPU required — thresholds are stored in thresholds.json)
#   3. If city geometry data is available, generates invariance plots
#      (translation, rotation, scale) for the Perceiver model (Figure 10)
#   4. If the feature-based paired dataset is available, evaluates the MLP
#      and generates its invariance plots (Figures 7, 8)
#
# Usage:
#   bash run_reproducibility.sh
#
# Prerequisites:
#   - uv (https://github.com/astral-sh/uv)
#   - Python >= 3.12
#
# Note:
#   This script uses pre-computed thresholds (data/thresholds.json) that were
#   derived on the Berlin dataset. To reproduce thresholds from scratch, run:
#     python end2end/checkpoints/evaluate.py --compute-thresholds --city berlin
#   This requires the full Berlin dataset (see REPRODUCE.md).
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VENV_DIR=".venv"
CITY="berlin"

# Data file paths
THRESHOLDS_FILE="data/thresholds.json"
GEOM_FILE="data/all_geoms_${CITY}.joblib"
IDX_FILE="data/${CITY}_idx_parameter.joblib"
X_FILE="data/${CITY}_X_pairs_dataset.npy"
Y_FILE="data/${CITY}_y_pairs_dataset.npy"
SCALER_FILE="data/scaler.joblib"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
log() {
    echo ""
    echo "======================================================================="
    echo "  $1"
    echo "======================================================================="
    echo ""
}

check_file() {
    if [ -f "$1" ]; then
        echo "  ✓ Found: $1"
        return 0
    else
        echo "  ✗ Not found: $1"
        return 1
    fi
}

elapsed() {
    local duration=$SECONDS
    echo "  ↳ Completed in $(( duration / 3600 ))h $(( (duration % 3600) / 60 ))m $(( duration % 60 ))s"
}

# ---------------------------------------------------------------------------
# Step 0: Environment setup
# ---------------------------------------------------------------------------
log "Step 0 — Setting up the virtual environment"

if ! command -v uv &> /dev/null; then
    echo "  Error: 'uv' is not installed. Install it via:"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

uv venv --python "python3.12" "${VENV_DIR}" 2>/dev/null || true
source "${VENV_DIR}/bin/activate"
uv sync

echo "  ✓ Virtual environment created and dependencies installed"
echo "  Python: $(python --version)"
echo "  Device: CPU (no GPU required for checkpoint-based reproduction)"

# ---------------------------------------------------------------------------
# Step 1: Check available data files
# ---------------------------------------------------------------------------
log "Step 1 — Checking available data files"

echo "  Pre-computed thresholds (required):"
if ! check_file "${THRESHOLDS_FILE}"; then
    echo ""
    echo "  ERROR: ${THRESHOLDS_FILE} is required but not found."
    echo "  This file should be included in the repository."
    exit 1
fi

echo ""
echo "  Optional data files (for full evaluation):"
HAS_GEOM=false
HAS_IDX=false
HAS_FEATURE_DATA=false

check_file "${GEOM_FILE}" && HAS_GEOM=true || true
check_file "${IDX_FILE}" && HAS_IDX=true || true

if check_file "${X_FILE}" && check_file "${Y_FILE}"; then
    HAS_FEATURE_DATA=true
fi
check_file "${SCALER_FILE}" || true

# ---------------------------------------------------------------------------
# Step 2: End-to-end model evaluation (Perceiver)
# ---------------------------------------------------------------------------
log "Step 2 — Evaluating End-to-End Model (Perceiver)"

echo "  Using pre-computed thresholds from ${THRESHOLDS_FILE}."
echo "  The Perceiver model checkpoint is included in the repository."
echo ""

# Show threshold values for reference
echo "  Pre-computed thresholds:"
python -c "
import json
with open('${THRESHOLDS_FILE}') as f:
    data = json.load(f)
for name, metrics in data.items():
    print(f\"    {name:20s}  threshold={metrics['threshold']:.6f}  F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}\")
"
echo ""

if [ "${HAS_GEOM}" = true ] && [ "${HAS_IDX}" = true ]; then
    # Full evaluation: compute F1 on the dataset AND generate invariance plots
    echo "  Running full evaluation (F1 + invariance plots)..."
    SECONDS=0
    python end2end/checkpoints/evaluate.py \
        --evaluate-f1 \
        --plot-invariance \
        --city "${CITY}" \
        --polygon-idx 42
    elapsed
    echo ""
    echo "  Invariance figures saved to: end2end/checkpoints/output/"

elif [ "${HAS_GEOM}" = true ]; then
    # Geometry data available: generate invariance plots only
    echo "  Running invariance evaluation only (no paired dataset for F1)..."
    SECONDS=0
    python end2end/checkpoints/evaluate.py \
        --plot-invariance \
        --city "${CITY}" \
        --polygon-idx 42
    elapsed
    echo ""
    echo "  Invariance figures saved to: end2end/checkpoints/output/"

else
    echo "  ⚠ Skipping Perceiver evaluation: city geometry data not available."
    echo "    To enable this step, generate the data files first:"
    echo "      python data/1_load_filter_polygons.py"
    echo "    See REPRODUCE.md for details."
fi

# ---------------------------------------------------------------------------
# Step 3: Feature-based MLP evaluation
# ---------------------------------------------------------------------------
log "Step 3 — Evaluating Feature-Based MLP"

if [ "${HAS_GEOM}" = true ]; then
    echo "  Running MLP invariance evaluation..."
    SECONDS=0

    if [ "${HAS_FEATURE_DATA}" = true ]; then
        # Full evaluation: invariance plots + F1 score
        python featurebased/checkpoints/evaluate_mlp.py \
            --city "${CITY}" \
            --polygon-idx 42 \
            --f1
    else
        # Invariance plots only (no paired dataset for F1)
        python featurebased/checkpoints/evaluate_mlp.py \
            --city "${CITY}" \
            --polygon-idx 42
    fi
    elapsed
    echo ""
    echo "  Figures saved to: featurebased/checkpoints/output/${CITY}/"
else
    echo "  ⚠ Skipping MLP evaluation: city geometry data not available."
    echo "    To enable this step, generate the data files first:"
    echo "      python data/1_load_filter_polygons.py"
    echo "    See REPRODUCE.md for details."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "Reproducibility Demo Complete"

echo "  What was demonstrated:"
echo "    ✓ Environment setup with pinned dependencies (uv sync)"
echo "    ✓ Pre-computed thresholds loaded from ${THRESHOLDS_FILE}"
if [ "${HAS_GEOM}" = true ]; then
    echo "    ✓ Model evaluation using pre-trained checkpoints"
    echo "    ✓ Invariance plot generation (Figures 7, 8, 10)"
fi
echo ""
echo "  For full reproduction of all paper results, see REPRODUCE.md."
echo "  Key commands:"
echo "    # Evaluate all end-to-end models"
echo "    python end2end/checkpoints/evaluate.py --plot-invariance --city berlin"
echo ""
echo "    # Evaluate the feature-based MLP (with F1)"
echo "    python featurebased/checkpoints/evaluate_mlp.py --city berlin --f1"
echo ""
echo "    # Feature importance (Figure 9)"
echo "    python featurebased/checkpoints/feature_importance.py --city berlin --skip-rf"
echo ""