#!/bin/bash
# Thin end-to-end pipeline: 100 problems, ~570 QA examples
# Validates the full pipeline before scaling to 25k problems
#
# Expected runtime: ~35 minutes on A100
# Expected output: checkpoints/ao_thin/final + results/thin/eval_results.json
#
# Usage:
#   bash scripts/run_thin_pipeline.sh
#   bash scripts/run_thin_pipeline.sh --skip-download  # If data already downloaded

set -euo pipefail

SKIP_DOWNLOAD=false
for arg in "$@"; do
    case $arg in
        --skip-download) SKIP_DOWNLOAD=true ;;
    esac
done

CONFIG="configs/thin.yaml"

echo "============================================"
echo "CODI-AO Thin Pipeline"
echo "============================================"

# Step 1: Download data and checkpoints
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    echo "[1/5] Downloading data and checkpoints..."
    python scripts/01_download_data.py
else
    echo ""
    echo "[1/5] Skipping download (--skip-download)"
fi

# Step 2: Extract activations
echo ""
echo "[2/5] Extracting CODI activations (100 problems)..."
python scripts/02_extract_activations.py --config "$CONFIG"

# Step 3: Generate QA dataset
echo ""
echo "[3/5] Generating QA training dataset..."
python scripts/03_generate_qa.py --config "$CONFIG"

# Step 4: Train the AO
echo ""
echo "[4/5] Training the Activation Oracle..."
python scripts/04_train.py --config "$CONFIG"

# Step 5: Evaluate
echo ""
echo "[5/5] Running evaluation..."
python scripts/05_eval.py --config "$CONFIG"

echo ""
echo "============================================"
echo "Thin pipeline complete!"
echo "Results: results/thin/eval_results.json"
echo "Model:   checkpoints/ao_thin/final"
echo "============================================"
