#!/bin/bash

# Script to download datasets for Hybrid RAG project
# Usage: bash scripts/download_datasets.sh [data_dir]

set -e  # Exit on error

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"

echo "=========================================="
echo "Downloading Datasets for Hybrid RAG"
echo "Data directory: $DATA_DIR"
echo "=========================================="

# Create subdirectories
mkdir -p "$DATA_DIR/hotpotqa"
mkdir -p "$DATA_DIR/nq"
mkdir -p "$DATA_DIR/wikipedia"
mkdir -p "$DATA_DIR/beir"

# Download HotpotQA
echo ""
echo "[1/4] Downloading HotpotQA dataset..."
if [ ! -f "$DATA_DIR/hotpotqa/hotpot_train_v1.1.json" ]; then
    echo "  Downloading training set..."
    curl -L "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json" \
        -o "$DATA_DIR/hotpotqa/hotpot_train_v1.1.json"
else
    echo "  Training set already exists, skipping..."
fi

if [ ! -f "$DATA_DIR/hotpotqa/hotpot_dev_distractor_v1.json" ]; then
    echo "  Downloading dev set..."
    curl -L "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json" \
        -o "$DATA_DIR/hotpotqa/hotpot_dev_distractor_v1.json"
else
    echo "  Dev set already exists, skipping..."
fi

echo "  HotpotQA download complete!"

# Download Natural Questions (using HuggingFace datasets - will be done in Python)
echo ""
echo "[2/4] Natural Questions will be downloaded via HuggingFace datasets"
echo "  (This will be handled in Python scripts)"

# Download Wikipedia subset (using HuggingFace datasets)
echo ""
echo "[3/4] Wikipedia corpus will be downloaded via HuggingFace datasets"
echo "  (This will be handled in Python scripts)"

# BEIR benchmark info
echo ""
echo "[4/4] BEIR benchmark datasets"
echo "  (These will be downloaded on-demand via the beir package)"

echo ""
echo "=========================================="
echo "Dataset download complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Install Python dependencies: pip install -r requirements.txt"
echo "2. Run preprocessing: python scripts/preprocess_data.py"
echo "3. Build indexes: python scripts/chunk_and_index.py"
echo ""
