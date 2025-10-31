#!/bin/bash
# Quick setup script - Run this to set up everything automatically
# Optimized for Apple Silicon (M4 Pro) with Python 3.10
# Usage: bash setup.sh

set -e

echo "======================================"
echo "Hybrid RAG: Automated Setup"
echo "Apple Silicon M4 Pro Optimized"
echo "======================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Check Python version in conda environment
echo ""
echo "[1/6] Checking conda environment 'research'..."
if conda env list | grep -q "^research "; then
    echo "Conda environment 'research' found"
else
    echo "Warning: Conda environment 'research' not found"
    echo "Please create it first with:"
    echo "  conda create -n research python=3.10"
    exit 1
fi

echo ""
echo "[2/6] Activating conda environment 'research'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate research

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version in 'research' environment: $PYTHON_VERSION"

if [[ ! "$PYTHON_VERSION" =~ ^3\.10\. ]]; then
    echo "Warning: Python 3.10 recommended. Current version: $PYTHON_VERSION"
    echo "Continuing anyway..."
fi

echo ""
echo "[3/6] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "[4/6] Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)"

echo ""
echo "[5/6] Downloading datasets..."
bash scripts/download_datasets.sh data

echo ""
echo "[6/6] Preprocessing data..."
python scripts/preprocess_data.py \
    --data_dir data \
    --datasets hotpotqa wikipedia \
    --wiki_max 10000

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Conda environment 'research' is activated."
echo ""
echo "Next steps:"
echo "  1. Build indexes (with Metal acceleration):"
echo "     python scripts/chunk_and_index.py --batch_size 128 --device mps"
echo ""
echo "  2. Test retrieval:"
echo "     python scripts/smoke_test.py --device mps"
echo ""
echo "To deactivate conda environment, run: conda deactivate"
