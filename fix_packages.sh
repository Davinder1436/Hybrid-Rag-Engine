#!/bin/bash
# Fix package installation in the 'research' conda environment
# This ensures compatible versions of sentence-transformers and huggingface-hub

set -e

echo "======================================"
echo "Fixing Package Versions"
echo "======================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found."
    exit 1
fi

# Check if research environment exists
if ! conda env list | grep -q "^research "; then
    echo "Error: 'research' environment not found."
    echo "Create it with: conda create -n research python=3.10 -y"
    exit 1
fi

echo ""
echo "Activating 'research' environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate research

echo ""
echo "Current Python version:"
python --version

echo ""
echo "Upgrading problematic packages..."
pip install --upgrade "sentence-transformers>=2.7.0" "huggingface-hub>=0.19.0"

echo ""
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)"

echo ""
echo "Verifying installation..."
python -c "from sentence_transformers import SentenceTransformer; print('✅ sentence-transformers imported successfully')" || echo "❌ Failed to import sentence-transformers"

python -c "import torch; print(f'✅ PyTorch version: {torch.__version__}'); print(f'✅ MPS available: {torch.backends.mps.is_available()}')"

echo ""
echo "======================================"
echo "Fix Complete!"
echo "======================================"
echo ""
echo "Try running your command again:"
echo "  python scripts/chunk_and_index.py --batch_size 128 --device mps"
