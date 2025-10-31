#!/bin/bash
# Build only the dense index (assumes chunking is already done)
# Usage: bash build_dense_index.sh

set -e

echo "======================================"
echo "Building Dense Index Only"
echo "======================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found."
    exit 1
fi

# Activate research environment
echo "Activating 'research' environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate research

echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(python --version)"
echo ""

# Check if chunked passages exist
if [ ! -f "data/indexes/chunked_passages.jsonl" ]; then
    echo "Error: chunked_passages.jsonl not found."
    echo "Run the full indexing first:"
    echo "  python scripts/chunk_and_index.py --batch_size 128 --device mps"
    exit 1
fi

echo "Building dense index with Metal acceleration..."
echo ""

# Build dense index only, skip chunking
python scripts/chunk_and_index.py \
    --batch_size 128 \
    --device mps \
    --index_types dense \
    --skip_chunking

echo ""
echo "======================================"
echo "Dense index build complete!"
echo "======================================"
echo ""
echo "Test retrieval with:"
echo "  python scripts/smoke_test.py --device mps"
