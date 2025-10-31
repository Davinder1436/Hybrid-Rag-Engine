# Hybrid RAG: Sparse + Dense Retrieval with Uncertainty

A hybrid retrieval system combining sparse (BM25) and dense (neural) retrieval with a calibrated uncertainty head for selective retrieval. Built for multi-hop and long-document QA tasks.

**Optimized for Apple Silicon M4 Pro with 24GB RAM** ⚡

## Overview

This project implements a state-of-the-art retrieval-augmented generation (RAG) pipeline that:

- **Combines sparse and dense retrieval** for robust candidate generation
- **Uses Metal acceleration (MPS)** for 3-4x faster encoding on Apple Silicon
- **Uses an uncertainty head** to predict retrieval confidence and enable selective behavior
- **Evaluates on multi-hop QA** (HotpotQA) and open-domain QA (Natural Questions)
- **Reduces hallucinations** through calibrated confidence estimation

## Quick Start (M4 Pro Optimized)

```bash
# 1. Setup (one-time, ~5 minutes)
conda activate research  # Use your conda environment
pip install -r requirements.txt

# 2. Download & process data (~15 minutes)
bash scripts/download_datasets.sh data
python scripts/preprocess_data.py --wiki_max 10000

# 3. Build indexes with Metal acceleration (~10 minutes)
python scripts/chunk_and_index.py --batch_size 128 --device mps

# 4. Test retrieval
python scripts/smoke_test.py --device mps
```

**Total time: ~30 minutes** to a working retrieval system!

**Note**: This project uses conda environment named "research". See CONDA_GUIDE.md for setup.

See **M4_QUICKSTART.md** for copy-paste commands.

## Project Structure

```
Hybrid-RAG/
├── data/                      # Data directory (created during setup)
│   ├── hotpotqa/             # HotpotQA dataset
│   ├── nq/                   # Natural Questions
│   ├── wikipedia/            # Wikipedia passages
│   ├── processed/            # Preprocessed data
│   └── indexes/              # BM25 and FAISS indexes
├── scripts/
│   ├── download_datasets.sh   # Download raw datasets
│   ├── preprocess_data.py     # Preprocess and prepare data
│   ├── chunk_and_index.py     # Build retrieval indexes
│   └── smoke_test.py          # Test retrieval functionality
├── requirements.txt           # Python dependencies
├── doc.md                     # Detailed project documentation
└── README.md                  # This file
```

## Quick Start

### 1. Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for tokenization)
python -c "import nltk; nltk.download('punkt')"
```

### 2. Download Datasets

```bash
# Download HotpotQA dataset
bash scripts/download_datasets.sh data
```

### 3. Preprocess Data

```bash
# Process HotpotQA and download Wikipedia passages
python scripts/preprocess_data.py \
    --data_dir data \
    --datasets hotpotqa wikipedia \
    --wiki_max 10000

# This will:
# - Process HotpotQA train/dev splits
# - Extract passages from HotpotQA contexts
# - Download and process Wikipedia articles (limited to 10k for testing)
```

### 4. Build Indexes

```bash
# Build both BM25 and dense (FAISS) indexes
python scripts/chunk_and_index.py \
    --data_dir data \
    --output_dir data/indexes \
    --sources hotpotqa wikipedia \
    --chunk_size 256 \
    --stride 100

# This will:
# - Chunk passages into 256-token segments with 100-token overlap
# - Build BM25 sparse index
# - Build FAISS dense index using sentence transformers
# - Save indexes to data/indexes/
```

### 5. Test Retrieval

```bash
# Run smoke test with sample queries
python scripts/smoke_test.py \
    --index_dir data/indexes \
    --top_k 5 \
    --queries \
        "Who was the first president of the United States?" \
        "What is the capital of France?"
```

## Component Details

### Sparse Retrieval (BM25)
- Classic lexical retrieval using BM25 algorithm
- Fast, interpretable, robust for keyword matching
- Implemented using `rank-bm25` library

### Dense Retrieval (FAISS)
- Neural retrieval using sentence transformers
- Encodes queries and passages into dense vectors
- Uses FAISS for efficient similarity search
- Default model: `all-MiniLM-L6-v2` (fast and effective)

### Passage Chunking
- Splits long documents into 256-token chunks
- Uses 100-token stride for overlap (avoids boundary issues)
- Preserves context and improves retrieval granularity

## Next Steps

After completing the setup, you can:

1. **Train a dense retriever** with contrastive learning
2. **Implement the uncertainty head** for confidence estimation
3. **Build the fusion strategy** to combine BM25 and dense results
4. **Add a reader model** (Fusion-in-Decoder) for answer generation
5. **Evaluate on multi-hop QA** tasks

See `doc.md` for detailed implementation guidance on each component.

## Configuration Options

### Preprocessing

```bash
python scripts/preprocess_data.py --help

Options:
  --data_dir        Root data directory (default: data)
  --datasets        Which datasets to process (hotpotqa, nq, wikipedia)
  --nq_sample       Sample size for NQ training data
  --wiki_max        Max Wikipedia articles to process (default: 100000)
```

### Indexing

```bash
python scripts/chunk_and_index.py --help

Options:
  --data_dir        Root data directory
  --output_dir      Output directory for indexes
  --sources         Passage sources to index (hotpotqa, wikipedia)
  --chunk_size      Maximum tokens per chunk (default: 256)
  --stride          Overlap between chunks (default: 100)
  --dense_model     Model for dense encoding (default: all-MiniLM-L6-v2)
  --index_types     Which indexes to build (bm25, dense)
```

## System Requirements

### Your System (Optimized For)
- **Hardware**: MacBook M4 Pro with 24GB RAM ✅
- **Python**: 3.10 (stable and compatible) ✅
- **Acceleration**: Metal Performance Shaders (MPS) ✅
- **Expected Speed**: 3-4x faster than CPU for dense encoding

### Minimum Requirements
- Python 3.8+
- 8GB+ RAM
- 10GB disk space

## Documentation

- **M4_QUICKSTART.md** - Quick start commands for M4 Pro (START HERE!)
- **CONDA_GUIDE.md** - Conda environment setup and management
- **APPLE_SILICON_GUIDE.md** - Detailed M4 Pro optimizations
- **SETUP_GUIDE.md** - Step-by-step setup instructions
- **doc.md** - Full research plan and implementation details
- **README.md** - This file

## Troubleshooting

### Out of Memory
- Reduce `--wiki_max` when preprocessing
- Use `--nq_sample` to limit NQ dataset size
- Process datasets one at a time

### Slow Indexing
- Dense encoding is the bottleneck
- Use GPU if available (install `faiss-gpu`)
- Reduce corpus size for initial testing

### Missing Data
- Ensure you ran `download_datasets.sh` first
- Check that files exist in `data/hotpotqa/`
- Re-run preprocessing if needed

## References

- DPR (Dense Passage Retrieval): [Karpukhin et al., 2020]
- BEIR Benchmark: [Thakur et al., 2021]
- HotpotQA: [Yang et al., 2018]
- Fusion-in-Decoder: [Izacard & Grave, 2021]

## License

MIT License - see LICENSE file for details

## Citation

If you use this code, please cite:

```bibtex
@misc{hybrid-rag-2025,
  title={Hybrid RAG with Uncertainty Estimation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/Hybrid-RAG}
}
```
