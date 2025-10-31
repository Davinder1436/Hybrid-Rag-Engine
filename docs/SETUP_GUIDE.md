# Setup Guide: Step-by-Step Instructions

This guide walks you through setting up the Hybrid RAG pipeline from scratch.

## Step 1: Environment Setup

### Create Virtual Environment

```bash
cd /Users/dave/Work/Research/Hybrid-RAG

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate
```

### Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# This installs:
# - PyTorch and Transformers for neural models
# - FAISS for dense retrieval indexing
# - Pyserini for BM25
# - sentence-transformers for encoding
# - datasets library for HuggingFace datasets
# - and more...
```

### Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt')"
```

## Step 2: Download Raw Datasets

### Download HotpotQA

```bash
# Make the script executable
chmod +x scripts/download_datasets.sh

# Run the download script
bash scripts/download_datasets.sh data
```

This downloads:
- HotpotQA training set (~90,000 examples)
- HotpotQA dev set with distractors (~7,400 examples)

**Note:** Natural Questions and Wikipedia will be downloaded automatically during preprocessing via HuggingFace datasets.

## Step 3: Preprocess Datasets

### Process All Datasets (Recommended for Quick Start)

```bash
# Process HotpotQA and a small Wikipedia subset
python scripts/preprocess_data.py \
    --data_dir data \
    --datasets hotpotqa wikipedia \
    --wiki_max 10000
```

This will:
1. Process HotpotQA train/dev splits into unified format
2. Extract passages from HotpotQA contexts
3. Download and process 10,000 Wikipedia articles
4. Save all processed data to `data/processed/`

**Expected time:** 10-20 minutes (depends on download speed)

### Process Individual Datasets

```bash
# Only HotpotQA
python scripts/preprocess_data.py \
    --data_dir data \
    --datasets hotpotqa

# Only Wikipedia (more articles)
python scripts/preprocess_data.py \
    --data_dir data \
    --datasets wikipedia \
    --wiki_max 50000

# Add Natural Questions
python scripts/preprocess_data.py \
    --data_dir data \
    --datasets nq \
    --nq_sample 10000
```

### Verify Preprocessing

```bash
# Check processed files
ls -lh data/processed/hotpotqa/
ls -lh data/processed/corpus/
ls -lh data/processed/wikipedia/
```

You should see:
- `data/processed/hotpotqa/train.json` and `dev.json`
- `data/processed/corpus/hotpotqa_passages.jsonl`
- `data/processed/wikipedia/passages.jsonl`

## Step 4: Build Retrieval Indexes

### Build Both Indexes (Recommended)

```bash
python scripts/chunk_and_index.py \
    --data_dir data \
    --output_dir data/indexes \
    --sources hotpotqa wikipedia \
    --chunk_size 256 \
    --stride 100 \
    --dense_model sentence-transformers/all-MiniLM-L6-v2
```

This will:
1. Load passages from HotpotQA and Wikipedia
2. Chunk passages into 256-token segments with 100-token overlap
3. Build BM25 sparse index (fast)
4. Build FAISS dense index using sentence transformers
5. Save indexes to `data/indexes/`

**Expected time:** 20-40 minutes (depends on corpus size and CPU/GPU)

### Build Individual Indexes

```bash
# Only BM25 (fast)
python scripts/chunk_and_index.py \
    --data_dir data \
    --output_dir data/indexes \
    --sources hotpotqa \
    --index_types bm25

# Only Dense (slower, but more accurate)
python scripts/chunk_and_index.py \
    --data_dir data \
    --output_dir data/indexes \
    --sources hotpotqa \
    --index_types dense
```

### Verify Indexes

```bash
# Check index files
ls -lh data/indexes/

# Should see:
# - bm25_index.pkl
# - faiss_index.bin
# - dense_metadata.pkl
# - chunked_passages.jsonl
```

## Step 5: Test Retrieval

### Run Smoke Test

```bash
python scripts/smoke_test.py \
    --index_dir data/indexes \
    --top_k 5 \
    --queries \
        "Who was the first president of the United States?" \
        "What is the capital of France?" \
        "When did World War II end?"
```

This will:
1. Load both BM25 and dense indexes
2. Run each query through both retrievers
3. Display top-5 results from each

**Expected output:** You should see retrieved passages with scores for each query.

### Test with Custom Queries

```bash
python scripts/smoke_test.py \
    --index_dir data/indexes \
    --top_k 10 \
    --queries "Your custom question here"
```

## Step 6: Next Steps

After completing the setup, you can proceed with:

### 1. Implement Hybrid Fusion
Combine BM25 and dense results:
- Union strategy
- Score normalization
- Reranking with cross-encoders

### 2. Train Uncertainty Head
Build the calibrated confidence estimator:
- Auxiliary head on retriever
- Train with reader success labels
- Temperature scaling for calibration

### 3. Add Reader Model
Implement Fusion-in-Decoder:
- T5-based generator
- Condition on retrieved passages
- Track provenance for faithfulness

### 4. Full Evaluation
Run experiments on:
- HotpotQA multi-hop questions
- Natural Questions open-domain QA
- BEIR benchmark for generalization

## Troubleshooting

### Installation Issues

**Problem:** FAISS installation fails
```bash
# Try CPU-only version
pip install faiss-cpu

# For GPU version (requires CUDA)
pip install faiss-gpu
```

**Problem:** PyTorch not found
```bash
# Install PyTorch for your system
# Visit: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio
```

### Preprocessing Issues

**Problem:** Out of memory during Wikipedia download
```bash
# Reduce the number of articles
python scripts/preprocess_data.py \
    --datasets wikipedia \
    --wiki_max 5000
```

**Problem:** HotpotQA files not found
```bash
# Re-run download script
bash scripts/download_datasets.sh data

# Check if files exist
ls data/hotpotqa/
```

### Indexing Issues

**Problem:** Dense encoding is very slow
```bash
# Use a smaller model
python scripts/chunk_and_index.py \
    --dense_model sentence-transformers/all-MiniLM-L6-v2

# Or reduce corpus size
python scripts/chunk_and_index.py \
    --sources hotpotqa  # Skip Wikipedia
```

**Problem:** FAISS build fails with memory error
```bash
# Process fewer passages
# Edit preprocess_data.py to limit passage count
# Or use IVF index (to be implemented)
```

### Testing Issues

**Problem:** No results returned
```bash
# Check that indexes were built correctly
ls -lh data/indexes/

# Verify passages were chunked
wc -l data/indexes/chunked_passages.jsonl
```

## Resource Requirements

### Minimum
- **CPU:** 4 cores
- **RAM:** 8 GB
- **Disk:** 10 GB
- **Time:** ~1 hour for small setup (10k Wikipedia)

### Recommended
- **CPU:** 8+ cores or GPU
- **RAM:** 16 GB
- **Disk:** 50 GB
- **Time:** ~2 hours for medium setup (100k Wikipedia)

### Full Scale
- **CPU:** Multi-core or GPU
- **RAM:** 32+ GB
- **Disk:** 100+ GB
- **Time:** Several hours for full Wikipedia

## Directory Structure After Setup

```
Hybrid-RAG/
├── data/
│   ├── hotpotqa/
│   │   ├── hotpot_train_v1.1.json
│   │   └── hotpot_dev_distractor_v1.json
│   ├── processed/
│   │   ├── hotpotqa/
│   │   │   ├── train.json
│   │   │   └── dev.json
│   │   ├── corpus/
│   │   │   └── hotpotqa_passages.jsonl
│   │   └── wikipedia/
│   │       ├── passages.jsonl
│   │       └── metadata.json
│   └── indexes/
│       ├── bm25_index.pkl
│       ├── faiss_index.bin
│       ├── dense_metadata.pkl
│       └── chunked_passages.jsonl
├── scripts/
│   ├── download_datasets.sh
│   ├── preprocess_data.py
│   ├── chunk_and_index.py
│   └── smoke_test.py
└── venv/  # Virtual environment
```

## Quick Reference Commands

```bash
# Full setup from scratch
source venv/bin/activate
pip install -r requirements.txt
bash scripts/download_datasets.sh data
python scripts/preprocess_data.py --wiki_max 10000
python scripts/chunk_and_index.py
python scripts/smoke_test.py

# Rebuild indexes only
python scripts/chunk_and_index.py --sources hotpotqa wikipedia

# Test with new queries
python scripts/smoke_test.py --queries "Your question"

# Clean and restart
rm -rf data/indexes data/processed
python scripts/preprocess_data.py
python scripts/chunk_and_index.py
```

## Getting Help

- Check `doc.md` for detailed component documentation
- Review `README.md` for project overview
- Inspect script help: `python scripts/[script].py --help`
- Debug with Python: `python -i scripts/[script].py [args]`

---

**Ready to start?** Run the commands in order and you'll have a working retrieval system in about an hour!
