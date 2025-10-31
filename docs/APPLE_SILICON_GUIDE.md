# Apple Silicon (M4 Pro) Optimization Guide

This guide contains specific optimizations and recommendations for running Hybrid RAG on your MacBook M4 Pro with 24GB RAM.

## Hardware Specifications

- **Chip**: Apple M4 Pro
- **RAM**: 24GB unified memory
- **GPU**: Metal Performance Shaders (MPS)
- **Python**: 3.10 (stable and compatible)

## Metal Acceleration

### What is Metal/MPS?

Metal Performance Shaders (MPS) is Apple's GPU acceleration framework for neural network operations on Apple Silicon. PyTorch 2.0+ supports MPS backend, providing significant speedups for:

- Dense passage encoding (2-4x faster than CPU)
- Neural network inference
- Matrix operations

### Automatic Device Detection

The code automatically detects and uses the best available device:

```python
def get_device():
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    elif torch.cuda.is_available():
        return "cuda"  # NVIDIA GPU
    else:
        return "cpu"
```

### Verify MPS is Available

```bash
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

Expected output: `MPS available: True`

## Optimized Package Versions

All packages in `requirements.txt` are pinned to versions compatible with:
- Python 3.10
- Apple Silicon ARM64 architecture
- Metal acceleration

Key packages:
- `torch>=2.0.0` - MPS support
- `faiss-cpu` - ARM64 optimized
- `transformers` - M1/M2/M3/M4 compatible
- `sentence-transformers` - Works with MPS

## Performance Optimizations

### 1. Batch Size

**Default**: 64 passages per batch (optimized for 24GB RAM)

You have plenty of RAM, so larger batches = faster encoding:

```bash
# Standard (default)
python scripts/chunk_and_index.py --batch_size 64

# Aggressive (use more RAM)
python scripts/chunk_and_index.py --batch_size 128

# Conservative (if you have other apps running)
python scripts/chunk_and_index.py --batch_size 32
```

### 2. Model Selection

**Recommended models for M4 Pro:**

| Model | Dimensions | Speed on MPS | Quality |
|-------|-----------|--------------|---------|
| `all-MiniLM-L6-v2` | 384 | Fast ⚡ | Good ✓ |
| `all-MiniLM-L12-v2` | 384 | Medium | Better ✓✓ |
| `all-mpnet-base-v2` | 768 | Medium | Best ✓✓✓ |
| `multi-qa-mpnet-base-dot-v1` | 768 | Medium | Best for QA ✓✓✓ |

**Recommendation**: Start with `all-MiniLM-L6-v2`, upgrade to `all-mpnet-base-v2` for better quality.

```bash
# Fast model (default)
python scripts/chunk_and_index.py \
    --dense_model sentence-transformers/all-MiniLM-L6-v2

# Better quality
python scripts/chunk_and_index.py \
    --dense_model sentence-transformers/all-mpnet-base-v2
```

### 3. Memory Usage

Your 24GB unified memory is shared between CPU and GPU. Monitor usage:

```bash
# Install memory monitoring
pip install psutil

# Check memory during indexing
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB')"
```

**Estimated memory usage:**

| Corpus Size | Passages | Batch 64 | Batch 128 |
|-------------|----------|----------|-----------|
| Small | 50k | ~4 GB | ~6 GB |
| Medium | 200k | ~8 GB | ~12 GB |
| Large | 500k | ~15 GB | ~20 GB |
| Full Wikipedia | 21M | Use streaming | Use streaming |

### 4. FAISS Index Optimization

For your RAM and CPU architecture:

**Current (Flat index)**: Exact search, fast for <1M vectors
- Works well for 50k-500k passages
- Uses ~1.5GB for 384-dim vectors with 500k passages

**Future (IVF index)**: Approximate search for >1M vectors
```python
# For larger corpora
index = faiss.IndexIVFFlat(quantizer, dimension, nlist=100)
```

## Performance Expectations

### Dense Encoding (100k passages, all-MiniLM-L6-v2)

| Device | Batch Size | Time | Speed |
|--------|-----------|------|-------|
| CPU (10 cores) | 32 | ~25 min | ~4k p/min |
| MPS (M4 Pro) | 64 | ~8 min | ~12k p/min |
| MPS (M4 Pro) | 128 | ~6 min | ~16k p/min |

**Your M4 Pro with MPS should be 3-4x faster than CPU!**

### Full Pipeline (HotpotQA + 50k Wikipedia)

| Step | Time (MPS) | Time (CPU) |
|------|-----------|-----------|
| Download | 5 min | 5 min |
| Preprocess | 8 min | 8 min |
| Chunk | 2 min | 2 min |
| BM25 index | 3 min | 3 min |
| Dense encode | 6 min | 20 min |
| FAISS build | 1 min | 1 min |
| **Total** | **~25 min** | **~40 min** |

## Best Practices for M4 Pro

### 1. Close Other Apps

For maximum performance during indexing:
```bash
# Close memory-intensive apps (Chrome, etc.)
# This frees up more unified memory for MPS
```

### 2. Monitor Temperature

M4 Pro has excellent thermals, but heavy workloads can throttle:
```bash
# Check if throttling occurs
sudo powermetrics --samplers smc -i1000 -n1 | grep -i "CPU die temperature"
```

### 3. Use Activity Monitor

Watch GPU utilization:
- Open Activity Monitor
- Window → GPU History
- You should see high GPU usage during dense encoding

### 4. Power Settings

For long indexing runs:
```bash
# Prevent sleep
caffeinate -i python scripts/chunk_and_index.py
```

## Optimized Workflow for M4 Pro

### Quick Start (10k Wikipedia)
```bash
# Should complete in ~15 minutes
python scripts/preprocess_data.py --wiki_max 10000
python scripts/chunk_and_index.py --batch_size 128
```

### Medium Scale (100k Wikipedia)
```bash
# Should complete in ~45 minutes
python scripts/preprocess_data.py --wiki_max 100000
python scripts/chunk_and_index.py --batch_size 128 --device mps
```

### Large Scale (500k passages)
```bash
# Should complete in ~2 hours
python scripts/preprocess_data.py --wiki_max 500000
python scripts/chunk_and_index.py \
    --batch_size 96 \
    --device mps \
    --dense_model sentence-transformers/all-MiniLM-L6-v2
```

## Troubleshooting

### MPS Not Available

```bash
# Check PyTorch version
python3 -c "import torch; print(torch.__version__)"

# Should be >= 2.0.0
# If not, upgrade:
pip install --upgrade torch torchvision torchaudio
```

### Out of Memory Errors

```bash
# Reduce batch size
python scripts/chunk_and_index.py --batch_size 32

# Or use a smaller model
python scripts/chunk_and_index.py \
    --dense_model sentence-transformers/all-MiniLM-L6-v2
```

### Slow Encoding (Not Using MPS)

```bash
# Explicitly specify device
python scripts/chunk_and_index.py --device mps

# Verify MPS is being used (should see in output):
# "Using device: mps"
# "✓ Using Apple Silicon Metal acceleration"
```

### FAISS Installation Issues

```bash
# Ensure ARM64 version is installed
pip uninstall faiss-cpu faiss-gpu
pip install faiss-cpu --no-cache-dir

# Verify installation
python3 -c "import faiss; print(f'FAISS version: {faiss.__version__}')"
```

## Advanced Optimizations

### 1. Mixed Precision (Future)

For even faster encoding with larger models:
```python
# Add to DenseIndex class
self.encoder = SentenceTransformer(model_name, device=device)
if device == "mps":
    # Use float16 for faster computation
    self.encoder = self.encoder.half()
```

### 2. Parallel Processing

For data preprocessing:
```python
# Use multiprocessing for chunking
from multiprocessing import Pool
with Pool(10) as p:  # M4 Pro has 10 performance cores
    results = p.map(chunk_function, passages)
```

### 3. Streaming for Large Corpora

For >1M passages:
```python
# Process in chunks to avoid memory issues
for chunk in chunks(passages, chunk_size=100000):
    encode_and_index(chunk)
```

## Benchmarking Your System

Run this to benchmark your M4 Pro:

```bash
# Create benchmark script
cat > benchmark_m4.py << 'EOF'
import torch
import time
from sentence_transformers import SentenceTransformer

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

texts = ["This is a test sentence"] * 1000

start = time.time()
embeddings = model.encode(texts, batch_size=64, convert_to_numpy=True)
elapsed = time.time() - start

print(f"Device: {device}")
print(f"Encoded 1000 sentences in {elapsed:.2f}s")
print(f"Speed: {1000/elapsed:.0f} sentences/sec")
EOF

python benchmark_m4.py
```

**Expected results on M4 Pro:**
- MPS: ~1500-2000 sentences/sec
- CPU: ~400-600 sentences/sec

## Summary

Your M4 Pro with 24GB RAM is **excellent** for this project:

✅ Plenty of unified memory for large batches
✅ Metal acceleration for 3-4x faster encoding  
✅ Can handle 500k+ passages comfortably
✅ Fast enough for iterative development

**Recommended settings:**
- Batch size: 64-128
- Device: mps (auto-detected)
- Model: all-MiniLM-L6-v2 (start) → all-mpnet-base-v2 (production)

**You should be able to:**
- Index 100k passages in ~10 minutes
- Index full HotpotQA + Wikipedia subset in ~30 minutes
- Iterate quickly on experiments

Enjoy the speed of Apple Silicon! 🚀
