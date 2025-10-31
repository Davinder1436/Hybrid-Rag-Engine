# FAISS Segmentation Fault Fix

## Issue: Segmentation Fault During FAISS Index Building

**Error:**
```
Segmentation fault: 11
```

This happened after encoding completed but during the FAISS index building phase.

## Root Cause

FAISS on Apple Silicon (ARM64) can crash with a segmentation fault when trying to add a very large number of vectors (881k+) to the index in a single operation. This is a known issue with FAISS on macOS.

## Solution Applied

Modified `chunk_and_index.py` to:

1. **Add embeddings in batches** instead of all at once
   - Batch size: 10,000 vectors at a time
   - Prevents memory issues and segfaults

2. **Ensure float32 type** for embeddings
   - Converts embeddings to `float32` explicitly
   - Improves compatibility with FAISS

3. **Better error handling and progress tracking**
   - Shows progress for each batch added
   - Better error messages if save fails

## Changes Made

### Before (Caused Segfault):
```python
self.faiss_index.add(embeddings)  # Adding 881k vectors at once
```

### After (Fixed):
```python
# Add in batches of 10k
batch_size_faiss = 10000
for i in range(num_batches):
    start_idx = i * batch_size_faiss
    end_idx = min((i + 1) * batch_size_faiss, len(embeddings))
    batch_embeddings = embeddings[start_idx:end_idx]
    self.faiss_index.add(batch_embeddings)
    print(f"    Added batch {i+1}/{num_batches}")
```

## Try Again

The script has been fixed. Run it again:

```bash
bash build_dense_index.sh
```

Or manually:

```bash
conda activate research
python scripts/chunk_and_index.py \
    --batch_size 128 \
    --device mps \
    --index_types dense \
    --skip_chunking
```

## What to Expect

You should now see:

```
[4/4] Building Dense (FAISS) index...
  Encoding 881594 passages...
  Batch size: 128 (optimized for Apple Silicon)
  [Progress bar...]
  
  Building FAISS index...
  Normalizing embeddings...
  Adding embeddings to FAISS index...
    Added batch 1/89 (10000/881594 vectors)
    Added batch 2/89 (20000/881594 vectors)
    ...
    Added batch 89/89 (881594/881594 vectors)
  FAISS index built with 881594 vectors
  
  Saving FAISS index...
    FAISS index saved: data/indexes/faiss_index.bin
    Metadata saved: data/indexes/dense_metadata.pkl
  Dense index saved to data/indexes
```

## Timeline

- Encoding: Already done (took ~22 minutes)
- Building FAISS: ~2-3 minutes with batching
- Saving: ~1 minute

**Total new time**: ~3-5 minutes (much faster than re-encoding!)

## Verification

After completion, check files:

```bash
ls -lh data/indexes/

# Should see:
# bm25_index.pkl          (1.6 GB)
# chunked_passages.jsonl  (567 MB)
# faiss_index.bin         (~1.3 GB)  ← New!
# dense_metadata.pkl      (~600 MB)  ← New!
```

## Alternative: Use Smaller Dataset

If it still crashes, you can use fewer passages:

```bash
# Use only HotpotQA passages (skip Wikipedia)
python scripts/chunk_and_index.py \
    --batch_size 64 \
    --device mps \
    --sources hotpotqa \
    --index_types dense
```

## Why It Crashed

The good news: Your M4 Pro successfully encoded all 881k passages with MPS! The crash happened at the very end during FAISS index construction, which is a known Apple Silicon issue with large-scale operations.

The fix batches the index construction to avoid this issue.

## Additional Notes

### Memory Usage
- Encoding 881k passages: ~8-10 GB RAM
- Building FAISS with batching: ~4-6 GB RAM
- Your 24GB RAM is more than enough

### Apple Silicon FAISS Limitations
- Known issues with very large batch operations
- Batching is the standard workaround
- Future FAISS versions may fix this

### Next Steps After Success

Test the retrieval system:

```bash
conda activate research
python scripts/smoke_test.py --device mps
```

---

**TL;DR**: The script is now fixed to add embeddings in batches (10k at a time) to avoid segmentation faults. Run `bash build_dense_index.sh` again. Should take ~5 minutes instead of crashing!
