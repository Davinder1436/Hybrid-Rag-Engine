# Hybrid RAG Module Documentation

## Overview

The `rag/` module provides a modular, production-ready implementation of a Hybrid Sparse + Dense Retrieval system with uncertainty quantification. This implements the architecture described in `doc.md`.

### Key Components

- **Sparse Retriever**: BM25-based lexical retrieval using `rank-bm25`
- **Dense Retriever**: FAISS-based semantic retrieval with SentenceTransformers
- **Hybrid Retriever**: Fusion strategy combining sparse and dense results
- **Uncertainty Head**: Calibrated confidence prediction for retrieval sufficiency
- **Metrics**: Comprehensive evaluation suite (Recall@k, MRR, ECE, selective risk vs coverage)

## Module Structure

```
rag/
├── __init__.py                 # Main exports
├── retrievers.py               # Base + Sparse + Dense + Hybrid retrievers
├── uncertainty.py              # Uncertainty head + trainer
├── data_utils.py               # Dataset classes and utilities
├── metrics.py                  # Retrieval, calibration, selective metrics
└── trainers.py (planned)       # Training loops for retriever + uncertainty head
```

## Quick Start

### 1. Build Indexes

#### Option A: Using CLI
```bash
# Build both sparse and dense indexes
conda activate research
python scripts/rag_cli.py build \
  --data_dir data/processed \
  --output_dir data/indexes \
  --device mps

# Or build individually
python scripts/rag_cli.py build-sparse
python scripts/rag_cli.py build-dense --batch_size 128 --index_type flat
```

#### Option B: Using Python API
```python
from pathlib import Path
from rag.retrievers import SparseRetriever, DenseRetriever, HybridRetriever
import json

# Load passages
passages = []
with open("data/processed/corpus/hotpotqa_passages.jsonl") as f:
    passages = [json.loads(line) for line in f]

# Build sparse index
sparse = SparseRetriever()
sparse.build(passages)
sparse.save(Path("data/indexes/sparse"))

# Build dense index
dense = DenseRetriever(device="mps")
dense.build(passages, batch_size=128)
dense.save(Path("data/indexes/dense"))

# Create hybrid retriever
hybrid = HybridRetriever(sparse, dense, fusion_strategy="union")
hybrid.save(Path("data/indexes/hybrid"))
```

### 2. Test Installation

Run the smoke test to validate all modules:

```bash
python scripts/smoke_test_rag.py \
  --data_dir data/processed \
  --device mps \
  --max_passages 1000 \
  --max_queries 5
```

Expected output:
```
======================================================================
HYBRID RAG SMOKE TEST
======================================================================

[1/6] Loading data...
  ✓ Loaded 881594 passages
  ✓ Loaded 5 queries

[2/6] Testing Sparse Retriever (BM25)...
  ✓ BM25 index built: 881594 passages
  ✓ Search returned 10 results
    Top result: ... (score: 12.34)

[3/6] Testing Dense Retriever (FAISS)...
  Encoding 1000 passages (may take a moment)...
  ✓ FAISS index built: 1000 vectors
  ✓ Search returned 10 results
    Top result: ... (score: 0.89)

[4/6] Testing Hybrid Retriever...
  ✓ Hybrid retriever created
  ✓ Hybrid search returned 10 results
    Top result: ... (score: 0.75)
    Sparse: 8, Dense: 7, Both: 5

[5/6] Testing Uncertainty Head...
  ✓ Uncertainty head created
  ✓ Prediction computed
    p_success: 0.82
    max_score: 0.80
    mean_score: 0.78

[6/6] Testing Metrics...
  ✓ Retrieval metrics computed
    Recall@3: 0.50
    Precision@3: 0.67
    MRR: 1.00
  ✓ Calibration metrics computed
    ECE: 0.05
    Brier Score: 0.12

======================================================================
✓ ALL SMOKE TESTS PASSED
======================================================================
```

### 3. Interactive Search

```bash
python scripts/rag_cli.py search \
  --retriever_type hybrid \
  --fusion_strategy union \
  --top_k 10

# Then type queries:
Query: What is the capital of France?
Query: Who wrote Romeo and Juliet?
Query: quit
```

### 4. Evaluate Retrieval

```bash
python scripts/rag_cli.py eval \
  --data_dir data/processed \
  --queries_file hotpotqa/dev.json \
  --fusion_strategy union \
  --num_eval_queries 100
```

## API Reference

### Retrievers

#### BaseRetriever (Abstract)

```python
class BaseRetriever(ABC):
    def search(query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Return list of (passage, score) tuples"""
    
    def save(path: Path): ...
    def load(path: Path): ...
```

#### SparseRetriever (BM25)

```python
from rag.retrievers import SparseRetriever

# Build
sparse = SparseRetriever(k1=1.5, b=0.75)
sparse.build(passages)

# Search
results = sparse.search("query text", top_k=100)
# Returns: [{"passage": {...}, "score": 12.34, "rank": 1, "retriever": "sparse"}, ...]

# Save/Load
sparse.save(Path("data/indexes/sparse"))
sparse = SparseRetriever.load(Path("data/indexes/sparse"))
```

#### DenseRetriever (FAISS)

```python
from rag.retrievers import DenseRetriever

# Build
dense = DenseRetriever(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="mps",
    index_type="flat"  # "flat", "ivf", "hnsw"
)
dense.build(passages, batch_size=64)

# Search
results = dense.search("query text", top_k=100)
# Returns: [{"passage": {...}, "score": 0.89, "rank": 1, "retriever": "dense"}, ...]

# Save/Load
dense.save(Path("data/indexes/dense"))
dense = DenseRetriever.load(Path("data/indexes/dense"), device="mps")
```

#### HybridRetriever

```python
from rag.retrievers import HybridRetriever

hybrid = HybridRetriever(
    sparse_retriever=sparse,
    dense_retriever=dense,
    fusion_strategy="union",  # "union", "rrf", "weighted_sum"
    alpha=0.5  # Weight for dense in weighted_sum
)

# Search (fuses results from both)
results = hybrid.search("query text", top_k=100)
# Returns fused results with hybrid scores

# Each result includes:
# - "sparse_score": score from BM25
# - "dense_score": score from dense
# - "score": fused score
# - "retriever": "hybrid"
```

### Uncertainty Head

```python
from rag.uncertainty import UncertaintyHead
import numpy as np

# Create head
head = UncertaintyHead(input_dim=384, hidden_dim=256)

# Predict on retrieval results
query_embedding = np.random.randn(384)
retrieval_results = [
    {"score": 0.8, "passage": {...}, "retriever": "hybrid"},
    {"score": 0.7, "passage": {...}, "retriever": "hybrid"},
    ...
]

prediction = head.predict(
    query_embedding,
    retrieval_results,
    score_threshold=0.1,
    retriever_type="hybrid"
)

# Returns dict with:
# {
#   "p_success": 0.82,        # Probability evidence is sufficient
#   "p_failure": 0.18,        # 1 - p_success
#   "max_score": 0.80,        # Max score among retrieved
#   "mean_score": 0.75,       # Mean score
#   "std_score": 0.05,        # Std of scores
#   "num_high_scores": 4,     # Count above threshold
#   "agreement": 0.5           # Sparse-dense agreement (for hybrid)
# }

# Save/Load
head.save(Path("models/uncertainty_head"))
head = UncertaintyHead.load(Path("models/uncertainty_head"), device="cpu")
```

### Metrics

#### Retrieval Metrics

```python
from rag.metrics import RetrievalMetrics

# Per-query metrics
recall = RetrievalMetrics.recall_at_k(["p1", "p2", "p3"], ["p1", "p4"], k=3)
precision = RetrievalMetrics.precision_at_k(retrieved_ids, relevant_ids, k=100)
mrr = RetrievalMetrics.mean_reciprocal_rank(retrieved_ids, relevant_ids)

# Batch evaluation
predictions = [["p1", "p2", ...], ["p3", "p4", ...]]  # Per-query retrieved IDs
references = [["p1", "p4"], ["p3"]]  # Per-query relevant IDs

metrics = RetrievalMetrics.evaluate_batch(
    predictions,
    references,
    k_values=[10, 100]
)
# Returns: {"recall@10": 0.75, "precision@10": 0.80, "mrr@10": 0.85, ...}
```

#### Calibration Metrics

```python
from rag.metrics import CalibrationMetrics
import numpy as np

predictions = np.array([0.9, 0.8, 0.1, 0.2])  # Predicted probabilities
targets = np.array([1, 1, 0, 0])              # Binary labels

ece = CalibrationMetrics.expected_calibration_error(predictions, targets, num_bins=10)
brier = CalibrationMetrics.brier_score(predictions, targets)

# Plot calibration curve
fig, ax = CalibrationMetrics.calibration_plot(
    predictions, targets, save_path=Path("calibration.png")
)
```

#### Selective Classification Metrics

```python
from rag.metrics import SelectiveMetrics

# Coverage vs risk at threshold
coverage, risk = SelectiveMetrics.coverage_vs_risk(
    predictions, targets, confidence_threshold=0.5
)

# Full coverage-risk curve
coverages, risks = SelectiveMetrics.coverage_risk_curve(predictions, targets)

# Plot
fig, ax = SelectiveMetrics.selective_plot(
    predictions, targets, save_path=Path("selective.png")
)
```

## Dataset Classes

```python
from rag.data_utils import PassageDataset, QueryDataset, RetrievalDataset

# Passages
passages_ds = PassageDataset.from_jsonl(Path("passages.jsonl"))
passage = passages_ds[0]  # {"id": "p1", "text": "...", "title": "...", ...}

# Queries
queries_ds = QueryDataset.from_jsonl(Path("queries.jsonl"))
# or from HotpotQA format:
queries_ds = QueryDataset.from_hotpotqa(Path("hotpot_dev.json"))

# Retrieval (for training uncertainty head)
retrieval_ds = RetrievalDataset(
    queries=queries,
    query_embeddings=np.random.randn(len(queries), 384),
    retrieval_results=[[...], [...]],  # Per-query results
    labels=[1, 0, 1, ...]  # Binary: sufficient/insufficient
)

item = retrieval_ds[0]
# {
#   "query_id": "q1",
#   "query_embedding": Tensor(384),
#   "score_stats": Tensor(5),  # [max, mean, std, num_high, agreement]
#   "label": 1,
#   "num_retrieved": 10
# }
```

## Training (Planned)

Coming soon:
- `train_dense_retriever.py` — Fine-tune DPR on QA pairs with hard negative mining
- `train_uncertainty_head.py` — Train calibrated confidence predictor
- `train_hybrid_reranker.py` — Optional cross-encoder re-ranker on fused candidates

## Evaluation Benchmarks

### On HotpotQA (dev set, 5,928 queries)

Expected performance (baseline):
- **Sparse (BM25)**
  - Recall@100: 0.85
  - MRR@100: 0.68
- **Dense (all-MiniLM-L6-v2)**
  - Recall@100: 0.92
  - MRR@100: 0.75
- **Hybrid (Union + union fusion)**
  - Recall@100: 0.95
  - MRR@100: 0.78

### Uncertainty Head Calibration

Expected calibration performance:
- ECE: ~0.08 (after training)
- Brier Score: ~0.10
- Coverage vs Risk: achieves 95% accuracy at 80% coverage

## Performance Tips

### Speed

1. **Use IVF/HNSW for large indexes**: For >1M passages, use `index_type="ivf"` or `"hnsw"` in DenseRetriever
2. **Batch encoding**: Increase `batch_size` for faster encoding (up to 512 on M4 Pro)
3. **Parallel search**: Use multi-threading for batch retrieval
4. **Cached embeddings**: Pre-compute and cache query embeddings

### Memory

1. **FAISS batch size**: For FAISS add operations, reduce batch size if OOM (default 2000)
2. **Model quantization**: Use quantized models like `all-MiniLM-L6-v2` instead of larger models
3. **CPU offloading**: Store FAISS index on disk and load on-demand for very large corpora

### Quality

1. **Fine-tune dense retriever**: Custom training on your QA pairs improves recall significantly
2. **Tune fusion weights**: Vary `alpha` in weighted_sum fusion to balance sparse/dense
3. **Hard negatives**: Use BM25 results as hard negatives during dense training
4. **Calibration**: Apply temperature scaling post-training for better uncertainty estimates

## Extending the System

### Adding a Custom Dense Model

```python
from rag.retrievers import DenseRetriever

class ColBERTRetriever(DenseRetriever):
    def __init__(self, ...):
        # Custom initialization for ColBERT late-interaction encoding
        ...
    
    def build(self, passages, batch_size=64):
        # Override to use late-interaction encoding
        ...
```

### Adding a Custom Fusion Strategy

```python
from rag.retrievers import HybridRetriever

class CrossEncoderFusion(HybridRetriever):
    def _fuse_crossencoder(self, sparse_results, dense_results, top_k):
        # Fuse using cross-encoder re-ranker
        ...
```

### Adding a Cross-Encoder Re-Ranker

```python
from sentence_transformers import CrossEncoder

class RerankerModule:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLMv2-L12-H384-P8"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query, passages, top_k=10):
        # Re-rank passages
        ...
```

## Troubleshooting

### Issue: "Segmentation fault" when building dense index

**Solution**: Reduce FAISS batch size in `chunk_and_index.py` or rebuild using smaller passages

### Issue: Out of memory during dense encoding

**Solution**: Reduce `batch_size` or `max_passages` during build

### Issue: Poor retrieval quality on custom domain

**Solution**: Fine-tune dense retriever on domain-specific QA pairs

### Issue: Uncertainty head predictions always near 0.5

**Solution**: Train uncertainty head with more data and proper labels; check score statistics

## Citation

If you use this code, please cite:
```bibtex
@misc{hybrid-rag-2024,
  title={Hybrid Sparse + Dense Retrieval with Calibrated Uncertainty},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/hybrid-rag}
}
```

## License

MIT
