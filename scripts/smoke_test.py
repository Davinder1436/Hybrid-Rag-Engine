"""
Smoke test for retrieval indexes.
Tests both BM25 and dense retrieval on sample queries.
Optimized for Apple Silicon M4 Pro.
"""

import argparse
from pathlib import Path
import json
import sys
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from chunk_and_index import BM25Index, DenseIndex, get_device


def test_retrieval(bm25_index: BM25Index, dense_index: DenseIndex, queries: list, top_k: int = 5):
    """Test retrieval with sample queries."""
    
    for i, query in enumerate(queries, 1):
        print("\n" + "="*80)
        print(f"Query {i}: {query}")
        print("="*80)
        
        # BM25 results
        print("\n[BM25 Results]")
        bm25_results = bm25_index.search(query, top_k=top_k)
        for j, result in enumerate(bm25_results[:top_k], 1):
            passage = result["passage"]
            score = result["score"]
            print(f"\n  {j}. [Score: {score:.3f}] {passage['title']}")
            print(f"     {passage['text'][:200]}...")
        
        # Dense results
        print("\n[Dense Results]")
        dense_results = dense_index.search(query, top_k=top_k)
        for j, result in enumerate(dense_results[:top_k], 1):
            passage = result["passage"]
            score = result["score"]
            print(f"\n  {j}. [Score: {score:.3f}] {passage['title']}")
            print(f"     {passage['text'][:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Test retrieval indexes")
    parser.add_argument(
        "--index_dir",
        type=str,
        default="data/indexes",
        help="Directory containing indexes"
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=[
            "Who was the first president of the United States?",
            "What is the capital of France?",
            "When did World War II end?"
        ],
        help="Test queries"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to show per query"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["mps", "cuda", "cpu"],
        help="Device for dense retrieval (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    index_dir = Path(args.index_dir)
    
    # Detect device
    device = args.device if args.device else get_device()
    
    print("="*80)
    print("Hybrid RAG: Retrieval Smoke Test")
    print("="*80)
    print(f"Index directory: {index_dir.absolute()}")
    print(f"Number of test queries: {len(args.queries)}")
    print(f"Device: {device}")
    if device == "mps":
        print("✓ Using Apple Silicon Metal acceleration")
    print("="*80)
    
    # Load indexes
    print("\nLoading indexes...")
    
    try:
        print("  Loading BM25 index...")
        bm25_index = BM25Index.load(index_dir)
        print(f"    Loaded {len(bm25_index.passages)} passages")
    except Exception as e:
        print(f"  Error loading BM25 index: {e}")
        return
    
    try:
        print("  Loading dense index...")
        dense_index = DenseIndex.load(index_dir, device=device)
        print(f"    Loaded {len(dense_index.passages)} passages")
        print(f"    Using device: {dense_index.device}")
    except Exception as e:
        print(f"  Error loading dense index: {e}")
        return
    
    # Run tests
    print("\n" + "="*80)
    print("Running retrieval tests...")
    print("="*80)
    
    test_retrieval(bm25_index, dense_index, args.queries, args.top_k)
    
    print("\n" + "="*80)
    print("Smoke test complete!")
    print("="*80)


if __name__ == "__main__":
    main()
