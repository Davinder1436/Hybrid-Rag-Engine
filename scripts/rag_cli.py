#!/usr/bin/env python
"""
CLI for building and evaluating Hybrid RAG system.
"""

import argparse
from pathlib import Path
import json
import logging
from typing import List, Dict, Any

from rag.retrievers import SparseRetriever, DenseRetriever, HybridRetriever
from rag.data_utils import PassageDataset, QueryDataset, batch_encode_passages, batch_encode_queries
from rag.metrics import RetrievalMetrics


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_sparse_index(args):
    """Build sparse (BM25) index."""
    logger.info("Building sparse index...")
    
    # Load passages
    passages_file = Path(args.data_dir) / args.passages_file
    with open(passages_file, 'r') as f:
        passages = [json.loads(line) for line in f]
    
    logger.info(f"Loaded {len(passages)} passages")
    
    # Build sparse index
    sparse = SparseRetriever()
    sparse.build(passages)
    
    # Save
    output_path = Path(args.output_dir)
    sparse.save(output_path / "sparse")
    
    logger.info(f"✓ Sparse index saved to {output_path / 'sparse'}")


def build_dense_index(args):
    """Build dense (FAISS) index."""
    logger.info("Building dense index...")
    
    # Load passages
    passages_file = Path(args.data_dir) / args.passages_file
    with open(passages_file, 'r') as f:
        passages = [json.loads(line) for line in f]
    
    logger.info(f"Loaded {len(passages)} passages")
    
    # Build dense index
    dense = DenseRetriever(device=args.device, index_type=args.index_type)
    dense.build(passages, batch_size=args.batch_size)
    
    # Save
    output_path = Path(args.output_dir)
    dense.save(output_path / "dense")
    
    logger.info(f"✓ Dense index saved to {output_path / 'dense'}")


def build_hybrid_index(args):
    """Build both sparse and dense indexes."""
    build_sparse_index(args)
    build_dense_index(args)
    
    logger.info("✓ Hybrid indexes built successfully")


def evaluate_retrieval(args):
    """Evaluate retrieval quality."""
    logger.info("Evaluating retrieval...")
    
    # Load indexes
    output_path = Path(args.output_dir)
    sparse = SparseRetriever.load(output_path / "sparse")
    dense = DenseRetriever.load(output_path / "dense", device=args.device)
    hybrid = HybridRetriever(sparse, dense, fusion_strategy=args.fusion_strategy)
    
    # Load queries
    queries_file = Path(args.data_dir) / args.queries_file
    with open(queries_file, 'r') as f:
        queries = [json.loads(line) for line in f]
    
    logger.info(f"Loaded {len(queries)} queries")
    
    # Evaluate
    retriever_types = {
        "sparse": sparse,
        "dense": dense,
        "hybrid": hybrid
    }
    
    for retriever_name, retriever in retriever_types.items():
        logger.info(f"\nEvaluating {retriever_name}...")
        
        predictions = []
        references = []
        
        for query in queries[:args.num_eval_queries]:
            results = retriever.search(query["question"], top_k=100)
            retrieved_ids = [r["passage"]["id"] for r in results]
            predictions.append(retrieved_ids)
            
            # For now, use supporting facts if available
            if "supporting_facts" in query:
                relevant_ids = [fact[0] for fact in query["supporting_facts"]]
                references.append(relevant_ids)
        
        if references:
            metrics = RetrievalMetrics.evaluate_batch(predictions, references, k_values=[10, 100])
            
            for metric_name, metric_value in metrics.items():
                logger.info(f"  {retriever_name}/{metric_name}: {metric_value:.4f}")


def interactive_search(args):
    """Interactive search interface."""
    logger.info("Loading indexes...")
    
    output_path = Path(args.output_dir)
    
    if args.retriever_type in ["sparse", "hybrid"]:
        sparse = SparseRetriever.load(output_path / "sparse")
    
    if args.retriever_type in ["dense", "hybrid"]:
        dense = DenseRetriever.load(output_path / "dense", device=args.device)
    
    if args.retriever_type == "hybrid":
        retriever = HybridRetriever(sparse, dense, fusion_strategy=args.fusion_strategy)
    elif args.retriever_type == "sparse":
        retriever = sparse
    else:
        retriever = dense
    
    logger.info(f"Using {args.retriever_type} retriever")
    logger.info("Type 'quit' to exit\n")
    
    while True:
        query = input("Query: ").strip()
        
        if query.lower() == "quit":
            break
        
        if not query:
            continue
        
        logger.info(f"Searching for: {query}")
        results = retriever.search(query, top_k=args.top_k)
        
        print(f"\nTop {len(results)} results:")
        for i, result in enumerate(results, 1):
            passage = result["passage"]
            print(f"\n{i}. [{passage['id']}] {passage['title']}")
            print(f"   Score: {result['score']:.4f}")
            print(f"   Text: {passage['text'][:200]}...")
        
        print()


def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG CLI")
    
    # Common arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory with processed data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/indexes",
        help="Output directory for indexes"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Device for dense retriever"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Build sparse command
    sparse_parser = subparsers.add_parser("build-sparse", help="Build sparse index")
    sparse_parser.add_argument("--passages_file", default="corpus/hotpotqa_passages.jsonl")
    sparse_parser.set_defaults(func=build_sparse_index)
    
    # Build dense command
    dense_parser = subparsers.add_parser("build-dense", help="Build dense index")
    dense_parser.add_argument("--passages_file", default="corpus/hotpotqa_passages.jsonl")
    dense_parser.add_argument("--batch_size", type=int, default=64)
    dense_parser.add_argument("--index_type", default="flat", choices=["flat", "ivf", "hnsw"])
    dense_parser.set_defaults(func=build_dense_index)
    
    # Build hybrid command
    hybrid_parser = subparsers.add_parser("build", help="Build hybrid indexes")
    hybrid_parser.add_argument("--passages_file", default="corpus/hotpotqa_passages.jsonl")
    hybrid_parser.add_argument("--batch_size", type=int, default=64)
    hybrid_parser.add_argument("--index_type", default="flat")
    hybrid_parser.set_defaults(func=build_hybrid_index)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate retrieval")
    eval_parser.add_argument("--queries_file", default="hotpotqa/dev.json")
    eval_parser.add_argument("--fusion_strategy", default="union", choices=["union", "rrf"])
    eval_parser.add_argument("--num_eval_queries", type=int, default=100)
    eval_parser.set_defaults(func=evaluate_retrieval)
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Interactive search")
    search_parser.add_argument("--retriever_type", default="hybrid", choices=["sparse", "dense", "hybrid"])
    search_parser.add_argument("--fusion_strategy", default="union")
    search_parser.add_argument("--top_k", type=int, default=10)
    search_parser.set_defaults(func=interactive_search)
    
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()
