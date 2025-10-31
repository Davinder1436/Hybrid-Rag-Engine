"""
Data utilities for loading passages and queries.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import numpy as np
from torch.utils.data import Dataset
import torch


class PassageDataset(Dataset):
    """Dataset of passages for encoding/indexing."""
    
    def __init__(self, passages: List[Dict[str, Any]]):
        """
        Args:
            passages: List of passage dicts with keys: 'id', 'text', 'title', 'source'
        """
        self.passages = passages
    
    def __len__(self) -> int:
        return len(self.passages)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.passages[idx]
    
    @classmethod
    def from_jsonl(cls, file_path: Path):
        """Load passages from JSONL file."""
        passages = []
        with open(file_path, 'r') as f:
            for line in f:
                passages.append(json.loads(line))
        return cls(passages)


class QueryDataset(Dataset):
    """Dataset of queries with retrieval and QA annotations."""
    
    def __init__(self, queries: List[Dict[str, Any]]):
        """
        Args:
            queries: List of query dicts with keys:
                - 'id': query id
                - 'question': query text
                - 'answers': list of acceptable answers (for QA)
                - 'supporting_facts' (optional): relevant passages for multi-hop QA
        """
        self.queries = queries
    
    def __len__(self) -> int:
        return len(self.queries)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.queries[idx]
    
    @classmethod
    def from_jsonl(cls, file_path: Path):
        """Load queries from JSONL file."""
        queries = []
        with open(file_path, 'r') as f:
            for line in f:
                queries.append(json.loads(line))
        return cls(queries)
    
    @classmethod
    def from_hotpotqa(cls, file_path: Path):
        """Load from HotpotQA format (JSON list)."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        queries = []
        for item in data:
            query = {
                "id": item.get("_id", item.get("id")),
                "question": item["question"],
                "answers": item.get("answer", []),
                "supporting_facts": item.get("supporting_facts", []),
                "type": item.get("type", "comparison")
            }
            queries.append(query)
        
        return cls(queries)


class RetrievalDataset(Dataset):
    """Dataset combining queries and retrieval results for training."""
    
    def __init__(
        self,
        queries: List[Dict[str, Any]],
        query_embeddings: np.ndarray,  # (num_queries, emb_dim)
        retrieval_results: List[List[Dict[str, Any]]],  # For each query, list of results
        labels: List[int]  # Binary: 0 (insufficient), 1 (sufficient)
    ):
        """
        Args:
            queries: List of query dicts
            query_embeddings: Pre-computed query embeddings
            retrieval_results: Retrieval results for each query
            labels: Binary labels for sufficiency
        """
        assert len(queries) == len(query_embeddings) == len(retrieval_results) == len(labels)
        
        self.queries = queries
        self.query_embeddings = query_embeddings
        self.retrieval_results = retrieval_results
        self.labels = np.array(labels, dtype=np.int64)
    
    def __len__(self) -> int:
        return len(self.queries)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        query = self.queries[idx]
        query_emb = self.query_embeddings[idx]
        results = self.retrieval_results[idx]
        label = self.labels[idx]
        
        # Extract scores
        scores = np.array([r["score"] for r in results])
        
        # Compute score statistics
        if len(scores) > 0:
            max_score = np.max(scores)
            mean_score = np.mean(scores)
            std_score = np.std(scores) if len(scores) > 1 else 0.0
            num_high = np.sum(scores > 0.1)
        else:
            max_score = mean_score = std_score = num_high = 0.0
        
        # Compute agreement (for hybrid)
        agreement = 0.0
        sparse_ids = {r["passage"]["id"] for r in results if r.get("retriever") == "sparse"}
        dense_ids = {r["passage"]["id"] for r in results if r.get("retriever") == "dense"}
        if sparse_ids and dense_ids:
            agreement = len(sparse_ids & dense_ids) / min(len(sparse_ids), len(dense_ids))
        
        return {
            "query_id": query.get("id"),
            "query_embedding": torch.from_numpy(query_emb).float(),
            "score_stats": torch.tensor(
                [max_score, mean_score, std_score, num_high, agreement],
                dtype=torch.float32
            ),
            "label": torch.tensor(label, dtype=torch.long),
            "num_retrieved": len(results)
        }


def create_retrieval_dataset(
    queries_file: Path,
    retrieval_results_file: Path,
    query_embeddings_file: Path,
    labels_file: Path
) -> RetrievalDataset:
    """
    Load a complete retrieval dataset.
    
    Args:
        queries_file: JSONL file with queries
        retrieval_results_file: JSONL file with retrieval results per query
        query_embeddings_file: NPZ file with query embeddings (key: 'embeddings')
        labels_file: Text file with one label per line (0 or 1)
    """
    # Load queries
    queries = []
    with open(queries_file, 'r') as f:
        for line in f:
            queries.append(json.loads(line))
    
    # Load embeddings
    emb_data = np.load(query_embeddings_file)
    query_embeddings = emb_data['embeddings']
    
    # Load retrieval results
    retrieval_results = []
    with open(retrieval_results_file, 'r') as f:
        for line in f:
            retrieval_results.append(json.loads(line))
    
    # Load labels
    labels = []
    with open(labels_file, 'r') as f:
        for line in f:
            labels.append(int(line.strip()))
    
    return RetrievalDataset(queries, query_embeddings, retrieval_results, labels)


def batch_encode_passages(
    passages: List[Dict[str, Any]],
    encoder,  # SentenceTransformer
    batch_size: int = 64,
    show_progress: bool = True
) -> np.ndarray:
    """
    Encode a batch of passages.
    
    Args:
        passages: List of passage dicts with 'text' key
        encoder: SentenceTransformer encoder
        batch_size: Encoding batch size
        show_progress: Show progress bar
        
    Returns:
        Embeddings array (num_passages, emb_dim)
    """
    texts = [p["text"] for p in passages]
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=False
    )
    return embeddings.astype('float32')


def batch_encode_queries(
    queries: List[Dict[str, Any]],
    encoder,  # SentenceTransformer
    batch_size: int = 64,
    show_progress: bool = True
) -> np.ndarray:
    """
    Encode a batch of queries.
    
    Args:
        queries: List of query dicts with 'question' key
        encoder: SentenceTransformer encoder
        batch_size: Encoding batch size
        show_progress: Show progress bar
        
    Returns:
        Embeddings array (num_queries, emb_dim)
    """
    texts = [q.get("question", q.get("text", "")) for q in queries]
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=False
    )
    return embeddings.astype('float32')
