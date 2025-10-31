"""
Core retriever classes for Hybrid RAG system.
Implements: BaseRetriever, SparseRetriever (BM25), DenseRetriever (DPR/SentenceTransformer), HybridRetriever.
"""

import json
import pickle
import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from tqdm import tqdm

# Sparse retrieval
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

# Dense retrieval
import torch
from sentence_transformers import SentenceTransformer
import faiss


# Ensure NLTK tokenizers are available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab')
    nltk.download('punkt')


class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""
    
    @abstractmethod
    def search(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Search for top-k passages given a query.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of dicts with keys: 'id', 'text', 'title', 'score', 'source'
        """
        pass
    
    @abstractmethod
    def save(self, path: Path):
        """Save retriever index/state to disk."""
        pass
    
    @abstractmethod
    def load(self, path: Path):
        """Load retriever index/state from disk."""
        pass


class SparseRetriever(BaseRetriever):
    """BM25 sparse retrieval."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: BM25 k1 parameter (default 1.5)
            b: BM25 b parameter (default 0.75)
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.passages = []
        self.tokenized_corpus = []
        self.passage_id_map = {}  # Map from corpus index to passage id
    
    def build(self, passages: List[Dict[str, Any]]):
        """Build BM25 index from passages.
        
        Args:
            passages: List of passage dicts with keys: 'id', 'text', 'title', 'source'
        """
        print(f"\n[BM25] Building index from {len(passages)} passages...")
        self.passages = passages
        self.passage_id_map = {i: p['id'] for i, p in enumerate(passages)}
        
        # Tokenize corpus
        print("  Tokenizing passages...")
        for passage in tqdm(passages, desc="  Tokenizing", leave=False):
            tokens = word_tokenize(passage["text"].lower())
            self.tokenized_corpus.append(tokens)
        
        # Build BM25
        print("  Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        print(f"  ✓ BM25 index built: {len(passages)} passages")
    
    def search(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Search BM25 index.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of results with 'passage', 'score', 'rank'
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first.")
        
        query_tokens = word_tokenize(query.lower())
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:  # Only include non-zero scores
                results.append({
                    "passage": self.passages[idx],
                    "score": float(scores[idx]),
                    "rank": rank + 1,
                    "retriever": "sparse"
                })
        
        return results
    
    def save(self, path: Path):
        """Save BM25 index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / "bm25_index.pkl", 'wb') as f:
            pickle.dump({
                "bm25": self.bm25,
                "passages": self.passages,
                "tokenized_corpus": self.tokenized_corpus,
                "passage_id_map": self.passage_id_map,
                "k1": self.k1,
                "b": self.b
            }, f)
        print(f"  ✓ BM25 index saved to {path / 'bm25_index.pkl'}")
    
    @classmethod
    def load(cls, path: Path):
        """Load BM25 index from disk."""
        path = Path(path)
        with open(path / "bm25_index.pkl", 'rb') as f:
            data = pickle.load(f)
        
        index = cls(k1=data["k1"], b=data["b"])
        index.bm25 = data["bm25"]
        index.passages = data["passages"]
        index.tokenized_corpus = data["tokenized_corpus"]
        index.passage_id_map = data.get("passage_id_map", {})
        
        print(f"  ✓ BM25 index loaded: {len(index.passages)} passages")
        return index


class DenseRetriever(BaseRetriever):
    """Dense retrieval using sentence transformers + FAISS."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        index_type: str = "flat"  # "flat", "ivf", "hnsw"
    ):
        """
        Args:
            model_name: HuggingFace model name for encoding
            device: Device ('mps', 'cuda', 'cpu', or None for auto-detect)
            index_type: FAISS index type ('flat', 'ivf', 'hnsw')
        """
        self.model_name = model_name
        self.device = device or self._get_device()
        self.index_type = index_type
        
        print(f"\n[Dense] Loading model: {model_name}")
        print(f"  Device: {self.device}")
        
        self.encoder = SentenceTransformer(model_name, device=self.device)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.faiss_index = None
        self.passages = []
        self.passage_id_map = {}
    
    @staticmethod
    def _get_device() -> str:
        """Auto-detect best device."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def build(self, passages: List[Dict[str, Any]], batch_size: int = 64):
        """Build FAISS index from passages.
        
        Args:
            passages: List of passage dicts
            batch_size: Encoding batch size
        """
        print(f"\n[Dense] Building index from {len(passages)} passages...")
        self.passages = passages
        self.passage_id_map = {i: p['id'] for i, p in enumerate(passages)}
        
        # Encode passages
        print(f"  Encoding {len(passages)} passages...")
        texts = [p["text"] for p in passages]
        
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False
        )
        
        # Ensure float32
        embeddings = embeddings.astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build FAISS index
        print(f"  Building FAISS {self.index_type} index...")
        if self.index_type == "flat":
            self.faiss_index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "ivf":
            # IVF with PQ for large-scale retrieval
            n_clusters = min(int(np.sqrt(len(passages))), 256)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.faiss_index = faiss.IndexIVFPQ(
                quantizer, self.dimension, n_clusters, 8, 8
            )
            self.faiss_index.train(embeddings)
        elif self.index_type == "hnsw":
            # HNSW for fast approximate search
            self.faiss_index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")
        
        # Add embeddings in batches
        print("  Adding embeddings to index...")
        
        # Set FAISS to single-threaded to avoid memory issues on Apple Silicon
        try:
            faiss.omp_set_num_threads(1)
        except:
            pass
        
        batch_size_faiss = 1000  # Smaller batch for stability
        num_batches = (len(embeddings) + batch_size_faiss - 1) // batch_size_faiss
        
        try:
            import gc
            for i in range(num_batches):
                start_idx = i * batch_size_faiss
                end_idx = min((i + 1) * batch_size_faiss, len(embeddings))
                batch = np.ascontiguousarray(embeddings[start_idx:end_idx], dtype='float32')
                faiss.normalize_L2(batch)
                self.faiss_index.add(batch)
                
                # Clean up memory
                del batch
                gc.collect()
                
                if (i + 1) % 5 == 0 or (i + 1) == num_batches:
                    print(f"    Added {end_idx}/{len(embeddings)} vectors")
        except Exception as e:
            print(f"  Error adding to FAISS: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print(f"  ✓ Dense index built: {self.faiss_index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Search dense index.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of results with 'passage', 'score', 'rank'
        """
        if self.faiss_index is None:
            raise RuntimeError("Dense index not built. Call build() first.")
        
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0:  # -1 means invalid result
                results.append({
                    "passage": self.passages[idx],
                    "score": float(score),
                    "rank": rank + 1,
                    "retriever": "dense"
                })
        
        return results
    
    def save(self, path: Path):
        """Save dense index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        try:
            faiss.write_index(self.faiss_index, str(path / "faiss_index.bin"))
            print(f"  ✓ FAISS index saved: {path / 'faiss_index.bin'}")
        except Exception as e:
            print(f"  Error saving FAISS: {e}")
            raise
        
        # Save metadata
        with open(path / "dense_metadata.pkl", 'wb') as f:
            pickle.dump({
                "passages": self.passages,
                "passage_id_map": self.passage_id_map,
                "model_name": self.model_name,
                "dimension": self.dimension,
                "device": self.device,
                "index_type": self.index_type
            }, f)
        print(f"  ✓ Dense metadata saved: {path / 'dense_metadata.pkl'}")
    
    @classmethod
    def load(cls, path: Path, device: Optional[str] = None):
        """Load dense index from disk."""
        path = Path(path)
        
        with open(path / "dense_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        use_device = device or metadata.get("device", "cpu")
        
        index = cls(
            model_name=metadata["model_name"],
            device=use_device,
            index_type=metadata.get("index_type", "flat")
        )
        index.passages = metadata["passages"]
        index.passage_id_map = metadata.get("passage_id_map", {})
        index.faiss_index = faiss.read_index(str(path / "faiss_index.bin"))
        
        print(f"  ✓ Dense index loaded: {index.faiss_index.ntotal} vectors")
        return index


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining sparse (BM25) and dense (FAISS) retrieval."""
    
    def __init__(
        self,
        sparse_retriever: SparseRetriever,
        dense_retriever: DenseRetriever,
        fusion_strategy: str = "union",  # "union", "rrf", "weighted_sum"
        alpha: float = 0.5,  # Weight for weighted_sum fusion
    ):
        """
        Args:
            sparse_retriever: BM25 retriever instance
            dense_retriever: Dense retriever instance
            fusion_strategy: How to combine results ("union", "rrf", "weighted_sum")
            alpha: Weight for dense in weighted_sum (sparse weight = 1-alpha)
        """
        self.sparse = sparse_retriever
        self.dense = dense_retriever
        self.fusion_strategy = fusion_strategy
        self.alpha = alpha
        
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    
    def search(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Search both retrievers and fuse results.
        
        Args:
            query: Query string
            top_k: Number of final results to return
            
        Returns:
            Fused list of results
        """
        # Retrieve from both
        sparse_results = self.sparse.search(query, top_k=top_k)
        dense_results = self.dense.search(query, top_k=top_k)
        
        # Fuse results
        if self.fusion_strategy == "union":
            return self._fuse_union(sparse_results, dense_results, top_k)
        elif self.fusion_strategy == "rrf":
            return self._fuse_rrf(sparse_results, dense_results, top_k)
        elif self.fusion_strategy == "weighted_sum":
            return self._fuse_weighted(sparse_results, dense_results, top_k)
        else:
            raise ValueError(f"Unknown fusion_strategy: {self.fusion_strategy}")
    
    def _fuse_union(
        self,
        sparse_results: List[Dict],
        dense_results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Union-based fusion: deduplicate and re-rank by sum of normalized scores."""
        # Combine and deduplicate by passage id
        passage_dict = {}
        for r in sparse_results + dense_results:
            pid = r["passage"]["id"]
            if pid not in passage_dict:
                passage_dict[pid] = {
                    "passage": r["passage"],
                    "sparse_score": 0.0,
                    "dense_score": 0.0,
                    "sparse_rank": float('inf'),
                    "dense_rank": float('inf')
                }
            if r["retriever"] == "sparse":
                passage_dict[pid]["sparse_score"] = r["score"]
                passage_dict[pid]["sparse_rank"] = r["rank"]
            else:
                passage_dict[pid]["dense_score"] = r["score"]
                passage_dict[pid]["dense_rank"] = r["rank"]
        
        # Normalize scores and compute final score
        for pid, data in passage_dict.items():
            # Normalize to [0, 1]
            sparse_norm = (1.0 - (data["sparse_rank"] / len(sparse_results))) if data["sparse_rank"] < float('inf') else 0.0
            dense_norm = (1.0 - (data["dense_rank"] / len(dense_results))) if data["dense_rank"] < float('inf') else 0.0
            
            # Weighted sum
            data["score"] = (1 - self.alpha) * sparse_norm + self.alpha * dense_norm
        
        # Sort by score
        sorted_results = sorted(
            passage_dict.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        # Format output
        results = []
        for rank, item in enumerate(sorted_results[:top_k]):
            results.append({
                "passage": item["passage"],
                "score": item["score"],
                "rank": rank + 1,
                "retriever": "hybrid",
                "sparse_score": item["sparse_score"],
                "dense_score": item["dense_score"]
            })
        
        return results
    
    def _fuse_rrf(
        self,
        sparse_results: List[Dict],
        dense_results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Reciprocal Rank Fusion: RRF = 1 / (k + rank)."""
        passage_dict = {}
        k = 60  # RRF parameter
        
        for r in sparse_results:
            pid = r["passage"]["id"]
            passage_dict[pid] = passage_dict.get(pid, {
                "passage": r["passage"],
                "rrf_score": 0.0
            })
            passage_dict[pid]["rrf_score"] += 1.0 / (k + r["rank"])
        
        for r in dense_results:
            pid = r["passage"]["id"]
            if pid not in passage_dict:
                passage_dict[pid] = {"passage": r["passage"], "rrf_score": 0.0}
            passage_dict[pid]["rrf_score"] += 1.0 / (k + r["rank"])
        
        # Sort by RRF score
        sorted_results = sorted(
            passage_dict.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )
        
        results = []
        for rank, item in enumerate(sorted_results[:top_k]):
            results.append({
                "passage": item["passage"],
                "score": item["rrf_score"],
                "rank": rank + 1,
                "retriever": "hybrid"
            })
        
        return results
    
    def _fuse_weighted(
        self,
        sparse_results: List[Dict],
        dense_results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Weighted sum of normalized scores."""
        return self._fuse_union(sparse_results, dense_results, top_k)
    
    def save(self, path: Path):
        """Save both retrievers."""
        path = Path(path)
        self.sparse.save(path / "sparse")
        self.dense.save(path / "dense")
        print(f"  ✓ Hybrid retriever saved to {path}")
    
    @classmethod
    def load(cls, path: Path, device: Optional[str] = None):
        """Load both retrievers."""
        path = Path(path)
        sparse = SparseRetriever.load(path / "sparse")
        dense = DenseRetriever.load(path / "dense", device=device)
        
        retriever = cls(sparse, dense)
        print(f"  ✓ Hybrid retriever loaded")
        return retriever
