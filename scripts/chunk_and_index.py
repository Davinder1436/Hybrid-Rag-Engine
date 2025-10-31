"""
Chunk documents and build both sparse (BM25) and dense (FAISS) indexes.
This is the core indexing pipeline for the Hybrid RAG system.
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import pickle
import gc

# Sparse retrieval
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# Dense retrieval
import torch
from sentence_transformers import SentenceTransformer
import faiss


# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab')
    nltk.download('punkt')  # Also download punkt for compatibility


# Configure device for Apple Silicon
def get_device():
    """Get the best available device (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)."""
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon Metal Performance Shaders
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


class PassageChunker:
    """Chunks passages into smaller segments for retrieval."""
    
    def __init__(self, chunk_size: int = 256, stride: int = 100):
        """
        Args:
            chunk_size: Maximum tokens per chunk
            stride: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.stride = stride
    
    def chunk_passage(self, passage: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a passage into overlapping chunks."""
        text = passage["text"]
        tokens = word_tokenize(text.lower())
        
        if len(tokens) <= self.chunk_size:
            # Passage is small enough, return as-is
            return [passage]
        
        chunks = []
        chunk_id = 0
        
        for i in range(0, len(tokens), self.stride):
            chunk_tokens = tokens[i:i + self.chunk_size]
            
            if len(chunk_tokens) < 50:  # Skip very small chunks at the end
                break
            
            chunk_text = " ".join(chunk_tokens)
            
            chunk = {
                "id": f"{passage['id']}_chunk{chunk_id}",
                "parent_id": passage["id"],
                "title": passage["title"],
                "text": chunk_text,
                "source": passage["source"],
                "chunk_index": chunk_id
            }
            chunks.append(chunk)
            chunk_id += 1
        
        return chunks if chunks else [passage]


class BM25Index:
    """BM25 sparse retrieval index."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.passages = []
        self.tokenized_corpus = []
    
    def build(self, passages: List[Dict[str, Any]]):
        """Build BM25 index from passages."""
        print("\n  Tokenizing corpus for BM25...")
        self.passages = passages
        
        for passage in tqdm(passages, desc="  Tokenizing"):
            # Tokenize passage text
            tokens = word_tokenize(passage["text"].lower())
            self.tokenized_corpus.append(tokens)
        
        print("  Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        print(f"  BM25 index built with {len(passages)} passages")
    
    def search(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Search BM25 index."""
        query_tokens = word_tokenize(query.lower())
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "passage": self.passages[idx],
                "score": float(scores[idx])
            })
        
        return results
    
    def save(self, path: Path):
        """Save BM25 index to disk."""
        with open(path / "bm25_index.pkl", 'wb') as f:
            pickle.dump({
                "bm25": self.bm25,
                "passages": self.passages,
                "tokenized_corpus": self.tokenized_corpus,
                "k1": self.k1,
                "b": self.b
            }, f)
        print(f"  BM25 index saved to {path / 'bm25_index.pkl'}")
    
    @classmethod
    def load(cls, path: Path):
        """Load BM25 index from disk."""
        with open(path / "bm25_index.pkl", 'rb') as f:
            data = pickle.load(f)
        
        index = cls(k1=data["k1"], b=data["b"])
        index.bm25 = data["bm25"]
        index.passages = data["passages"]
        index.tokenized_corpus = data["tokenized_corpus"]
        
        return index


class DenseIndex:
    """Dense retrieval index using sentence transformers and FAISS."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        """
        Args:
            model_name: HuggingFace model for encoding passages
            device: Device to use ('mps', 'cuda', 'cpu'). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device if device else get_device()
        
        print(f"\n  Loading sentence transformer: {model_name}")
        print(f"  Using device: {self.device}")
        
        # Load model with device specification
        self.encoder = SentenceTransformer(model_name, device=self.device)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.faiss_index = None
        self.passages = []
    
    def build(self, passages: List[Dict[str, Any]], batch_size: int = 64):
        """Build FAISS index from passages.
        
        Args:
            passages: List of passage dictionaries
            batch_size: Batch size for encoding (increased for M4 Pro)
        """
        print(f"\n  Encoding {len(passages)} passages...")
        print(f"  Batch size: {batch_size} (optimized for Apple Silicon)")
        self.passages = passages
        
        # Encode all passages
        texts = [p["text"] for p in passages]
        
        # Use larger batch size for M4 Pro with 24GB RAM
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False  # We'll normalize separately
        )
        
        # Convert to float32 to ensure compatibility
        embeddings = embeddings.astype('float32')

        # Build FAISS index (optimized for Apple Silicon)
        print("  Building FAISS index...")

        # Try to limit FAISS/OpenMP threads to reduce memory contention
        try:
            faiss.omp_set_num_threads(1)
        except Exception:
            # Some FAISS builds don't expose omp_set_num_threads
            pass

        # Use IndexFlatIP for exact search (efficient on Apple Silicon with ARM64)
        self.faiss_index = faiss.IndexFlatIP(self.dimension)

        # Add embeddings to index in smaller batches to avoid segfaults/memory spikes
        print("  Adding embeddings to FAISS index (safe batching)...")
        batch_size_faiss = 2000  # smaller batches are more stable on macOS
        num_batches = (len(embeddings) + batch_size_faiss - 1) // batch_size_faiss

        try:
            for i in range(num_batches):
                start_idx = i * batch_size_faiss
                end_idx = min((i + 1) * batch_size_faiss, len(embeddings))
                # Ensure contiguous float32 array
                batch_embeddings = np.ascontiguousarray(embeddings[start_idx:end_idx], dtype='float32')

                # Normalize this batch for cosine similarity
                faiss.normalize_L2(batch_embeddings)

                # Add to FAISS index
                self.faiss_index.add(batch_embeddings)
                print(f"    Added batch {i+1}/{num_batches} ({end_idx}/{len(embeddings)} vectors)")

                # Free memory proactively
                del batch_embeddings
                gc.collect()

        except Exception as e:
            print(f"    Error while adding to FAISS: {e}")
            print("    Attempting to persist partial index before raising...")
            try:
                tmp_index_path = Path.cwd() / "partial_faiss_index_temp.bin"
                faiss.write_index(self.faiss_index, str(tmp_index_path))
                tmp_meta_path = Path.cwd() / "partial_dense_metadata_temp.pkl"
                with open(tmp_meta_path, 'wb') as f:
                    pickle.dump({
                        "passages": self.passages[: self.faiss_index.ntotal],
                        "model_name": self.model_name,
                        "dimension": self.dimension,
                        "device": self.device
                    }, f)
                print(f"    Partial index saved: {tmp_index_path}")
                print(f"    Partial metadata saved: {tmp_meta_path}")
            except Exception as e2:
                print(f"    Failed to save partial index: {e2}")
            raise

        print(f"  FAISS index built with {self.faiss_index.ntotal} vectors")
        print(f"  Index type: Flat (exact search, optimized for <1M vectors)")
    
    def search(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Search dense index."""
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "passage": self.passages[idx],
                "score": float(score)
            })
        
        return results
    
    def save(self, path: Path):
        """Save dense index to disk."""
        print("  Saving FAISS index...")
        
        # Save FAISS index
        try:
            faiss.write_index(self.faiss_index, str(path / "faiss_index.bin"))
            print(f"    FAISS index saved: {path / 'faiss_index.bin'}")
        except Exception as e:
            print(f"    Error saving FAISS index: {e}")
            raise
        
        # Save passages and metadata
        try:
            with open(path / "dense_metadata.pkl", 'wb') as f:
                pickle.dump({
                    "passages": self.passages,
                    "model_name": self.model_name,
                    "dimension": self.dimension,
                    "device": self.device
                }, f)
            print(f"    Metadata saved: {path / 'dense_metadata.pkl'}")
        except Exception as e:
            print(f"    Error saving metadata: {e}")
            raise
        
        print(f"  Dense index saved to {path}")
        print(f"  Device used: {self.device}")
    
    @classmethod
    def load(cls, path: Path, device: str = None):
        """Load dense index from disk.
        
        Args:
            path: Path to index directory
            device: Device to use (auto-detected if None)
        """
        # Load metadata
        with open(path / "dense_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        # Use saved device or override with provided device
        saved_device = metadata.get("device", "cpu")
        use_device = device if device else saved_device
        
        index = cls(model_name=metadata["model_name"], device=use_device)
        index.passages = metadata["passages"]
        index.faiss_index = faiss.read_index(str(path / "faiss_index.bin"))
        
        print(f"  Loaded index with device: {use_device}")
        
        return index


def load_passages(data_dir: Path, sources: List[str]) -> List[Dict[str, Any]]:
    """Load passages from processed data."""
    passages = []
    
    for source in sources:
        if source == "wikipedia":
            file_path = data_dir / "processed" / "wikipedia" / "passages.jsonl"
        elif source == "hotpotqa":
            file_path = data_dir / "processed" / "corpus" / "hotpotqa_passages.jsonl"
        else:
            print(f"  Warning: Unknown source '{source}', skipping")
            continue
        
        if not file_path.exists():
            print(f"  Warning: {file_path} not found, skipping")
            continue
        
        print(f"  Loading passages from {file_path}")
        with open(file_path, 'r') as f:
            for line in f:
                passages.append(json.loads(line))
    
    return passages


def main():
    parser = argparse.ArgumentParser(description="Build retrieval indexes for Hybrid RAG")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Root data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/indexes",
        help="Output directory for indexes"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["hotpotqa", "wikipedia"],
        help="Passage sources to index"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help="Maximum tokens per chunk"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=100,
        help="Overlap between chunks"
    )
    parser.add_argument(
        "--dense_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model for dense retrieval encoding"
    )
    parser.add_argument(
        "--skip_chunking",
        action="store_true",
        help="Skip chunking step"
    )
    parser.add_argument(
        "--index_types",
        nargs="+",
        default=["bm25", "dense"],
        choices=["bm25", "dense"],
        help="Which indexes to build"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["mps", "cuda", "cpu"],
        help="Device for dense encoding (auto-detected if not specified)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for dense encoding (default: 64, good for M4 Pro 24GB)"
    )
    parser.add_argument(
        "--max_passages",
        type=int,
        default=None,
        help="(Optional) Limit number of passages (useful for testing)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Hybrid RAG: Building Retrieval Indexes")
    print("="*60)
    print(f"Data directory: {data_dir.absolute()}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Sources: {', '.join(args.sources)}")
    print(f"Index types: {', '.join(args.index_types)}")
    print("="*60)
    
    # Load passages
    print("\n[1/4] Loading passages...")
    passages = load_passages(data_dir, args.sources)
    # Optionally limit passages for quick tests
    if args.max_passages is not None:
        passages = passages[: args.max_passages]
    print(f"  Loaded {len(passages)} passages")
    
    # Chunk passages
    if not args.skip_chunking:
        print("\n[2/4] Chunking passages...")
        chunker = PassageChunker(chunk_size=args.chunk_size, stride=args.stride)
        
        chunked_passages = []
        for passage in tqdm(passages, desc="  Chunking"):
            chunks = chunker.chunk_passage(passage)
            chunked_passages.extend(chunks)
        
        print(f"  Created {len(chunked_passages)} chunks from {len(passages)} passages")
        
        # Save chunked passages
        chunk_file = output_dir / "chunked_passages.jsonl"
        with open(chunk_file, 'w') as f:
            for passage in chunked_passages:
                f.write(json.dumps(passage) + "\n")
        print(f"  Saved chunked passages to {chunk_file}")
        
        passages = chunked_passages
    else:
        print("\n[2/4] Skipping chunking (using passages as-is)")
    
    # Build BM25 index
    if "bm25" in args.index_types:
        print("\n[3/4] Building BM25 (sparse) index...")
        bm25_index = BM25Index()
        bm25_index.build(passages)
        bm25_index.save(output_dir)
    else:
        print("\n[3/4] Skipping BM25 index")
    
    # Build dense index
    if "dense" in args.index_types:
        print("\n[4/4] Building Dense (FAISS) index...")
        device = args.device if args.device else get_device()
        print(f"  Detected/Selected device: {device}")
        
        if device == "mps":
            print("  ✓ Using Apple Silicon Metal acceleration")
        
        dense_index = DenseIndex(model_name=args.dense_model, device=device)
        dense_index.build(passages, batch_size=args.batch_size)
        dense_index.save(output_dir)
    else:
        print("\n[4/4] Skipping dense index")
    
    print("\n" + "="*60)
    print("Indexing complete!")
    print("="*60)
    print(f"\nIndexes saved to: {output_dir.absolute()}")
    print("\nNext steps:")
    print("  1. Test retrieval: python scripts/smoke_test.py")
    print("  2. Train retriever: python scripts/train_retriever.py")


if __name__ == "__main__":
    main()
