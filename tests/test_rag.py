"""
Unit and integration tests for Hybrid RAG modules.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import json

# Import from rag module
from rag.retrievers import BaseRetriever, SparseRetriever, DenseRetriever, HybridRetriever
from rag.uncertainty import UncertaintyHead
from rag.data_utils import PassageDataset, QueryDataset, RetrievalDataset
from rag.metrics import RetrievalMetrics, CalibrationMetrics, SelectiveMetrics


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_passages():
    """Sample passages for testing."""
    return [
        {
            "id": "p1",
            "text": "The capital of France is Paris. Paris is known for the Eiffel Tower.",
            "title": "France",
            "source": "wikipedia"
        },
        {
            "id": "p2",
            "text": "London is the capital of the United Kingdom. The UK includes England, Scotland, Wales.",
            "title": "UK",
            "source": "wikipedia"
        },
        {
            "id": "p3",
            "text": "Berlin is the capital of Germany and the largest city in Germany.",
            "title": "Germany",
            "source": "wikipedia"
        },
        {
            "id": "p4",
            "text": "The Eiffel Tower is a wrought iron lattice tower on the Champ de Mars in Paris.",
            "title": "Eiffel Tower",
            "source": "wikipedia"
        },
    ]


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        {
            "id": "q1",
            "question": "What is the capital of France?",
            "answers": ["Paris"],
            "type": "single-hop"
        },
        {
            "id": "q2",
            "question": "What is the Eiffel Tower?",
            "answers": ["A tower in Paris"],
            "type": "single-hop"
        }
    ]


# =============================================================================
# Tests for Sparse Retriever (BM25)
# =============================================================================

class TestSparseRetriever:
    """Tests for BM25 sparse retriever."""
    
    def test_sparse_build(self, sample_passages):
        """Test building BM25 index."""
        retriever = SparseRetriever()
        retriever.build(sample_passages)
        
        assert retriever.bm25 is not None
        assert len(retriever.passages) == len(sample_passages)
        assert len(retriever.tokenized_corpus) == len(sample_passages)
    
    def test_sparse_search(self, sample_passages):
        """Test BM25 search."""
        retriever = SparseRetriever()
        retriever.build(sample_passages)
        
        results = retriever.search("Paris", top_k=5)
        
        assert len(results) > 0
        assert results[0]["passage"]["id"] in ["p1", "p4"]  # Should find Paris-related docs
        assert "score" in results[0]
        assert "retriever" in results[0]
    
    def test_sparse_save_load(self, sample_passages):
        """Test saving and loading BM25 index."""
        retriever = SparseRetriever()
        retriever.build(sample_passages)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            
            # Save
            retriever.save(save_path)
            assert (save_path / "bm25_index.pkl").exists()
            
            # Load
            loaded_retriever = SparseRetriever.load(save_path)
            assert len(loaded_retriever.passages) == len(sample_passages)
            
            # Verify same search results
            results1 = retriever.search("Paris", top_k=3)
            results2 = loaded_retriever.search("Paris", top_k=3)
            
            assert [r["passage"]["id"] for r in results1] == [r["passage"]["id"] for r in results2]


# =============================================================================
# Tests for Dense Retriever
# =============================================================================

class TestDenseRetriever:
    """Tests for dense FAISS retriever."""
    
    @pytest.mark.slow
    def test_dense_build(self, sample_passages):
        """Test building dense index (requires model download)."""
        retriever = DenseRetriever(device="cpu")
        retriever.build(sample_passages, batch_size=2)
        
        assert retriever.faiss_index is not None
        assert retriever.faiss_index.ntotal == len(sample_passages)
    
    @pytest.mark.slow
    def test_dense_search(self, sample_passages):
        """Test dense search."""
        retriever = DenseRetriever(device="cpu")
        retriever.build(sample_passages, batch_size=2)
        
        results = retriever.search("Paris", top_k=3)
        
        assert len(results) <= 3
        assert "score" in results[0]
        assert results[0]["retriever"] == "dense"


# =============================================================================
# Tests for Hybrid Retriever
# =============================================================================

class TestHybridRetriever:
    """Tests for hybrid retriever."""
    
    def test_hybrid_creation(self, sample_passages):
        """Test creating hybrid retriever."""
        sparse = SparseRetriever()
        sparse.build(sample_passages)
        
        # Mock dense retriever (avoid slow model download in test)
        dense = Mock(spec=DenseRetriever)
        dense.search.return_value = []
        
        hybrid = HybridRetriever(sparse, dense)
        assert hybrid.sparse is not None
        assert hybrid.dense is not None
    
    def test_hybrid_fusion_strategies(self, sample_passages):
        """Test different fusion strategies."""
        sparse = SparseRetriever()
        sparse.build(sample_passages)
        
        dense = Mock(spec=DenseRetriever)
        dense.search.return_value = [
            {"passage": p, "score": 0.9, "rank": i, "retriever": "dense"}
            for i, p in enumerate(sample_passages[:2])
        ]
        
        # Test union fusion
        hybrid_union = HybridRetriever(sparse, dense, fusion_strategy="union")
        results = hybrid_union.search("Paris", top_k=5)
        assert len(results) > 0
        
        # Test RRF fusion
        hybrid_rrf = HybridRetriever(sparse, dense, fusion_strategy="rrf")
        results = hybrid_rrf.search("Paris", top_k=5)
        assert len(results) > 0


# =============================================================================
# Tests for Uncertainty Head
# =============================================================================

class TestUncertaintyHead:
    """Tests for uncertainty head."""
    
    def test_uncertainty_head_creation(self):
        """Test creating uncertainty head."""
        head = UncertaintyHead(input_dim=384, hidden_dim=256)
        
        assert isinstance(head, torch.nn.Module)
        assert head.input_dim == 384
        assert head.hidden_dim == 256
    
    def test_uncertainty_forward(self):
        """Test forward pass."""
        head = UncertaintyHead(input_dim=384, hidden_dim=256)
        
        batch_size = 4
        query_embs = torch.randn(batch_size, 384)
        score_stats = torch.randn(batch_size, 5)
        
        output = head(query_embs, score_stats)
        
        assert output.shape == (batch_size, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output
    
    def test_uncertainty_predict(self):
        """Test prediction interface."""
        head = UncertaintyHead(input_dim=384, hidden_dim=256)
        head.eval()
        
        query_emb = np.random.randn(384)
        results = [
            {"score": 0.8, "passage": {"id": "p1"}, "retriever": "sparse"},
            {"score": 0.7, "passage": {"id": "p2"}, "retriever": "dense"},
        ]
        
        pred = head.predict(query_emb, results)
        
        assert "p_success" in pred
        assert "p_failure" in pred
        assert 0 <= pred["p_success"] <= 1
        assert abs(pred["p_success"] + pred["p_failure"] - 1.0) < 1e-5
    
    def test_uncertainty_save_load(self):
        """Test saving and loading."""
        head = UncertaintyHead(input_dim=384, hidden_dim=256)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            
            # Save
            head.save(save_path)
            assert (save_path / "uncertainty_head.pt").exists()
            assert (save_path / "uncertainty_config.pkl").exists()
            
            # Load
            loaded = UncertaintyHead.load(save_path, device="cpu")
            assert loaded.input_dim == 384
            assert loaded.hidden_dim == 256


# =============================================================================
# Tests for Data Utils
# =============================================================================

class TestDataUtils:
    """Tests for data utilities."""
    
    def test_passage_dataset(self, sample_passages):
        """Test PassageDataset."""
        dataset = PassageDataset(sample_passages)
        
        assert len(dataset) == len(sample_passages)
        assert dataset[0]["id"] == "p1"
    
    def test_query_dataset(self, sample_queries):
        """Test QueryDataset."""
        dataset = QueryDataset(sample_queries)
        
        assert len(dataset) == len(sample_queries)
        assert dataset[0]["question"] == "What is the capital of France?"
    
    def test_retrieval_dataset(self, sample_passages, sample_queries):
        """Test RetrievalDataset."""
        query_embs = np.random.randn(len(sample_queries), 384)
        retrieval_results = [
            [
                {
                    "score": 0.8,
                    "passage": p,
                    "retriever": "dense"
                }
                for p in sample_passages[:2]
            ]
            for _ in sample_queries
        ]
        labels = [1, 0]  # First query has sufficient evidence, second doesn't
        
        dataset = RetrievalDataset(sample_queries, query_embs, retrieval_results, labels)
        
        assert len(dataset) == len(sample_queries)
        item = dataset[0]
        assert "query_embedding" in item
        assert "score_stats" in item
        assert "label" in item


# =============================================================================
# Tests for Metrics
# =============================================================================

class TestRetrievalMetrics:
    """Tests for retrieval metrics."""
    
    def test_recall_at_k(self):
        """Test Recall@k."""
        retrieved = ["p1", "p2", "p3", "p4"]
        relevant = ["p1", "p3"]
        
        recall = RetrievalMetrics.recall_at_k(retrieved, relevant, k=2)
        assert recall == 0.5  # 1 out of 2 relevant docs in top-2
    
    def test_precision_at_k(self):
        """Test Precision@k."""
        retrieved = ["p1", "p2", "p3", "p4"]
        relevant = ["p1", "p3"]
        
        precision = RetrievalMetrics.precision_at_k(retrieved, relevant, k=3)
        assert precision == 2/3  # 2 relevant out of top-3
    
    def test_mrr(self):
        """Test Mean Reciprocal Rank."""
        retrieved = ["p2", "p4", "p1"]
        relevant = ["p1"]
        
        mrr = RetrievalMetrics.mean_reciprocal_rank(retrieved, relevant)
        assert mrr == 1/3  # First relevant at rank 3


class TestCalibrationMetrics:
    """Tests for calibration metrics."""
    
    def test_ece(self):
        """Test Expected Calibration Error."""
        predictions = np.array([0.9, 0.8, 0.1, 0.2])
        targets = np.array([1, 1, 0, 0])
        
        ece = CalibrationMetrics.expected_calibration_error(predictions, targets, num_bins=2)
        assert 0 <= ece <= 1
    
    def test_brier_score(self):
        """Test Brier Score."""
        predictions = np.array([0.8, 0.2])
        targets = np.array([1, 0])
        
        score = CalibrationMetrics.brier_score(predictions, targets)
        assert score == 0.04  # (0.2)^2 + (0.2)^2 / 2


class TestSelectiveMetrics:
    """Tests for selective classification metrics."""
    
    def test_coverage_vs_risk(self):
        """Test coverage vs risk."""
        predictions = np.array([0.9, 0.8, 0.3, 0.2])
        targets = np.array([1, 1, 0, 0])
        
        cov, risk = SelectiveMetrics.coverage_vs_risk(predictions, targets, 0.5)
        assert cov == 0.5  # 2 out of 4 above threshold
        assert risk == 0.0  # Both covered are correct


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_sparse_retrieval_pipeline(self, sample_passages, sample_queries):
        """Test full sparse retrieval pipeline."""
        retriever = SparseRetriever()
        retriever.build(sample_passages)
        
        for query in sample_queries:
            results = retriever.search(query["question"], top_k=3)
            assert len(results) > 0
            assert "passage" in results[0]
            assert "score" in results[0]
    
    def test_uncertainty_on_retrieval_results(self, sample_passages, sample_queries):
        """Test uncertainty head on retrieval results."""
        # Build sparse retriever
        sparse = SparseRetriever()
        sparse.build(sample_passages)
        
        # Create uncertainty head
        head = UncertaintyHead(input_dim=384)
        
        # Mock query embeddings
        query_emb = np.random.randn(384)
        
        # Get retrieval results
        for query in sample_queries:
            results = sparse.search(query["question"], top_k=3)
            
            # Predict uncertainty
            pred = head.predict(query_emb, results)
            
            assert "p_success" in pred
            assert 0 <= pred["p_success"] <= 1


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
