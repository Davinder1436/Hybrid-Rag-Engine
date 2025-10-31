"""
Hybrid RAG system: Sparse + Dense Retrieval with Uncertainty Head.
"""

__version__ = "0.1.0"

from .retrievers import (
    BaseRetriever,
    SparseRetriever,
    DenseRetriever,
    HybridRetriever,
)
from .uncertainty import UncertaintyHead
from .data_utils import PassageDataset, QueryDataset
from .metrics import RetrievalMetrics, CalibrationMetrics

__all__ = [
    "BaseRetriever",
    "SparseRetriever",
    "DenseRetriever",
    "HybridRetriever",
    "UncertaintyHead",
    "PassageDataset",
    "QueryDataset",
    "RetrievalMetrics",
    "CalibrationMetrics",
]
