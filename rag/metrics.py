"""
Evaluation metrics for retrieval, QA, and calibration.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
from pathlib import Path
import matplotlib.pyplot as plt


class RetrievalMetrics:
    """Metrics for evaluating retrieval quality."""
    
    @staticmethod
    def recall_at_k(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = 100
    ) -> float:
        """
        Recall@k: what fraction of relevant docs appear in top-k.
        
        Args:
            retrieved_ids: Ordered list of retrieved doc IDs
            relevant_ids: Set of relevant doc IDs
            k: Cutoff
            
        Returns:
            Recall@k in [0, 1]
        """
        if len(relevant_ids) == 0:
            return 1.0
        
        top_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        return len(top_k & relevant_set) / len(relevant_set)
    
    @staticmethod
    def precision_at_k(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = 100
    ) -> float:
        """
        Precision@k: what fraction of top-k are relevant.
        
        Args:
            retrieved_ids: Ordered list of retrieved doc IDs
            relevant_ids: Set of relevant doc IDs
            k: Cutoff
            
        Returns:
            Precision@k in [0, 1]
        """
        if k == 0:
            return 0.0
        
        top_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        return len(top_k & relevant_set) / k
    
    @staticmethod
    def mean_reciprocal_rank(
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        MRR: 1 / (rank of first relevant doc), or 0 if none found.
        
        Args:
            retrieved_ids: Ordered list of retrieved doc IDs
            relevant_ids: Set of relevant doc IDs
            
        Returns:
            MRR in [0, 1]
        """
        relevant_set = set(relevant_ids)
        
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_set:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def ndcg_at_k(
        retrieved_scores: List[float],
        relevant_ids: List[str],
        retrieved_ids: List[str],
        k: int = 100
    ) -> float:
        """
        NDCG@k: normalized discounted cumulative gain.
        
        Args:
            retrieved_scores: Relevance scores of retrieved docs
            relevant_ids: Set of relevant doc IDs
            retrieved_ids: Ordered list of retrieved doc IDs
            k: Cutoff
            
        Returns:
            NDCG@k in [0, 1]
        """
        # Compute DCG
        dcg = 0.0
        relevant_set = set(relevant_ids)
        
        for i, doc_id in enumerate(retrieved_ids[:k], start=1):
            relevance = 1.0 if doc_id in relevant_set else 0.0
            dcg += relevance / np.log2(i + 1)
        
        # Compute ideal DCG
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant_ids), k) + 1))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def evaluate_batch(
        predictions: List[List[str]],  # Batch of retrieved ID lists
        references: List[List[str]],   # Batch of relevant ID lists
        k_values: List[int] = [10, 100]
    ) -> Dict[str, float]:
        """
        Evaluate a batch of retrieval predictions.
        
        Args:
            predictions: List of retrieved ID lists
            references: List of relevant ID lists
            k_values: List of k values for recall/precision
            
        Returns:
            Dict of metric names to values
        """
        metrics = {}
        
        for k in k_values:
            recalls = [
                RetrievalMetrics.recall_at_k(pred, ref, k)
                for pred, ref in zip(predictions, references)
            ]
            precisions = [
                RetrievalMetrics.precision_at_k(pred, ref, k)
                for pred, ref in zip(predictions, references)
            ]
            mrrs = [
                RetrievalMetrics.mean_reciprocal_rank(pred, ref)
                for pred, ref in zip(predictions, references)
            ]
            
            metrics[f"recall@{k}"] = np.mean(recalls)
            metrics[f"precision@{k}"] = np.mean(precisions)
            metrics[f"mrr@{k}"] = np.mean(mrrs)
        
        return metrics


class CalibrationMetrics:
    """Metrics for evaluating prediction calibration."""
    
    @staticmethod
    def expected_calibration_error(
        predictions: np.ndarray,  # Predicted probabilities (batch_size,)
        targets: np.ndarray,      # Binary targets (batch_size,)
        num_bins: int = 10
    ) -> float:
        """
        Expected Calibration Error (ECE).
        
        Args:
            predictions: Predicted probabilities in [0, 1]
            targets: Binary labels (0 or 1)
            num_bins: Number of bins for histogram
            
        Returns:
            ECE value
        """
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0
        
        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Mask for samples in this bin
            in_bin = (predictions >= bin_lower) & (predictions < bin_upper)
            
            if not np.any(in_bin):
                continue
            
            # Confidence: average predicted probability
            confidence = predictions[in_bin].mean()
            
            # Accuracy: fraction of correct predictions
            accuracy = targets[in_bin].mean()
            
            # Weight by fraction of samples in bin
            bin_weight = np.sum(in_bin) / len(predictions)
            ece += bin_weight * np.abs(confidence - accuracy)
        
        return ece
    
    @staticmethod
    def brier_score(
        predictions: np.ndarray,  # Predicted probabilities
        targets: np.ndarray       # Binary targets
    ) -> float:
        """
        Brier Score: mean squared error between predictions and targets.
        
        Args:
            predictions: Predicted probabilities in [0, 1]
            targets: Binary labels (0 or 1)
            
        Returns:
            Brier score (lower is better)
        """
        return np.mean((predictions - targets) ** 2)
    
    @staticmethod
    def calibration_plot(
        predictions: np.ndarray,
        targets: np.ndarray,
        num_bins: int = 10,
        save_path: Path = None
    ):
        """
        Plot calibration curve: predicted probability vs actual frequency.
        
        Args:
            predictions: Predicted probabilities
            targets: Binary labels
            num_bins: Number of bins
            save_path: Optional path to save figure
        """
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        confidences = []
        accuracies = []
        bin_sizes = []
        
        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (predictions >= bin_lower) & (predictions < bin_upper)
            
            if not np.any(in_bin):
                continue
            
            confidence = predictions[in_bin].mean()
            accuracy = targets[in_bin].mean()
            bin_size = np.sum(in_bin)
            
            confidences.append(confidence)
            accuracies.append(accuracy)
            bin_sizes.append(bin_size)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
        
        # Calibration curve
        ax.scatter(confidences, accuracies, s=bin_sizes, alpha=0.7, label='Observed')
        
        ax.set_xlabel('Confidence (predicted probability)')
        ax.set_ylabel('Accuracy (true frequency)')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Calibration plot saved: {save_path}")
        
        return fig, ax


class SelectiveMetrics:
    """Metrics for selective classification (coverage vs risk trade-off)."""
    
    @staticmethod
    def coverage_vs_risk(
        predictions: np.ndarray,    # Predicted probabilities
        targets: np.ndarray,        # Binary labels
        confidence_threshold: float = 0.5
    ) -> Tuple[float, float]:
        """
        Compute coverage and risk at a confidence threshold.
        
        Coverage: fraction of examples above threshold.
        Risk: error rate on covered examples.
        
        Args:
            predictions: Predicted probabilities
            targets: Binary labels
            confidence_threshold: Confidence cutoff
            
        Returns:
            (coverage, risk)
        """
        above_threshold = predictions >= confidence_threshold
        
        if not np.any(above_threshold):
            return 0.0, 1.0
        
        coverage = np.sum(above_threshold) / len(predictions)
        risk = 1.0 - np.mean(targets[above_threshold])
        
        return coverage, risk
    
    @staticmethod
    def coverage_risk_curve(
        predictions: np.ndarray,
        targets: np.ndarray,
        num_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute coverage vs risk curve by varying threshold.
        
        Args:
            predictions: Predicted probabilities
            targets: Binary labels
            num_points: Number of threshold points
            
        Returns:
            (coverages, risks)
        """
        thresholds = np.linspace(0, 1, num_points)
        coverages = []
        risks = []
        
        for threshold in thresholds:
            cov, risk = SelectiveMetrics.coverage_vs_risk(predictions, targets, threshold)
            coverages.append(cov)
            risks.append(risk)
        
        return np.array(coverages), np.array(risks)
    
    @staticmethod
    def selective_plot(
        predictions: np.ndarray,
        targets: np.ndarray,
        save_path: Path = None
    ):
        """
        Plot coverage vs risk curve.
        
        Args:
            predictions: Predicted probabilities
            targets: Binary labels
            save_path: Optional path to save figure
        """
        coverages, risks = SelectiveMetrics.coverage_risk_curve(predictions, targets)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(coverages, risks, 'b-', linewidth=2, label='System')
        ax.set_xlabel('Coverage (fraction of queries answered)')
        ax.set_ylabel('Risk (error rate on answered set)')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Selective plot saved: {save_path}")
        
        return fig, ax


def compute_qa_metrics(
    predictions: List[str],  # Generated answers
    references: List[List[str]],  # Multiple valid answers per question
    metric: str = "f1"  # "em" or "f1"
) -> float:
    """
    Compute QA metrics (EM, F1).
    
    Args:
        predictions: List of predicted answers
        references: List of lists of valid answers per question
        metric: "em" or "f1"
        
    Returns:
        Metric value
    """
    from collections import Counter
    
    def f1_score_qa(pred, refs):
        """Compute F1 between prediction and list of references."""
        best_f1 = 0.0
        for ref in refs:
            pred_tokens = Counter(pred.lower().split())
            ref_tokens = Counter(ref.lower().split())
            
            common = pred_tokens & ref_tokens
            if len(pred_tokens) + len(ref_tokens) == 0:
                continue
            
            p = sum(common.values()) / (len(pred_tokens) + 1e-10)
            r = sum(common.values()) / (len(ref_tokens) + 1e-10)
            f1 = 2 * p * r / (p + r + 1e-10)
            best_f1 = max(best_f1, f1)
        
        return best_f1
    
    def em_score_qa(pred, refs):
        """Exact match."""
        pred_norm = pred.lower().strip()
        for ref in refs:
            if pred_norm == ref.lower().strip():
                return 1.0
        return 0.0
    
    if metric == "f1":
        scores = [f1_score_qa(pred, refs) for pred, refs in zip(predictions, references)]
    elif metric == "em":
        scores = [em_score_qa(pred, refs) for pred, refs in zip(predictions, references)]
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return np.mean(scores)
