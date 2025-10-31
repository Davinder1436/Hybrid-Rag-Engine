"""
Uncertainty head for retrieval confidence prediction.
Predicts calibrated probability that retrieved evidence is sufficient for answering.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import pickle


class UncertaintyHead(nn.Module):
    """
    Lightweight MLP-based uncertainty head that predicts retrieval sufficiency.
    
    Input features:
    - Query embedding (from encoder)
    - Max score from retrieved passages
    - Mean score from retrieved passages
    - Std of scores
    - Number of retrieved passages with score > threshold
    
    Output:
    - p_success: probability that retrieved evidence is sufficient for correct answer
    """
    
    def __init__(
        self,
        input_dim: int = 384,  # Dimension of embeddings (e.g., MiniLM-L6)
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_classes: int = 1  # Binary: sufficient/insufficient
    ):
        """
        Args:
            input_dim: Feature dimension (query embedding dim)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            num_classes: Number of output classes (1 for binary with sigmoid)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Feature dimension: query_emb + score_stats
        # score_stats: [max_score, mean_score, std_score, num_high_scores, retriever_agreement]
        feature_dim = input_dim + 5
        
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Sigmoid for binary classification
        self.sigmoid = nn.Sigmoid()
    
    def forward(
        self,
        query_embedding: torch.Tensor,
        score_stats: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            query_embedding: Shape (batch_size, input_dim)
            score_stats: Shape (batch_size, 5) with [max_score, mean_score, std_score, num_high, agreement]
            
        Returns:
            p_success: Shape (batch_size, 1) with values in [0, 1]
        """
        # Concatenate embeddings and stats
        features = torch.cat([query_embedding, score_stats], dim=1)
        
        # MLP
        logits = self.mlp(features)
        
        # Sigmoid for probability
        p_success = self.sigmoid(logits)
        
        return p_success
    
    def compute_score_stats(
        self,
        scores: np.ndarray,
        score_threshold: float = 0.1
    ) -> Dict[str, float]:
        """
        Compute statistics from retrieved scores.
        
        Args:
            scores: Array of scores from retrieved passages
            score_threshold: Threshold for "high score" count
            
        Returns:
            Dict with keys: max_score, mean_score, std_score, num_high_scores
        """
        if len(scores) == 0:
            return {
                "max_score": 0.0,
                "mean_score": 0.0,
                "std_score": 0.0,
                "num_high_scores": 0
            }
        
        return {
            "max_score": float(np.max(scores)),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)) if len(scores) > 1 else 0.0,
            "num_high_scores": float(np.sum(scores > score_threshold))
        }
    
    def predict(
        self,
        query_embedding: np.ndarray,
        retrieval_results: List[Dict[str, Any]],
        score_threshold: float = 0.1,
        retriever_type: str = "hybrid"
    ) -> Dict[str, float]:
        """
        Predict retrieval sufficiency.
        
        Args:
            query_embedding: Query embedding (1D array, shape (emb_dim,))
            retrieval_results: List of retrieved results with 'score' key
            score_threshold: Threshold for high-score count
            retriever_type: Type of retriever ("sparse", "dense", "hybrid")
            
        Returns:
            Dict with 'p_success' (probability) and score statistics
        """
        # Extract scores
        scores = np.array([r["score"] for r in retrieval_results])
        
        # Compute statistics
        stats = self.compute_score_stats(scores, score_threshold)
        
        # Compute retriever agreement (for hybrid: how many results appear in both)
        agreement = 0.0
        if retriever_type == "hybrid" and len(retrieval_results) > 0:
            sparse_ids = set(r["passage"]["id"] for r in retrieval_results if r.get("retriever") == "sparse")
            dense_ids = set(r["passage"]["id"] for r in retrieval_results if r.get("retriever") == "dense")
            if len(sparse_ids) > 0 and len(dense_ids) > 0:
                agreement = len(sparse_ids & dense_ids) / min(len(sparse_ids), len(dense_ids))
        
        # Convert to tensors
        query_emb_t = torch.from_numpy(query_embedding).float().unsqueeze(0)  # (1, emb_dim)
        score_stats_t = torch.tensor(
            [[stats["max_score"], stats["mean_score"], stats["std_score"], 
              stats["num_high_scores"], agreement]],
            dtype=torch.float32
        )  # (1, 5)
        
        # Forward pass
        with torch.no_grad():
            p_success = self.forward(query_emb_t, score_stats_t)
        
        return {
            "p_success": float(p_success.item()),
            "p_failure": 1.0 - float(p_success.item()),
            **stats,
            "agreement": agreement
        }
    
    def save(self, path: Path):
        """Save model weights and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.state_dict(), path / "uncertainty_head.pt")
        
        config = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes
        }
        with open(path / "uncertainty_config.pkl", 'wb') as f:
            pickle.dump(config, f)
        
        print(f"  ✓ Uncertainty head saved to {path}")
    
    @classmethod
    def load(cls, path: Path, device: str = "cpu"):
        """Load model weights and config."""
        path = Path(path)
        
        with open(path / "uncertainty_config.pkl", 'rb') as f:
            config = pickle.load(f)
        
        model = cls(**config)
        model.load_state_dict(torch.load(path / "uncertainty_head.pt", map_location=device))
        model.to(device)
        model.eval()
        
        print(f"  ✓ Uncertainty head loaded from {path}")
        return model


class UncertaintyTrainer:
    """Trainer for uncertainty head."""
    
    def __init__(
        self,
        model: UncertaintyHead,
        device: str = "cpu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        """
        Args:
            model: UncertaintyHead instance
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.BCELoss()
    
    def train_step(
        self,
        batch_query_embs: torch.Tensor,
        batch_score_stats: torch.Tensor,
        batch_labels: torch.Tensor
    ) -> float:
        """
        Single training step.
        
        Args:
            batch_query_embs: (batch_size, emb_dim)
            batch_score_stats: (batch_size, 5)
            batch_labels: (batch_size, 1) with values in {0, 1}
            
        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward
        logits = self.model(batch_query_embs, batch_score_stats)
        
        # Loss
        loss = self.criterion(logits, batch_labels.float())
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def eval_step(
        self,
        batch_query_embs: torch.Tensor,
        batch_score_stats: torch.Tensor,
        batch_labels: torch.Tensor
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluation step.
        
        Args:
            batch_query_embs: (batch_size, emb_dim)
            batch_score_stats: (batch_size, 5)
            batch_labels: (batch_size, 1)
            
        Returns:
            (loss, predictions, labels)
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(batch_query_embs, batch_score_stats)
            loss = self.criterion(logits, batch_labels.float())
        
        preds = logits.cpu().numpy()
        labels = batch_labels.cpu().numpy()
        
        return loss.item(), preds, labels
