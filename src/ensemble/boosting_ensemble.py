"""
Boosting-based weighted majority vote ensemble.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


class BoostingEnsemble:
    """Weighted majority vote ensemble with boosting."""

    def __init__(self, model_names: List[str], config: Dict[str, Any] = None):
        """
        Initialize boosting ensemble.
        
        Args:
            model_names: List of model names
            config: Configuration dictionary
        """
        self.model_names = model_names
        self.n_models = len(model_names)
        self.config = config or {}
        
        # Initialize weights uniformly
        self.weights = {name: 1.0 / self.n_models for name in model_names}
        self.model_accuracies = {name: 0.5 for name in model_names}
        self.total_samples = 0

    def update_weights(self, model_accuracies: Dict[str, float], 
                      difficulty_levels: Optional[List[float]] = None):
        """
        Update model weights based on accuracy.
        
        Args:
            model_accuracies: Accuracy per model
            difficulty_levels: Optional difficulty levels for each sample
        """
        self.model_accuracies = model_accuracies.copy()
        
        # Base weights from accuracy
        total_accuracy = sum(model_accuracies.values())
        if total_accuracy > 0:
            new_weights = {
                name: acc / total_accuracy 
                for name, acc in model_accuracies.items()
            }
        else:
            new_weights = self.weights.copy()
        
        # Smooth weights to avoid extreme differences
        alpha = 0.9  # Smoothing factor
        for name in self.model_names:
            self.weights[name] = alpha * new_weights[name] + (1 - alpha) * self.weights[name]
        
        logger.info(f"Updated weights: {self.weights}")

    def predict(self, predictions: Dict[str, str], options: Optional[List[str]] = None,
                confidences: Optional[Dict[str, float]] = None) -> Tuple[str, float]:
        """
        Make ensemble prediction using weighted majority vote.
        
        Args:
            predictions: Dict of model_name -> predicted_answer
            options: List of valid options
            confidences: Optional confidence scores per model
            
        Returns:
            Tuple of (predicted_answer, confidence_score)
        """
        if not predictions or len(predictions) == 0:
            return "", 0.0
        
        # Weighted vote counting
        votes = {}
        vote_weights = {}
        
        for model_name, prediction in predictions.items():
            if model_name not in self.model_names:
                logger.warning(f"Unknown model: {model_name}")
                continue
            
            weight = self.weights.get(model_name, 1.0 / self.n_models)
            confidence = confidences.get(model_name, 1.0) if confidences else 1.0
            weighted_vote = weight * confidence
            
            if prediction not in votes:
                votes[prediction] = 0.0
                vote_weights[prediction] = []
            
            votes[prediction] += weighted_vote
            vote_weights[prediction].append((model_name, weight, confidence))
        
        if not votes:
            return "", 0.0
        
        # Get answer(s) with highest weighted vote
        total_votes = sum(votes.values())
        best_value = max(votes.values())
        # Tolerance for numeric tie
        eps = 1e-12
        candidates = [label for label, v in votes.items() if abs(v - best_value) <= eps]

        if len(candidates) == 1:
            predicted_answer = candidates[0]
        else:
            # Deterministic tie-breaker: prefer the prediction from the highest-weight model
            sorted_models = sorted(self.model_names, key=lambda n: self.weights.get(n, 0.0), reverse=True)
            chosen = None
            for m in sorted_models:
                pred = predictions.get(m)
                if pred in candidates:
                    chosen = pred
                    break
            predicted_answer = chosen if chosen is not None else candidates[0]

        vote_confidence = (votes.get(predicted_answer, 0.0) / total_votes) if total_votes > 0 else 0.0
        
        logger.debug(
            f"Votes: {votes}\n"
            f"Prediction: {predicted_answer} (confidence: {vote_confidence:.3f})"
        )
        
        return predicted_answer, vote_confidence

    def batch_predict(self, batch_predictions: List[Dict[str, str]], 
                     batch_options: Optional[List[List[str]]] = None,
                     batch_confidences: Optional[List[Dict[str, float]]] = None) -> List[Tuple[str, float]]:
        """
        Make ensemble predictions for a batch.
        
        Args:
            batch_predictions: List of prediction dicts
            batch_options: List of option lists (one per sample)
            batch_confidences: List of confidence dicts
            
        Returns:
            List of (prediction, confidence) tuples
        """
        results = []
        for i, predictions in enumerate(batch_predictions):
            options = batch_options[i] if batch_options else None
            confidences = batch_confidences[i] if batch_confidences else None
            
            result = self.predict(predictions, options, confidences)
            results.append(result)
        
        return results

    def get_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.weights.copy()

    def reset_weights(self):
        """Reset weights to uniform distribution."""
        self.weights = {name: 1.0 / self.n_models for name in self.model_names}
        logger.info("Reset weights to uniform distribution")

    def analyze_disagreement(self, predictions: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze disagreement between models.
        
        Args:
            predictions: Dict of model_name -> predicted_answer
            
        Returns:
            Dictionary with disagreement metrics
        """
        unique_predictions = set(predictions.values())
        
        return {
            "n_unique_predictions": len(unique_predictions),
            "n_agreeing": self.n_models - len(unique_predictions) + 1,
            "disagreement_rate": (len(unique_predictions) - 1) / max(1, self.n_models - 1),
            "predictions": predictions.copy()
        }
