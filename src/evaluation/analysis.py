"""
Comprehensive analysis tools for ensemble results.
"""

import logging
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import Counter

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyze ensemble results in detail."""

    def __init__(self, results_dir: str = "./results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def save_results(self, results: Dict[str, Any], filename: str):
        """Save results to JSON file."""
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved results to {filepath}")

    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load results from JSON file."""
        filepath = self.results_dir / filename
        with open(filepath, 'r') as f:
            results = json.load(f)
        logger.info(f"Loaded results from {filepath}")
        return results

    def compare_models(self, model_results: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
        """
        Compare predictions across models.
        
        Args:
            model_results: Dict of model_name -> {dataset -> predictions}
            
        Returns:
            Comparison analysis
        """
        comparison = {}
        
        for dataset, model_preds in model_results.items():
            n_models = len(model_preds)
            if n_models < 2:
                continue
            
            # Calculate agreement matrix
            model_names = list(model_preds.keys())
            n_samples = len(list(model_preds.values())[0])
            
            agreement_matrix = np.zeros((n_models, n_models))
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i <= j:
                        preds1 = np.array(model_preds[model1])
                        preds2 = np.array(model_preds[model2])
                        agreement = np.mean(preds1 == preds2)
                        agreement_matrix[i, j] = agreement
                        agreement_matrix[j, i] = agreement
            
            comparison[dataset] = {
                "model_names": model_names,
                "agreement_matrix": agreement_matrix.tolist(),
                "pairwise_agreements": {
                    f"{m1}_vs_{m2}": agreement_matrix[i, j]
                    for i, m1 in enumerate(model_names)
                    for j, m2 in enumerate(model_names)
                    if i < j
                }
            }
        
        return comparison

    def analyze_ensemble_contribution(self, 
                                    individual_accuracy: Dict[str, float],
                                    ensemble_accuracy: float,
                                    dataset_name: str) -> Dict[str, Any]:
        """
        Analyze ensemble improvement over individual models.
        
        Args:
            individual_accuracy: Dict of model_name -> accuracy
            ensemble_accuracy: Ensemble accuracy
            dataset_name: Name of dataset
            
        Returns:
            Contribution analysis
        """
        best_individual = max(individual_accuracy.values())
        improvement = ensemble_accuracy - best_individual
        improvement_pct = (improvement / best_individual * 100) if best_individual > 0 else 0
        
        return {
            "dataset": dataset_name,
            "individual_accuracies": individual_accuracy,
            "best_individual_accuracy": best_individual,
            "best_individual_model": max(individual_accuracy.items(), key=lambda x: x[1])[0],
            "ensemble_accuracy": ensemble_accuracy,
            "absolute_improvement": improvement,
            "relative_improvement_pct": improvement_pct,
            "ensemble_beats_individuals": ensemble_accuracy >= best_individual
        }

    def generate_error_distribution(self, predictions: List[str], 
                                   ground_truth: List[str]) -> Dict[str, Any]:
        """
        Generate distribution of error types.
        
        Args:
            predictions: List of predictions
            ground_truth: List of ground truth labels
            
        Returns:
            Error distribution analysis
        """
        errors = Counter()
        correct = Counter()
        
        for pred, truth in zip(predictions, ground_truth):
            if pred == truth:
                correct[truth] += 1
            else:
                errors[f"{pred} -> {truth}"] += 1
        
        return {
            "total_correct": sum(correct.values()),
            "total_errors": sum(errors.values()),
            "correct_by_label": dict(correct),
            "top_errors": dict(errors.most_common(20))
        }

    def calculate_confidence_gap(self, confidences: List[float],
                                predictions: List[str],
                                ground_truth: List[str]) -> Dict[str, Any]:
        """
        Calculate gap between confidence and accuracy.
        
        Args:
            confidences: Confidence scores
            predictions: Predictions
            ground_truth: Ground truth labels
            
        Returns:
            Confidence gap analysis
        """
        confidences = np.array(confidences)
        is_correct = np.array(predictions) == np.array(ground_truth)
        
        correct_confidence = confidences[is_correct].mean() if is_correct.sum() > 0 else 0
        incorrect_confidence = confidences[~is_correct].mean() if (~is_correct).sum() > 0 else 0
        
        gap = correct_confidence - incorrect_confidence
        
        return {
            "avg_confidence_correct": float(correct_confidence),
            "avg_confidence_incorrect": float(incorrect_confidence),
            "confidence_gap": float(gap),
            "n_correct": int(is_correct.sum()),
            "n_incorrect": int((~is_correct).sum())
        }

    def create_summary_table(self, results_dict: Dict[str, Dict[str, float]]) -> str:
        """
        Create summary table of results.
        
        Args:
            results_dict: Dictionary of model_name -> {metric_name -> value}
            
        Returns:
            Formatted table string
        """
        lines = []
        
        # Header
        models = list(results_dict.keys())
        metrics = set()
        for model_results in results_dict.values():
            metrics.update(model_results.keys())
        metrics = sorted(metrics)
        
        header = "Model".ljust(20) + "| " + " | ".join(m[:15].ljust(15) for m in metrics)
        lines.append(header)
        lines.append("-" * len(header))
        
        # Rows
        for model in models:
            row = model.ljust(20) + "| "
            values = [results_dict[model].get(m, 0) for m in metrics]
            row += " | ".join(f"{v:.4f}".ljust(15) for v in values)
            lines.append(row)
        
        return "\n".join(lines)
