"""
Evaluation metrics for medical QA.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate evaluation metrics for QA systems."""

    @staticmethod
    def calculate_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
        """
        Calculate accuracy.
        
        Args:
            predictions: List of predictions
            ground_truth: List of ground truth labels
            
        Returns:
            Accuracy score
        """
        return accuracy_score(ground_truth, predictions)

    @staticmethod
    def calculate_per_category_accuracy(predictions: List[str], ground_truth: List[str],
                                       categories: List[str]) -> Dict[str, float]:
        """
        Calculate accuracy per category.
        
        Args:
            predictions: List of predictions
            ground_truth: List of ground truth labels
            categories: List of category labels
            
        Returns:
            Dict of category -> accuracy
        """
        unique_categories = set(categories)
        per_category_acc = {}
        
        for category in unique_categories:
            mask = np.array(categories) == category
            if mask.sum() > 0:
                cat_acc = accuracy_score(
                    np.array(ground_truth)[mask],
                    np.array(predictions)[mask]
                )
                per_category_acc[category] = cat_acc
        
        return per_category_acc

    @staticmethod
    def calculate_calibration(confidences: List[float], predictions: List[str],
                             ground_truth: List[str], n_bins: int = 10) -> Dict[str, Any]:
        """
        Calculate calibration metrics (ECE, MCE).
        
        Args:
            confidences: List of confidence scores
            predictions: List of predictions
            ground_truth: List of ground truth labels
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary with calibration metrics
        """
        # Convert to binary correctness
        correctness = np.array(predictions) == np.array(ground_truth)
        confidences = np.array(confidences)
        
        # Calculate Expected Calibration Error (ECE)
        ece = np.mean(np.abs(confidences - correctness.astype(float)))
        
        # Calculate Maximum Calibration Error (MCE)
        mce = np.max(np.abs(confidences - correctness.astype(float)))
        
        return {
            "ece": ece,
            "mce": mce,
            "avg_confidence": np.mean(confidences),
            "avg_accuracy": np.mean(correctness)
        }

    @staticmethod
    def calculate_disagreement_rate(model_predictions: Dict[str, List[str]]) -> float:
        """
        Calculate rate of model disagreement.
        
        Args:
            model_predictions: Dict of model_name -> list of predictions
            
        Returns:
            Disagreement rate (0-1)
        """
        n_models = len(model_predictions)
        if n_models < 2:
            return 0.0
        
        predictions_array = np.array([v for v in model_predictions.values()])
        n_samples = predictions_array.shape[1]
        
        disagreement_count = 0
        for i in range(n_samples):
            unique_preds = len(set(predictions_array[:, i]))
            if unique_preds > 1:
                disagreement_count += 1
        
        return disagreement_count / n_samples

    @staticmethod
    def analyze_errors(predictions: List[str], ground_truth: List[str],
                      examples: Optional[List[Dict]] = None, top_k: int = 20) -> Dict[str, Any]:
        """
        Analyze error patterns.
        
        Args:
            predictions: List of predictions
            ground_truth: List of ground truth labels
            examples: Optional list of example dicts
            top_k: Number of top errors to return
            
        Returns:
            Dictionary with error analysis
        """
        errors = []
        
        for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
            if pred != truth:
                error_info = {
                    "index": i,
                    "prediction": pred,
                    "ground_truth": truth,
                }
                if examples:
                    error_info["question"] = examples[i].get("question", "")
                    error_info["dataset"] = examples[i].get("dataset", "")
                
                errors.append(error_info)
        
        # Sort by index to keep most recent errors
        errors = errors[-top_k:]
        
        return {
            "total_errors": len(errors),
            "error_rate": len(errors) / len(predictions),
            "top_errors": errors
        }


class EvaluationReport:
    """Generate comprehensive evaluation report."""

    def __init__(self):
        self.metrics = {}

    def add_dataset_evaluation(self, dataset_name: str, metrics: Dict[str, Any]):
        """Add evaluation metrics for a dataset."""
        self.metrics[dataset_name] = metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        if not self.metrics:
            return {}
        
        # Calculate averages across datasets
        all_accuracies = [m.get("accuracy", 0) for m in self.metrics.values()]
        all_eces = [m.get("calibration", {}).get("ece", 0) for m in self.metrics.values()]
        
        summary = {
            "n_datasets": len(self.metrics),
            "avg_accuracy": np.mean(all_accuracies) if all_accuracies else 0,
            "avg_ece": np.mean(all_eces) if all_eces else 0,
            "metrics_by_dataset": self.metrics
        }
        
        return summary

    def print_report(self):
        """Print formatted evaluation report."""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("EVALUATION REPORT")
        print("="*70)
        
        print(f"\nNumber of Datasets: {summary.get('n_datasets', 'N/A')}")
        print(f"Average Accuracy: {summary.get('avg_accuracy', 0):.4f}")
        print(f"Average ECE: {summary.get('avg_ece', 0):.4f}")
        
        print("\nPer-Dataset Results:")
        print("-"*70)
        
        for dataset, metrics in summary.get('metrics_by_dataset', {}).items():
            print(f"\n{dataset}:")
            print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"  Calibration (ECE): {metrics.get('calibration', {}).get('ece', 0):.4f}")
            print(f"  Disagreement Rate: {metrics.get('disagreement_rate', 0):.4f}")
        
        print("\n" + "="*70)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return self.get_summary()
