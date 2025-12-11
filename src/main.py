"""
Main orchestration script for the multi-LLM ensemble medical QA system.
"""

import logging
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules
from src.utils.config import ConfigLoader
from src.utils.dataset_loader import DatasetLoader
from src.utils.embedder import QuestionEmbedder
from src.llm_clients.gpt4_client import GPT4Client
from src.llm_clients.llama2_client import LLaMA2Client
from src.llm_clients.medical_chatgpt_client import MedicalChatGPTClient
from src.ensemble.boosting_ensemble import BoostingEnsemble
from src.ensemble.dynamic_selection import ClusterBasedEnsemble
from src.evaluation.metrics import MetricsCalculator, EvaluationReport
from src.evaluation.analysis import ResultsAnalyzer


class EnsembleOrchestrator:
    """Main orchestration class for ensemble evaluation."""

    def __init__(self, config_dir: str = "./config"):
        """Initialize orchestrator."""
        self.config_loader = ConfigLoader(config_dir)
        self.api_keys = self.config_loader.get_api_keys()
        self.exp_config = self.config_loader.get_experiment_config()
        
        self.llm_clients = {}
        self.datasets = {}
        self.results = {}
        self.analyzer = ResultsAnalyzer()

    def initialize_llm_clients(self):
        """Initialize all LLM clients."""
        logger.info("Initializing LLM clients...")
        
        try:
            # GPT-4 client
            if self.api_keys.get('openai', {}).get('api_key'):
                self.llm_clients['gpt4'] = GPT4Client(
                    self.api_keys['openai']['api_key'],
                    self.api_keys['openai']
                )
                logger.info("✓ GPT-4 client initialized")
            else:
                logger.warning("GPT-4 API key not found")
        except Exception as e:
            logger.error(f"Failed to initialize GPT-4 client: {e}")

        try:
            # LLaMA-2 client (Purdue University instance)
            if self.api_keys.get('purdue_llama', {}).get('api_key'):
                self.llm_clients['llama2'] = LLaMA2Client(
                    self.api_keys['purdue_llama']['api_key'],
                    self.api_keys['purdue_llama']
                )
                logger.info("✓ LLaMA-2 client initialized (Purdue)")
            else:
                logger.warning("Purdue LLaMA API key not found")
        except Exception as e:
            logger.error(f"Failed to initialize LLaMA-2 client: {e}")

        try:
            # Medical AI 4o client (Oxford Medical Edition)
            if self.api_keys.get('openai', {}).get('api_key'):
                self.llm_clients['medical_ai_4o'] = MedicalChatGPTClient(
                    self.api_keys['openai']['api_key'],
                    self.api_keys['openai']
                )
                logger.info("✓ Medical AI 4o client initialized")
            else:
                logger.warning("Medical AI 4o requires OpenAI API key")
        except Exception as e:
            logger.error(f"Failed to initialize Medical AI 4o client: {e}")

        if not self.llm_clients:
            logger.error("No LLM clients initialized!")
            sys.exit(1)

        logger.info(f"Initialized {len(self.llm_clients)} LLM client(s)")

    def load_datasets(self):
        """Load all configured datasets."""
        logger.info("Loading datasets...")
        
        loader = DatasetLoader()
        self.datasets = loader.load_all_datasets(self.exp_config)
        
        for dataset_name, examples in self.datasets.items():
            logger.info(f"Loaded {len(examples)} examples from {dataset_name}")

    def evaluate_baseline(self, sample_size: int = 100):
        """
        Evaluate individual models on a small sample.
        
        Args:
            sample_size: Number of samples to evaluate on
        """
        logger.info(f"Starting baseline evaluation (sample_size={sample_size})...")
        
        baseline_results = {}
        
        for dataset_name, examples in self.datasets.items():
            logger.info(f"\nEvaluating on {dataset_name}...")
            examples_sample = examples[:sample_size]
            
            dataset_results = {}
            
            for model_name, client in self.llm_clients.items():
                logger.info(f"  Evaluating {model_name}...")
                
                predictions = []
                
                for example in tqdm(examples_sample, desc=f"{model_name}"):
                    question = example.get("question", "")
                    options = example.get("options", [])
                    
                    # Format question
                    prompt = client.format_question(question, options)
                    
                    # Get prediction
                    try:
                        response = client.query(prompt)
                        pred_text = response.get("text", "")
                        prediction = client.extract_answer(pred_text, options)
                        predictions.append(prediction)
                    except Exception as e:
                        logger.warning(f"Error querying {model_name}: {e}")
                        predictions.append("")
                
                # Calculate accuracy
                gold_labels = [ex.get("gold_label", "") for ex in examples_sample]
                accuracy = MetricsCalculator.calculate_accuracy(predictions, gold_labels)
                
                dataset_results[model_name] = {
                    "accuracy": float(accuracy),
                    "n_samples": len(examples_sample),
                    "predictions": predictions[:20]  # Save first 20
                }
                
                logger.info(f"  {model_name} accuracy: {accuracy:.4f}")
            
            baseline_results[dataset_name] = dataset_results
        
        self.results['baseline'] = baseline_results
        self.analyzer.save_results(baseline_results, "baseline_results.json")
        logger.info("Baseline evaluation completed!")
        
        return baseline_results

    def evaluate_boosting_ensemble(self, sample_size: int = 100):
        """Evaluate boosting ensemble."""
        logger.info("Evaluating boosting ensemble...")
        
        if 'baseline' not in self.results:
            logger.error("Baseline results not found. Run evaluate_baseline first.")
            return
        
        ensemble_results = {}
        
        for dataset_name, baseline in self.results['baseline'].items():
            logger.info(f"\nEvaluating boosting ensemble on {dataset_name}...")
            
            examples_sample = self.datasets[dataset_name][:sample_size]
            gold_labels = [ex.get("gold_label", "") for ex in examples_sample]
            
            # Initialize ensemble
            ensemble = BoostingEnsemble(list(self.llm_clients.keys()))
            
            # Update weights based on baseline
            model_accs = {
                model: results.get("accuracy", 0.5)
                for model, results in baseline.items()
            }
            ensemble.update_weights(model_accs)
            
            # Get predictions from all models
            all_predictions = {}
            for model_name in self.llm_clients.keys():
                all_predictions[model_name] = baseline[model_name].get("predictions", [])
            
            # Make ensemble predictions (using available predictions)
            ensemble_preds = []
            disagreement_rates = []
            
            for i in range(len(examples_sample)):
                model_preds = {
                    model: all_predictions[model][i] if i < len(all_predictions[model]) else ""
                    for model in self.llm_clients.keys()
                }
                
                pred, conf = ensemble.predict(model_preds)
                ensemble_preds.append(pred)
                
                disagree = ensemble.analyze_disagreement(model_preds)
                disagreement_rates.append(disagree.get("disagreement_rate", 0))
            
            # Calculate ensemble accuracy
            ensemble_accuracy = MetricsCalculator.calculate_accuracy(ensemble_preds, gold_labels)
            
            ensemble_results[dataset_name] = {
                "ensemble_accuracy": float(ensemble_accuracy),
                "individual_accuracies": model_accs,
                "weights": ensemble.get_weights(),
                "avg_disagreement_rate": float(np.mean(disagreement_rates)),
                "ensemble_predictions": ensemble_preds[:20]
            }
            
            logger.info(f"  Ensemble accuracy: {ensemble_accuracy:.4f}")
            logger.info(f"  Disagreement rate: {np.mean(disagreement_rates):.4f}")
        
        self.results['boosting_ensemble'] = ensemble_results
        self.analyzer.save_results(ensemble_results, "boosting_ensemble_results.json")
        logger.info("Boosting ensemble evaluation completed!")
        
        return ensemble_results

    def generate_report(self):
        """Generate final evaluation report."""
        logger.info("Generating evaluation report...")
        
        report = EvaluationReport()
        
        # Add baseline results
        if 'baseline' in self.results:
            for dataset, results in self.results['baseline'].items():
                avg_acc = np.mean([r.get("accuracy", 0) for r in results.values()])
                report.add_dataset_evaluation(
                    f"{dataset}_baseline",
                    {
                        "accuracy": avg_acc,
                        "model_results": results
                    }
                )
        
        # Add ensemble results
        if 'boosting_ensemble' in self.results:
            for dataset, results in self.results['boosting_ensemble'].items():
                report.add_dataset_evaluation(
                    f"{dataset}_boosting",
                    {
                        "accuracy": results.get("ensemble_accuracy", 0),
                        "calibration": {"ece": 0.0},
                        "disagreement_rate": results.get("avg_disagreement_rate", 0)
                    }
                )
        
        report.print_report()
        
        # Save report
        summary = report.to_dict()
        self.analyzer.save_results(summary, "evaluation_report.json")
        
        return report

    def run_full_pipeline(self, sample_size: int = 100):
        """Run complete evaluation pipeline."""
        logger.info("Starting full evaluation pipeline...")
        
        try:
            self.initialize_llm_clients()
            self.load_datasets()
            self.evaluate_baseline(sample_size)
            self.evaluate_boosting_ensemble(sample_size)
            self.generate_report()
            
            logger.info("✓ Pipeline completed successfully!")
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            sys.exit(1)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-LLM Ensemble for Medical QA")
    parser.add_argument(
        "--config-dir",
        default="./config",
        help="Path to config directory"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only run baseline evaluation"
    )
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = EnsembleOrchestrator(config_dir=args.config_dir)
    
    if args.baseline_only:
        orchestrator.initialize_llm_clients()
        orchestrator.load_datasets()
        orchestrator.evaluate_baseline(args.sample_size)
    else:
        orchestrator.run_full_pipeline(args.sample_size)


if __name__ == "__main__":
    main()
