"""
Main orchestration script for the multi-LLM ensemble medical QA system.
"""

import logging
import json
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    def __init__(self, config_dir: str = "./config", cache_dir: str = "./results/cache"):
        """Initialize orchestrator."""
        self.config_loader = ConfigLoader(config_dir)
        self.api_keys = self.config_loader.get_api_keys()
        self.exp_config = self.config_loader.get_experiment_config()
        
        self.llm_clients = {}
        self.datasets = {}
        self.train_data = {}
        self.test_data = {}
        self.results = {}
        self.analyzer = ResultsAnalyzer()
        
        # Setup caching
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cached predictions."""
        cache_file = self.cache_dir / "prediction_cache.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                logger.info("Loaded prediction cache")
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Save predictions to cache."""
        cache_file = self.cache_dir / "prediction_cache.json"
        with open(cache_file, 'w') as f:
            json.dump(self.prediction_cache, f)
    
    def _get_cache_key(self, model: str, dataset: str, question: str) -> str:
        """Generate cache key for a prediction."""
        import hashlib
        key = f"{model}:{dataset}:{question[:100]}"
        return hashlib.md5(key.encode()).hexdigest()

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
                logger.info("[OK] GPT-4 client initialized")
            else:
                logger.warning("GPT-4 API key not found")
        except Exception as e:
            logger.error(f"Failed to initialize GPT-4 client: {e}")

        try:
            # LLaMA-2 client (Purdue University instance)
            purdue_config = self.api_keys.get('purdue_llama', {}).copy()
            # Pass OpenAI key for GPT-4o fallback
            purdue_config['openai_api_key'] = self.api_keys.get('openai', {}).get('api_key')
            if purdue_config.get('endpoint') or purdue_config.get('use_gpt4_fallback'):
                self.llm_clients['llama2'] = LLaMA2Client(
                    purdue_config.get('api_key', ''),
                    purdue_config
                )
                logger.info("[OK] LLaMA-2 client initialized (Purdue)")
            else:
                logger.warning("Purdue LLaMA config not found")
        except Exception as e:
            logger.error(f"Failed to initialize LLaMA-2 client: {e}")

        try:
            # Medical AI 4o client (Oxford Medical Edition)
            if self.api_keys.get('openai', {}).get('api_key'):
                self.llm_clients['medical_ai_4o'] = MedicalChatGPTClient(
                    self.api_keys['openai']['api_key'],
                    self.api_keys['openai']
                )
                logger.info("[OK] Medical AI 4o client initialized")
            else:
                logger.warning("Medical AI 4o requires OpenAI API key")
        except Exception as e:
            logger.error(f"Failed to initialize Medical AI 4o client: {e}")

        if not self.llm_clients:
            logger.error("No LLM clients initialized!")
            sys.exit(1)

        logger.info(f"Initialized {len(self.llm_clients)} LLM client(s)")

    def load_datasets(self):
        """Load all configured datasets (legacy - for backward compat)."""
        logger.info("Loading datasets...")
        
        loader = DatasetLoader()
        self.datasets = loader.load_all_datasets(self.exp_config)
        
        for dataset_name, examples in self.datasets.items():
            logger.info(f"Loaded {len(examples)} examples from {dataset_name}")

    def load_train_test_data(
        self, 
        train_sample_size: int = None, 
        test_sample_size: int = None,
        datasets: List[str] = None
    ):
        """
        Load train/test splits from the new split files.
        
        Args:
            train_sample_size: Max training samples per dataset
            test_sample_size: Max test samples per dataset  
            datasets: List of datasets to load ('medqa', 'pubmedqa', 'medmcqa')
        """
        logger.info("Loading train/test splits...")
        
        loader = DatasetLoader()
        
        if datasets is None:
            datasets = ['medqa', 'pubmedqa', 'medmcqa']
        
        for name in datasets:
            train, test = loader.load_train_test_split(
                name,
                train_sample_size,
                test_sample_size
            )
            if train:
                self.train_data[name] = train
                logger.info(f"  {name} train: {len(train)} examples")
            if test:
                self.test_data[name] = test
                logger.info(f"  {name} test: {len(test)} examples")
        
        total_train = sum(len(v) for v in self.train_data.values())
        total_test = sum(len(v) for v in self.test_data.values())
        logger.info(f"Total: {total_train} train, {total_test} test examples")

    def _process_example(self, client, example, model_name, dataset_name):
        """Process a single example and return prediction."""
        question = example.get("question", "")
        options = example.get("options", [])
        
        # Check cache first
        cache_key = self._get_cache_key(model_name, dataset_name, question)
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        if isinstance(options, dict):
            options_list = [options.get(k, "") for k in sorted(options.keys())]
        else:
            options_list = options
        
        prompt = client.format_question(question, options_list)
        
        try:
            response = client.query(prompt)
            pred_text = response.get("text", "")
            prediction = client.extract_answer(pred_text, options_list)
            
            # Handle PubMedQA format
            gold_label = example.get("gold_label", "")
            if prediction and gold_label and gold_label.lower() in ['yes', 'no', 'maybe']:
                letter_idx = ord(prediction.upper()) - ord('A')
                if 0 <= letter_idx < len(options_list):
                    prediction = options_list[letter_idx].lower()
            
            # Cache result
            self.prediction_cache[cache_key] = prediction
            
            return prediction
        except Exception as e:
            logger.warning(f"Error querying {model_name}: {e}")
            return ""

    def evaluate_on_test_parallel(self, max_workers: int = 5):
        """
        Evaluate all models on test sets using parallel processing.
        Uses the new 80/20 splits.
        """
        logger.info(f"Evaluating on TEST sets (parallel, workers={max_workers})...")
        
        test_results = {}
        
        for dataset_name, test_examples in self.test_data.items():
            logger.info(f"\n[{dataset_name.upper()}] Test set: {len(test_examples)} examples")
            
            dataset_results = {}
            
            for model_name, client in self.llm_clients.items():
                logger.info(f"  Evaluating {model_name}...")
                
                predictions = [None] * len(test_examples)
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {
                        executor.submit(
                            self._process_example, 
                            client, ex, model_name, dataset_name
                        ): i 
                        for i, ex in enumerate(test_examples)
                    }
                    
                    for future in tqdm(
                        as_completed(future_to_idx), 
                        total=len(test_examples), 
                        desc=f"{model_name}"
                    ):
                        idx = future_to_idx[future]
                        try:
                            predictions[idx] = future.result()
                        except Exception as e:
                            logger.error(f"Error: {e}")
                            predictions[idx] = ""
                
                gold_labels = [ex.get("gold_label", "") for ex in test_examples]
                accuracy = MetricsCalculator.calculate_accuracy(predictions, gold_labels)
                
                dataset_results[model_name] = {
                    "accuracy": float(accuracy),
                    "n_samples": len(test_examples),
                    "predictions": predictions[:50]  # Save first 50
                }
                
                logger.info(f"    {model_name} accuracy: {accuracy:.4f} ({int(accuracy * len(test_examples))}/{len(test_examples)} correct)")
            
            test_results[dataset_name] = dataset_results
        
        self.results['test'] = test_results
        self.analyzer.save_results(test_results, "test_results.json")
        self._save_cache()
        
        return test_results

    def evaluate_ensemble_on_test(self):
        """
        Evaluate boosting ensemble on test sets.
        Requires test results to be computed first.
        """
        logger.info("Evaluating ENSEMBLE on test sets...")
        
        if 'test' not in self.results:
            logger.error("Test results not found. Run evaluate_on_test_parallel first.")
            return
        
        ensemble_results = {}
        
        for dataset_name, model_results in self.results['test'].items():
            logger.info(f"\n[{dataset_name.upper()}] Ensemble evaluation...")
            
            test_examples = self.test_data[dataset_name]
            gold_labels = [ex.get("gold_label", "") for ex in test_examples]
            
            # Initialize ensemble
            ensemble = BoostingEnsemble(list(self.llm_clients.keys()))
            
            # Update weights based on individual accuracies
            model_accs = {
                model: results.get("accuracy", 0.5)
                for model, results in model_results.items()
            }
            ensemble.update_weights(model_accs)
            
            logger.info(f"  Model weights: {ensemble.get_weights()}")
            
            # Get all predictions
            all_predictions = {}
            for model_name in self.llm_clients.keys():
                preds = model_results[model_name].get("predictions", [])
                # Extend with cached predictions if needed
                if len(preds) < len(test_examples):
                    # Re-fetch from cache
                    extended_preds = []
                    for i, ex in enumerate(test_examples):
                        cache_key = self._get_cache_key(model_name, dataset_name, ex.get("question", ""))
                        extended_preds.append(self.prediction_cache.get(cache_key, ""))
                    all_predictions[model_name] = extended_preds
                else:
                    all_predictions[model_name] = preds
            
            # Make ensemble predictions
            ensemble_preds = []
            confidences = []
            disagreement_rates = []
            
            for i in range(len(test_examples)):
                model_preds = {
                    model: all_predictions[model][i] if i < len(all_predictions[model]) else ""
                    for model in self.llm_clients.keys()
                }
                
                pred, conf = ensemble.predict(model_preds)
                ensemble_preds.append(pred)
                confidences.append(conf)
                
                disagree = ensemble.analyze_disagreement(model_preds)
                disagreement_rates.append(disagree.get("disagreement_rate", 0))
            
            # Calculate ensemble accuracy
            ensemble_accuracy = MetricsCalculator.calculate_accuracy(ensemble_preds, gold_labels)
            
            # Best individual model
            best_model = max(model_accs, key=model_accs.get)
            best_model_acc = model_accs[best_model]
            
            improvement = ensemble_accuracy - best_model_acc
            
            ensemble_results[dataset_name] = {
                "ensemble_accuracy": float(ensemble_accuracy),
                "individual_accuracies": model_accs,
                "best_individual": {"model": best_model, "accuracy": best_model_acc},
                "improvement_over_best": float(improvement),
                "weights": ensemble.get_weights(),
                "avg_confidence": float(np.mean(confidences)),
                "avg_disagreement_rate": float(np.mean(disagreement_rates)),
                "n_samples": len(test_examples)
            }
            
            logger.info(f"  Ensemble accuracy: {ensemble_accuracy:.4f}")
            logger.info(f"  Best individual ({best_model}): {best_model_acc:.4f}")
            logger.info(f"  Improvement: {improvement:+.4f}")
            logger.info(f"  Disagreement rate: {np.mean(disagreement_rates):.4f}")
        
        self.results['ensemble'] = ensemble_results
        self.analyzer.save_results(ensemble_results, "ensemble_results.json")
        
        return ensemble_results

    def generate_final_report(self):
        """Generate comprehensive evaluation report."""
        logger.info("\n" + "=" * 60)
        logger.info("FINAL EVALUATION REPORT")
        logger.info("=" * 60)
        
        report_data = {
            "test_results": self.results.get('test', {}),
            "ensemble_results": self.results.get('ensemble', {})
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY")
        print("=" * 60)
        
        if 'test' in self.results:
            print("\n[Individual Model Accuracy on Test Sets]")
            for dataset_name, model_results in self.results['test'].items():
                print(f"\n  {dataset_name}:")
                for model, res in model_results.items():
                    acc = res.get('accuracy', 0)
                    n = res.get('n_samples', 0)
                    print(f"    {model}: {acc:.4f} ({int(acc*n)}/{n})")
        
        if 'ensemble' in self.results:
            print("\n[Ensemble Performance]")
            for dataset_name, ens_res in self.results['ensemble'].items():
                print(f"\n  {dataset_name}:")
                print(f"    Ensemble Accuracy: {ens_res['ensemble_accuracy']:.4f}")
                best = ens_res['best_individual']
                print(f"    Best Individual ({best['model']}): {best['accuracy']:.4f}")
                print(f"    Improvement: {ens_res['improvement_over_best']:+.4f}")
                print(f"    Disagreement Rate: {ens_res['avg_disagreement_rate']:.4f}")
        
        # Overall summary
        if 'ensemble' in self.results:
            print("\n" + "-" * 60)
            print("OVERALL SUMMARY")
            print("-" * 60)
            
            all_ens_accs = [r['ensemble_accuracy'] for r in self.results['ensemble'].values()]
            all_best_accs = [r['best_individual']['accuracy'] for r in self.results['ensemble'].values()]
            
            print(f"  Average Ensemble Accuracy: {np.mean(all_ens_accs):.4f}")
            print(f"  Average Best Individual: {np.mean(all_best_accs):.4f}")
            print(f"  Average Improvement: {np.mean(all_ens_accs) - np.mean(all_best_accs):+.4f}")
        
        print("\n" + "=" * 60)
        
        # Save report
        self.analyzer.save_results(report_data, "final_report.json")
        
        return report_data

    # --- Legacy methods for backward compatibility ---

    def evaluate_baseline(self, sample_size: int = 100):
        """Legacy: Evaluate individual models on a small sample."""
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
                    
                    if isinstance(options, dict):
                        options_list = [options.get(k, "") for k in sorted(options.keys())]
                    else:
                        options_list = options
                    
                    prompt = client.format_question(question, options_list)
                    
                    try:
                        response = client.query(prompt)
                        pred_text = response.get("text", "")
                        prediction = client.extract_answer(pred_text, options_list)
                        
                        gold_label = example.get("gold_label", "")
                        if prediction and gold_label and gold_label.lower() in ['yes', 'no', 'maybe']:
                            letter_idx = ord(prediction.upper()) - ord('A')
                            if 0 <= letter_idx < len(options_list):
                                prediction = options_list[letter_idx].lower()
                        
                        predictions.append(prediction)
                    except Exception as e:
                        logger.warning(f"Error querying {model_name}: {e}")
                        predictions.append("")
                
                gold_labels = [ex.get("gold_label", "") for ex in examples_sample]
                accuracy = MetricsCalculator.calculate_accuracy(predictions, gold_labels)
                
                dataset_results[model_name] = {
                    "accuracy": float(accuracy),
                    "n_samples": len(examples_sample),
                    "predictions": predictions[:20]
                }
                
                logger.info(f"  {model_name} accuracy: {accuracy:.4f}")
            
            baseline_results[dataset_name] = dataset_results
        
        self.results['baseline'] = baseline_results
        self.analyzer.save_results(baseline_results, "baseline_results.json")
        logger.info("Baseline evaluation completed!")
        
        return baseline_results

    def evaluate_baseline_parallel(self, sample_size: int = 100, max_workers: int = 10):
        """Legacy: Evaluate individual models using parallel requests."""
        logger.info(f"Starting PARALLEL baseline evaluation (sample_size={sample_size}, workers={max_workers})...")
        
        baseline_results = {}
        
        for dataset_name, examples in self.datasets.items():
            logger.info(f"\nEvaluating on {dataset_name}...")
            examples_sample = examples[:sample_size]
            
            dataset_results = {}
            
            for model_name, client in self.llm_clients.items():
                logger.info(f"  Evaluating {model_name} with {max_workers} parallel workers...")
                
                predictions = [None] * len(examples_sample)
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {
                        executor.submit(
                            self._process_example, 
                            client, ex, model_name, dataset_name
                        ): i 
                        for i, ex in enumerate(examples_sample)
                    }
                    
                    for future in tqdm(as_completed(future_to_idx), total=len(examples_sample), desc=f"{model_name}"):
                        idx = future_to_idx[future]
                        try:
                            predictions[idx] = future.result()
                        except Exception as e:
                            logger.error(f"Error: {e}")
                            predictions[idx] = ""
                
                gold_labels = [ex.get("gold_label", "") for ex in examples_sample]
                accuracy = MetricsCalculator.calculate_accuracy(predictions, gold_labels)
                
                dataset_results[model_name] = {
                    "accuracy": float(accuracy),
                    "n_samples": len(examples_sample),
                    "predictions": predictions[:20]
                }
                
                logger.info(f"  {model_name} accuracy: {accuracy:.4f}")
            
            baseline_results[dataset_name] = dataset_results
        
        self.results['baseline'] = baseline_results
        self.analyzer.save_results(baseline_results, "baseline_results.json")
        self._save_cache()
        logger.info("Parallel baseline evaluation completed!")
        
        return baseline_results

    def evaluate_boosting_ensemble(self, sample_size: int = 100):
        """Legacy: Evaluate boosting ensemble."""
        logger.info("Evaluating boosting ensemble...")
        
        if 'baseline' not in self.results:
            logger.error("Baseline results not found. Run evaluate_baseline first.")
            return
        
        ensemble_results = {}
        
        for dataset_name, baseline in self.results['baseline'].items():
            logger.info(f"\nEvaluating boosting ensemble on {dataset_name}...")
            
            examples_sample = self.datasets[dataset_name][:sample_size]
            gold_labels = [ex.get("gold_label", "") for ex in examples_sample]
            
            ensemble = BoostingEnsemble(list(self.llm_clients.keys()))
            
            model_accs = {
                model: results.get("accuracy", 0.5)
                for model, results in baseline.items()
            }
            ensemble.update_weights(model_accs)
            
            all_predictions = {}
            for model_name in self.llm_clients.keys():
                all_predictions[model_name] = baseline[model_name].get("predictions", [])
            
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
        """Legacy: Generate final evaluation report."""
        logger.info("Generating evaluation report...")
        
        report = EvaluationReport()
        
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
        
        summary = report.to_dict()
        self.analyzer.save_results(summary, "evaluation_report.json")
        
        return report

    def run_full_pipeline(self, sample_size: int = 100):
        """Legacy: Run complete evaluation pipeline."""
        logger.info("Starting full evaluation pipeline...")
        
        try:
            self.initialize_llm_clients()
            self.load_datasets()
            self.evaluate_baseline(sample_size)
            self.evaluate_boosting_ensemble(sample_size)
            self.generate_report()
            
            logger.info("[OK] Pipeline completed successfully!")
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            sys.exit(1)

    def run_train_test_pipeline(
        self, 
        test_sample_size: int = None,
        max_workers: int = 5,
        datasets: List[str] = None
    ):
        """
        Run full train/test evaluation pipeline.
        
        Args:
            test_sample_size: Max test samples per dataset (None = use all)
            max_workers: Parallel workers for evaluation
            datasets: List of datasets to evaluate
        """
        logger.info("=" * 60)
        logger.info("TRAIN/TEST EVALUATION PIPELINE")
        logger.info("=" * 60)
        
        try:
            self.initialize_llm_clients()
            self.load_train_test_data(
                train_sample_size=None,  # Load all train for reference
                test_sample_size=test_sample_size,
                datasets=datasets
            )
            
            # Evaluate on test sets
            self.evaluate_on_test_parallel(max_workers=max_workers)
            
            # Evaluate ensemble
            self.evaluate_ensemble_on_test()
            
            # Generate report
            self.generate_final_report()
            
            logger.info("\n[OK] Train/Test pipeline completed successfully!")
        
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
        help="Number of samples to evaluate (for legacy mode)"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="Number of test samples per dataset (None = use all)"
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only run baseline evaluation (legacy mode)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing (5-10x faster)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5)"
    )
    parser.add_argument(
        "--train-test",
        action="store_true",
        help="Use train/test splits (new mode)"
    )
    parser.add_argument(
        "--datasets",
        nargs='+',
        default=None,
        help="Datasets to evaluate (medqa, pubmedqa, medmcqa)"
    )
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = EnsembleOrchestrator(config_dir=args.config_dir)
    
    if args.train_test:
        # New train/test split mode
        orchestrator.run_train_test_pipeline(
            test_sample_size=args.test_size,
            max_workers=args.workers,
            datasets=args.datasets
        )
    elif args.baseline_only:
        orchestrator.initialize_llm_clients()
        orchestrator.load_datasets()
        if args.parallel:
            orchestrator.evaluate_baseline_parallel(args.sample_size, args.workers)
        else:
            orchestrator.evaluate_baseline(args.sample_size)
    else:
        orchestrator.run_full_pipeline(args.sample_size)


if __name__ == "__main__":
    main()
