"""
Generate final report from cached predictions.
Uses complete data from MedQA and PubMedQA.
"""

import json
from pathlib import Path
from collections import Counter
import numpy as np

# Paths
CACHE_FILE = Path("results/cache/prediction_cache.json")
RESULTS_DIR = Path("results")
SPLITS_DIR = Path("data/splits")

def load_cache():
    """Load cached predictions."""
    with open(CACHE_FILE, 'r') as f:
        return json.load(f)

def load_test_data(dataset_name):
    """Load test data for a dataset."""
    filepath = SPLITS_DIR / f"{dataset_name}_test.jsonl"
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples

def get_cache_key(model, dataset, question):
    """Generate cache key matching the main.py format."""
    import hashlib
    key = f"{model}:{dataset}:{question[:100]}"
    return hashlib.md5(key.encode()).hexdigest()

def calculate_accuracy(predictions, gold_labels):
    """Calculate accuracy."""
    correct = sum(1 for p, g in zip(predictions, gold_labels) if p and g and p.upper() == g.upper())
    return correct / len(gold_labels) if gold_labels else 0

def weighted_vote(model_preds, weights):
    """Perform weighted majority voting."""
    vote_weights = {}
    for model, pred in model_preds.items():
        if pred:
            pred_upper = pred.upper() if pred else ""
            if pred_upper:
                vote_weights[pred_upper] = vote_weights.get(pred_upper, 0) + weights.get(model, 1.0)
    
    if not vote_weights:
        return "", 0.0
    
    winner = max(vote_weights, key=vote_weights.get)
    total_weight = sum(vote_weights.values())
    confidence = vote_weights[winner] / total_weight if total_weight > 0 else 0
    return winner, confidence

def analyze_dataset(cache, test_data, dataset_name, models):
    """Analyze a single dataset."""
    results = {}
    
    # Map dataset names
    dataset_key_map = {
        'medqa': 'medqa',
        'pubmedqa': 'pubmedqa', 
        'medmcqa': 'medmcqa'
    }
    dataset_key = dataset_key_map.get(dataset_name, dataset_name)
    
    for model in models:
        predictions = []
        for ex in test_data:
            question = ex.get("question", "")
            cache_key = get_cache_key(model, dataset_key, question)
            pred = cache.get(cache_key, "")
            predictions.append(pred)
        
        gold_labels = [ex.get("gold_label", "") for ex in test_data]
        
        # Count how many predictions we have
        valid_preds = sum(1 for p in predictions if p)
        
        if valid_preds > 0:
            accuracy = calculate_accuracy(predictions, gold_labels)
            results[model] = {
                "accuracy": accuracy,
                "correct": int(accuracy * len(gold_labels)),
                "total": len(gold_labels),
                "coverage": valid_preds / len(gold_labels)
            }
        else:
            results[model] = None
    
    return results

def compute_ensemble(cache, test_data, dataset_name, models, model_accuracies):
    """Compute ensemble predictions."""
    # Calculate weights based on accuracy
    total_acc = sum(model_accuracies.values())
    weights = {m: acc / total_acc for m, acc in model_accuracies.items()}
    
    dataset_key_map = {
        'medqa': 'medqa',
        'pubmedqa': 'pubmedqa',
        'medmcqa': 'medmcqa'
    }
    dataset_key = dataset_key_map.get(dataset_name, dataset_name)
    
    ensemble_preds = []
    confidences = []
    disagreements = []
    
    for ex in test_data:
        question = ex.get("question", "")
        
        model_preds = {}
        for model in models:
            cache_key = get_cache_key(model, dataset_key, question)
            model_preds[model] = cache.get(cache_key, "")
        
        pred, conf = weighted_vote(model_preds, weights)
        ensemble_preds.append(pred)
        confidences.append(conf)
        
        # Check disagreement
        unique_preds = set(p.upper() for p in model_preds.values() if p)
        disagreements.append(1 if len(unique_preds) > 1 else 0)
    
    gold_labels = [ex.get("gold_label", "") for ex in test_data]
    ensemble_accuracy = calculate_accuracy(ensemble_preds, gold_labels)
    
    return {
        "accuracy": ensemble_accuracy,
        "correct": int(ensemble_accuracy * len(gold_labels)),
        "total": len(gold_labels),
        "avg_confidence": float(np.mean(confidences)),
        "disagreement_rate": float(np.mean(disagreements)),
        "weights": weights
    }

def main():
    print("=" * 60)
    print("GENERATING FINAL REPORT")
    print("=" * 60)
    
    # Load cache
    print("\nLoading cached predictions...")
    cache = load_cache()
    print(f"  Loaded {len(cache)} cached predictions")
    
    models = ['gpt4', 'llama2', 'medical_ai_4o']
    datasets = ['medqa', 'pubmedqa']  # Only complete datasets
    
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n[{dataset_name.upper()}]")
        
        # Load test data
        test_data = load_test_data(dataset_name)
        print(f"  Test samples: {len(test_data)}")
        
        # Analyze individual models
        model_results = analyze_dataset(cache, test_data, dataset_name, models)
        
        # Print individual results
        model_accuracies = {}
        for model, res in model_results.items():
            if res:
                print(f"  {model}: {res['accuracy']:.4f} ({res['correct']}/{res['total']})")
                model_accuracies[model] = res['accuracy']
            else:
                print(f"  {model}: No data")
        
        # Compute ensemble
        if len(model_accuracies) >= 2:
            ensemble_results = compute_ensemble(cache, test_data, dataset_name, models, model_accuracies)
            print(f"  ENSEMBLE: {ensemble_results['accuracy']:.4f} ({ensemble_results['correct']}/{ensemble_results['total']})")
            print(f"    Disagreement rate: {ensemble_results['disagreement_rate']:.4f}")
            
            # Find best individual
            best_model = max(model_accuracies, key=model_accuracies.get)
            improvement = ensemble_results['accuracy'] - model_accuracies[best_model]
            print(f"    Best individual ({best_model}): {model_accuracies[best_model]:.4f}")
            print(f"    Ensemble improvement: {improvement:+.4f}")
            
            all_results[dataset_name] = {
                "individual": model_results,
                "ensemble": ensemble_results,
                "best_individual": {"model": best_model, "accuracy": model_accuracies[best_model]},
                "improvement": improvement
            }
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    print("\n[Individual Model Performance]")
    print("-" * 50)
    print(f"{'Model':<20} {'MedQA':<15} {'PubMedQA':<15} {'Average':<10}")
    print("-" * 50)
    
    for model in models:
        medqa_acc = all_results.get('medqa', {}).get('individual', {}).get(model, {})
        pubmedqa_acc = all_results.get('pubmedqa', {}).get('individual', {}).get(model, {})
        
        medqa_val = medqa_acc.get('accuracy', 0) if medqa_acc else 0
        pubmedqa_val = pubmedqa_acc.get('accuracy', 0) if pubmedqa_acc else 0
        avg = (medqa_val + pubmedqa_val) / 2 if medqa_acc and pubmedqa_acc else 0
        
        print(f"{model:<20} {medqa_val:.4f}         {pubmedqa_val:.4f}         {avg:.4f}")
    
    print("\n[Ensemble Performance]")
    print("-" * 50)
    
    total_ensemble_correct = 0
    total_samples = 0
    total_best_correct = 0
    
    for dataset_name in datasets:
        res = all_results.get(dataset_name, {})
        ens = res.get('ensemble', {})
        best = res.get('best_individual', {})
        imp = res.get('improvement', 0)
        
        if ens:
            print(f"{dataset_name.upper()}:")
            print(f"  Ensemble Accuracy: {ens['accuracy']:.4f}")
            print(f"  Best Individual ({best['model']}): {best['accuracy']:.4f}")
            print(f"  Improvement: {imp:+.4f}")
            print(f"  Disagreement Rate: {ens['disagreement_rate']:.4f}")
            print()
            
            total_ensemble_correct += ens['correct']
            total_samples += ens['total']
            total_best_correct += int(best['accuracy'] * ens['total'])
    
    # Overall
    overall_ensemble_acc = total_ensemble_correct / total_samples if total_samples > 0 else 0
    overall_best_acc = total_best_correct / total_samples if total_samples > 0 else 0
    
    print("-" * 50)
    print(f"OVERALL ({total_samples} questions):")
    print(f"  Ensemble Accuracy: {overall_ensemble_acc:.4f}")
    print(f"  Avg Best Individual: {overall_best_acc:.4f}")
    print(f"  Overall Improvement: {overall_ensemble_acc - overall_best_acc:+.4f}")
    
    # Save results
    report = {
        "datasets_evaluated": datasets,
        "total_questions": total_samples,
        "models": models,
        "results_by_dataset": all_results,
        "overall": {
            "ensemble_accuracy": overall_ensemble_acc,
            "avg_best_individual": overall_best_acc,
            "improvement": overall_ensemble_acc - overall_best_acc
        }
    }
    
    report_path = RESULTS_DIR / "final_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")
    
    print("\n" + "=" * 60)
    print("REPORT COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()

