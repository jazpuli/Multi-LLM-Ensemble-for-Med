"""
Create 80/20 train/test split from original datasets.
Formats all datasets consistently for the ensemble evaluation.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any

# Set seed for reproducibility
random.seed(42)

DATA_DIR = Path("data")


def load_medqa_usmle() -> List[Dict[str, Any]]:
    """Load MedQA-USMLE from original JSONL file."""
    filepath = DATA_DIR / "MedQA-USMLE.jsonl"
    examples = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            item = json.loads(line.strip())
            
            # Normalize options to list
            options_dict = item.get("options", {})
            options_list = [options_dict.get(k, "") for k in sorted(options_dict.keys())]
            
            examples.append({
                "id": f"medqa_{i}",
                "dataset": "medqa_usmle",
                "question": item.get("question", ""),
                "options": options_list,
                "options_dict": options_dict,
                "gold_label": item.get("answer", ""),
                "meta_info": item.get("meta_info", "")
            })
    
    print(f"Loaded {len(examples)} MedQA-USMLE examples")
    return examples


def load_pubmedqa() -> List[Dict[str, Any]]:
    """Load PubMedQA from original JSON file."""
    filepath = DATA_DIR / "PubMedQA.json"
    examples = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for pubmed_id, item in data.items():
        # Combine contexts into one string
        context = " ".join(item.get("CONTEXTS", []))
        question = item.get("QUESTION", "")
        
        # Full question with context
        full_question = f"Context: {context}\n\nQuestion: {question}"
        
        examples.append({
            "id": f"pubmedqa_{pubmed_id}",
            "dataset": "pubmedqa",
            "question": full_question,
            "question_only": question,
            "context": context,
            "options": ["yes", "no", "maybe"],
            "gold_label": item.get("final_decision", "").lower(),
            "long_answer": item.get("LONG_ANSWER", ""),
            "meshes": item.get("MESHES", []),
            "year": item.get("YEAR", "")
        })
    
    print(f"Loaded {len(examples)} PubMedQA examples")
    return examples


def load_medmcqa() -> List[Dict[str, Any]]:
    """Load MedMCQA from original JSON file."""
    filepath = DATA_DIR / "MedMCQA.json"
    examples = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        # Check if it's array or line-by-line
        content = f.read().strip()
        
    # Try parsing as array first, then as JSONL
    try:
        if content.startswith('['):
            data = json.loads(content)
        else:
            # JSONL format
            data = [json.loads(line) for line in content.split('\n') if line.strip()]
    except:
        # Line-by-line JSON objects
        data = []
        for line in content.split('\n'):
            if line.strip():
                try:
                    data.append(json.loads(line.strip()))
                except:
                    pass
    
    for i, item in enumerate(data):
        # Convert cop (1-4) to letter (A-D)
        cop = item.get("cop", 1)
        gold_label = chr(64 + cop) if isinstance(cop, int) and 1 <= cop <= 4 else "A"
        
        # Get options
        options = [
            item.get("opa", ""),
            item.get("opb", ""),
            item.get("opc", ""),
            item.get("opd", "")
        ]
        
        examples.append({
            "id": item.get("id", f"medmcqa_{i}"),
            "dataset": "medmcqa",
            "question": item.get("question", ""),
            "options": options,
            "gold_label": gold_label,
            "explanation": item.get("exp", ""),
            "subject": item.get("subject_name", ""),
            "topic": item.get("topic_name", "")
        })
    
    print(f"Loaded {len(examples)} MedMCQA examples")
    return examples


def split_dataset(examples: List[Dict], train_ratio: float = 0.8) -> tuple:
    """Split dataset into train and test sets."""
    random.shuffle(examples)
    split_idx = int(len(examples) * train_ratio)
    return examples[:split_idx], examples[split_idx:]


def save_jsonl(examples: List[Dict], filepath: Path):
    """Save examples as JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    print(f"Saved {len(examples)} examples to {filepath}")


def save_json(examples: List[Dict], filepath: Path):
    """Save examples as JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(examples)} examples to {filepath}")


def main():
    print("=" * 60)
    print("Creating 80/20 Train/Test Split from Original Datasets")
    print("=" * 60)
    
    # Load all datasets
    print("\n[Loading datasets...]")
    medqa = load_medqa_usmle()
    pubmedqa = load_pubmedqa()
    medmcqa = load_medmcqa()
    
    # Create splits
    print("\n[Creating 80/20 splits...]")
    medqa_train, medqa_test = split_dataset(medqa)
    pubmedqa_train, pubmedqa_test = split_dataset(pubmedqa)
    medmcqa_train, medmcqa_test = split_dataset(medmcqa)
    
    # Print statistics
    print("\n[Split Statistics]")
    print(f"  MedQA-USMLE: {len(medqa_train)} train, {len(medqa_test)} test")
    print(f"  PubMedQA:    {len(pubmedqa_train)} train, {len(pubmedqa_test)} test")
    print(f"  MedMCQA:     {len(medmcqa_train)} train, {len(medmcqa_test)} test")
    
    total_train = len(medqa_train) + len(pubmedqa_train) + len(medmcqa_train)
    total_test = len(medqa_test) + len(pubmedqa_test) + len(medmcqa_test)
    print(f"\n  TOTAL: {total_train} train, {total_test} test")
    
    # Create output directory
    output_dir = DATA_DIR / "splits"
    output_dir.mkdir(exist_ok=True)
    
    # Save individual dataset splits
    print("\n[Saving individual dataset splits...]")
    save_jsonl(medqa_train, output_dir / "medqa_train.jsonl")
    save_jsonl(medqa_test, output_dir / "medqa_test.jsonl")
    save_jsonl(pubmedqa_train, output_dir / "pubmedqa_train.jsonl")
    save_jsonl(pubmedqa_test, output_dir / "pubmedqa_test.jsonl")
    save_jsonl(medmcqa_train, output_dir / "medmcqa_train.jsonl")
    save_jsonl(medmcqa_test, output_dir / "medmcqa_test.jsonl")
    
    # Create combined datasets
    print("\n[Creating combined datasets...]")
    all_train = medqa_train + pubmedqa_train + medmcqa_train
    all_test = medqa_test + pubmedqa_test + medmcqa_test
    
    # Shuffle combined datasets
    random.shuffle(all_train)
    random.shuffle(all_test)
    
    save_jsonl(all_train, output_dir / "combined_train.jsonl")
    save_jsonl(all_test, output_dir / "combined_test.jsonl")
    
    # Save summary
    summary = {
        "datasets": {
            "medqa_usmle": {"train": len(medqa_train), "test": len(medqa_test)},
            "pubmedqa": {"train": len(pubmedqa_train), "test": len(pubmedqa_test)},
            "medmcqa": {"train": len(medmcqa_train), "test": len(medmcqa_test)}
        },
        "total": {"train": total_train, "test": total_test},
        "split_ratio": "80/20",
        "random_seed": 42
    }
    save_json(summary, output_dir / "split_summary.json")
    
    print("\n" + "=" * 60)
    print("Done! Files saved to data/splits/")
    print("=" * 60)
    
    # Print file listing
    print("\n[Created files]")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
