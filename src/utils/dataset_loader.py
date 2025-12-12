"""
Dataset utilities for loading medical QA datasets.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import os

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load and manage medical QA datasets."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or "./data"
        self.splits_dir = Path(self.cache_dir) / "splits"
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_split_file(self, filename: str, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load dataset from a split JSONL file.
        
        Args:
            filename: Name of the file in splits directory
            sample_size: Maximum number of samples to load
            
        Returns:
            List of question examples
        """
        filepath = self.splits_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Split file not found: {filepath}")
            return []
        
        examples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if sample_size and i >= sample_size:
                    break
                examples.append(json.loads(line.strip()))
        
        logger.info(f"Loaded {len(examples)} examples from {filename}")
        return examples

    def load_train_test_split(
        self, 
        dataset_name: str, 
        train_sample_size: Optional[int] = None,
        test_sample_size: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Load train and test split for a specific dataset.
        
        Args:
            dataset_name: One of 'medqa', 'pubmedqa', 'medmcqa', 'combined'
            train_sample_size: Maximum training samples
            test_sample_size: Maximum test samples
            
        Returns:
            Tuple of (train_examples, test_examples)
        """
        train_file = f"{dataset_name}_train.jsonl"
        test_file = f"{dataset_name}_test.jsonl"
        
        train_examples = self.load_split_file(train_file, train_sample_size)
        test_examples = self.load_split_file(test_file, test_sample_size)
        
        return train_examples, test_examples

    def load_all_splits(
        self,
        train_sample_size: Optional[int] = None,
        test_sample_size: Optional[int] = None
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Load train/test splits for all datasets.
        
        Returns:
            Dict mapping dataset_name -> {"train": [...], "test": [...]}
        """
        datasets = {}
        
        for name in ['medqa', 'pubmedqa', 'medmcqa']:
            train, test = self.load_train_test_split(
                name, 
                train_sample_size, 
                test_sample_size
            )
            if train or test:
                datasets[name] = {"train": train, "test": test}
        
        return datasets

    def load_combined_split(
        self,
        train_sample_size: Optional[int] = None,
        test_sample_size: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load combined train/test datasets."""
        return self.load_train_test_split(
            'combined',
            train_sample_size,
            test_sample_size
        )

    # --- Legacy methods for backward compatibility ---
    
    def load_pubmedqa(self, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load PubMedQA dataset (legacy: loads test split)."""
        logger.info("Loading PubMedQA from split file...")
        examples = self.load_split_file("pubmedqa_test.jsonl", sample_size)
        
        # If splits don't exist, fallback to formatted file
        if not examples:
            return self._load_pubmedqa_formatted(sample_size)
        return examples

    def load_medqa_usmle(self, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load MedQA-USMLE dataset (legacy: loads test split)."""
        logger.info("Loading MedQA-USMLE from split file...")
        examples = self.load_split_file("medqa_test.jsonl", sample_size)
        
        if not examples:
            return self._load_medqa_formatted(sample_size)
        return examples

    def load_medmcqa(self, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load MedMCQA dataset (legacy: loads test split)."""
        logger.info("Loading MedMCQA from split file...")
        examples = self.load_split_file("medmcqa_test.jsonl", sample_size)
        
        if not examples:
            return self._load_medmcqa_formatted(sample_size)
        return examples

    def _load_pubmedqa_formatted(self, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fallback: Load PubMedQA from formatted JSON file."""
        try:
            filepath = Path(self.cache_dir) / "PubMedQA_formatted.json"
            
            if not filepath.exists():
                logger.warning(f"PubMedQA formatted file not found at {filepath}")
                return []
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            examples = []
            for i, item in enumerate(data):
                if sample_size and i >= sample_size:
                    break
                
                examples.append({
                    "dataset": "pubmedqa",
                    "id": item.get("id", f"pubmedqa_{i}"),
                    "question": item.get("question", ""),
                    "long_answer": item.get("long_answer", ""),
                    "answer_type": item.get("answer_type", ""),
                    "gold_label": item.get("gold_label", item.get("answer_type", "")),
                    "options": ["yes", "no", "maybe"],
                    "pubmed_id": item.get("pubmed_id"),
                    "raw_data": item
                })
            
            logger.info(f"Loaded {len(examples)} PubMedQA examples from formatted file")
            return examples
        
        except Exception as e:
            logger.error(f"Error loading PubMedQA: {e}")
            return []

    def _load_medqa_formatted(self, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fallback: Load MedQA-USMLE from formatted JSONL file."""
        try:
            filepath = Path(self.cache_dir) / "MedQA-USMLE_formatted.jsonl"
            
            if not filepath.exists():
                logger.warning(f"MedQA-USMLE formatted file not found at {filepath}")
                return []
            
            examples = []
            with open(filepath, 'r') as f:
                for i, line in enumerate(f):
                    if sample_size and i >= sample_size:
                        break
                    
                    item = json.loads(line.strip())
                    
                    raw_options = item.get("options", {})
                    if isinstance(raw_options, dict):
                        options_list = [raw_options.get(k, "") for k in sorted(raw_options.keys())]
                    else:
                        options_list = raw_options

                    examples.append({
                        "dataset": "medqa_usmle",
                        "id": f"medqa_{i}",
                        "question": item.get("question", ""),
                        "options": options_list,
                        "options_raw": raw_options,
                        "gold_label": item.get("gold_label", item.get("answer", "")),
                        "answer": item.get("answer", ""),
                        "meta_info": item.get("meta_info", ""),
                        "raw_data": item
                    })
            
            logger.info(f"Loaded {len(examples)} MedQA-USMLE examples from formatted file")
            return examples
        
        except Exception as e:
            logger.error(f"Error loading MedQA-USMLE: {e}")
            return []

    def _load_medmcqa_formatted(self, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fallback: Load MedMCQA from formatted JSON file."""
        try:
            filepath = Path(self.cache_dir) / "MedMCQA_formatted.json"
            
            if not filepath.exists():
                logger.warning(f"MedMCQA formatted file not found at {filepath}")
                return []
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            examples = []
            for i, item in enumerate(data):
                if sample_size and i >= sample_size:
                    break
                
                options = item.get("options", {})
                correct_answer = item.get("correct_answer", "")
                
                if isinstance(options, dict):
                    options_list = [options.get(k, "") for k in sorted(options.keys())]
                else:
                    options_list = options
                
                examples.append({
                    "dataset": "medmcqa",
                    "id": item.get("id", f"medmcqa_{i}"),
                    "question": item.get("question", ""),
                    "options": options,
                    "options_list": options_list,
                    "gold_label": correct_answer,
                    "correct_answer": correct_answer,
                    "subject": item.get("subject", "General"),
                    "topic": item.get("topic"),
                    "choice_type": item.get("choice_type", "single"),
                    "explanation": item.get("explanation"),
                    "raw_data": item
                })
            
            logger.info(f"Loaded {len(examples)} MedMCQA examples from formatted file")
            return examples
        
        except Exception as e:
            logger.error(f"Error loading MedMCQA: {e}")
            return []

    def load_all_datasets(self, config: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load all configured datasets (legacy method - uses test splits).
        
        Args:
            config: Experiment configuration
            
        Returns:
            Dictionary mapping dataset names to examples
        """
        datasets = {}
        dataset_config = config.get("datasets", {})
        
        if dataset_config.get("pubmedqa", {}).get("enabled"):
            datasets["pubmedqa"] = self.load_pubmedqa(
                sample_size=dataset_config["pubmedqa"].get("sample_size")
            )
        
        if dataset_config.get("medqa_usmle", {}).get("enabled"):
            datasets["medqa_usmle"] = self.load_medqa_usmle(
                sample_size=dataset_config["medqa_usmle"].get("sample_size")
            )
        
        if dataset_config.get("medmcqa", {}).get("enabled"):
            datasets["medmcqa"] = self.load_medmcqa(
                sample_size=dataset_config["medmcqa"].get("sample_size")
            )
        
        return datasets

    def save_examples(self, examples: List[Dict], filename: str):
        """Save examples to JSON file."""
        filepath = Path(self.cache_dir) / filename
        with open(filepath, 'w') as f:
            json.dump(examples, f, indent=2)
        logger.info(f"Saved {len(examples)} examples to {filepath}")

    def load_examples(self, filename: str) -> List[Dict]:
        """Load examples from JSON file."""
        filepath = Path(self.cache_dir) / filename
        with open(filepath, 'r') as f:
            examples = json.load(f)
        logger.info(f"Loaded {len(examples)} examples from {filepath}")
        return examples
