"""
Dataset utilities for loading medical QA datasets.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import os

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load and manage medical QA datasets."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or "./data"
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_pubmedqa(self, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load PubMedQA dataset from formatted JSON file.
        
        Args:
            sample_size: Maximum number of samples to load
            
        Returns:
            List of question examples
        """
        logger.info("Loading PubMedQA from formatted JSON...")
        
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
            
            logger.info(f"Loaded {len(examples)} PubMedQA examples")
            return examples
        
        except Exception as e:
            logger.error(f"Error loading PubMedQA: {e}")
            return []

    def load_medqa_usmle(self, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load MedQA-USMLE dataset from formatted JSONL file.
        
        Args:
            sample_size: Maximum number of samples to load
            
        Returns:
            List of question examples
        """
        logger.info("Loading MedQA-USMLE from formatted JSONL...")
        
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
                    
                    # Normalize options: some files store options as dicts ("A": text)
                    raw_options = item.get("options", {})
                    if isinstance(raw_options, dict):
                        # Sort by key to preserve A, B, C... ordering
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
            
            logger.info(f"Loaded {len(examples)} MedQA-USMLE examples")
            return examples
        
        except Exception as e:
            logger.error(f"Error loading MedQA-USMLE: {e}")
            return []

    def load_medmcqa(self, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load MedMCQA dataset from formatted JSON file.
        
        Args:
            sample_size: Maximum number of samples to load
            
        Returns:
            List of question examples
        """
        logger.info("Loading MedMCQA from formatted JSON...")
        
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
                
                # Convert options dict to list if needed
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
            
            logger.info(f"Loaded {len(examples)} MedMCQA examples")
            return examples
        
        except Exception as e:
            logger.error(f"Error loading MedMCQA: {e}")
            return []

    def load_all_datasets(self, config: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load all configured datasets from formatted files.
        
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

