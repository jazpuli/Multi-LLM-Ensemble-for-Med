"""
Base class for LLM clients.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Base class for LLM API clients."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initialize LLM client.
        
        Args:
            model_name: Name of the model
            config: Configuration dictionary
        """
        self.model_name = model_name
        self.config = config
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 500)

    @abstractmethod
    def query(self, prompt: str, temperature: Optional[float] = None, 
              max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Query the LLM with a prompt.
        
        Args:
            prompt: Input prompt
            temperature: Override temperature
            max_tokens: Override max tokens
            
        Returns:
            Response dictionary with 'text', 'confidence', 'usage'
        """
        pass

    @abstractmethod
    def batch_query(self, prompts: list, **kwargs) -> list:
        """
        Query the LLM with multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional arguments
            
        Returns:
            List of response dictionaries
        """
        pass

    def format_question(self, question: str, options: Optional[list] = None) -> str:
        """
        Format a medical question for the LLM.
        
        Args:
            question: Question text
            options: Multiple choice options
            
        Returns:
            Formatted prompt
        """
        if options:
            options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            last_letter = chr(65 + max(0, len(options) - 1))
            # Strict instruction to force single-letter output
            instruction = f"Answer with exactly one letter (A–{last_letter}) and nothing else."
            return f"{question}\n\n{options_str}\n\n{instruction}\n\nAnswer:"
        return f"{question}\n\nAnswer:"
