"""
LLaMA-2 client supporting multiple backends (Ollama, Together AI, Purdue HPC).
"""

import logging
import time
from typing import Dict, Any, Optional, List
from .base_client import BaseLLMClient
import requests
import json

logger = logging.getLogger(__name__)


class LLaMA2Client(BaseLLMClient):
    """Client for LLaMA-2 supporting multiple backends."""

    def __init__(self, api_key: str, config: Dict[str, Any]):
        """
        Initialize LLaMA-2 client.
        
        Args:
            api_key: API key or credentials
            config: Configuration dictionary with backend info
        """
        super().__init__("llama-2", config)
        self.api_key = api_key
        self.endpoint = config.get("endpoint", "http://localhost:11434")
        self.model = config.get("model", "llama2")
        self.use_local = config.get("use_local", False)
        self.use_gpt4_fallback = config.get("use_gpt4_fallback", False)
        self.backend = self._detect_backend()
        
        logger.info(f"Initialized LLaMA-2 client with backend: {self.backend}")
        logger.info(f"Endpoint: {self.endpoint}, Model: {self.model}")

    def _detect_backend(self) -> str:
        """Detect which backend is being used."""
        if self.use_local or self.endpoint == "local":
            return "local_hpc"
        elif "localhost" in self.endpoint or "127.0.0.1" in self.endpoint:
            return "ollama"
        else:
            return "together_ai"

    def query(self, prompt: str, temperature: Optional[float] = None,
              max_tokens: Optional[int] = None, retries: int = 3) -> Dict[str, Any]:
        """
        Query LLaMA-2 with a prompt.
        
        Args:
            prompt: Input prompt
            temperature: Override temperature
            max_tokens: Override max tokens
            retries: Number of retries on failure
            
        Returns:
            Response dictionary
        """
        temp = temperature or self.temperature
        max_tok = max_tokens or self.max_tokens

        if self.backend == "ollama":
            return self._query_ollama(prompt, temp, max_tok, retries)
        elif self.backend == "together_ai":
            return self._query_together(prompt, temp, max_tok, retries)
        elif self.backend == "local_hpc":
            return self._query_local_hpc(prompt, temp, max_tok, retries)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _query_ollama(self, prompt: str, temperature: float, max_tokens: int, retries: int) -> Dict[str, Any]:
        """Query Ollama local instance."""
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.endpoint}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": temperature
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    text = data.get("response", "").strip()
                    
                    return {
                        "text": text,
                        "finish_reason": "stop",
                        "tokens_used": len(text.split())
                    }
                else:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                logger.error(f"Failed to connect to Ollama at {self.endpoint}")
                if self.use_gpt4_fallback:
                    logger.info("Falling back to GPT-4o")
                    return self._query_gpt4_fallback(prompt, temperature, max_tokens)
                if attempt < retries - 1:
                    logger.info(f"Retrying... ({attempt + 1}/{retries})")
            except Exception as e:
                logger.error(f"Error querying Ollama: {e}")
                if attempt < retries - 1:
                    logger.info(f"Retrying... ({attempt + 1}/{retries})")
        
        if self.use_gpt4_fallback:
            logger.info("Falling back to GPT-4o after Ollama timeout")
            return self._query_gpt4_fallback(prompt, temperature, max_tokens)
        
        raise RuntimeError(f"Failed to query Ollama after {retries} attempts")

    def _query_together(self, prompt: str, temperature: float, max_tokens: int, retries: int) -> Dict[str, Any]:
        """Query Together AI."""
        try:
            import together
        except ImportError:
            raise ImportError("Please install together: pip install together")
        
        together.api_key = self.api_key
        
        for attempt in range(retries):
            try:
                response = together.Complete.create(
                    prompt=prompt,
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["\n\n"]
                )
                
                text = response["output"]["choices"][0]["text"].strip()
                
                return {
                    "text": text,
                    "confidence": 0.8,  # LLaMA-2 doesn't provide explicit confidence
                    "usage": {
                        "input_tokens": len(prompt.split()),
                        "output_tokens": len(text.split())
                    },
                    "model": "llama-2-70b"
                }

            except Exception as e:
                if "rate" in str(e).lower() and attempt < retries - 1:
                    logger.warning(f"Rate limited, retrying... (attempt {attempt + 1}/{retries})")
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Error querying LLaMA-2: {e}")
                    if attempt == retries - 1:
                        return {"text": "", "confidence": 0.0, "error": str(e)}
                    raise

        return {"text": "", "confidence": 0.0, "error": "Max retries exceeded"}

    def _query_local_hpc(self, prompt: str, temperature: float, max_tokens: int, retries: int) -> Dict[str, Any]:
        """Query LLaMA-2 on Purdue HPC (SSH or API endpoint)."""
        logger.warning("Local HPC query not yet fully implemented. Please use Ollama or Together AI.")
        
        # Stub for future implementation - would connect to Purdue HPC
        # This could use SSH or HTTP if HPC provides an API endpoint
        return {
            "text": "Local HPC backend not yet configured",
            "confidence": 0.0,
            "error": "HPC backend not implemented"
        }

    def _query_gpt4_fallback(self, prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Fallback to GPT-4o when Ollama is not available."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        # Get API key from environment or config
        api_key = self.api_key if self.api_key else None
        if not api_key:
            # Try to get from parent config
            api_key = self.config.get("api_key") if self.config else None
        
        if not api_key:
            raise ValueError("OpenAI API key not found for GPT-4o fallback")
        
        client = OpenAI(api_key=api_key)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            text = response.choices[0].message.content.strip()
            
            return {
                "text": text,
                "finish_reason": response.choices[0].finish_reason,
                "tokens_used": response.usage.total_tokens,
                "model": "gpt-4o-fallback"
            }
        except Exception as e:
            logger.error(f"Error with GPT-4o fallback: {e}")
            raise

    def batch_query(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Query LLaMA-2 with multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional arguments
            
        Returns:
            List of response dictionaries
        """
        responses = []
        for i, prompt in enumerate(prompts):
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(prompts)} prompts")
            response = self.query(prompt, **kwargs)
            responses.append(response)
        
        return responses

    def extract_answer(self, response_text: str, options: Optional[List[str]] = None) -> str:
        """
        Extract answer from LLaMA-2 response.
        
        Args:
            response_text: Raw response text
            options: Multiple choice options
            
        Returns:
            Extracted answer
        """
        import re

        text = response_text.strip()

        if not options:
            return text.split('\n')[0].strip()[:200]

        last_letter = chr(65 + max(0, len(options) - 1))
        letter_pattern = rf"(?i)\b([A-{last_letter}])\b"

        m = re.search(rf"(?im)answer\s*[:\-]?\s*([A-{last_letter}])", text)
        if not m:
            m = re.search(rf"(?im)^\s*([A-{last_letter}])(?:[\.\)])", text)
        if not m:
            m = re.search(letter_pattern, text)

        if m:
            return m.group(1).upper()

        # Match option text -> return corresponding letter
        for i, option in enumerate(options):
            if isinstance(option, str) and re.search(re.escape(option), text, re.I):
                return chr(65 + i)

        return ""
