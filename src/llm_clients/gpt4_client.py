"""
OpenAI GPT-4 client for medical QA.
"""

import logging
from typing import Dict, Any, Optional, List
from .base_client import BaseLLMClient

try:
    from openai import OpenAI, RateLimitError, APIError
except ImportError:
    raise ImportError("Please install openai: pip install openai")

logger = logging.getLogger(__name__)


class GPT4Client(BaseLLMClient):
    """Client for OpenAI's GPT-4 API."""

    def __init__(self, api_key: str, config: Dict[str, Any]):
        """
        Initialize GPT-4 client.
        
        Args:
            api_key: OpenAI API key
            config: Configuration dictionary
        """
        super().__init__("gpt-4", config)
        self.client = OpenAI(api_key=api_key)
        logger.info("Initialized GPT-4 client")

    def query(self, prompt: str, temperature: Optional[float] = None,
              max_tokens: Optional[int] = None, retries: int = 3) -> Dict[str, Any]:
        """
        Query GPT-4o with a prompt.
        
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

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a medical expert assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temp,
                    max_tokens=max_tok
                )

                text = response.choices[0].message.content
                
                return {
                    "text": text,
                    "confidence": 0.9,  # GPT-4 doesn't provide explicit confidence
                    "usage": {
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens
                    },
                    "model": "gpt-4"
                }

            except RateLimitError:
                if attempt < retries - 1:
                    logger.warning(f"Rate limited, retrying... (attempt {attempt + 1}/{retries})")
                    import time
                    time.sleep(2 ** attempt)
                else:
                    logger.error("Max retries exceeded for rate limit")
                    raise

            except APIError as e:
                logger.error(f"API error: {e}")
                raise

        return {"text": "", "confidence": 0.0, "error": "Max retries exceeded"}

    def batch_query(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Query GPT-4 with multiple prompts.
        
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
        Extract answer from GPT-4 response.
        
        Args:
            response_text: Raw response text
            options: Multiple choice options
            
        Returns:
            Extracted answer
        """
        import re

        text = response_text.strip()

        # If no options provided, return first line
        if not options:
            return text.split('\n')[0].strip()[:200]

        # Normalize: try to find a letter A..(last) in the response and return it uppercase
        last_letter = chr(65 + max(0, len(options) - 1))
        letter_pattern = rf"(?i)\b([A-{last_letter}])\b"

        # Common patterns: 'Answer: D', 'D.', 'D)'
        m = re.search(rf"(?im)answer\s*[:\-]?\s*([A-{last_letter}])", text)
        if not m:
            m = re.search(rf"(?im)^\s*([A-{last_letter}])(?:[\.\)])", text)
        if not m:
            m = re.search(letter_pattern, text)

        if m:
            return m.group(1).upper()

        # If model repeated the option text, match option text to its index and return letter
        for i, option in enumerate(options):
            if isinstance(option, str) and re.search(re.escape(option), text, re.I):
                return chr(65 + i)

        # Fallback: empty string (no valid A.. letter found)
        return ""
