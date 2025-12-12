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
                system_prompt = """You are a highly skilled medical expert with extensive knowledge of clinical medicine, pathophysiology, pharmacology, and medical diagnostics. You have years of experience in medical practice and board exam preparation.

When answering medical questions:
1. Carefully analyze all clinical findings and patient information
2. Consider differential diagnoses and rule out alternatives
3. Apply evidence-based medical reasoning
4. Provide your final answer clearly

Always end your response with "Final Answer: X" where X is the letter of your chosen option."""
                
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,  # Lower temperature for more consistent answers
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

    async def query_async(self, prompt: str, temperature: Optional[float] = None,
                          max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Async version of query for parallel processing."""
        import asyncio
        from openai import AsyncOpenAI
        
        temp = temperature or 0.3
        max_tok = max_tokens or self.max_tokens
        
        async_client = AsyncOpenAI(api_key=self.client.api_key)
        
        system_prompt = """You are a highly skilled medical expert. Analyze the question and provide your answer. Always end with "Final Answer: X" where X is your chosen letter."""
        
        try:
            response = await async_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=max_tok
            )
            
            return {
                "text": response.choices[0].message.content,
                "confidence": 0.9,
                "usage": {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens
                }
            }
        except Exception as e:
            logger.error(f"Async query error: {e}")
            return {"text": "", "confidence": 0.0, "error": str(e)}

    def extract_answer(self, response_text: str, options: Optional[List[str]] = None) -> str:
        """
        Extract answer from GPT-4 response (handles chain-of-thought format).
        
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

        last_letter = chr(65 + max(0, len(options) - 1))
        
        # Priority 1: Look for "Final Answer: X" pattern (chain-of-thought)
        m = re.search(rf"(?i)final\s*answer\s*[:\-]?\s*([A-{last_letter}])", text)
        if m:
            return m.group(1).upper()
        
        # Priority 2: Look for "The answer is X" or "Answer: X"
        m = re.search(rf"(?i)(?:the\s+)?answer\s+(?:is\s+)?[:\-]?\s*([A-{last_letter}])\b", text)
        if m:
            return m.group(1).upper()
        
        # Priority 3: Look for standalone letter at end of response
        m = re.search(rf"(?im)[:\-]?\s*\(?([A-{last_letter}])\)?\s*$", text)
        if m:
            return m.group(1).upper()
        
        # Priority 4: Look for "Option X" or "X." at start of line
        m = re.search(rf"(?im)(?:option\s+)?([A-{last_letter}])(?:[\.\)]|\s+is)", text)
        if m:
            return m.group(1).upper()
        
        # Priority 5: First standalone letter in valid range
        m = re.search(rf"(?i)\b([A-{last_letter}])\b", text)
        if m:
            return m.group(1).upper()

        # Fallback: match option text to its index
        for i, option in enumerate(options):
            if isinstance(option, str) and re.search(re.escape(option), text, re.I):
                return chr(65 + i)

        return ""
