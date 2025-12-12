"""
Medical AI 4o client (OpenAI) for medical QA.
"""

import logging
from typing import Dict, Any, Optional, List
from .base_client import BaseLLMClient

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai")

logger = logging.getLogger(__name__)


class MedicalChatGPTClient(BaseLLMClient):
    """Client for OpenAI's Medical AI 4o model."""

    def __init__(self, api_key: str, config: Dict[str, Any]):
        """
        Initialize Medical AI 4o client.
        
        Args:
            api_key: OpenAI API key
            config: Configuration dictionary
        """
        super().__init__("medical-ai-4o", config)
        self.model = config.get("medical_model", "gpt-4o-medical")
        self.client = OpenAI(api_key=api_key)
        logger.info(f"Initialized Medical AI 4o client with model: {self.model}")

    def query(self, prompt: str, temperature: Optional[float] = None,
              max_tokens: Optional[int] = None, retries: int = 3) -> Dict[str, Any]:
        """
        Query Medical AI 4o with a prompt.
        
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
                system_prompt = """You are an expert medical AI assistant specialized in clinical medicine, diagnostics, and medical board exam questions. You excel at:
- Analyzing clinical presentations and patient histories
- Identifying key diagnostic clues and symptoms
- Applying medical knowledge to multiple choice questions
- Providing evidence-based reasoning

Analyze each question carefully, consider all options, and provide your answer. Always conclude with "Final Answer: X" where X is your chosen letter."""

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,  # Lower temperature for consistency
                    max_tokens=max_tok
                )

                text = response.choices[0].message.content.strip()
                
                return {
                    "text": text,
                    "confidence": 0.85,  # Medical models often have good calibration
                    "usage": {
                        "input_tokens": len(prompt.split()),
                        "output_tokens": len(text.split())
                    },
                    "model": "medical-chatgpt"
                }

            except Exception as e:
                if "rate" in str(e).lower() and attempt < retries - 1:
                    logger.warning(f"Rate limited, retrying... (attempt {attempt + 1}/{retries})")
                    import time
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Error querying Medical ChatGPT: {e}")
                    if attempt == retries - 1:
                        return {"text": "", "confidence": 0.0, "error": str(e)}
                    raise

        return {"text": "", "confidence": 0.0, "error": "Max retries exceeded"}

    def batch_query(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Query Medical ChatGPT with multiple prompts.
        
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
        Extract answer from Medical ChatGPT response (handles chain-of-thought format).
        
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
        
        # Priority 1: Look for "Final Answer: X" pattern
        m = re.search(rf"(?i)final\s*answer\s*[:\-]?\s*([A-{last_letter}])", text)
        if m:
            return m.group(1).upper()
        
        # Priority 2: Look for "The answer is X" or "Answer: X"
        m = re.search(rf"(?i)(?:the\s+)?answer\s+(?:is\s+)?[:\-]?\s*([A-{last_letter}])\b", text)
        if m:
            return m.group(1).upper()
        
        # Priority 3: Standalone letter at end
        m = re.search(rf"(?im)[:\-]?\s*\(?([A-{last_letter}])\)?\s*$", text)
        if m:
            return m.group(1).upper()
        
        # Priority 4: First letter in valid range
        m = re.search(rf"(?i)\b([A-{last_letter}])\b", text)
        if m:
            return m.group(1).upper()

        # Fallback: match option text
        for i, option in enumerate(options):
            if isinstance(option, str) and re.search(re.escape(option), text, re.I):
                return chr(65 + i)

        return ""
