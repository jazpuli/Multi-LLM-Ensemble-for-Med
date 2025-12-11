"""
Embedder utilities for question clustering.
"""

import logging
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)


class QuestionEmbedder:
    """Embed medical questions using pre-trained models."""

    def __init__(self, model_name: str = "distiluse-base-multilingual-cased-v2"):
        """
        Initialize embedder with specified model.
        
        Args:
            model_name: Name of sentence-transformers model
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Check for GPU availability
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("Using GPU for embeddings")
        else:
            logger.info("Using CPU for embeddings")

    def embed_questions(self, questions: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of questions.
        
        Args:
            questions: List of question texts
            batch_size: Batch size for embedding
            
        Returns:
            Array of embeddings (n_questions, embedding_dim)
        """
        logger.info(f"Embedding {len(questions)} questions...")
        embeddings = self.model.encode(
            questions,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        logger.info(f"Embedding shape: {embeddings.shape}")
        return embeddings

    def embed_single(self, question: str) -> np.ndarray:
        """
        Embed a single question.
        
        Args:
            question: Question text
            
        Returns:
            Embedding vector
        """
        return self.model.encode(question, convert_to_numpy=True)

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity([embedding1], [embedding2])[0][0]
