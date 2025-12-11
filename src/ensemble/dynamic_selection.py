"""
Cluster-based dynamic model selection ensemble.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances

logger = logging.getLogger(__name__)


class ClusterBasedEnsemble:
    """Dynamic model selection based on question clustering."""

    def __init__(self, model_names: List[str], embedding_dim: int = 384, 
                 n_clusters: int = 10, config: Dict[str, Any] = None):
        """
        Initialize cluster-based ensemble.
        
        Args:
            model_names: List of model names
            embedding_dim: Dimension of embeddings
            n_clusters: Number of clusters
            config: Configuration dictionary
        """
        self.model_names = model_names
        self.n_models = len(model_names)
        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters
        self.config = config or {}
        
        # Initialize clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_centers = None
        self.fitted = False
        
        # Cluster-to-model mapping
        self.cluster_model_mapping = {}
        self.cluster_accuracies = {}

    def fit(self, embeddings: np.ndarray, cluster_labels: Optional[np.ndarray] = None):
        """
        Fit clustering on embeddings.
        
        Args:
            embeddings: Question embeddings (n_questions, embedding_dim)
            cluster_labels: Optional pre-computed cluster labels
        """
        logger.info(f"Fitting KMeans clustering with {self.n_clusters} clusters...")
        
        if cluster_labels is None:
            self.kmeans.fit(embeddings)
            cluster_labels = self.kmeans.labels_
        else:
            self.kmeans.cluster_centers_ = np.array([
                embeddings[cluster_labels == i].mean(axis=0)
                for i in range(self.n_clusters)
            ])
        
        self.cluster_centers = self.kmeans.cluster_centers_
        self.fitted = True
        
        logger.info(f"Cluster sizes: {np.bincount(cluster_labels)}")

    def update_cluster_model_mapping(self, cluster_model_accuracies: Dict[int, Dict[str, float]]):
        """
        Update which models perform best in each cluster.
        
        Args:
            cluster_model_accuracies: Dict of cluster_id -> {model_name -> accuracy}
        """
        self.cluster_model_mapping = {}
        self.cluster_accuracies = cluster_model_accuracies.copy()
        
        for cluster_id, model_accs in cluster_model_accuracies.items():
            if model_accs:
                best_model = max(model_accs.items(), key=lambda x: x[1])[0]
                self.cluster_model_mapping[cluster_id] = best_model
                logger.info(
                    f"Cluster {cluster_id}: Best model = {best_model} "
                    f"(acc: {model_accs[best_model]:.3f})"
                )

    def assign_cluster(self, embedding: np.ndarray) -> int:
        """
        Assign embedding to nearest cluster.
        
        Args:
            embedding: Question embedding
            
        Returns:
            Cluster ID
        """
        if not self.fitted:
            raise ValueError("Clustering not fitted. Call fit() first.")
        
        distances = cosine_distances([embedding], self.cluster_centers)[0]
        cluster_id = np.argmin(distances)
        return cluster_id

    def select_models(self, embedding: np.ndarray, 
                     selection_strategy: str = "best") -> List[str]:
        """
        Select optimal model(s) for a question based on cluster.
        
        Args:
            embedding: Question embedding
            selection_strategy: "best" (single best) or "ensemble" (weighted ensemble)
            
        Returns:
            List of model names to use
        """
        cluster_id = self.assign_cluster(embedding)
        
        if selection_strategy == "best":
            # Return single best model for cluster
            if cluster_id in self.cluster_model_mapping:
                return [self.cluster_model_mapping[cluster_id]]
            else:
                logger.warning(f"No mapping for cluster {cluster_id}, using all models")
                return self.model_names
        
        elif selection_strategy == "ensemble":
            # Return all models (for weighted ensemble within cluster)
            return self.model_names
        
        else:
            raise ValueError(f"Unknown strategy: {selection_strategy}")

    def predict(self, embedding: np.ndarray, 
               predictions: Dict[str, str],
               selection_strategy: str = "best",
               options: Optional[List[str]] = None) -> Tuple[str, float, int]:
        """
        Make prediction for a single question.
        
        Args:
            embedding: Question embedding
            predictions: Dict of model_name -> predicted_answer
            selection_strategy: Model selection strategy
            options: Optional valid options
            
        Returns:
            Tuple of (predicted_answer, confidence, cluster_id)
        """
        cluster_id = self.assign_cluster(embedding)
        selected_models = self.select_models(embedding, selection_strategy)
        
        # Filter predictions to selected models
        selected_predictions = {
            m: predictions[m] for m in selected_models if m in predictions
        }
        
        if not selected_predictions:
            return "", 0.0, cluster_id
        
        # Simple majority vote among selected models
        from collections import Counter
        votes = Counter(selected_predictions.values())
        predicted_answer = votes.most_common(1)[0][0]
        confidence = votes.most_common(1)[0][1] / len(selected_predictions)
        
        return predicted_answer, confidence, cluster_id

    def batch_predict(self, embeddings: np.ndarray,
                     batch_predictions: List[Dict[str, str]],
                     selection_strategy: str = "best",
                     batch_options: Optional[List[List[str]]] = None) -> List[Tuple[str, float, int]]:
        """
        Make predictions for a batch of questions.
        
        Args:
            embeddings: Batch of embeddings
            batch_predictions: Batch of prediction dicts
            selection_strategy: Model selection strategy
            batch_options: Batch of option lists
            
        Returns:
            List of (prediction, confidence, cluster_id) tuples
        """
        results = []
        
        for i, (embedding, predictions) in enumerate(zip(embeddings, batch_predictions)):
            options = batch_options[i] if batch_options else None
            result = self.predict(embedding, predictions, selection_strategy, options)
            results.append(result)
        
        return results

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about clusters and model assignments."""
        return {
            "n_clusters": self.n_clusters,
            "fitted": self.fitted,
            "cluster_model_mapping": self.cluster_model_mapping.copy(),
            "cluster_accuracies": self.cluster_accuracies.copy()
        }
