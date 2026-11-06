"""Shared foundation: token-vector store, exact MaxSim, codebooks (spherical k-means) and metrics."""
from .codebook import assign_to_codebook, nearest_centroid, train_spherical_kmeans, training_sample
from .exact import maxsim_matrix, maxsim_padded
from .metrics import ndcg_at_k, ndcg_per_query, recall_vs_exact
from .store import TokenStore

__all__ = ["TokenStore", "maxsim_matrix", "maxsim_padded", "assign_to_codebook", "nearest_centroid",
           "train_spherical_kmeans", "training_sample", "ndcg_at_k", "ndcg_per_query", "recall_vs_exact"]
