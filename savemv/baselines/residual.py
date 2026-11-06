"""ColBERTv2 / PLAID residual compression (Santhanam et al., NAACL 2022): the standard lossy index.

Every vector is stored as its nearest centroid id plus a b-bit code per dimension of the residual
x - mu_c. Codes index global bucket weights taken from quantiles of held-out residuals, and
decompression is normalize(mu_c + weights[codes]), as in colbert/indexing/codecs/residual.py.
We evaluate exhaustive MaxSim over the decompressed vectors, an upper bound on PLAID's quality.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..core.codebook import nearest_centroid, train_spherical_kmeans, training_sample
from ..core.store import TokenStore
from ..utils import get_logger

log = get_logger(__name__)


def plaid_num_centroids(n_vectors: int) -> int:
    """ColBERT's codebook size: 2 ** floor(log2(16 * sqrt(n_vectors)))."""
    return int(2 ** math.floor(math.log2(16 * math.sqrt(max(1, n_vectors)))))


@dataclass
class ResidualCodec:
    centroids: np.ndarray    # (K, dim) float32, unit rows
    cutoffs: np.ndarray      # (2**nbits - 1,) bucket boundaries of residual values
    weights: np.ndarray      # (2**nbits,) reconstruction value of each bucket
    nbits: int

    @property
    def k(self) -> int:
        return len(self.centroids)

    @classmethod
    def fit(cls, vectors: np.ndarray, nbits: list[int], k: int, iters: int, seed: int, device: torch.device,
            points_per_centroid: int = 32, min_train_points: int = 200_000,
            heldout: int = 50_000) -> list["ResidualCodec"]:
        """One k-means codebook (on a sample) shared by codecs for each `nbits`; bucket statistics come
        from disjoint held-out vectors, as in ColBERT's CollectionIndexer."""
        n = len(vectors)
        rows = training_sample(n, k, points_per_centroid, min_train_points, seed)
        centroids = train_spherical_kmeans(np.asarray(vectors[rows]), k, iters, seed, device)
        rest = np.setdiff1d(np.arange(n), rows, assume_unique=True)
        pool = rest if len(rest) >= min(heldout, n // 20) else np.arange(n)
        held = np.sort(np.random.default_rng(seed + 1).choice(pool, min(heldout, len(pool)), replace=False))
        x = torch.from_numpy(np.asarray(vectors[held], dtype=np.float32)).to(device)
        c = torch.from_numpy(centroids).to(device)
        resid = (x - c[nearest_centroid(x, c)[0]]).flatten()
        codecs = []
        for b in nbits:
            levels = 2 ** b
            q = torch.arange(levels, device=device, dtype=torch.float32) / levels
            cutoffs = torch.quantile(resid, q[1:]).cpu().numpy().astype(np.float32)
            weights = torch.quantile(resid, q + 0.5 / levels).cpu().numpy().astype(np.float32)
            codecs.append(cls(centroids, cutoffs, weights, b))
        return codecs

    @torch.inference_mode()
    def reconstruct(self, vectors: np.ndarray, device: torch.device, chunk: int = 1 << 20) -> np.ndarray:
        """Compress then decompress every vector; returns float16 rows (normalized, as ColBERT's decompress)."""
        c = torch.from_numpy(self.centroids).to(device)
        cutoffs = torch.from_numpy(self.cutoffs).to(device)
        weights = torch.from_numpy(self.weights).to(device)
        out = np.empty(vectors.shape, dtype=np.float16)
        for s in range(0, len(vectors), chunk):
            x = torch.from_numpy(np.asarray(vectors[s:s + chunk], dtype=np.float32)).to(device)
            mu = c[nearest_centroid(x, c)[0]]
            codes = torch.bucketize(x - mu, cutoffs)
            out[s:s + len(x)] = F.normalize(mu + weights[codes], dim=-1).half().cpu().numpy()
        return out

    def compress_store(self, store: TokenStore, device: torch.device) -> TokenStore:
        return TokenStore(self.reconstruct(store.vectors, device), store.offsets, store.token_ids)

    def bytes_per_vector(self, dim: int) -> float:
        """Centroid id at ceil(log2 K) bits (ColBERT uses 4 bytes) plus dim * nbits bits of residual."""
        return math.ceil(math.log2(self.k)) / 8 + dim * self.nbits / 8

    def shared_bytes(self) -> float:
        """fp16 codebook plus the bucket tables."""
        return self.centroids.size * 2 + (self.cutoffs.size + self.weights.size) * 4
