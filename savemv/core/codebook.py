"""Spherical k-means codebook over token vectors (PyTorch; runs on GPU when available)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import get_logger

log = get_logger(__name__)


def _chunk_rows(device: torch.device, k: int, budget_elems: int = 1 << 27) -> int:
    """Rows per chunk so that a (rows x k) similarity block stays within the budget."""
    return max(1024, budget_elems // max(1, k)) if device.type == "cuda" else max(256, (1 << 24) // max(1, k))


@torch.inference_mode()
def nearest_centroid(x: torch.Tensor, centroids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Index of and similarity to the nearest centroid (by inner product) for every row."""
    rows = _chunk_rows(x.device, len(centroids))
    ids, sims = [], []
    for s in range(0, len(x), rows):
        best, idx = (x[s:s + rows] @ centroids.T).max(dim=1)
        ids.append(idx)
        sims.append(best)
    return torch.cat(ids), torch.cat(sims)


@torch.inference_mode()
def train_spherical_kmeans(points: np.ndarray, k: int, iters: int, seed: int, device: torch.device) -> np.ndarray:
    """Lloyd iterations on the unit sphere; empty clusters are re-seeded with random points."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.from_numpy(np.ascontiguousarray(points, dtype=np.float32)).to(device)
    x = F.normalize(x, dim=1)
    n = len(x)
    if n < k:
        raise ValueError(f"need at least k={k} training points, got {n}")
    centroids = x[torch.randperm(n, generator=gen)[:k].to(device)].clone()
    for it in range(iters):
        assign, sim = nearest_centroid(x, centroids)
        sums = torch.zeros_like(centroids).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k)
        empty = counts == 0
        if empty.any():
            refill = torch.randint(0, n, (int(empty.sum()),), generator=gen).to(device)
            sums[empty] = x[refill]
        centroids = F.normalize(sums, dim=1)
        if it == iters - 1 or it % 5 == 0:
            log.info("k-means K=%d iter %d/%d: mean cos=%.4f, empty=%d", k, it + 1, iters,
                     float(sim.mean()), int(empty.sum()))
    return centroids.cpu().numpy()


@torch.inference_mode()
def assign_to_codebook(vectors: np.ndarray, centroids: np.ndarray, device: torch.device,
                       chunk: int = 1 << 20) -> tuple[np.ndarray, np.ndarray]:
    """Nearest centroid id (int32) and exact residual norm ||x - mu_c|| (float32) per vector."""
    c = torch.from_numpy(np.ascontiguousarray(centroids, dtype=np.float32)).to(device)
    ids = np.empty(len(vectors), dtype=np.int32)
    resid = np.empty(len(vectors), dtype=np.float32)
    for s in range(0, len(vectors), chunk):
        x = torch.from_numpy(np.asarray(vectors[s:s + chunk], dtype=np.float32)).to(device)
        idx, sim = nearest_centroid(x, c)
        sq = (x * x).sum(1) + 1.0 - 2.0 * sim       # ||x||^2 + ||mu||^2 - 2 <x, mu>, ||mu|| = 1
        ids[s:s + len(x)] = idx.to(torch.int32).cpu().numpy()
        resid[s:s + len(x)] = sq.clamp_min(0).sqrt().cpu().numpy()
    return ids, resid


def training_sample(n_tokens: int, k: int, points_per_centroid: int, min_points: int, seed: int) -> np.ndarray:
    """Sorted row indices of the k-means training sample."""
    size = min(n_tokens, max(k * points_per_centroid, min_points))
    return np.sort(np.random.default_rng(seed).choice(n_tokens, size, replace=False))
