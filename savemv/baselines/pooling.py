"""Hierarchical token pooling (Clavie et al., 2024), the training-free way to shrink ColBERT indexes.

Mirrors PyLate's `ColBERT.pool_embeddings_hierarchical`: the first `protected` tokens ([CLS]) are
kept, the rest are clustered with Ward linkage on the cosine distance 1 - <x, y> into
max(n // factor, 1) clusters, and every cluster is replaced by its (unnormalized) mean.
Documents are pooled in parallel worker processes when the platform can fork.
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np
from scipy.cluster import hierarchy

from ..core.store import TokenStore
from ..utils import get_logger

log = get_logger(__name__)

_STORE: TokenStore | None = None   # read by forked workers


def pool_document(vectors: np.ndarray, factor: int, protected: int = 1) -> np.ndarray:
    """Pooled vectors of one document (float32), protected tokens first.

    Inputs are re-normalized in float32 (PyLate pools the model's unit-norm outputs; ours are stored in
    fp16) and distances are clamped at 0, since Ward linkage rejects the tiny negative values that
    near-duplicate tokens would otherwise produce.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    keep, rest = vectors[:protected], vectors[protected:]
    n = len(rest)
    clusters = max(n // factor, 1)
    if clusters >= n:
        return vectors
    dist = np.maximum(1.0 - rest @ rest.T, 0.0)
    labels = hierarchy.fcluster(hierarchy.linkage(dist[np.triu_indices(n, k=1)], method="ward"),
                                t=clusters, criterion="maxclust")
    # contiguous ids: fcluster may return fewer clusters than requested, and (older scipy, tied merge
    # heights) occasionally more, where PyLate's scatter would fail
    _, labels = np.unique(labels, return_inverse=True)
    sums = np.zeros((labels.max() + 1, rest.shape[1]), dtype=np.float32)
    np.add.at(sums, labels, rest)
    return np.concatenate([keep, sums / np.bincount(labels)[:, None]])


def _pool_range(args: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    start, stop, factor, protected = args
    docs = [pool_document(_STORE.doc(i), factor, protected) for i in range(start, stop)]
    return np.concatenate(docs).astype(np.float16), np.array([len(d) for d in docs], dtype=np.int64)


def pool_store(store: TokenStore, factor: int, protected: int = 1, workers: int = 0) -> TokenStore:
    """Pool every document of the store; `workers` = 0 uses all cores (serial where fork is unavailable)."""
    global _STORE
    workers = workers or os.cpu_count() or 1
    chunks = max(1, min(store.n_docs, workers * 16))
    bounds = np.linspace(0, store.n_docs, chunks + 1).astype(int)
    tasks = [(int(a), int(b), factor, protected) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    _STORE = store
    try:
        if workers > 1 and "fork" in mp.get_all_start_methods():
            with mp.get_context("fork").Pool(workers) as pool:
                parts = pool.map(_pool_range, tasks)
        else:
            parts = [_pool_range(t) for t in tasks]
    finally:
        _STORE = None
    vectors = np.concatenate([v for v, _ in parts])
    lengths = np.concatenate([n for _, n in parts])
    offsets = np.zeros(store.n_docs + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    log.info("pooled factor %d: %.1f -> %.1f vectors / doc", factor, store.n_tokens / store.n_docs,
             len(vectors) / store.n_docs)
    return TokenStore(vectors, offsets, np.full(len(vectors), -1, dtype=np.int32))
