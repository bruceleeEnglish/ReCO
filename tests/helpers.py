"""Synthetic data shared by the tests."""
from __future__ import annotations

import numpy as np

from savemv.core.store import TokenStore


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def random_store(n_docs: int, dim: int, max_len: int, seed: int = 0) -> TokenStore:
    rng = np.random.default_rng(seed)
    docs = [unit(rng.standard_normal((int(rng.integers(1, max_len + 1)), dim))).astype(np.float16)
            for _ in range(n_docs)]
    toks = [rng.integers(0, 1000, len(d)).astype(np.int32) for d in docs]
    return TokenStore.from_docs(docs, toks)


def random_queries(n: int, m: int, dim: int, seed: int = 1) -> np.ndarray:
    return unit(np.random.default_rng(seed).standard_normal((n, m, dim))).astype(np.float32)


def naive_maxsim(store: TokenStore, queries: np.ndarray) -> np.ndarray:
    out = np.empty((len(queries), store.n_docs), dtype=np.float64)
    for qi, q in enumerate(queries):
        for d in range(store.n_docs):
            out[qi, d] = (q @ store.doc(d).astype(np.float64).T).max(axis=1).sum()
    return out
