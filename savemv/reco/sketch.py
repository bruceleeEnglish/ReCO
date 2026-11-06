"""Centroid-bag sketch: each document keeps only the set of codebook centroids its tokens use.

Storage per document is min(|C_d| * ceil(log2 K), K) bits (id list or bitmap, whichever is
smaller) plus a 4-byte CSR pointer. Lemma: for a unit query vector q,
|max_x <q,x> - max_{c in C_d} <q,mu_c>| <= max_x ||x - mu_c(x)||, so the estimate is consistent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CentroidBagSketch:
    k: int
    ptr: np.ndarray            # (n_docs + 1,) int64 CSR pointer into `cent`
    cent: np.ndarray           # (nnz,) int32 unique centroid ids, sorted within each doc
    rmax: np.ndarray | None    # (nnz,) float32 max residual of the doc's tokens at that centroid

    @property
    def n_docs(self) -> int:
        return len(self.ptr) - 1

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(self.ptr)

    @classmethod
    def build(cls, doc_of_token: np.ndarray, centroid_ids: np.ndarray, residuals: np.ndarray,
              k: int, n_docs: int) -> "CentroidBagSketch":
        """Vectorized construction from per-token centroid assignments."""
        key = doc_of_token.astype(np.int64) * k + centroid_ids.astype(np.int64)
        order = np.argsort(key, kind="stable")
        key, res = key[order], residuals[order]
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        ukey = key[starts]
        doc, cent = ukey // k, (ukey % k).astype(np.int32)
        ptr = np.zeros(n_docs + 1, dtype=np.int64)
        np.cumsum(np.bincount(doc, minlength=n_docs), out=ptr[1:])
        rmax = np.maximum.reduceat(res, starts).astype(np.float32)
        return cls(k, ptr, cent, rmax)

    def bytes_per_doc(self) -> np.ndarray:
        bits = np.minimum(self.sizes * math.ceil(math.log2(self.k)), self.k)
        return bits / 8.0 + 4.0

    def storage_bytes(self) -> float:
        """Total sketch size (the shared codebook, K x dim fp16, is reported separately)."""
        return float(self.bytes_per_doc().sum())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"k": np.array(self.k), "ptr": self.ptr, "cent": self.cent}
        if self.rmax is not None:
            arrays["rmax"] = self.rmax
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "CentroidBagSketch":
        z = np.load(path)
        return cls(int(z["k"]), z["ptr"], z["cent"], z["rmax"] if "rmax" in z.files else None)
