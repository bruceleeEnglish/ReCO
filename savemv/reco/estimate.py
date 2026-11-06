"""Centroid-interaction estimator: s_hat(q, d) = sum_i max_{c in C_d} <q_i, mu_c>.

Needs only the sketch and the shared codebook; one query costs O(sum_d |C_d| * m) lookups.
"""
from __future__ import annotations

import numpy as np
import torch
from numba import njit, prange

from .sketch import CentroidBagSketch


@njit(parallel=True, fastmath=True, cache=True)
def _bag_maxsim(ptr: np.ndarray, cent: np.ndarray, sim_cq: np.ndarray, out: np.ndarray) -> None:
    n_docs = len(ptr) - 1
    m = sim_cq.shape[1]
    for d in prange(n_docs):
        best = np.full(m, -1e30, dtype=np.float32)
        for j in range(ptr[d], ptr[d + 1]):
            row = sim_cq[cent[j]]
            for i in range(m):
                if row[i] > best[i]:
                    best[i] = row[i]
        total = 0.0
        for i in range(m):
            total += best[i]
        out[d] = total


class CentroidBagEstimator:
    def __init__(self, sketch: CentroidBagSketch, centroids: np.ndarray) -> None:
        if len(centroids) != sketch.k:
            raise ValueError("codebook size does not match the sketch")
        self.sketch = sketch
        self.centroids = np.ascontiguousarray(centroids, dtype=np.float32)

    def score(self, query: np.ndarray) -> np.ndarray:
        """(n_docs,) estimates for one query of shape (m, dim)."""
        sim = np.ascontiguousarray(self.centroids @ np.asarray(query, dtype=np.float32).T)   # (K, m)
        out = np.empty(self.sketch.n_docs, dtype=np.float32)
        _bag_maxsim(self.sketch.ptr, self.sketch.cent, sim, out)
        return out

    def score_many(self, queries: np.ndarray) -> np.ndarray:
        """(n_queries, n_docs) estimates."""
        return np.stack([self.score(q) for q in queries])


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _bag_maxsim_kernel(ptr, cent, table, out, n_docs, M: tl.constexpr, BLOCK: tl.constexpr):
        """One program = BLOCK consecutive documents; running max of their entries' table rows in registers."""
        docs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = docs < n_docs
        start = tl.load(ptr + docs, mask=valid, other=0)
        size = tl.load(ptr + docs + 1, mask=valid, other=0) - start
        cols = tl.arange(0, M)
        best = tl.full((BLOCK, M), float("-inf"), tl.float32)
        for j in range(0, tl.max(size, axis=0)):
            has = j < size
            c = tl.load(cent + start + j, mask=has, other=0)
            rows = tl.load(table + c[:, None] * M + cols[None, :], mask=has[:, None], other=float("-inf"))
            best = tl.maximum(best, rows.to(tl.float32))
        tl.store(out + docs, tl.sum(best, axis=1), mask=valid)

    @triton.jit
    def _bag_maxsim_multi_kernel(ptr, cent, table, out, n_docs, Q: tl.constexpr, M: tl.constexpr,
                                 BLOCK: tl.constexpr):
        """Q queries in one pass over the sketch: `table` is (K, Q * M), query-major; `out` is (Q, n_docs)."""
        docs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = docs < n_docs
        start = tl.load(ptr + docs, mask=valid, other=0)
        size = tl.load(ptr + docs + 1, mask=valid, other=0) - start
        cols = tl.arange(0, Q * M)
        best = tl.full((BLOCK, Q * M), float("-inf"), tl.float32)
        for j in range(0, tl.max(size, axis=0)):
            has = j < size
            c = tl.load(cent + start + j, mask=has, other=0)
            rows = tl.load(table + c[:, None] * (Q * M) + cols[None, :], mask=has[:, None], other=float("-inf"))
            best = tl.maximum(best, rows.to(tl.float32))
        for qi in tl.static_range(Q):
            sel = (cols >= qi * M) & (cols < (qi + 1) * M)
            tl.store(out + qi * n_docs + docs, tl.sum(tl.where(sel[None, :], best, 0.0), axis=1), mask=valid)
except ImportError:          # CPU-only installs
    triton = None


class TorchCentroidBagEstimator:
    """Same estimate on a torch device (float32 throughout, so it matches the numba version).

    kernel="scatter": a query gathers the centroid similarities of the sketch entries and reduces them per
    document with scatter_reduce('amax'), in document-aligned chunks of at most `chunk_entries` (indices
    kept as int32 on the device), so memory stays bounded; a corpus of ~5M entries is one chunk and takes a
    few ms on an RTX 3090. kernel="triton" (CUDA): a fused kernel that reads every centroid id once and keeps
    the running maxima in registers; its traffic is the sketch itself, which matters at millions of documents.
    `half_table` stores the per-query (K x m) similarity table in fp16 for the triton kernel (about 2x faster
    at K = 65536; estimates move by ~1e-3, so use the same setting offline and online).
    """

    def __init__(self, sketch: CentroidBagSketch, centroids: np.ndarray, device: torch.device,
                 chunk_entries: int = 1 << 24, kernel: str = "scatter", half_table: bool = False) -> None:
        if len(centroids) != sketch.k:
            raise ValueError("codebook size does not match the sketch")
        if kernel not in ("scatter", "triton"):
            raise ValueError(f"unknown kernel {kernel!r}")
        if kernel == "triton" and (triton is None or device.type != "cuda"):
            raise ValueError("kernel='triton' needs triton and a CUDA device")
        self.kernel = kernel
        self.half_table = half_table
        self.n_docs = sketch.n_docs
        self.device = device
        self.cent = torch.from_numpy(sketch.cent.astype(np.int32)).to(device)
        ptr = sketch.ptr.astype(np.int64)
        self.ptr = torch.from_numpy(ptr).to(device) if kernel == "triton" else None
        self.centroids = torch.from_numpy(np.ascontiguousarray(centroids, dtype=np.float32)).to(device)
        self.chunks: list[tuple[int, int, int, int]] = []          # (doc0, doc1, entry0, entry1)
        if kernel == "triton":
            return
        d0 = 0
        while d0 < self.n_docs:
            d1 = int(np.searchsorted(ptr, ptr[d0] + chunk_entries, side="right")) - 1
            d1 = min(max(d1, d0 + 1), self.n_docs)
            self.chunks.append((d0, d1, int(ptr[d0]), int(ptr[d1])))
            d0 = d1
        local = np.repeat(np.arange(self.n_docs, dtype=np.int64), sketch.sizes)
        for c0, _, e0, e1 in self.chunks:
            local[e0:e1] -= c0
        self.local = torch.from_numpy(local.astype(np.int32)).to(device)

    @torch.inference_mode()
    def score_tensor(self, query: torch.Tensor) -> torch.Tensor:
        """(n_docs,) float32 estimates on the device for one query (m, dim)."""
        m = query.shape[0]
        table = self.centroids @ query.float().T                                       # (K, m)
        out = torch.empty(self.n_docs, device=self.device)
        if self.kernel == "triton":
            width = 1 << max(0, (m - 1).bit_length())          # tl.arange needs a power of two; pad with zeros
            if width != m:
                table = torch.nn.functional.pad(table, (0, width - m))
            table = (table.half() if self.half_table else table).contiguous()
            block = 32
            _bag_maxsim_kernel[(triton.cdiv(self.n_docs, block),)](self.ptr, self.cent, table, out, self.n_docs,
                                                                   M=width, BLOCK=block, num_warps=4)
            return out
        for d0, d1, e0, e1 in self.chunks:
            vals = table.index_select(0, self.cent[e0:e1])                             # (entries, m)
            best = torch.full((d1 - d0, m), -torch.inf, device=self.device)
            best.scatter_reduce_(0, self.local[e0:e1].long()[:, None].expand(-1, m), vals, reduce="amax",
                                 include_self=True)
            out[d0:d1] = best.sum(dim=1)
        return out

    @torch.inference_mode()
    def score_batch_tensor(self, queries: torch.Tensor, group: int = 1) -> torch.Tensor:
        """(Q, n_docs) estimates for queries (Q, m, dim). With the triton kernel, `group` > 1 queries share one pass
        over the sketch (each entry read once per group). On an RTX 3090 this is not faster: the scan is bound by
        lookups into the (K x m) table, which fits the L2 cache for one query (K = 64k, fp16) but not for four."""
        if self.kernel != "triton":
            return torch.stack([self.score_tensor(q) for q in queries])
        n_q, m, _ = queries.shape
        width = 1 << max(0, (m - 1).bit_length())
        out = torch.empty((n_q, self.n_docs), device=self.device)
        for g0 in range(0, n_q, group):
            qs = queries[g0:g0 + group].float()
            g = 1 << max(0, (len(qs) - 1).bit_length())                    # power of two for tl.arange
            table = torch.zeros((self.centroids.shape[0], g, width), device=self.device)
            table[:, :len(qs), :m] = torch.einsum("kd,qmd->kqm", self.centroids, qs)
            table = table.reshape(len(table), g * width)
            table = (table.half() if self.half_table else table).contiguous()
            res = torch.empty((g, self.n_docs), device=self.device)
            block = 32 if g * width <= 128 else 16
            _bag_maxsim_multi_kernel[(triton.cdiv(self.n_docs, block),)](self.ptr, self.cent, table, res,
                                                                         self.n_docs, Q=g, M=width, BLOCK=block,
                                                                         num_warps=4)
            out[g0:g0 + len(qs)] = res[:len(qs)]
        return out

    def score(self, query: np.ndarray) -> np.ndarray:
        return self.score_tensor(torch.from_numpy(np.asarray(query, dtype=np.float32)).to(self.device)).cpu().numpy()
