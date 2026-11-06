"""Exact estimate prefixes from an inverted index over the sketches (CPU).

The estimate decomposes over query tokens, s_hat(q, d) = sum_i f_i(d) with f_i(d) = max_{c in C_d} <q_i, mu_c>, and a
token's centroids sorted by <q_i, mu_c> enumerate documents in decreasing f_i through posting lists centroid ->
documents. After visiting each token's n best centroids, every unvisited document satisfies
s_hat <= UB = sum_i s_i^(n+1), the (n+1)-th best centroid score of each token (the bound of the threshold algorithm,
Fagin et al.). The visited documents with s_hat > UB are therefore exactly the head of the full-scan order, in the
same order, and recomputation can proceed on them as if the whole corpus had been scanned. When verification needs
more of the order, the index visits more centroids (n x 2) and scores only the new candidates.

The posting lists are a permutation of the sketch entries (one int32 document id per entry), derived from the sketch
when it is loaded; the stored index is unchanged.
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit, prange

from .sketch import CentroidBagSketch


@njit(cache=True)
def _postings(ptr: np.ndarray, cent: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    starts = np.zeros(k + 1, np.int64)
    for c in cent:
        starts[c + 1] += 1
    for c in range(k):
        starts[c + 1] += starts[c]
    fill = starts[:-1].copy()
    docs = np.empty(len(cent), np.int32)
    for d in range(len(ptr) - 1):
        for j in range(ptr[d], ptr[d + 1]):
            c = cent[j]
            docs[fill[c]] = d
            fill[c] += 1
    return starts, docs


@njit(cache=True)
def _collect(post_ptr, post_doc, chosen, stamp, tag, out, n_out):
    """Append the unseen documents of the chosen centroids' postings to out[n_out:]; returns the new length."""
    for c in chosen:
        for j in range(post_ptr[c], post_ptr[c + 1]):
            d = post_doc[j]
            if stamp[d] != tag:
                stamp[d] = tag
                out[n_out] = d
                n_out += 1
    return n_out


@njit(parallel=True, fastmath=True, cache=True)
def _score(ptr, cent, sim, docs, out):
    """estimate._bag_maxsim restricted to `docs`, with the same arithmetic (float64 sum of float32 maxima under
    the same flags), so candidate estimates equal the full scan's bit for bit."""
    m = sim.shape[1]
    for t in prange(len(docs)):
        d = docs[t]
        best = np.full(m, -1e30, dtype=np.float32)
        for j in range(ptr[d], ptr[d + 1]):
            row = sim[cent[j]]
            for i in range(m):
                if row[i] > best[i]:
                    best[i] = row[i]
        total = 0.0
        for i in range(m):
            total += best[i]
        out[t] = total


class InvertedSketchIndex:
    def __init__(self, sketch: CentroidBagSketch, centroids: np.ndarray) -> None:
        t = time.perf_counter()
        self.sketch = sketch
        self.centroids = np.ascontiguousarray(centroids, dtype=np.float32)
        self.post_ptr, self.post_doc = _postings(sketch.ptr, sketch.cent, sketch.k)
        self.stamp = np.zeros(sketch.n_docs, np.int32)
        self.seen_c = np.zeros(sketch.k, np.int32)
        self.tag = 0
        self.build_s = time.perf_counter() - t

    @property
    def memory_bytes(self) -> int:
        return self.post_doc.nbytes + self.post_ptr.nbytes + self.stamp.nbytes

    def prefix(self, query: np.ndarray, need: int, start: int = 4, a: float = 1.0, b: float = 0.0):
        """The first `need` documents of the full-scan order (decreasing estimate, ties by id) and their
        calibrated estimates a * s_hat + b, or the whole exact prefix found when the corpus is exhausted.
        Returns (order, values, stats)."""
        sk, k = self.sketch, self.sketch.k
        sim = np.ascontiguousarray(self.centroids @ np.asarray(query, dtype=np.float32).T)      # (K, m)
        neg = -np.ascontiguousarray(sim.T)                                                       # (m, K)
        self.tag += 1
        tag = self.tag
        cand = np.empty(sk.n_docs, np.int64)
        vals = np.empty(sk.n_docs, np.float32)
        n_cand, n, rounds = 0, start, 0
        while True:
            rounds += 1
            if n >= k:
                chosen_all = np.arange(k)
                ub = -np.inf
            else:
                part = np.argpartition(neg, n, axis=1)
                chosen_all = part[:, :n].ravel()
                ub = float(-neg[np.arange(len(neg)), part[:, n]].sum())
            new_c = chosen_all[self.seen_c[chosen_all] != tag]
            new_c = np.unique(new_c)
            self.seen_c[new_c] = tag
            n_new = _collect(self.post_ptr, self.post_doc, new_c.astype(np.int64), self.stamp, tag, cand, n_cand)
            _score(sk.ptr, sk.cent, sim, cand[n_cand:n_new], vals[n_cand:n_new])
            n_cand = n_new
            exact = vals[:n_cand] > ub
            if exact.sum() >= need or n >= k:
                break
            n *= 2
        docs, v = cand[:n_cand][exact], vals[:n_cand][exact]
        order = np.lexsort((docs, -v))[:need]
        return docs[order], a * v[order] + b, {"probe": min(n, k), "rounds": rounds, "candidates": n_cand,
                                               "exact_prefix": int(exact.sum())}
