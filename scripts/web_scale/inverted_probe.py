"""Feasibility probe for an inverted index over the sketches (MS MARCO, CPU).

The sketch estimate is s_hat(q, d) = sum_i f_i(d) with f_i(d) = max_{c in C_d} <q_i, mu_c>. Visiting, for every query
token i, its n best centroids and the documents that contain them (posting lists centroid -> documents) yields the
candidates; any document outside them has s_hat <= UB = sum_i s_i^(n+1), the (n+1)-th best centroid score of each
token (a threshold-algorithm bound). Candidates with s_hat > UB therefore form an exact prefix of the full-scan
order. This probe measures, per n, the candidates, the exact-prefix length, the coverage of the exact top-10 and CPU
time, against the full scan.

    python scripts/web_scale/inverted_probe.py --out outputs/msmarco --queries 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from savemv.reco import CentroidBagEstimator, CentroidBagSketch  # noqa: E402


@njit(cache=True)
def _postings(ptr: np.ndarray, cent: np.ndarray, k: int):
    counts = np.zeros(k + 1, np.int64)
    for c in cent:
        counts[c + 1] += 1
    for c in range(k):
        counts[c + 1] += counts[c]
    fill = counts[:-1].copy()
    docs = np.empty(len(cent), np.int32)
    for d in range(len(ptr) - 1):
        for j in range(ptr[d], ptr[d + 1]):
            c = cent[j]
            docs[fill[c]] = d
            fill[c] += 1
    return counts, docs


@njit(parallel=True, cache=True)
def _mark(post_ptr, post_doc, chosen, mark):
    for t in prange(len(chosen)):
        c = chosen[t]
        for j in range(post_ptr[c], post_ptr[c + 1]):
            mark[post_doc[j]] = 1


@njit(parallel=True, cache=True)
def _score_subset(ptr, cent, sim, docs, out):
    m = sim.shape[1]
    for t in prange(len(docs)):
        d = docs[t]
        best = np.full(m, -1e30, dtype=np.float32)
        for j in range(ptr[d], ptr[d + 1]):
            row = sim[cent[j]]
            for i in range(m):
                if row[i] > best[i]:
                    best[i] = row[i]
        out[t] = best.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/msmarco")
    ap.add_argument("--k", type=int, default=65536)
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--probes", type=int, nargs="+", default=[4, 16, 64, 256, 1024])
    args = ap.parse_args()
    out = Path(args.out)
    sk = CentroidBagSketch.load(out / f"sketch_K{args.k}.npz")
    cents = np.load(out / f"codebook_K{args.k}.npy").astype(np.float32)
    queries = np.load(out / "queries.npy")[:args.queries]
    top10 = np.load(out / "exact_top.npy")[:args.queries, :10]
    t = time.perf_counter()
    post_ptr, post_doc = _postings(sk.ptr, sk.cent, args.k)
    print(f"postings built in {time.perf_counter() - t:.1f} s; {len(post_doc) * 4 / 1e9:.2f} GB")
    full = CentroidBagEstimator(sk, cents)
    full.score(queries[0])
    mark = np.zeros(sk.n_docs, np.uint8)
    rows = []
    for qi, q in enumerate(queries):
        sim = np.ascontiguousarray(cents @ q.T)                     # (K, m)
        t = time.perf_counter()
        shat_full = full.score(q)
        t_full = time.perf_counter() - t
        order_full = np.argsort(-shat_full, kind="stable")
        rank_top = np.empty(10, np.int64)
        inv = np.empty(sk.n_docs, np.int64)
        inv[order_full] = np.arange(sk.n_docs)
        rank_top = inv[top10[qi]]
        srt = -np.sort(-sim, axis=0)                               # per-token centroid scores, descending
        for n in args.probes:
            t = time.perf_counter()
            chosen = np.unique(np.argpartition(-sim, n, axis=0)[:n].ravel()).astype(np.int64)
            mark[:] = 0
            _mark(post_ptr, post_doc, chosen, mark)
            docs = np.flatnonzero(mark).astype(np.int64)
            vals = np.empty(len(docs), np.float32)
            _score_subset(sk.ptr, sk.cent, sim, docs, vals)
            ub = float(srt[n].sum())
            prefix = int((vals > ub).sum())
            t_inv = time.perf_counter() - t
            cover = float(np.isin(top10[qi], docs).mean())
            rows.append((n, len(docs), prefix, cover, t_inv, t_full, int(rank_top.max())))
    rows = np.array(rows)
    print("n | candidates | exact prefix (p10 / p50) | top-10 coverage | time ms | full ms | deepest exact top-10 rank p50")
    for n in args.probes:
        r = rows[rows[:, 0] == n]
        print(f"{n:5d} | {np.median(r[:, 1]):10.0f} | {np.percentile(r[:, 2], 10):8.0f} / {np.median(r[:, 2]):8.0f} | "
              f"{r[:, 3].mean():.3f} | {1000 * r[:, 4].mean():7.1f} | {1000 * r[:, 5].mean():7.1f} | {np.median(r[:, 6]):.0f}")


if __name__ == "__main__":
    main()
