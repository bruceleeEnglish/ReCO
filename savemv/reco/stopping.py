"""Stopping rules for estimate-then-recompute search, and their offline evaluation.

Documents are recomputed in decreasing order of the calibrated estimate s_hat = a*E + b, `batch` at a
time, keeping the k best exact scores (theta_k = the k-th). Two guarantees are targeted:
recall (E[fraction of the exact top-k found] >= 1 - delta) and strict (P[all of it found] >= 1 - delta).

  fixed / fixed_strict   recompute the first R documents; R by conformal risk control (CRC)
  adaptive               stop once theta_k - s_hat(next) >= tau, with tau from
    naive   pooled split-conformal scores  s(d) - s_hat(d),          d in exact top-k   (recall)
    rank    pooled scores                  s*_k - s_hat(d)                              (recall)
    rank1   pooled scores                  s*_(k+1) - s_hat(d)                          (recall)
    strict  per query                      s*_k - min_d s_hat(d)                        (strict)
    strict1 per query                      s*_(k+1) - min_d s_hat(d)                    (strict)
    crc / crc_strict   CRC on the miss loss of the stopping procedure itself
  patience               stop once the running top-k has not changed for W recomputations; W by CRC
  hazard                 stop once -logit of a learned hazard (hazard.py) reaches lambda; lambda by CRC
  hazard_doc             hazard with the pooled per-document correction of rank1 (like-for-like with it)
  hazard_u               hazard with a fixed model trained on other corpora (when one is given)
A missed top-k document d was never reached, so s_hat(d) <= s_hat(next) <= theta_k - tau, and theta_k is
at most s*_(k+1) because d is missing: hence s*_(k+1) - s_hat(d) >= tau, the "rank1" score (pointwise
below "rank"). CRC (Angelopoulos et al., 2022) instead uses that the miss loss of each query is
non-increasing in tau and picks the smallest tau with (sum_i loss_i(tau) + 1) / (n + 1) <= delta, which
gives E[loss] <= delta exactly; the pooled quantiles are approximately valid (scores within a query
are dependent).

Evaluation takes one or more QuerySplits with disjoint test sets: a single fit/cal/test split, or
k-fold cross-calibration where every query is tested exactly once. Each query is scanned once into a
StopPath (the running theta_k and next estimate at every batch boundary); every rule and margin is
then a binary search on that path.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from numba import njit

from ..core.metrics import ndcg_per_query
from .hazard import HazardModel, HazardSettings, fit_hazard, hazard_gmax

RECALL_RULES = ("fixed", "naive", "rank", "rank1", "crc", "patience", "hazard", "hazard_doc")
STRICT_RULES = ("fixed_strict", "strict", "strict1", "crc_strict")
RULES = RECALL_RULES + STRICT_RULES
STATISTIC = {"patience": "patience", "hazard": "hazard", "hazard_doc": "hazard", "hazard_u": "hazard_u"}


@dataclass
class QuerySplit:
    fit: np.ndarray
    cal: np.ndarray
    test: np.ndarray


def split_queries(n_queries: int, fractions: list[float], seed: int) -> QuerySplit:
    """Random fit / calibration / test split of query indices."""
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("split fractions must be three numbers summing to 1")
    perm = np.random.default_rng(seed).permutation(n_queries)
    a = int(fractions[0] * n_queries)
    b = a + int(fractions[1] * n_queries)
    return QuerySplit(perm[:a], perm[a:b], perm[b:])


def kfold_splits(n_queries: int, folds: int, fit_fraction: float, seed: int) -> list[QuerySplit]:
    """k folds; fold f tests its own queries and splits all other queries into fit / calibration."""
    if folds < 2 or n_queries < folds:
        raise ValueError(f"need 2 <= folds <= n_queries, got folds={folds}, n_queries={n_queries}")
    parts = np.array_split(np.random.default_rng(seed).permutation(n_queries), folds)
    out = []
    for f in range(folds):
        rest = np.random.default_rng(seed + 1 + f).permutation(np.concatenate(parts[:f] + parts[f + 1:]))
        n_fit = int(round(fit_fraction * len(rest)))
        out.append(QuerySplit(np.sort(rest[:n_fit]), np.sort(rest[n_fit:]), np.sort(parts[f])))
    return out


def best_docs(docs: np.ndarray, scores: np.ndarray, n: int) -> np.ndarray:
    """The n entries of `docs` with the highest `scores`, best first; ties go to the smaller doc index."""
    if len(scores) > n:
        cand = np.flatnonzero(scores >= np.partition(scores, len(scores) - n)[len(scores) - n])
    else:
        cand = np.arange(len(scores))
    return docs[cand[np.lexsort((docs[cand], -scores[cand]))][:n]]


def exact_topk(scores: np.ndarray, k: int) -> np.ndarray:
    """(n_queries, k) indices of the k best documents per row, best first (ties by doc index)."""
    docs = np.arange(scores.shape[1])
    return np.stack([best_docs(docs, row, k) for row in scores])


def topk_ranks(est: np.ndarray, topk: np.ndarray) -> np.ndarray:
    """(n_queries, k) position of every exact top-k document in the stable descending order of `est`.

    position(d) = #{j : est_j > est_d} + #{j < d : est_j = est_d}, i.e. np.argsort(-est, kind="stable").
    """
    idx = np.arange(est.shape[1])
    out = np.empty(topk.shape, dtype=np.int64)
    for q in range(len(est)):
        row, docs = est[q], topk[q]
        vals = row[docs][:, None]
        out[q] = (row[None, :] > vals).sum(1) + ((row[None, :] == vals) & (idx[None, :] < docs[:, None])).sum(1)
    return out


def fit_linear_map(est: np.ndarray, exact: np.ndarray, rows: np.ndarray, top: int) -> tuple[float, float]:
    """Least-squares exact ~ a * est + b on each fit query's `top` highest estimates (a > 0)."""
    top = min(top, est.shape[1] - 1)
    xs, ys = [], []
    for q in rows:
        head = np.argpartition(-est[q], top)[:top]
        xs.append(est[q, head])
        ys.append(exact[q, head])
    x, y = np.concatenate(xs).astype(np.float64), np.concatenate(ys).astype(np.float64)
    a, b = np.linalg.lstsq(np.c_[x, np.ones_like(x)], y, rcond=None)[0]
    return (float(a), float(b)) if a > 0 else (1.0, 0.0)


def conformal_quantile(values: np.ndarray, delta: float) -> float:
    """Split-conformal threshold: the ceil((n+1)(1-delta))-th smallest value (inf if that exceeds n)."""
    n = len(values)
    rank = math.ceil(round((n + 1) * (1.0 - delta), 9))
    return math.inf if rank > n else float(np.sort(values)[rank - 1])


def recall_quantile_r(ranks: np.ndarray, target: float) -> int:
    """Smallest R such that a fraction >= target of the given top-k positions is < R (descriptive)."""
    flat = np.sort(ranks.ravel())
    return int(flat[max(0, math.ceil(round(target * len(flat), 9)) - 1)]) + 1


def nonconformity(est: np.ndarray, exact: np.ndarray, topk: np.ndarray, rows: np.ndarray, a: float, b: float,
                  s_next: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Bound-based scores of the calibration rows; `s_next` (n_queries,) holds s*_(k+1) for rank1 / strict1."""
    shat = a * est[rows[:, None], topk[rows]] + b            # (n_rows, k) calibrated estimates of true top-k
    s_top = exact[rows[:, None], topk[rows]]                 # (n_rows, k) exact scores
    s_kth = s_top[:, -1:]                                    # k-th best exact score
    out = {"naive": (s_top - shat).ravel(), "rank": (s_kth - shat).ravel(), "strict": s_kth[:, 0] - shat.min(axis=1)}
    if s_next is not None:
        out["rank1"] = (s_next[rows, None] - shat).ravel()
        out["strict1"] = s_next[rows] - shat.min(axis=1)
    return out


@njit(cache=True)
def adaptive_stop(est_sorted: np.ndarray, exact_sorted: np.ndarray, a: float, b: float, tau: float,
                  k: int, batch: int) -> int:
    """Documents recomputed before theta_k - (a * est(next) + b) >= tau (reference simulator).

    Both arrays are in processing order (decreasing estimate); only the visited prefix is read.
    """
    n = len(est_sorted)
    best = np.full(k, -np.inf)
    p = 0
    while p < n:
        end = min(p + batch, n)
        for j in range(p, end):
            lo = 0
            for t in range(1, k):
                if best[t] < best[lo]:
                    lo = t
            if exact_sorted[j] > best[lo]:
                best[lo] = exact_sorted[j]
        p = end
        if p >= k and p < n:
            theta = best[0]
            for t in range(1, k):
                theta = min(theta, best[t])
            if theta - (a * est_sorted[p] + b) >= tau:
                break
    return p


@njit(cache=True, nogil=True)
def _scan_path(est_sorted: np.ndarray, exact_sorted: np.ndarray, k: int, batch: int):
    n = len(est_sorted)
    m = max(0, (n - 1) // batch + 1)
    ps = np.empty(m, np.int64)
    theta = np.empty(m, np.float64)
    nxt = np.empty(m, np.float64)
    best = np.full(k, -np.inf)
    p = 0
    c = 0
    while p < n:
        end = min(p + batch, n)
        for j in range(p, end):
            lo = 0
            for t in range(1, k):
                if best[t] < best[lo]:
                    lo = t
            if exact_sorted[j] > best[lo]:
                best[lo] = exact_sorted[j]
        p = end
        if p >= k and p < n:
            th = best[0]
            for t in range(1, k):
                th = min(th, best[t])
            ps[c] = p
            theta[c] = th
            nxt[c] = est_sorted[p]
            c += 1
    return ps[:c], theta[:c], nxt[:c]


@njit(cache=True, nogil=True)
def _scan_state(est_sorted: np.ndarray, exact_sorted: np.ndarray, k: int, batch: int, ahead: int):
    """_scan_path plus what the hazard features need at every boundary: the estimate `ahead` positions
    further, running means of exact scores and estimates, and recomputations since the top-k changed."""
    n = len(est_sorted)
    m = max(0, (n - 1) // batch + 1)
    ps = np.empty(m, np.int64)
    theta = np.empty(m, np.float64)
    nxt = np.empty(m, np.float64)
    fwd = np.empty(m, np.float64)
    mean_s = np.empty(m, np.float64)
    mean_e = np.empty(m, np.float64)
    since = np.empty(m, np.int64)
    best = np.full(k, -np.inf)
    last, p, c = -1, 0, 0
    sum_s, sum_e = 0.0, 0.0
    while p < n:
        end = min(p + batch, n)
        for j in range(p, end):
            lo = 0
            for t in range(1, k):
                if best[t] < best[lo]:
                    lo = t
            if exact_sorted[j] > best[lo]:
                best[lo] = exact_sorted[j]
                last = j
            sum_s += exact_sorted[j]
            sum_e += est_sorted[j]
        p = end
        if p >= k and p < n:
            th = best[0]
            for t in range(1, k):
                th = min(th, best[t])
            ps[c] = p
            theta[c] = th
            nxt[c] = est_sorted[p]
            fwd[c] = est_sorted[min(p + ahead, n - 1)]
            mean_s[c] = sum_s / p
            mean_e[c] = sum_e / p
            since[c] = p - (last + 1)
            c += 1
    return ps[:c], theta[:c], nxt[:c], fwd[:c], mean_s[:c], mean_e[:c], since[:c]


class StopPath:
    """Everything the stopping rules need about one query: positions of its exact top-k in the
    processing order, and theta_k / next estimate at every batch boundary (plus, with state=True,
    the hazard features' raw ingredients, which are affine in the score map (a, b))."""

    def __init__(self, est_row: np.ndarray, exact_row: np.ndarray, topk_row: np.ndarray, k: int, batch: int,
                 order: np.ndarray | None = None, state: bool = False, ahead: int = 64) -> None:
        order = np.argsort(-est_row, kind="stable") if order is None else order
        self.n = len(order)
        inv = np.empty(self.n, dtype=np.int64)
        inv[order] = np.arange(self.n)
        self.ranks = inv[topk_row]
        est_sorted, exact_sorted = est_row[order].astype(np.float64), exact_row[order].astype(np.float64)
        self.state = state
        if not state:
            ps, theta, nxt = _scan_path(est_sorted, exact_sorted, k, batch)
        else:
            ps, theta, nxt, fwd, mean_s, mean_e, since = _scan_state(est_sorted, exact_sorted, k, batch, ahead)
            self.fwd, self.mean_s, self.mean_e = fwd.astype(np.float32), mean_s.astype(np.float32), mean_e.astype(np.float32)
            self.since = since.astype(np.int32)
            self.est_top, self.ahead = float(est_sorted[0]) if self.n else 0.0, ahead
        self.ps, self.theta, self.nxt = ps.astype(np.int32), theta.astype(np.float32), nxt.astype(np.float32)

    def gmax(self, a: float, b: float) -> np.ndarray:
        """Running max of theta_k - s_hat(next): the rule with margin tau stops at the first boundary where it is >= tau."""
        g = self.theta.astype(np.float64) - (a * self.nxt.astype(np.float64) + b)
        return np.maximum.accumulate(g) if len(g) else g

    def patience_gmax(self) -> np.ndarray:
        """Running max of the recomputations since the top-k last changed (the patience rule's statistic)."""
        return np.maximum.accumulate(self.since.astype(np.float64)) if len(self.since) else np.empty(0)

    def features(self, a: float, b: float) -> np.ndarray:
        """(n_boundaries, 7) hazard state under the score map s_hat = a * E + b (hazard.FEATURES)."""
        theta, nxt = self.theta.astype(np.float64), self.nxt.astype(np.float64)
        shat_next = a * nxt + b
        offset = self.mean_s.astype(np.float64) - a * self.mean_e.astype(np.float64)
        return np.column_stack([theta - shat_next, theta - (a * nxt + offset), np.log(self.ps.astype(np.float64)),
                                np.log1p(self.since.astype(np.float64)), a * (nxt - self.fwd.astype(np.float64)),
                                (a * self.est_top + b) - theta, np.full(len(theta), np.log(max(self.n, 1)))])

    def hazard_labels(self, horizon: int) -> np.ndarray:
        """1 where an exact top-k document sits within the next `horizon` positions of the boundary."""
        r = np.sort(self.ranks)
        ps = self.ps.astype(np.int64)
        return np.searchsorted(r, ps + horizon, side="left") > np.searchsorted(r, ps, side="left")

    def stop(self, gmax: np.ndarray, tau: float) -> int:
        i = int(np.searchsorted(gmax, tau, side="left"))
        return int(self.ps[i]) if i < len(self.ps) else self.n

    def loss(self, p: int, strict: bool) -> float:
        missed = self.ranks >= p
        return float(missed.any()) if strict else float(missed.mean())


def crc_margin(paths: list[StopPath], gmaxes: list[np.ndarray], delta: float, strict: bool, iters: int = 64,
               correction: str = "query") -> float:
    """Smallest tau with (sum_i loss_i(tau) + 1) / (n + 1) <= delta (inf if even recomputing everything fails).

    correction="doc" pools the k top-k documents of every query instead: (k sum_i loss_i + 1) / (n k + 1),
    the per-document finite-sample correction of the pooled rank / rank1 quantiles (approximately valid;
    used only to compare like for like with them)."""
    n = len(paths)
    k = len(paths[0].ranks) if paths and correction == "doc" and not strict else 1

    def ok(tau: float) -> bool:
        total = sum(q.loss(q.stop(g, tau), strict) for q, g in zip(paths, gmaxes))
        return (k * total + 1.0) / (n * k + 1) <= delta

    tops = [float(g[-1]) for g in gmaxes if len(g)]
    hi = (max(tops) if tops else 0.0) + 1.0
    if not ok(hi):
        return math.inf
    lows = [float(g[0]) for g in gmaxes if len(g)]
    lo = (min(lows) if lows else 0.0) - 1.0
    if ok(lo):
        return lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


def crc_budget(ranks: np.ndarray, n_docs: int, delta: float, strict: bool, correction: str = "query") -> int:
    """Smallest fixed R with (sum_i loss_i(R) + 1) / (n + 1) <= delta; loss = missed fraction (or any miss).

    correction="doc" pools the k top-k documents of every query, as crc_margin does: (k sum_i loss_i + 1) / (n k + 1)."""
    n, k = ranks.shape
    worst = np.sort(ranks.max(axis=1)) if strict else np.sort(ranks.ravel())
    cand = np.unique(np.r_[worst + 1, n_docs])
    beyond = len(worst) - np.searchsorted(worst, cand, side="left")      # entries with rank >= R
    total = beyond if strict else beyond / k
    units = k if correction == "doc" and not strict else 1
    ok = np.flatnonzero((units * total + 1.0) / (n * units + 1) <= delta)
    return int(cand[ok[0]]) if len(ok) else n_docs


@dataclass
class FoldCalibration:
    a: float
    b: float
    tau: dict[float, dict[str, float]]   # delta -> adaptive rule -> margin / patience W / hazard lambda
    budget: dict[float, dict[str, int]]  # delta -> fixed rule -> R
    hazard: HazardModel | None = None    # fitted on this fold's fit queries
    universal: HazardModel | None = None

    def to_dict(self) -> dict:
        out = {"a": self.a, "b": self.b, "tau": {str(d): v for d, v in self.tau.items()},
               "fixed_R": {str(d): v for d, v in self.budget.items()}}
        if self.hazard is not None:
            out["hazard"] = self.hazard.to_dict()
        return out

    def statistic(self, name: str, path: StopPath) -> np.ndarray:
        """The running statistic a rule thresholds on this path (margin, patience or hazard)."""
        if name == "patience":
            return path.patience_gmax()
        if name in ("hazard", "hazard_u"):
            return hazard_gmax(self.hazard if name == "hazard" else self.universal, path, self.a, self.b)
        return path.gmax(self.a, self.b)


def calibrate(est: np.ndarray, exact: np.ndarray, topk: np.ndarray, s_next: np.ndarray, paths: dict[int, StopPath],
              split: QuerySplit, deltas: list[float], calib_top: int,
              hazard: HazardSettings | None = None) -> FoldCalibration:
    """Fit the score map and the hazard model on split.fit; margins, thresholds and budgets on split.cal."""
    hazard = hazard or HazardSettings()
    a, b = fit_linear_map(est, exact, split.fit, calib_top)
    scores = nonconformity(est, exact, topk, split.cal, a, b, s_next)
    cal = [paths[q] for q in split.cal]
    fit = [paths[q] for q in split.fit]
    model = fit_hazard(fit, [(a, b)] * len(fit), hazard) if fit else None
    fc = FoldCalibration(a, b, {}, {}, model, hazard.universal)
    stats = {"margin": [p.gmax(a, b) for p in cal], "patience": [p.patience_gmax() for p in cal]}
    if model is not None:
        stats["hazard"] = [fc.statistic("hazard", p) for p in cal]
    if hazard.universal is not None:
        stats["hazard_u"] = [fc.statistic("hazard_u", p) for p in cal]
    ranks = np.stack([p.ranks for p in cal])
    for d in deltas:
        fc.tau[d] = {name: conformal_quantile(vals, d) for name, vals in scores.items()}
        fc.tau[d]["crc"] = crc_margin(cal, stats["margin"], d, strict=False)
        fc.tau[d]["crc_strict"] = crc_margin(cal, stats["margin"], d, strict=True)
        fc.tau[d]["patience"] = crc_margin(cal, stats["patience"], d, strict=False)
        if model is not None:
            fc.tau[d]["hazard"] = crc_margin(cal, stats["hazard"], d, strict=False)
            fc.tau[d]["hazard_doc"] = crc_margin(cal, stats["hazard"], d, strict=False, correction="doc")
        if hazard.universal is not None:
            fc.tau[d]["hazard_u"] = crc_margin(cal, stats["hazard_u"], d, strict=False)
        fc.budget[d] = {"fixed": crc_budget(ranks, est.shape[1], d, strict=False),
                        "fixed_strict": crc_budget(ranks, est.shape[1], d, strict=True)}
    return fc


def _recall_grid(n_docs: int, points: int = 40) -> np.ndarray:
    return np.unique(np.round(np.geomspace(10, n_docs, points)).astype(np.int64))


def evaluate_stopping(est: np.ndarray, exact: np.ndarray, topk: np.ndarray, splits: QuerySplit | list[QuerySplit],
                      deltas: list[float], k: int, batch: int, calib_top: int,
                      qids: list[str], doc_ids: list[str], qrels: dict[str, dict[str, int]],
                      recall_targets: tuple[float, ...] = (0.9, 0.95, 0.99),
                      hazard: HazardSettings | None = None) -> dict:
    """Estimator-only quality plus every stopping rule, reported over the union of the test sets.

    Per-query outputs (R_list, per_query) follow the order of the returned `test_rows`.
    """
    hazard = hazard or HazardSettings()
    splits = [splits] if isinstance(splits, QuerySplit) else list(splits)
    n_docs = est.shape[1]
    test_rows = np.concatenate([s.test for s in splits])
    if len(np.unique(test_rows)) != len(test_rows):
        raise ValueError("test sets of the splits overlap")
    needed = np.unique(np.concatenate([np.concatenate([s.fit, s.cal, s.test]) for s in splits]))
    s_next = np.full(len(est), -np.inf)
    if n_docs > k:
        s_next[needed] = [np.partition(exact[q], n_docs - k - 1)[n_docs - k - 1] for q in needed]
    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as pool:
        paths = dict(zip(needed.tolist(), pool.map(
            lambda q: StopPath(est[q], exact[q], topk[q], k, batch, state=True, ahead=hazard.ahead), needed)))
        calibs = [calibrate(est, exact, topk, s_next, paths, s, deltas, calib_top, hazard) for s in splits]
        fold_of = np.concatenate([np.full(len(s.test), f) for f, s in enumerate(splits)])
        rules = [r for r in RULES + ("hazard_u",)
                 if all(r in c.budget[deltas[0]] or r in c.tau[deltas[0]] for c in calibs)]

        def run_query(i: int):
            q, c, path = test_rows[i], calibs[fold_of[i]], paths[test_rows[i]]
            order = np.argsort(-est[q], kind="stable")
            x = exact[q, order]
            head = order[:min(calib_top, n_docs)]
            std = float((exact[q, head] - (c.a * est[q, head].astype(np.float64) + c.b)).std())
            stats: dict[str, np.ndarray] = {}
            memo: dict[int, np.ndarray] = {}
            out = {}
            for d in deltas:
                for r in rules:
                    if r in c.budget[d]:
                        p = c.budget[d][r]
                    else:
                        name = STATISTIC.get(r, "margin")
                        if name not in stats:
                            stats[name] = c.statistic(name, path)
                        p = path.stop(stats[name], c.tau[d][r])
                    if p not in memo:
                        memo[p] = best_docs(order[:p], x[:p], 10)
                    out[d, r] = (p, memo[p])
            return order[:10], std, out

        per_query = list(pool.map(run_query, range(len(test_rows))))

    est_rank = [r[0] for r in per_query]
    head_std = np.array([r[1] for r in per_query])
    stops = {(d, r): np.array([pq[2][d, r][0] for pq in per_query], np.int64) for d in deltas for r in rules}
    rankings = {(d, r): [pq[2][d, r][1] for pq in per_query] for d in deltas for r in rules}

    test_ranks = np.stack([paths[q].ranks for q in test_rows])
    test_qids = [qids[i] for i in test_rows]
    est_ndcg = ndcg_per_query(est_rank, test_qids, doc_ids, qrels, 10)
    grid = _recall_grid(n_docs)
    result = {
        "test_rows": test_rows.tolist(),
        "calibration": [c.to_dict() for c in calibs],
        "head_error_std": float(head_std.mean()),
        "estimate_ndcg10": float(est_ndcg.mean()),
        "estimate_recall10": float((test_ranks < 10).mean()),
        "R_at_recall": {str(t): recall_quantile_r(test_ranks, t) for t in recall_targets},
        "recall_curve": {"R": grid.tolist(), "recall": [float((test_ranks < r).mean()) for r in grid]},
        "per_query": {"estimate_ndcg10": est_ndcg.tolist(), "ndcg10": {}},
        "deltas": {},
    }
    for d in deltas:
        entry, per_query = {}, {}
        for r in rules:
            rs = stops[d, r]
            found = test_ranks < rs[:, None]
            ndcg = ndcg_per_query(rankings[d, r], test_qids, doc_ids, qrels, 10)
            param = [c.budget[d][r] if r in c.budget[d] else c.tau[d][r] for c in calibs]
            entry[r] = {"param": param, "R_mean": float(rs.mean()), "R_p50": float(np.percentile(rs, 50)),
                        "R_p90": float(np.percentile(rs, 90)), "R_max": int(rs.max()),
                        "recall": float(found.mean()), "full_recall_rate": float(found.all(axis=1).mean()),
                        "ndcg10": float(ndcg.mean()), "R_list": rs.tolist()}
            per_query[r] = ndcg.tolist()
        result["deltas"][str(d)] = entry
        result["per_query"]["ndcg10"][str(d)] = per_query
    return result
