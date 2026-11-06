"""Online ReCo search: sketch estimate -> recompute candidates from raw text -> calibrated stop."""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

from ..utils import Timer
from ..core.exact import maxsim_padded
from .hazard import HazardModel


class DocumentEncoder(Protocol):
    """What the engine needs from an encoder (ColBERTEncoder satisfies it)."""
    device: torch.device

    def encode_query_tensor(self, text: str) -> torch.Tensor: ...

    def tokenize_docs(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]: ...

    def embed_docs(self, ids: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True)
class Calibration:
    a: float          # s_hat = a * estimate + b
    b: float
    tau: float        # threshold on the rule's statistic (inf for a pure fixed budget)
    k: int = 10
    batch: int = 8
    budget: int | None = None   # fixed rule: recompute exactly this many documents
    rule: str = "margin"        # statistic: margin theta_k - s_hat(next), patience, or hazard (-logit)
    hazard: HazardModel | None = None


@dataclass
class SearchResult:
    doc_indices: np.ndarray          # top-k documents, best first
    scores: np.ndarray               # their exact MaxSim scores
    n_recomputed: int
    recomputed: np.ndarray           # all recomputed doc indices, in processing order
    recomputed_scores: np.ndarray
    timings_ms: dict[str, float] = field(default_factory=dict)


class ReCoSearcher:
    """`estimator` is a CentroidBagEstimator (numpy) or TorchCentroidBagEstimator (on the encoder device).

    With `max_candidates`, only the top of the estimate order is materialized (a top-k selection instead of a
    full sort; for corpora of millions of documents); a query never recomputes more than that many documents.
    """

    def __init__(self, encoder: DocumentEncoder, estimator, doc_texts: list[str], calibration: Calibration,
                 max_candidates: int | None = None) -> None:
        self.encoder = encoder
        self.estimator = estimator
        self.doc_texts = doc_texts
        self.cal = calibration
        self.max_candidates = max_candidates
        # tokenizing the next batch on a worker thread overlaps the GPU forward of the current one
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="savemv-tokenize")

    def _estimate(self, q: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Processing order (decreasing estimate) and the calibrated estimates in that order."""
        a, b = self.cal.a, self.cal.b
        n = len(self.doc_texts) if self.max_candidates is None else min(self.max_candidates, len(self.doc_texts))
        if hasattr(self.estimator, "score_tensor"):
            shat = a * self.estimator.score_tensor(q) + b
            if n < len(shat):
                # selection, then a stable sort by (estimate desc, index asc): ties keep the offline order
                order = torch.topk(shat, n, sorted=False).indices.sort().values
                vals, pos = shat[order].sort(descending=True, stable=True)
                order = order[pos]
            else:
                order = torch.argsort(shat, descending=True, stable=True)
                vals = shat[order]
            return order.cpu().numpy(), vals.cpu().numpy()
        shat = a * self.estimator.score(q.cpu().numpy()) + b
        order = np.argsort(-shat, kind="stable")[:n]
        return order, shat[order]

    def _tokenize(self, order: np.ndarray, start: int) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        end = start + self.cal.batch if self.cal.budget is None else min(start + self.cal.batch, self.cal.budget)
        ids = order[start:end]
        return ids, self.encoder.tokenize_docs([self.doc_texts[i] for i in ids])

    def _statistic(self, vals: np.ndarray, p: int, theta: float, last: int, mean_resid: float) -> float:
        """The rule's statistic at boundary p (same definitions as StopPath / hazard.FEATURES)."""
        cal, nxt = self.cal, float(vals[p])
        if cal.rule == "margin":
            return theta - nxt
        since = p - (last + 1)
        if cal.rule == "patience":
            return float(since)
        fwd = float(vals[min(p + cal.hazard.ahead, len(vals) - 1)])
        state = [theta - nxt, theta - nxt - mean_resid, math.log(p), math.log1p(since), nxt - fwd,
                 float(vals[0]) - theta, math.log(len(self.doc_texts))]
        return float(cal.hazard.statistic(np.asarray([state]))[0])

    def search(self, query_text: str) -> SearchResult:
        dev, cal = self.encoder.device, self.cal
        with Timer(dev) as t_query:
            q = self.encoder.encode_query_tensor(query_text)
        with Timer(dev) as t_est:
            order, vals = self._estimate(q)
        n = len(order) if cal.budget is None else min(len(order), max(cal.budget, cal.k))
        scores = np.empty(n, dtype=np.float32)
        best = np.full(cal.k, -np.inf)           # running top-k, updated document by document as offline
        last, sum_resid = -1, 0.0                # last top-k change; sum of s - s_hat over recomputed docs
        p = 0
        with Timer(dev) as t_rec:
            pending = self._pool.submit(self._tokenize, order, 0)
            while p < n:
                ids, (tok_ids, mask) = pending.result()
                nxt = p + len(ids)
                if nxt < n:
                    pending = self._pool.submit(self._tokenize, order, nxt)
                emb, keep, _ = self.encoder.embed_docs(tok_ids, mask)
                batch_scores = maxsim_padded(q, emb, keep).float().cpu().numpy()
                scores[p:nxt] = batch_scores
                for j, s in enumerate(batch_scores.astype(np.float64)):
                    lo = int(np.argmin(best))
                    if s > best[lo]:
                        best[lo], last = s, p + j
                sum_resid += float(batch_scores.astype(np.float64).sum() - vals[p:nxt].astype(np.float64).sum())
                p = nxt
                if p >= cal.k and p < n and \
                        self._statistic(vals, p, float(best.min()), last, sum_resid / p) >= cal.tau:
                    pending.cancel()   # drop the prefetched batch if it has not started
                    break
        done = order[:p]
        top = np.argsort(-scores[:p], kind="stable")[:cal.k]
        return SearchResult(done[top], scores[top], p, done, scores[:p].copy(),
                            {"encode_query": t_query.ms, "estimate": t_est.ms, "recompute": t_rec.ms,
                             "total": t_query.ms + t_est.ms + t_rec.ms})
