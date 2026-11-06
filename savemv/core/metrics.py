"""Retrieval metrics."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def ndcg_per_query(rankings: Sequence[np.ndarray], qids: list[str], doc_ids: list[str],
                   qrels: dict[str, dict[str, int]], k: int = 10) -> np.ndarray:
    """nDCG@k per query with linear gains (as pytrec_eval / BEIR). Rankings hold doc indices, best first."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    values = np.zeros(len(qids))
    for i, (row, qid) in enumerate(zip(rankings, qids)):
        rel = qrels.get(qid, {})
        gains = np.array([rel.get(doc_ids[d], 0) for d in row[:k]], dtype=np.float64)
        ideal = np.sort(np.fromiter(rel.values(), dtype=np.float64))[::-1][:k]
        idcg = float((ideal * discounts[:len(ideal)]).sum())
        values[i] = float((gains * discounts[:len(gains)]).sum()) / idcg if idcg > 0 else 0.0
    return values


def ndcg_at_k(rankings: Sequence[np.ndarray], qids: list[str], doc_ids: list[str],
              qrels: dict[str, dict[str, int]], k: int = 10) -> float:
    """Mean nDCG@k over queries."""
    values = ndcg_per_query(rankings, qids, doc_ids, qrels, k)
    return float(values.mean()) if len(values) else 0.0


def recall_vs_exact(found: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of the exact top-k (`truth`) that appears in `found`."""
    return len(set(found.tolist()) & set(truth.tolist())) / max(1, len(truth))
