"""Exact MaxSim scoring: s(q, d) = sum_i max_{x in d} <q_i, x>."""
from __future__ import annotations

import numpy as np
import torch

from .store import TokenStore
from ..utils import get_logger

log = get_logger(__name__)


@torch.inference_mode()
def maxsim_matrix(store: TokenStore, queries: np.ndarray, device: torch.device,
                  queries_per_pass: int = 128, max_block_elems: int = 1 << 28,
                  resident_bytes: int = 6 << 30) -> np.ndarray:
    """(n_queries, n_docs) exact scores for all queries against the whole store.

    Token vectors are scanned in blocks of whole documents; a block's (tokens x query-tokens)
    similarities are reduced per document with scatter_reduce('amax'). The store is uploaded
    once when it fits in `resident_bytes`, otherwise streamed block by block.
    """
    nq, m, dim = queries.shape
    n_docs, offsets = store.n_docs, store.offsets
    out = np.empty((nq, n_docs), dtype=np.float32)
    resident = store.vectors.nbytes <= resident_bytes
    all_vecs = torch.from_numpy(np.array(store.vectors)).to(device) if resident else None
    lengths = torch.from_numpy(store.lengths).to(device)
    q_all = torch.from_numpy(np.ascontiguousarray(queries, dtype=np.float32)).to(device)

    for qs in range(0, nq, queries_per_pass):
        qb = q_all[qs:qs + queries_per_pass]
        cols = qb.shape[0] * m
        q_mat = qb.reshape(cols, dim)
        tok_budget = max(1, max_block_elems // cols)
        d0 = 0
        while d0 < n_docs:
            d1 = int(np.searchsorted(offsets, offsets[d0] + tok_budget, side="right")) - 1
            d1 = min(max(d1, d0 + 1), n_docs)
            a, b = int(offsets[d0]), int(offsets[d1])
            block = all_vecs[a:b] if resident else torch.from_numpy(np.array(store.vectors[a:b])).to(device)
            sims = block.float() @ q_mat.T                                   # (tokens, cols)
            seg = torch.repeat_interleave(torch.arange(d1 - d0, device=device), lengths[d0:d1])
            best = torch.full((d1 - d0, cols), -torch.inf, device=device)
            best.scatter_reduce_(0, seg[:, None].expand(-1, cols), sims, reduce="amax", include_self=True)
            out[qs:qs + qb.shape[0], d0:d1] = best.view(d1 - d0, qb.shape[0], m).sum(-1).T.cpu().numpy()
            d0 = d1
    return out


@torch.inference_mode()
def maxsim_padded(query: torch.Tensor, docs: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Scores of one query (m, dim) against a padded document batch (B, L, dim) with mask (B, L)."""
    sims = docs @ query.T                                                    # (B, L, m)
    sims = sims.masked_fill(~keep[..., None], -torch.inf)
    return sims.max(dim=1).values.sum(dim=-1)
