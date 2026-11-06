"""MUVERA fixed-dimensional encodings (Dhulipala et al., NeurIPS 2024) and PQ compression.

Follows google/graph-mining sketching/point_cloud/fixed_dimensional_encoding.cc: per repetition,
SimHash partitions (Gray-coded ids), an AMS-sketch inner projection, per-partition sums for queries
and centroids for documents, and (optionally) empty document partitions filled with the point
whose SimHash bits are closest in Hamming distance. Random matrices use the seeding of the
muvera-py port (numpy default_rng(seed + repetition)), so outputs match it exactly when
fill_empty is off; for the fill, we follow the C++ code (the port decodes partition ids differently).
Documents are encoded on a torch device, many at a time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class FDEConfig:
    reps: int = 20
    simhash_bits: int = 5
    proj_dim: int = 16          # AMS-sketch inner projection; 0 keeps the input dimension (identity)
    fill_empty: bool = True     # documents only
    seed: int = 42

    def out_dim(self, dim: int) -> int:
        return self.reps * 2 ** self.simhash_bits * (self.proj_dim or dim)


def _rep_matrices(dim: int, cfg: FDEConfig, rep: int) -> tuple[np.ndarray, np.ndarray | None]:
    """SimHash Gaussian matrix (dim, bits) and AMS projection (dim, proj) of one repetition."""
    seed = cfg.seed + rep
    simhash = np.random.default_rng(seed).normal(0.0, 1.0, size=(dim, cfg.simhash_bits)).astype(np.float32)
    if not cfg.proj_dim:
        return simhash, None
    rng = np.random.default_rng(seed)
    index = rng.integers(0, cfg.proj_dim, size=dim)      # drawn before the signs, as in muvera-py
    signs = rng.choice([-1.0, 1.0], size=dim)
    ams = np.zeros((dim, cfg.proj_dim), dtype=np.float32)
    ams[np.arange(dim), index] = signs
    return simhash, ams


def _popcount_table(bits: int) -> torch.Tensor:
    v = torch.arange(2 ** bits)
    return sum((((v >> i) & 1) for i in range(bits)), torch.zeros_like(v))


@torch.inference_mode()
def encode_fde(points: torch.Tensor, seg: torch.Tensor, n_sets: int, cfg: FDEConfig, document: bool) -> torch.Tensor:
    """FDEs of `n_sets` point sets; `points` (N, dim) float32, `seg` (N,) set index, sorted ascending.

    document=True averages each partition (and fills empty ones if cfg.fill_empty); queries sum.
    """
    dev, dim = points.device, points.shape[1]
    parts = 2 ** cfg.simhash_bits
    proj = cfg.proj_dim or dim
    weights = (2 ** torch.arange(cfg.simhash_bits - 1, -1, -1, device=dev))
    popcount = _popcount_table(cfg.simhash_bits).to(dev)
    gray = torch.arange(parts, device=dev)
    target = gray ^ (gray >> 1)                        # sign pattern of each partition id (MSB = first projection)
    out = torch.zeros(n_sets, cfg.reps, parts, proj, device=dev)
    for rep in range(cfg.reps):
        simhash, ams = _rep_matrices(dim, cfg, rep)
        bits = (points @ torch.from_numpy(simhash).to(dev) > 0).long()           # (N, bits)
        part = torch.zeros(len(points), dtype=torch.long, device=dev)
        for j in range(cfg.simhash_bits):                                          # AppendToGrayCode
            part = (part << 1) + (bits[:, j] ^ (part & 1))
        x = points if ams is None else points @ torch.from_numpy(ams).to(dev)
        key = seg * parts + part
        block = torch.zeros(n_sets * parts, proj, device=dev).index_add_(0, key, x)
        if document:
            counts = torch.bincount(key, minlength=n_sets * parts).float()
            block = block / counts.clamp_min(1)[:, None]
            if cfg.fill_empty:
                empty = counts == 0
                if empty.any():
                    pattern = (bits * weights).sum(1)                                    # (N,)
                    ham = popcount[pattern[:, None] ^ target[None, :]]                   # (N, parts)
                    local = torch.arange(len(points), device=dev)
                    code = ham * (1 << 40) + local[:, None]                              # min distance, then first point
                    best = torch.full((n_sets, parts), torch.iinfo(torch.long).max, device=dev)
                    best.scatter_reduce_(0, seg[:, None].expand(-1, parts), code, reduce="amin")
                    nearest = (best & ((1 << 40) - 1)).flatten()
                    block[empty] = x[nearest[empty]]
        out[:, rep] = block.view(n_sets, parts, proj)
    return out.view(n_sets, -1)


def query_fdes(queries: np.ndarray, cfg: FDEConfig, device: torch.device) -> np.ndarray:
    """(n, D) query FDEs from (n, m, dim) query token vectors."""
    n, m, dim = queries.shape
    pts = torch.from_numpy(np.ascontiguousarray(queries, dtype=np.float32).reshape(n * m, dim)).to(device)
    seg = torch.arange(n, device=device).repeat_interleave(m)
    return encode_fde(pts, seg, n, cfg, document=False).cpu().numpy()


def document_fdes(vectors: np.ndarray, offsets: np.ndarray, docs: np.ndarray, cfg: FDEConfig,
                  device: torch.device) -> torch.Tensor:
    """(len(docs), D) FDEs (on `device`) of the given document indices of a CSR token store."""
    lengths = offsets[docs + 1] - offsets[docs]
    rows = np.concatenate([np.arange(offsets[d], offsets[d + 1]) for d in docs])
    pts = torch.from_numpy(np.asarray(vectors[rows], dtype=np.float32)).to(device)
    seg = torch.from_numpy(np.repeat(np.arange(len(docs)), lengths)).to(device)
    return encode_fde(pts, seg, len(docs), cfg, document=True)


@dataclass
class ProductQuantizer:
    """PQ with `sub_dim`-dimensional subspaces and 2**8 L2 k-means centroids each (MUVERA's PQ-256-8)."""
    codebooks: np.ndarray        # (S, C, sub_dim) float32

    @property
    def sub_dim(self) -> int:
        return self.codebooks.shape[2]

    @classmethod
    @torch.inference_mode()
    def fit(cls, x: torch.Tensor, sub_dim: int = 8, n_centroids: int = 256, iters: int = 20, seed: int = 0,
            subspaces_per_pass: int = 16) -> "ProductQuantizer":
        n, d = x.shape
        if d % sub_dim:
            raise ValueError(f"dimension {d} is not a multiple of {sub_dim}")
        gen = torch.Generator().manual_seed(seed)
        books = []
        for s0 in range(0, d // sub_dim, subspaces_per_pass):
            s1 = min(s0 + subspaces_per_pass, d // sub_dim)
            xs = x[:, s0 * sub_dim:s1 * sub_dim].reshape(n, s1 - s0, sub_dim).transpose(0, 1).float()  # (S, n, g)
            init = torch.stack([torch.randperm(n, generator=gen)[:n_centroids] if n >= n_centroids
                                else torch.randint(0, n, (n_centroids,), generator=gen)
                                for _ in range(s1 - s0)]).to(x.device)
            cents = torch.gather(xs, 1, init[..., None].expand(-1, -1, sub_dim)).clone()
            for _ in range(iters):
                assign = torch.cdist(xs, cents).argmin(dim=2)                                  # (S, n)
                sums = torch.zeros_like(cents).scatter_add_(1, assign[..., None].expand(-1, -1, sub_dim), xs)
                counts = torch.zeros(cents.shape[:2], device=x.device).scatter_add_(1, assign,
                                                                                    torch.ones_like(assign, dtype=torch.float))
                cents = torch.where(counts[..., None] > 0, sums / counts.clamp_min(1)[..., None], cents)
            books.append(cents.cpu())
        return cls(torch.cat(books).numpy())

    @torch.inference_mode()
    def encode(self, x: torch.Tensor, subspaces_per_pass: int = 64) -> np.ndarray:
        """(n, S) uint8 codes."""
        n = len(x)
        s, _, g = self.codebooks.shape
        books = torch.from_numpy(self.codebooks).to(x.device)
        xs = x.float().reshape(n, s, g).transpose(0, 1)
        codes = [torch.cdist(xs[a:a + subspaces_per_pass], books[a:a + subspaces_per_pass]).argmin(dim=2)
                 for a in range(0, s, subspaces_per_pass)]
        return torch.cat(codes).T.to(torch.uint8).cpu().numpy()

    @torch.inference_mode()
    def decode(self, codes: np.ndarray, device: torch.device) -> torch.Tensor:
        """(n, D) float32 reconstruction on `device`."""
        books = torch.from_numpy(self.codebooks).to(device)                                     # (S, C, g)
        c = torch.from_numpy(codes.astype(np.int64)).to(device)                                 # (n, S)
        return books[torch.arange(books.shape[0], device=device)[None, :], c].reshape(len(codes), -1)

    def bytes_per_vector(self) -> float:
        return float(self.codebooks.shape[0])          # one byte per subspace

    def shared_bytes(self) -> float:
        return float(self.codebooks.size * 4)
