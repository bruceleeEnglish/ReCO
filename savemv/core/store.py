"""Token-vector storage in CSR layout (one contiguous fp16 matrix + document offsets).

The store holds the full multi-vector index. ReCo never needs it at query time; it is kept
to compute exact ground truth and to simulate recomputation cheaply during evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TokenStore:
    vectors: np.ndarray     # (n_tokens, dim) float16, L2-normalized rows
    offsets: np.ndarray     # (n_docs + 1,) int64; doc i owns vectors[offsets[i]:offsets[i+1]]
    token_ids: np.ndarray   # (n_tokens,) int32 wordpiece ids of the kept tokens

    def __post_init__(self) -> None:
        if self.offsets[0] != 0 or self.offsets[-1] != len(self.vectors):
            raise ValueError("offsets do not cover the vector matrix")
        if len(self.token_ids) != len(self.vectors):
            raise ValueError("token_ids and vectors differ in length")

    @property
    def n_docs(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_tokens(self) -> int:
        return len(self.vectors)

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def doc(self, i: int) -> np.ndarray:
        return self.vectors[self.offsets[i]:self.offsets[i + 1]]

    def doc_of_token(self) -> np.ndarray:
        """(n_tokens,) document index of every token."""
        return np.repeat(np.arange(self.n_docs, dtype=np.int64), self.lengths)

    @classmethod
    def from_docs(cls, docs: list[np.ndarray], token_ids: list[np.ndarray]) -> "TokenStore":
        lengths = np.fromiter((len(d) for d in docs), dtype=np.int64, count=len(docs))
        offsets = np.zeros(len(docs) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        return cls(np.concatenate(docs).astype(np.float16, copy=False), offsets,
                   np.concatenate(token_ids).astype(np.int32, copy=False))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        np.save(directory / "offsets.npy", self.offsets)
        np.save(directory / "token_ids.npy", self.token_ids)

    @classmethod
    def load(cls, directory: Path, mmap: bool = True) -> "TokenStore":
        mode = "r" if mmap else None
        return cls(np.load(directory / "vectors.npy", mmap_mode=mode), np.load(directory / "offsets.npy"),
                   np.load(directory / "token_ids.npy", mmap_mode=mode))

    @staticmethod
    def exists(directory: Path) -> bool:
        return all((directory / f).exists() for f in ("vectors.npy", "offsets.npy", "token_ids.npy"))
