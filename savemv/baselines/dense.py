"""Single-vector baselines: a dense bi-encoder (BGE-style, CLS pooling) and mean-pooled ColBERT tokens,
each stored as fp16, 8-bit scalar-quantized, or 1-bit (sign) codes scored asymmetrically.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..core.store import TokenStore

QUANTIZATIONS = ("fp16", "int8", "binary")


def quantize(x: np.ndarray, mode: str) -> tuple[np.ndarray, float, float]:
    """Reconstruction used for scoring, bytes per vector, shared bytes (quantizer tables)."""
    d = x.shape[1]
    if mode == "fp16":
        return x.astype(np.float16).astype(np.float32), 2.0 * d, 0.0
    if mode == "int8":
        lo, hi = x.min(axis=0), x.max(axis=0)
        scale = np.maximum(hi - lo, 1e-12) / 255.0
        codes = np.clip(np.rint((x - lo) / scale), 0, 255)
        return (lo + codes * scale).astype(np.float32), float(d), 2.0 * d * 4
    if mode == "binary":
        return np.where(x > 0, 1.0, -1.0).astype(np.float32), d / 8.0, 0.0
    raise ValueError(f"unknown quantization {mode!r}; expected one of {QUANTIZATIONS}")


@torch.inference_mode()
def mean_pool_store(store: TokenStore, device: torch.device, chunk: int = 1 << 22) -> np.ndarray:
    """(n_docs, dim) normalized mean of every document's token vectors."""
    sums = torch.zeros(store.n_docs, store.dim, device=device)
    doc = store.doc_of_token()
    for s in range(0, store.n_tokens, chunk):
        x = torch.from_numpy(np.asarray(store.vectors[s:s + chunk], dtype=np.float32)).to(device)
        sums.index_add_(0, torch.from_numpy(doc[s:s + chunk]).to(device), x)
    return F.normalize(sums, dim=1).cpu().numpy()


class DenseEncoder:
    """CLS-pooled, L2-normalized bi-encoder (e.g. BAAI/bge-small-en-v1.5)."""

    def __init__(self, path: str, device: torch.device, fp16: bool = True, batch_size: int = 128,
                 max_length: int = 512, query_prefix: str = "") -> None:
        from transformers import AutoModel, AutoTokenizer
        self.device, self.batch_size, self.max_length, self.query_prefix = device, batch_size, max_length, query_prefix
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(device).eval()
        if fp16 and device.type == "cuda":
            self.model.half()

    @torch.inference_mode()
    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        if query:
            texts = [self.query_prefix + t for t in texts]
        order = np.argsort([len(t) for t in texts], kind="stable")
        out = np.empty((len(texts), self.model.config.hidden_size), dtype=np.float32)
        for s in range(0, len(texts), self.batch_size):
            idx = order[s:s + self.batch_size]
            tok = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            cls = self.model(**tok).last_hidden_state[:, 0].float()
            out[idx] = F.normalize(cls, dim=-1).cpu().numpy()
        return out
