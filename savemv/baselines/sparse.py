"""Inverted-index first stages: BM25 (bm25s) and SPLADE++ (Formal et al., SIGIR 2022).

Index size is estimated as a compressed inverted index: per term, document-id gaps in VByte plus
one payload byte per posting (term frequency or quantized impact).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from ..utils import get_logger

log = get_logger(__name__)


def postings_bytes(term_ptr: np.ndarray, doc_ids: np.ndarray, payload: int = 1) -> int:
    """Bytes of a term-major inverted index with VByte doc-id gaps and `payload` bytes per posting."""
    terms = np.repeat(np.arange(len(term_ptr) - 1), np.diff(term_ptr))
    order = np.lexsort((doc_ids, terms))
    t, d = terms[order], doc_ids[order].astype(np.int64)
    gaps = np.diff(d, prepend=-1)
    first = np.r_[True, t[1:] != t[:-1]]
    gaps[first] = d[first] + 1
    vbyte = (np.floor(np.log2(gaps)).astype(np.int64) // 7) + 1
    return int(vbyte.sum()) + payload * len(d)


class BM25:
    """Lucene-variant BM25 with English stopwords and Snowball stemming (bm25s)."""

    def __init__(self, k1: float = 0.9, b: float = 0.4) -> None:   # Pyserini's BEIR defaults
        import bm25s
        import Stemmer
        self._bm25s = bm25s
        self.stemmer = Stemmer.Stemmer("english")
        self.retriever = bm25s.BM25(method="lucene", k1=k1, b=b)

    def _tokenize(self, texts: list[str], ids: bool):
        return self._bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer, return_ids=ids,
                                    show_progress=False)

    def index(self, doc_texts: list[str]) -> None:
        self.retriever.index(self._tokenize(doc_texts, ids=True), show_progress=False)

    def scores(self, query_texts: list[str]) -> np.ndarray:
        """(n_queries, n_docs) float32 BM25 scores."""
        n_docs = self.retriever.scores["num_docs"]
        out = np.zeros((len(query_texts), n_docs), dtype=np.float32)
        for i, toks in enumerate(self._tokenize(query_texts, ids=False)):
            toks = [t for t in toks if t in self.retriever.vocab_dict]
            if toks:
                out[i] = self.retriever.get_scores(toks)
        return out

    def index_bytes(self) -> int:
        s = self.retriever.scores
        return postings_bytes(np.asarray(s["indptr"]), np.asarray(s["indices"]), payload=1)


class SpladeEncoder:
    """SPLADE max-pooled log-saturated MLM logits; works with naver/splade-cocondenser-ensembledistil."""

    def __init__(self, path: str, device: torch.device, fp16: bool = True, batch_size: int = 32,
                 max_length: int = 256) -> None:
        from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
        self.device, self.batch_size, self.max_length = device, batch_size, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(path))
        # transformers refuses pytorch_model.bin on torch < 2.6; load the tensors directly (weights only)
        state = torch.load(Path(path) / "pytorch_model.bin", map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        model.tie_weights()
        unexpected = [k for k in unexpected if not k.endswith("position_ids")]   # buffer saved by old versions
        if [k for k in missing if "decoder" not in k and "position_ids" not in k] or unexpected:
            raise RuntimeError(f"SPLADE checkpoint mismatch: missing={missing} unexpected={unexpected}")
        self.model = model.to(device).eval()
        if fp16 and device.type == "cuda":
            self.model.half()

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> sp.csr_matrix:
        """(n, vocab) sparse float32 representations, in input order."""
        order = np.argsort([len(t) for t in texts], kind="stable")
        rows, cols, vals = [], [], []
        for s in range(0, len(texts), self.batch_size):
            idx = order[s:s + self.batch_size]
            tok = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            logits = self.model(**tok).logits
            w = (torch.log1p(torch.relu(logits)) * tok["attention_mask"][..., None]).amax(dim=1).float()
            r, c = torch.nonzero(w, as_tuple=True)
            rows.append(idx[r.cpu().numpy()])
            cols.append(c.cpu().numpy())
            vals.append(w[r, c].cpu().numpy())
        vocab = self.model.config.vocab_size
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(texts), vocab), dtype=np.float32)


def sparse_index_bytes(docs: sp.csr_matrix, payload: int = 1) -> int:
    """Compressed inverted-index size of a (n_docs, vocab) sparse document matrix."""
    csc = docs.tocsc()
    return postings_bytes(csc.indptr, csc.indices, payload)


def sparse_scores(queries: sp.csr_matrix, docs: sp.csr_matrix, chunk: int = 256) -> np.ndarray:
    """(n_queries, n_docs) dense float32 dot products."""
    out = np.empty((queries.shape[0], docs.shape[0]), dtype=np.float32)
    dt = docs.T.tocsr()
    for s in range(0, queries.shape[0], chunk):
        out[s:s + chunk] = (queries[s:s + chunk] @ dt).toarray()
    return out
