"""ConstBERT (MacAvaney, Mallia & Tonellotto, ECIR 2025): a fixed number of learned document vectors.

Re-implements `encode_queries` / `encode_documents` of pinecone/ConstBERT (modeling.py,
tokenization_utils.py): documents are padded to doc_maxlen tokens and a learned linear map over the
token axis (doc_project, doc_maxlen -> 32) mixes BERT outputs into 32 vectors before the 768 -> 128
projection; queries are standard ColBERT queries ([MASK]-augmented to query_maxlen, all positions kept).

The official HF wrapper builds `ColBERTConfig(name_or_path)`, whose first positional field is
`query_token_id`, so it marks queries with the model path, i.e. [UNK], rather than the trained
[unused0]. `query_marker="hf"` reproduces that; the default uses the trained marker.
With the same marker, outputs match the official code (scripts/official_baselines/check_constbert.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from ..core.store import TokenStore
from ..utils import get_logger

log = get_logger(__name__)


class ConstBERTEncoder:
    def __init__(self, path: str, device: torch.device, fp16: bool = True, batch_size: int = 128,
                 query_marker: str = "trained", query_maxlen: int | None = None) -> None:
        from transformers import AutoTokenizer, BertConfig, BertModel
        root = Path(path)
        meta = json.loads((root / "artifact.metadata").read_text(encoding="utf-8"))
        self.query_maxlen, self.doc_maxlen, self.dim = meta["query_maxlen"], meta["doc_maxlen"], meta["dim"]
        if query_maxlen is not None:      # e.g. long ArguAna queries; documents keep the trained length
            self.query_maxlen = query_maxlen
        if meta.get("attend_to_mask_tokens"):
            raise NotImplementedError("attend_to_mask_tokens=True is not supported")
        self.device, self.batch_size = device, batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        weights = load_file(str(root / "model.safetensors"))
        self.bert = BertModel(BertConfig.from_pretrained(path), add_pooling_layer=False)
        state = {k[len("bert."):]: v for k, v in weights.items() if k.startswith("bert.") and ".pooler." not in k}
        missing, unexpected = self.bert.load_state_dict(state, strict=False)
        if [k for k in missing if "position_ids" not in k] or unexpected:
            raise RuntimeError(f"ConstBERT checkpoint mismatch: missing={missing} unexpected={unexpected}")
        self.bert = self.bert.to(device).eval()
        self.half = fp16 and device.type == "cuda"
        if self.half:
            self.bert.half()
        self.linear = weights["linear.weight"].float().to(device)              # (dim, hidden)
        self.doc_project = weights["doc_project.weight"].float().to(device)    # (n_vectors, doc_maxlen)
        self.n_vectors = self.doc_project.shape[0]
        tok = self.tokenizer
        if query_marker not in ("trained", "hf"):
            raise ValueError("query_marker must be 'trained' or 'hf'")
        self.q_marker = tok.convert_tokens_to_ids(meta["query_token_id"] if query_marker == "trained" else path)
        self.d_marker = tok.convert_tokens_to_ids(meta["doc_token_id"])
        self.pad_id, self.mask_id = tok.pad_token_id, tok.mask_token_id

    def _tokenize(self, texts: list[str], maxlen: int, marker: int) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer([". " + t for t in texts], padding="max_length", truncation=True,
                             max_length=maxlen, return_tensors="pt")
        ids, mask = enc["input_ids"], enc["attention_mask"]
        ids[:, 1] = marker
        return ids, mask

    @torch.inference_mode()
    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """(n, query_maxlen, dim) float32."""
        out = []
        for s in range(0, len(texts), self.batch_size):
            ids, mask = self._tokenize(texts[s:s + self.batch_size], self.query_maxlen, self.q_marker)
            ids[ids == self.pad_id] = self.mask_id
            h = self.bert(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).last_hidden_state
            out.append(F.normalize(h.float() @ self.linear.T, dim=-1).cpu().numpy())
        return np.concatenate(out)

    @torch.inference_mode()
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """(n, n_vectors, dim) float16."""
        out = np.empty((len(texts), self.n_vectors, self.dim), dtype=np.float16)
        for s in range(0, len(texts), self.batch_size):
            ids, mask = self._tokenize(texts[s:s + self.batch_size], self.doc_maxlen, self.d_marker)
            h = self.bert(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).last_hidden_state
            d = torch.einsum("cl,blh->bch", self.doc_project, h.float())
            out[s:s + len(ids)] = F.normalize(d @ self.linear.T, dim=-1).half().cpu().numpy()
        return out

    def encode_corpus(self, texts: list[str], log_every: int = 200) -> TokenStore:
        """Fixed-length documents as a TokenStore (n_vectors rows per document)."""
        chunks = []
        step = self.batch_size * log_every
        for s in range(0, len(texts), step):
            chunks.append(self.encode_documents(texts[s:s + step]))
            log.info("ConstBERT: encoded %d / %d documents", min(s + step, len(texts)), len(texts))
        docs = np.concatenate(chunks)
        n, c, d = docs.shape
        offsets = np.arange(0, n * c + 1, c, dtype=np.int64)
        return TokenStore(docs.reshape(n * c, d), offsets, np.full(n * c, -1, dtype=np.int32))
