"""ColBERT encoder with colbert-ai semantics (validated for answerai-colbert-small-v1).

Query: [CLS] [Q] tokens [SEP], padded to `query_maxlen` with [MASK]; mask tokens are not
attended to (attend_to_mask_tokens = false) but all positions are kept for MaxSim.
Doc:   [CLS] [D] tokens [SEP], truncated at `doc_maxlen`; punctuation tokens are dropped.
Both:  BERT -> linear(hidden -> dim, no bias) -> L2 normalize.
"""
from __future__ import annotations

import string
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers import AutoTokenizer, BertConfig, BertModel

from ..config import EncoderConfig
from ..core.store import TokenStore
from ..utils import get_logger, resolve_device

log = get_logger(__name__)


def _model_file(repo: str, filename: str) -> str:
    """`repo` is either a local model directory or a Hugging Face model id."""
    local = Path(repo)
    return str(local / filename) if local.is_dir() else hf_hub_download(repo, filename)


class ColBERTEncoder:
    def __init__(self, cfg: EncoderConfig) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.repo)
        state = load_file(_model_file(cfg.repo, "model.safetensors"))
        self.bert = BertModel(BertConfig.from_pretrained(cfg.repo), add_pooling_layer=False)
        bert_state = {k[5:]: v for k, v in state.items()
                      if k.startswith("bert.") and ".pooler." not in k and not k.endswith("position_ids")}
        missing, unexpected = self.bert.load_state_dict(bert_state, strict=False)
        if unexpected or any("position_ids" not in m for m in missing):
            raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
        self.bert.to(self.device).eval()
        self.half = cfg.fp16 and self.device.type == "cuda"
        if self.half:
            self.bert.half()   # fp16 weights: ~20% faster than autocast on Ampere
        self.linear = state["linear.weight"].float().to(self.device)   # (dim, hidden), kept in fp32
        self.dim = int(self.linear.shape[0])
        self.pad_id = self.tokenizer.pad_token_id
        self.mask_id = self.tokenizer.mask_token_id
        self.q_marker = self.tokenizer.convert_tokens_to_ids("[unused0]")
        self.d_marker = self.tokenizer.convert_tokens_to_ids("[unused1]")
        skip = sorted({self.tokenizer.encode(p, add_special_tokens=False)[0] for p in string.punctuation})
        self.skip_ids = torch.tensor(skip, device=self.device)
        self._doc_tok = self._rust_tokenizer(cfg.doc_maxlen, pad_to=None)
        self._query_tok = self._rust_tokenizer(cfg.query_maxlen, pad_to=cfg.query_maxlen)
        log.info("encoder %s on %s (dim=%d, fp16 weights=%s)", cfg.repo, self.device, self.dim, self.half)

    # ------------------------------------------------------------------ core
    def _rust_tokenizer(self, maxlen: int, pad_to: int | None) -> Tokenizer:
        """Private copy of the fast (Rust) tokenizer with truncation and padding preset.

        Calling it directly skips transformers' Python-side tensor conversion, which costs more
        than tokenization itself (about 0.7 vs 0.2 ms per document at batch 32).
        """
        tok = Tokenizer.from_str(self.tokenizer.backend_tokenizer.to_str())
        tok.enable_truncation(max_length=maxlen)
        tok.enable_padding(pad_id=self.pad_id, pad_token=self.tokenizer.pad_token, length=pad_to)
        return tok

    @staticmethod
    def _encode_ids(tok: Tokenizer, texts: list[str], marker: int) -> tuple[np.ndarray, np.ndarray]:
        encs = tok.encode_batch([". " + t for t in texts])
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        ids[:, 1] = marker   # the "." placeholder becomes the [Q]/[D] marker
        return ids, mask

    def _to_device(self, *arrays: np.ndarray) -> list[torch.Tensor]:
        return [torch.from_numpy(a).to(self.device, non_blocking=True) for a in arrays]

    @torch.inference_mode()
    def _embed(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, L) ids -> (B, L, dim) float32 unit vectors."""
        hidden = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        return F.normalize(hidden.float() @ self.linear.T, dim=-1)

    # ------------------------------------------------------------- queries
    def _query_ids(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        ids, mask = self._encode_ids(self._query_tok, texts, self.q_marker)
        ids[ids == self.pad_id] = self.mask_id   # query augmentation with [MASK]
        return ids, mask

    def encode_queries(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """(n, query_maxlen, dim) float32."""
        out = [self._embed(*self._to_device(*self._query_ids(texts[s:s + batch_size]))).cpu().numpy()
               for s in range(0, len(texts), batch_size)]
        return np.concatenate(out) if out else np.zeros((0, self.cfg.query_maxlen, self.dim), np.float32)

    def encode_query_tensor(self, text: str) -> torch.Tensor:
        """(query_maxlen, dim) on the encoder device, for the online search path."""
        return self._embed(*self._to_device(*self._query_ids([text])))[0]

    # ----------------------------------------------------------- documents
    def tokenize_docs(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """CPU part of document encoding; safe in a worker thread (the Rust tokenizer releases the GIL)."""
        return self._encode_ids(self._doc_tok, texts, self.d_marker)

    def embed_docs(self, ids: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GPU part: (B, L, dim) vectors, (B, L) keep mask of scored tokens, (B, L) token ids."""
        ids_t, mask_t = self._to_device(ids, mask)
        emb = self._embed(ids_t, mask_t)
        keep = mask_t.bool() & ~torch.isin(ids_t, self.skip_ids)
        return emb, keep, ids_t

    def encode_docs_padded(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.embed_docs(*self.tokenize_docs(texts))

    def encode_corpus(self, texts: list[str], batch_size: int | None = None, log_every: int = 100) -> TokenStore:
        """Encode all documents; batches are formed from length-sorted texts to cut padding."""
        bs = batch_size or self.cfg.batch_size
        order = np.argsort([-len(t) for t in texts], kind="stable")
        vecs: list[np.ndarray | None] = [None] * len(texts)
        toks: list[np.ndarray | None] = [None] * len(texts)
        for step, s in enumerate(range(0, len(order), bs)):
            idx = order[s:s + bs]
            emb, keep, ids = self.encode_docs_padded([texts[i] for i in idx])
            counts = keep.sum(1).tolist()
            flat_vec = emb[keep].to(torch.float16).cpu().numpy()
            flat_tok = ids[keep].to(torch.int32).cpu().numpy()
            bounds = np.concatenate([[0], np.cumsum(counts)])
            for j, doc in enumerate(idx):
                vecs[doc] = flat_vec[bounds[j]:bounds[j + 1]]
                toks[doc] = flat_tok[bounds[j]:bounds[j + 1]]
            if step % log_every == 0:
                log.info("encoded %d / %d documents", min(s + bs, len(order)), len(order))
        return TokenStore.from_docs(vecs, toks)  # type: ignore[arg-type]
