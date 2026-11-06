"""Methods of the systematic comparison.

Each method builds some index of the corpus and scores every query against it. The scores are
evaluated twice by the runner: as a ranking on their own (a lossy index) and as the estimator that
orders recomputation from raw text under the same conformal stopping rule (a candidate generator).
Storage is reported per document plus corpus-level shared structures (codebooks, PQ tables).

A config entry {kind: <name>, ...params} selects a class from METHODS; `variants()` lists the
results it produces with the parameters that determine each one (used for result caching).
"""
from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch

from ..baselines.constbert import ConstBERTEncoder
from ..baselines.dense import DenseEncoder, mean_pool_store, quantize
from ..baselines.fde import FDEConfig, ProductQuantizer, document_fdes, query_fdes
from ..baselines.pooling import pool_store
from ..baselines.residual import ResidualCodec, plaid_num_centroids
from ..baselines.sparse import BM25, SpladeEncoder, sparse_index_bytes, sparse_scores
from ..config import ExperimentConfig
from ..data import RetrievalDataset
from ..core.codebook import assign_to_codebook
from ..core.exact import maxsim_matrix
from ..core.store import TokenStore
from ..reco.estimate import CentroidBagEstimator, TorchCentroidBagEstimator
from ..reco.sketch import CentroidBagSketch
from ..utils import get_logger

log = get_logger(__name__)


@dataclass
class MethodOutput:
    name: str
    family: str              # reference | ours | lossy-mv | learned-mv | single-vector | sparse
    est: np.ndarray          # (n_queries, n_docs) float32 scores
    bytes_per_doc: float     # mean per-document index bytes (including per-document pointers)
    shared_bytes: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    cfg: ExperimentConfig
    pipeline: Any            # savemv.pipeline.Pipeline (sketch cache, ColBERT encoder, exact scores)
    data: RetrievalDataset
    store: TokenStore
    queries: np.ndarray
    device: torch.device
    artifacts: Path

    def free_gpu(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def _vector_store_bytes(store: TokenStore, bytes_per_vector: float, doclen_bytes: float = 4.0) -> float:
    return store.n_tokens / store.n_docs * bytes_per_vector + doclen_bytes


def _codec_variants(ctx: Context, store: TokenStore, nbits: list[int], seed_offset: int = 0) -> list[ResidualCodec]:
    sc = ctx.cfg.sketch
    k = plaid_num_centroids(store.n_tokens)
    return ResidualCodec.fit(store.vectors, nbits, k, sc.kmeans_iters, sc.seed + seed_offset, ctx.device,
                             sc.points_per_centroid, sc.min_train_points)


def _cached_store(path: Path, build) -> TokenStore:
    if TokenStore.exists(path):
        return TokenStore.load(path)
    store = build()
    store.save(path)
    return TokenStore.load(path)


class Method:
    family = ""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec

    def variants(self) -> dict[str, dict[str, Any]]:
        raise NotImplementedError

    def run(self, ctx: Context) -> Iterator[MethodOutput]:
        raise NotImplementedError


class Full(Method):
    """Uncompressed ColBERT (fp16 token vectors): the quality reference."""
    family = "reference"

    def variants(self):
        return {"full-fp16": {"kind": "full"}}

    def run(self, ctx):
        est = ctx.pipeline.exact_scores(ctx.store, ctx.queries)
        yield MethodOutput("full-fp16", self.family, est, _vector_store_bytes(ctx.store, 2 * ctx.store.dim),
                           info={"vectors_per_doc": ctx.store.n_tokens / ctx.store.n_docs})


class ReCo(Method):
    """Ours: centroid-bag sketch; alone it is PLAID's centroid interaction without residuals.

    With `codebook` (a path template with {dataset} and {k}), the codebook is a pre-trained model
    artifact, e.g. trained on other corpora (scripts/stopping/train_universal_codebook.py); it is then not counted
    in the index size and the variant is named reco-K<k>-U.
    """
    family = "ours"

    def _name(self) -> str:
        return f"reco-K{self.spec['codebook_size']}" + ("-U" if self.spec.get("codebook") else "")

    def variants(self):
        k = self.spec["codebook_size"]
        return {self._name(): {"kind": "reco", "codebook_size": k, "sketch": "v1",
                               "codebook": self.spec.get("codebook")}}

    def _external_sketch(self, ctx: Context, k: int):
        centroids = np.load(self.spec["codebook"].format(dataset=ctx.cfg.dataset, k=k)).astype(np.float32)
        path = ctx.artifacts / f"reco_U_K{k}.npz"
        if path.exists():
            return CentroidBagSketch.load(path), centroids
        ids, resid = assign_to_codebook(ctx.store.vectors, centroids, ctx.device)
        sketch = CentroidBagSketch.build(ctx.store.doc_of_token(), ids, resid, k, ctx.store.n_docs)
        sketch.save(path)
        return sketch, centroids

    def run(self, ctx):
        k = self.spec["codebook_size"]
        external = bool(self.spec.get("codebook"))
        sketch, centroids = self._external_sketch(ctx, k) if external else ctx.pipeline.sketch(ctx.store, k)
        if ctx.device.type == "cuda":
            estimator = TorchCentroidBagEstimator(sketch, centroids, ctx.device)
        else:
            estimator = CentroidBagEstimator(sketch, centroids)
        est = np.stack([estimator.score(q) for q in ctx.queries])
        del estimator
        ctx.free_gpu()
        yield MethodOutput(self._name(), self.family, est, float(sketch.bytes_per_doc().mean()),
                           0.0 if external else float(k * ctx.store.dim * 2),
                           {"centroids_per_doc": float(sketch.sizes.mean()), "universal_codebook": external,
                            "codebook_size": k})


class Residual(Method):
    """ColBERTv2 / PLAID residual compression of every token vector."""
    family = "lossy-mv"

    def variants(self):
        return {f"residual-{b}bit": {"kind": "residual", "nbits": b} for b in self.spec.get("nbits", [2, 1])}

    def run(self, ctx):
        for codec in _codec_variants(ctx, ctx.store, self.spec.get("nbits", [2, 1])):
            est = maxsim_matrix(codec.compress_store(ctx.store, ctx.device), ctx.queries, ctx.device)
            yield MethodOutput(f"residual-{codec.nbits}bit", self.family, est,
                               _vector_store_bytes(ctx.store, codec.bytes_per_vector(ctx.store.dim)),
                               codec.shared_bytes(), {"codebook_size": codec.k})


class Pool(Method):
    """Hierarchical token pooling (PyLate), stored in fp16 or with ColBERTv2 residual codes.

    PyLate keeps the raw cluster means; with `normalize` (default) the fp16 variant stores them
    re-normalized, which is what the residual codec's decompression yields anyway and scored higher
    in our tests. `normalize: false` reproduces PyLate's raw means (names get a `-raw` suffix).
    """
    family = "lossy-mv"

    def _nbits(self) -> list[int | None]:
        return self.spec.get("nbits", [None, 2, 1])

    @staticmethod
    def _label(b: int | None) -> str:
        return "fp16" if b is None else f"{b}bit"

    def _name(self, b: int | None) -> str:
        return f"pool{self.spec['factor']}-{self._label(b)}" + ("" if self.spec.get("normalize", True) else "-raw")

    def variants(self):
        return {self._name(b): {"kind": "pool", "factor": self.spec["factor"], "nbits": b, "protected": 1,
                                "normalize": self.spec.get("normalize", True)} for b in self._nbits()}

    def run(self, ctx):
        f = self.spec["factor"]
        pooled = _cached_store(ctx.artifacts / f"pool_f{f}",
                               lambda: pool_store(ctx.store, f, protected=1, workers=ctx.cfg.bench.workers))
        info = {"vectors_per_doc": pooled.n_tokens / pooled.n_docs}
        if None in self._nbits():
            store = pooled
            if self.spec.get("normalize", True):
                vecs = np.asarray(pooled.vectors, dtype=np.float32)
                vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
                store = TokenStore(vecs.astype(np.float16), pooled.offsets, pooled.token_ids)
            yield MethodOutput(self._name(None), self.family, maxsim_matrix(store, ctx.queries, ctx.device),
                               _vector_store_bytes(pooled, 2 * pooled.dim), info=info)
        nbits = [b for b in self._nbits() if b is not None]
        for codec in (_codec_variants(ctx, pooled, nbits) if nbits else []):
            est = maxsim_matrix(codec.compress_store(pooled, ctx.device), ctx.queries, ctx.device)
            yield MethodOutput(self._name(codec.nbits), self.family, est,
                               _vector_store_bytes(pooled, codec.bytes_per_vector(pooled.dim)), codec.shared_bytes(),
                               {**info, "codebook_size": codec.k})


class FDE(Method):
    """MUVERA fixed-dimensional encodings, PQ-256-8 compressed (or float32 with pq: false).

    fill_empty defaults to off: filling empty document partitions lowered recall in our tests.
    """
    family = "lossy-mv"

    def _configs(self) -> list[FDEConfig]:
        s = self.spec
        return [FDEConfig(r, s.get("simhash_bits", 5), s.get("proj_dim", 16), s.get("fill_empty", False),
                          s.get("seed", 42)) for r in s.get("reps", [20])]

    def _name(self, c: FDEConfig) -> str:
        return f"fde-r{c.reps}{'-fill' if c.fill_empty else ''}-{'pq' if self.spec.get('pq', True) else 'fp32'}"

    def variants(self):
        return {self._name(c): {"kind": "fde", "config": vars(c), "pq": self.spec.get("pq", True)}
                for c in self._configs()}

    def run(self, ctx):
        store, dev = ctx.store, ctx.device
        chunk = max(1, int(self.spec.get("docs_per_chunk", 4096)))
        for c in self._configs():
            q = torch.from_numpy(query_fdes(ctx.queries, c, dev)).to(dev)
            est = np.empty((len(q), store.n_docs), dtype=np.float32)
            dim = c.out_dim(store.dim)
            if self.spec.get("pq", True):
                rng = np.random.default_rng(c.seed)
                sample = np.sort(rng.choice(store.n_docs, min(store.n_docs, self.spec.get("pq_train_docs", 50_000)),
                                            replace=False))
                train = torch.cat([document_fdes(store.vectors, store.offsets, sample[s:s + chunk], c, dev)
                                   for s in range(0, len(sample), chunk)])
                pq = ProductQuantizer.fit(train, sub_dim=8, n_centroids=256, iters=self.spec.get("pq_iters", 20),
                                          seed=c.seed)
                del train
                for s in range(0, store.n_docs, chunk):
                    docs = np.arange(s, min(s + chunk, store.n_docs))
                    codes = pq.encode(document_fdes(store.vectors, store.offsets, docs, c, dev))
                    est[:, s:s + len(docs)] = (q @ pq.decode(codes, dev).T).cpu().numpy()
                bytes_per_doc, shared = pq.bytes_per_vector(), pq.shared_bytes()
            else:
                for s in range(0, store.n_docs, chunk):
                    docs = np.arange(s, min(s + chunk, store.n_docs))
                    est[:, s:s + len(docs)] = (q @ document_fdes(store.vectors, store.offsets, docs, c, dev).T
                                               ).cpu().numpy()
                bytes_per_doc, shared = 4.0 * dim, 0.0
            ctx.free_gpu()
            yield MethodOutput(self._name(c), self.family, est, bytes_per_doc, shared, {"fde_dim": dim})


def _quantized_variants(prefix: str, kind: str, spec: dict) -> dict[str, dict]:
    return {f"{prefix}-{q}": {"kind": kind, "quant": q, **{k: v for k, v in spec.items() if k not in ("kind", "quant")}}
            for q in spec.get("quant", ["fp16", "int8", "binary"])}


class MeanPool(Method):
    """One vector per document: the normalized mean of its ColBERT token vectors (an aggregation sketch)."""
    family = "single-vector"

    def variants(self):
        return _quantized_variants("meanpool", "meanpool", self.spec)

    def run(self, ctx):
        docs = mean_pool_store(ctx.store, ctx.device)
        q = ctx.queries.mean(axis=1)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        for mode in self.spec.get("quant", ["fp16", "int8", "binary"]):
            rec, per_doc, shared = quantize(docs, mode)
            yield MethodOutput(f"meanpool-{mode}", self.family, (q @ rec.T).astype(np.float32), per_doc, shared)


class Dense(Method):
    """A single-vector bi-encoder of the same size class (BAAI/bge-small-en-v1.5 by default)."""
    family = "single-vector"

    def variants(self):
        return _quantized_variants("dense", "dense", self.spec)

    def run(self, ctx):
        s = self.spec
        enc = DenseEncoder(s["model"], ctx.device, ctx.cfg.encoder.fp16, s.get("batch_size", 128),
                           s.get("max_length", 512),
                           s.get("query_prefix", "Represent this sentence for searching relevant passages: "))
        path = ctx.artifacts / "dense_docs.npy"
        if path.exists():
            docs = np.load(path)
        else:
            docs = enc.encode(ctx.data.doc_texts)
            np.save(path, docs)
        q = enc.encode(ctx.data.query_texts, query=True)
        del enc
        ctx.free_gpu()
        for mode in s.get("quant", ["fp16", "int8", "binary"]):
            rec, per_doc, shared = quantize(docs, mode)
            yield MethodOutput(f"dense-{mode}", self.family, (q @ rec.T).astype(np.float32), per_doc, shared,
                               {"dim": docs.shape[1]})


class BM25Method(Method):
    family = "sparse"

    def variants(self):
        return {"bm25": {"kind": "bm25", "k1": self.spec.get("k1", 0.9), "b": self.spec.get("b", 0.4)}}

    def run(self, ctx):
        bm25 = BM25(self.spec.get("k1", 0.9), self.spec.get("b", 0.4))
        bm25.index(ctx.data.doc_texts)
        est = bm25.scores(ctx.data.query_texts)
        # postings (VByte gaps + 1-byte tf) and a 1-byte length norm per document, as Lucene stores
        yield MethodOutput("bm25", self.family, est, bm25.index_bytes() / ctx.data.n_docs + 1.0)


class Splade(Method):
    family = "sparse"

    def variants(self):
        return {"splade": {"kind": "splade", "max_length": self.spec.get("max_length", 256)}}

    def run(self, ctx):
        s = self.spec
        enc = SpladeEncoder(s["model"], ctx.device, ctx.cfg.encoder.fp16, s.get("batch_size", 32),
                            s.get("max_length", 256))
        path = ctx.artifacts / "splade_docs.npz"
        if path.exists():
            docs = sp.load_npz(path)
        else:
            docs = enc.encode(ctx.data.doc_texts)
            sp.save_npz(path, docs)
        q = enc.encode(ctx.data.query_texts)
        del enc
        ctx.free_gpu()
        yield MethodOutput("splade", self.family, sparse_scores(q, docs), sparse_index_bytes(docs) / ctx.data.n_docs,
                           info={"terms_per_doc": docs.nnz / docs.shape[0], "terms_per_query": q.nnz / q.shape[0]})


class ConstBERT(Method):
    """ConstBERT (32 learned vectors x 128 dims per document), fp16 or with ColBERTv2 residual codes.

    query_marker: trained (default, [unused0]) or hf (the official wrapper's [UNK]; names get `-hf`).
    """
    family = "learned-mv"

    def _nbits(self) -> list[int | None]:
        return self.spec.get("nbits", [None, 2, 1])

    def _name(self, b: int | None) -> str:
        hf = "-hf" if self.spec.get("query_marker", "trained") == "hf" else ""
        return f"constbert{hf}-{Pool._label(b)}"

    def variants(self):
        return {self._name(b): {"kind": "constbert", "nbits": b, "query_marker": self.spec.get("query_marker", "trained")}
                for b in self._nbits()}

    def run(self, ctx):
        s = self.spec
        enc = ConstBERTEncoder(s["model"], ctx.device, ctx.cfg.encoder.fp16, s.get("batch_size", 128),
                               s.get("query_marker", "trained"), s.get("query_maxlen", ctx.cfg.encoder.query_maxlen))
        store = _cached_store(ctx.artifacts / "constbert", lambda: enc.encode_corpus(ctx.data.doc_texts))
        q = enc.encode_queries(ctx.data.query_texts)
        del enc
        ctx.free_gpu()
        per_doc = store.n_tokens / store.n_docs
        if None in self._nbits():
            yield MethodOutput(self._name(None), self.family, maxsim_matrix(store, q, ctx.device),
                               per_doc * store.dim * 2, info={"vectors_per_doc": per_doc})
        nbits = [b for b in self._nbits() if b is not None]
        for codec in (_codec_variants(ctx, store, nbits) if nbits else []):
            est = maxsim_matrix(codec.compress_store(store, ctx.device), q, ctx.device)
            yield MethodOutput(self._name(codec.nbits), self.family, est,
                               per_doc * codec.bytes_per_vector(store.dim), codec.shared_bytes(),
                               {"vectors_per_doc": per_doc, "codebook_size": codec.k})


METHODS: dict[str, type[Method]] = {
    "full": Full, "reco": ReCo, "residual": Residual, "pool": Pool, "fde": FDE, "meanpool": MeanPool,
    "dense": Dense, "bm25": BM25Method, "splade": Splade, "constbert": ConstBERT,
}


def build_method(spec: dict[str, Any]) -> Method:
    kind = spec.get("kind")
    if kind not in METHODS:
        raise ValueError(f"unknown method kind {kind!r}; expected one of {sorted(METHODS)}")
    return METHODS[kind](spec)


def storage_ratio(bytes_per_doc: float, shared_bytes: float, n_docs: int, raw_bytes_per_doc: float) -> float:
    return (bytes_per_doc + shared_bytes / n_docs) / raw_bytes_per_doc if raw_bytes_per_doc > 0 else math.nan
