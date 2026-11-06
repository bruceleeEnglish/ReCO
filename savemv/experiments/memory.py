"""Agent-memory retrieval on per-user memories (LongMemEval, LoCoMo).

Compares exact ColBERT, ReCo (sketch alone, and with recomputation under a calibrated budget), BM25 and
dense retrievers; every question is answered from its own memory. ReCo uses a codebook trained on other
corpora (no per-user training). Also reports index bytes per memory unit and recompute cost per document.

    python -m savemv memory --bench longmemeval --data data/memory/longmemeval_s_cleaned.json
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..baselines.dense import DenseEncoder
from ..baselines.sparse import BM25
from ..config import EncoderConfig
from ..core.codebook import assign_to_codebook
from ..core.exact import maxsim_matrix
from ..core.store import TokenStore
from ..data.memory import LOADERS, MemoryBenchmark, MemoryTask
from ..encoders import ColBERTEncoder
from ..reco.estimate import CentroidBagEstimator
from ..reco.hazard import HazardSettings, fit_hazard, hazard_gmax
from ..reco.sketch import CentroidBagSketch
from ..reco.stopping import StopPath, crc_budget, crc_margin, kfold_splits, recall_quantile_r
from ..utils import get_logger, write_json

log = get_logger(__name__)
TOP = 10
BGE_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class MemoryConfig:
    bench: str
    data: str
    out: str | None = None
    colbert: str = "models/answerai-colbert-small-v1"
    doc_maxlen: int = 300
    codebooks: list[str] = field(default_factory=lambda: ["models/codebooks/loo-scifact_K16384.npy",
                                                         "models/codebooks/loo-scifact_K65536.npy"])
    dense: list[str] = field(default_factory=lambda: ["models/bge-small-en-v1.5", "models/bge-base-en-v1.5"])
    delta: float = 0.05
    cpu_threads: int = 8


# ------------------------------------------------------------------------------------------ metrics
def ranking_metrics(order: np.ndarray, task: MemoryTask, sessions: list[str]) -> dict[str, float]:
    """Evidence recall / hit / nDCG@10 over the task's ranked units, and session recall when available."""
    out: dict[str, float] = {}
    if len(task.relevant):
        rel = set(task.relevant.tolist())
        hits = np.array([u in rel for u in order[:TOP]])
        for k in (5, 10):
            out[f"recall@{k}"] = hits[:k].sum() / len(rel)
            out[f"hit@{k}"] = float(hits[:k].any())
        disc = 1.0 / np.log2(np.arange(2, TOP + 2))
        out["ndcg@10"] = float((hits * disc).sum() / disc[:min(TOP, len(rel))].sum())
    if task.relevant_sessions:
        seen: list[str] = []
        for u in order:
            if sessions[u] not in seen:
                seen.append(sessions[u])
                if len(seen) == TOP:
                    break
        for k in (5, 10):
            out[f"session_recall@{k}"] = len(set(seen[:k]) & task.relevant_sessions) / len(task.relevant_sessions)
    return out


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for r in rows for k in r})
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}


def rerank(order: np.ndarray, exact: dict[int, float], r: int) -> np.ndarray:
    """Recompute the first r units of `order` and sort them by exact score (the rest keeps its order)."""
    head = order[:r]
    head = head[np.argsort([-exact[u] for u in head], kind="stable")]
    return np.concatenate([head, order[r:]])


def _fit_map(est_rows: list[np.ndarray], exact_rows: list[np.ndarray], idx: np.ndarray, top: int = 200):
    """exact ~ a * estimate + b on the `top` highest estimates of each listed task (a > 0)."""
    xs, ys = [], []
    for i in idx:
        head = np.argsort(-est_rows[i], kind="stable")[:top]
        xs.append(est_rows[i][head])
        ys.append(exact_rows[i][head])
    x, y = np.concatenate(xs).astype(np.float64), np.concatenate(ys).astype(np.float64)
    a, b = np.linalg.lstsq(np.c_[x, np.ones_like(x)], y, rcond=None)[0]
    return (float(a), float(b)) if a > 0 else (1.0, 0.0)


def calibrated_stops(est_rows: list[np.ndarray], exact_rows: list[np.ndarray], delta: float,
                     settings: HazardSettings, batch: int = 8, folds: int = 5) -> dict[str, np.ndarray]:
    """Per-task recompute counts of the fixed budget and the hazard rule, cross-calibrated over users:
    each fold's hazard model and score map are fitted on 40% of the other users' questions, the CRC
    budget / threshold on the rest. Each task is its own small corpus (its memory)."""
    n_tasks = len(est_rows)
    paths = [StopPath(e, x, np.argsort(-x, kind="stable")[:TOP], TOP, batch, state=True, ahead=settings.ahead)
             for e, x in zip(est_rows, exact_rows)]
    size = max(len(e) for e in est_rows)
    stops = {"fixed": np.empty(n_tasks, np.int64), "hazard": np.empty(n_tasks, np.int64)}
    for sp in kfold_splits(n_tasks, folds, 0.4, seed=0):
        a, b = _fit_map(est_rows, exact_rows, sp.fit)
        model = fit_hazard([paths[i] for i in sp.fit], [(a, b)] * len(sp.fit), settings)
        cal = [paths[i] for i in sp.cal]
        lam = crc_margin(cal, [hazard_gmax(model, p, a, b) for p in cal], delta, strict=False)
        budget = crc_budget(np.stack([p.ranks for p in cal]), size, delta, strict=False)
        for i in sp.test:
            stops["hazard"][i] = paths[i].stop(hazard_gmax(model, paths[i], a, b), lam)
            stops["fixed"][i] = min(budget, paths[i].n)
    return stops


# ------------------------------------------------------------------------------------------- scoring
def subsketch(sketch: CentroidBagSketch, units: np.ndarray) -> CentroidBagSketch:
    """A user's own sketch: only the given units (CSR slices of the global sketch)."""
    ptr = np.zeros(len(units) + 1, dtype=np.int64)
    np.cumsum(sketch.sizes[units], out=ptr[1:])
    cent = np.concatenate([sketch.cent[sketch.ptr[u]:sketch.ptr[u + 1]] for u in units])
    return CentroidBagSketch(sketch.k, ptr, cent, None)


def bm25_scores(bench: MemoryBenchmark) -> tuple[list[np.ndarray], float]:
    """Per-user BM25 indexes (IDF from the user's own memory) and their mean index bytes per unit."""
    rows, total_bytes, total_units = [], 0, 0
    for t in bench.tasks:
        bm = BM25()
        bm.index([bench.texts[u] for u in t.units])
        rows.append(bm.scores([t.query])[0])
        total_bytes += bm.index_bytes() + len(t.units)          # postings + 1-byte length norm
        total_units += len(t.units)
    return rows, total_bytes / total_units


def score_all(cfg: MemoryConfig, bench: MemoryBenchmark, out: Path, device: torch.device):
    """Per-task scores of every retriever, index bytes per unit, and the encoder / ReCo sketches used."""
    tasks = bench.tasks
    enc = ColBERTEncoder(EncoderConfig(repo=cfg.colbert, doc_maxlen=cfg.doc_maxlen, batch_size=256,
                                       device=str(device)))
    tok_dir = out / f"tokens_d{cfg.doc_maxlen}"
    if not TokenStore.exists(tok_dir):
        enc.encode_corpus(bench.texts).save(tok_dir)
    store = TokenStore.load(tok_dir)
    queries = enc.encode_queries([t.query for t in tasks])
    exact_all = maxsim_matrix(store, queries, device)
    scores = {"colbert": [exact_all[i, t.units] for i, t in enumerate(tasks)]}
    storage = {"raw text": bench.raw_bytes_per_unit(),
               "ColBERT fp16": store.n_tokens / store.n_docs * store.dim * 2 + 4,
               "ColBERTv2 1-bit (ref.)": store.n_tokens / store.n_docs * (2 + store.dim / 8) + 4}
    sketches = {}
    for path in cfg.codebooks:
        centroids = np.load(path).astype(np.float32)
        k = len(centroids)
        ids, resid = assign_to_codebook(store.vectors, centroids, device)
        sketch = CentroidBagSketch.build(store.doc_of_token(), ids, resid, k, store.n_docs)
        name = f"reco-K{k // 1024}k"
        scores[name] = [CentroidBagEstimator(subsketch(sketch, t.units), centroids).score(queries[i])
                        for i, t in enumerate(tasks)]
        storage[f"ReCo sketch K={k // 1024}k"] = float(sketch.bytes_per_doc().mean())
        sketches[name] = (sketch, centroids)
    for path in cfg.dense:
        if not Path(path).is_dir():
            log.warning("skipping missing dense model %s", path)
            continue
        de = DenseEncoder(path, device, fp16=True, query_prefix=BGE_PREFIX)
        d, q = de.encode(bench.texts), de.encode([t.query for t in tasks], query=True)
        name = Path(path).name.replace("-en-v1.5", "")
        scores[name] = [q[i] @ d[t.units].T for i, t in enumerate(tasks)]
        storage[f"{name} fp16"] = 2.0 * d.shape[1]
        del de
    scores["bm25"], storage["BM25 index"] = bm25_scores(bench)
    return scores, storage, enc, store, queries, sketches


# ---------------------------------------------------------------------------------------- evaluation
def evaluate(scores: dict[str, list[np.ndarray]], bench: MemoryBenchmark, delta: float,
             settings: HazardSettings | None = None) -> dict:
    """Each retriever alone, and as the order of recomputation under a CRC-calibrated budget
    (5-fold over users; loss = missed fraction of ColBERT's top-10), plus the hazard rule."""
    settings = settings or HazardSettings(horizon=32, ahead=32)
    tasks = bench.tasks
    exact = scores["colbert"]
    best = [t.units[np.argsort(-e, kind="stable")[:TOP]] for t, e in zip(tasks, exact)]
    exact_maps = [dict(zip(t.units.tolist(), e.tolist())) for t, e in zip(tasks, exact)]
    sizes = np.array([len(t.units) for t in tasks])
    splits = kfold_splits(len(tasks), 5, 0.0, seed=0)
    methods = {}
    for name, rows in scores.items():
        orders = [t.units[np.argsort(-s, kind="stable")] for t, s in zip(tasks, rows)]
        pos = np.array([[int(np.flatnonzero(o == u)[0]) for u in b] for o, b in zip(orders, best)])
        budget = np.empty(len(tasks), dtype=np.int64)
        for sp in splits:
            budget[sp.test] = np.minimum(crc_budget(pos[sp.cal], int(sizes.max()), delta, strict=False),
                                         sizes[sp.test])
        reranked = [rerank(o, m, int(b)) for o, m, b in zip(orders, exact_maps, budget)]
        methods[name] = {
            "alone": mean_metrics([ranking_metrics(o, t, bench.sessions) for o, t in zip(orders, tasks)]),
            "R95": recall_quantile_r(pos, 0.95),
            "crc_budget_mean": float(budget.mean()), "crc_budget_p90": float(np.percentile(budget, 90)),
            "recall_of_colbert_top10": float((pos < budget[:, None]).mean()),
            "after_recompute": mean_metrics([ranking_metrics(o, t, bench.sessions) for o, t in zip(reranked, tasks)]),
        }
        if name != "colbert":     # fixed budget vs hazard rule on the same cross-user splits (fit 40% / cal 60%)
            stops = calibrated_stops(rows, exact, delta, settings)
            for rule, st in stops.items():
                done = [rerank(o, m, int(s)) for o, m, s in zip(orders, exact_maps, st)]
                methods[name][f"{rule}_stop"] = {
                    "R_mean": float(st.mean()), "R_p90": float(np.percentile(st, 90)), "R_p99": float(np.percentile(st, 99)),
                    "recall_of_colbert_top10": float((pos < st[:, None]).mean()),
                    "after_recompute": mean_metrics([ranking_metrics(o, t, bench.sessions) for o, t in zip(done, tasks)]),
                    "R_list": st.tolist()}
        a, b = methods[name]["alone"], methods[name]["after_recompute"]
        extra = "" if name == "colbert" else " | 40/60 split: fixed R %.1f, hazard R %.1f (recall %.3f)" % (
            methods[name]["fixed_stop"]["R_mean"], methods[name]["hazard_stop"]["R_mean"],
            methods[name]["hazard_stop"]["recall_of_colbert_top10"])
        log.info("%-10s alone R@10 %.3f nDCG %.3f | R95 %d | CRC budget %.1f -> R@10 %.3f nDCG %.3f%s", name,
                 a.get("recall@10", math.nan), a.get("ndcg@10", math.nan), methods[name]["R95"],
                 methods[name]["crc_budget_mean"], b.get("recall@10", math.nan), b.get("ndcg@10", math.nan), extra)
    return methods


def ms_per_doc(encoder: ColBERTEncoder, texts: list[str], batch: int) -> float:
    encoder.encode_docs_padded(texts[:batch])                    # warm-up
    t0 = time.perf_counter()
    for s in range(0, len(texts), batch):
        emb, keep, _ = encoder.encode_docs_padded(texts[s:s + batch])
        keep.sum().item()
    return 1000 * (time.perf_counter() - t0) / len(texts)


def latency(cfg: MemoryConfig, bench: MemoryBenchmark, enc: ColBERTEncoder, queries: np.ndarray, sketches: dict,
            device: torch.device) -> dict:
    """Recompute cost per unit (GPU, and CPU with `cpu_threads` threads), query encoding and sketch estimate on CPU."""
    sample = [bench.texts[i] for i in np.random.default_rng(0).choice(len(bench.texts), min(512, len(bench.texts)),
                                                                     replace=False)]
    lat: dict = {"gpu_ms_per_doc_b32": ms_per_doc(enc, sample, 32) if device.type == "cuda" else None}
    torch.set_num_threads(cfg.cpu_threads)
    cpu = ColBERTEncoder(EncoderConfig(repo=cfg.colbert, doc_maxlen=cfg.doc_maxlen, device="cpu", fp16=False))
    lat["cpu_threads"] = cfg.cpu_threads
    lat["cpu_ms_per_doc_b16"] = ms_per_doc(cpu, sample[:128], 16)
    t0 = time.perf_counter()
    for t in bench.tasks[:50]:
        cpu.encode_query_tensor(t.query)
    lat["cpu_ms_query_encode"] = 1000 * (time.perf_counter() - t0) / min(50, len(bench.tasks))
    for name, (sketch, centroids) in sketches.items():
        est = CentroidBagEstimator(subsketch(sketch, bench.tasks[0].units), centroids)
        est.score(queries[0])
        t0 = time.perf_counter()
        for i in range(20):
            est.score(queries[i % len(queries)])
        lat[f"cpu_ms_estimate_{name}"] = 1000 * (time.perf_counter() - t0) / 20
    return lat


def run_memory(cfg: MemoryConfig) -> dict:
    out = Path(cfg.out or f"outputs/memory/{cfg.bench}")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bench = LOADERS[cfg.bench](cfg.data)
    sizes = np.array([len(t.units) for t in bench.tasks])
    log.info("%s: %d unique units, %d tasks, %.0f units / task, %.0f raw B / unit", bench.name, len(bench.texts),
             len(bench.tasks), sizes.mean(), bench.raw_bytes_per_unit())
    scores, storage, enc, store, queries, sketches = score_all(cfg, bench, out, device)
    probe = np.random.default_rng(1).choice(len(bench.texts), min(5000, len(bench.texts)), replace=False)
    wordpieces = np.array([len(ids) for ids in enc.tokenizer([bench.texts[i] for i in probe])["input_ids"]])
    report = {"benchmark": bench.name, "n_units": len(bench.texts), "n_tasks": len(bench.tasks),
              "units_per_task": float(sizes.mean()), "tokens_per_unit": store.n_tokens / store.n_docs,
              "truncated_share": float(np.mean(wordpieces > cfg.doc_maxlen)), "storage_bytes_per_unit": storage,
              "methods": evaluate(scores, bench, cfg.delta),
              "latency": latency(cfg, bench, enc, queries, sketches, device)}
    write_json(out / "memory.json", report)
    (out / "memory.md").write_text(render(report), encoding="utf-8")
    log.info("wrote %s", out / "memory.md")
    return report


def render(report: dict) -> str:
    lines = [f"# {report['benchmark']}: {report['n_tasks']} questions, {report['units_per_task']:.0f} memory units "
             f"per user, {report['tokens_per_unit']:.0f} tokens / unit", "",
             "| retriever | R@5 | R@10 | nDCG@10 | session R@10 | R95 vs ColBERT top-10 | CRC budget (p90) "
             "| after recompute R@10 / nDCG@10 |", "|---|---|---|---|---|---|---|---|"]
    for name, e in report["methods"].items():
        a, b = e["alone"], e["after_recompute"]
        lines.append(f"| {name} | {a.get('recall@5', math.nan):.3f} | {a.get('recall@10', math.nan):.3f} "
                     f"| {a.get('ndcg@10', math.nan):.3f} | {a.get('session_recall@10', math.nan):.3f} | {e['R95']} "
                     f"| {e['crc_budget_mean']:.1f} ({e['crc_budget_p90']:.0f}) "
                     f"| {b.get('recall@10', math.nan):.3f} / {b.get('ndcg@10', math.nan):.3f} |")
    lines += ["", "Index bytes per memory unit: " + ", ".join(f"{k} {v:.0f}"
                                                              for k, v in report["storage_bytes_per_unit"].items()),
              "", "Latency: " + ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
                                         for k, v in report["latency"].items())]
    return "\n".join(lines) + "\n"
