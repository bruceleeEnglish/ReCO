"""MS MARCO passage ranking (8.8M passages): ReCo at web scale, against exact ColBERT on the same collection.

Stages (each reads what the previous one wrote under --out):
  encode   one streaming pass over the corpus: every passage is encoded once, its kept tokens are assigned to
           the codebooks (sketch entries) and scored exactly against the evaluation queries. No token store is
           written. -> codebook_K*.npy, sketch_K*.npz, queries.npy, lengths.npy, exact.npy (queries x passages,
           float32, evaluation only), meta.json
  eval     per codebook: the sketch estimate of every query (fused GPU kernel, timed), then the benchmark's
           stopping protocol (5-fold cross-calibration, k = 10) on the top-D estimate prefix plus the exact
           top documents it misses. Stops and recall are exact as long as no stop exceeds D (reported).
           --bm25 adds the BM25 -> ColBERT-from-text cascade under the same protocol.
           -> exact_top.npy, eval_K*.json, eval_bm25.json
  latency  the online engine over the full sketch (calibration and test queries of fold 0), and the sketch scan
           alone on CPU. -> latency_K*.json

  python scripts/web_scale/msmarco_scale.py encode --out outputs/msmarco [--max-docs 300000 for a smoke test]
  python scripts/web_scale/msmarco_scale.py eval --out outputs/msmarco --bm25
  python scripts/web_scale/msmarco_scale.py latency --out outputs/msmarco --k 65536
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from savemv.config import EncoderConfig  # noqa: E402
from savemv.core import nearest_centroid, train_spherical_kmeans, training_sample  # noqa: E402
from savemv.encoders.colbert import ColBERTEncoder  # noqa: E402
from savemv.reco import (Calibration, CentroidBagEstimator, CentroidBagSketch, HazardModel,  # noqa: E402
                         HazardSettings, ReCoSearcher, TorchCentroidBagEstimator)
from savemv.reco.stopping import evaluate_stopping, kfold_splits  # noqa: E402
from savemv.utils import get_logger, read_json, synchronize, write_json  # noqa: E402

log = get_logger("savemv.msmarco")
DELTAS = [0.1, 0.05, 0.01]


# ----------------------------------------------------------------------------- data
def load_corpus(path: Path, max_docs: int | None) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            ids.append(str(d["_id"]))
            texts.append(f"{d.get('title') or ''} {d.get('text') or ''}".strip())
            if max_docs and len(ids) >= max_docs:
                break
    log.info("corpus: %d passages", len(ids))
    return ids, texts


def load_queries(root: Path, doc_ids: list[str], n: int, seed: int) -> tuple[list[str], list[str], list[list[int]]]:
    """A seeded sample of dev queries that have a relevant passage in the (possibly truncated) corpus."""
    index = {d: i for i, d in enumerate(doc_ids)}
    rel: dict[str, list[int]] = {}
    with open(root / "qrels" / "dev.tsv", encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0 and did in index:
                rel.setdefault(qid, []).append(index[did])
    queries = []
    with open(root / "queries.jsonl", encoding="utf-8") as fh:
        for line in fh:
            q = json.loads(line)
            if str(q["_id"]) in rel:
                queries.append((str(q["_id"]), q["text"]))
    if len(queries) > n:
        keep = np.sort(np.random.default_rng(seed).choice(len(queries), n, replace=False))
        queries = [queries[i] for i in keep]
    log.info("queries: %d dev queries with judged passages", len(queries))
    return [q for q, _ in queries], [t for _, t in queries], [rel[q] for q, _ in queries]


# ----------------------------------------------------------------------------- metrics
def mrr_at_10(rankings: list[np.ndarray], rel: list[list[int]]) -> np.ndarray:
    out = np.zeros(len(rankings))
    for i, (row, r) in enumerate(zip(rankings, rel)):
        hits = np.flatnonzero(np.isin(row[:10], r))
        out[i] = 1.0 / (hits[0] + 1) if len(hits) else 0.0
    return out


def ndcg_at_10(rankings: list[np.ndarray], rel: list[list[int]]) -> np.ndarray:
    disc = 1.0 / np.log2(np.arange(2, 12))
    out = np.zeros(len(rankings))
    for i, (row, r) in enumerate(zip(rankings, rel)):
        gains = np.isin(row[:10], r).astype(np.float64)
        out[i] = float((gains * disc[:len(gains)]).sum()) / float(disc[:min(10, len(r))].sum())
    return out


def recall_at(rankings: list[np.ndarray], rel: list[list[int]], depth: int) -> np.ndarray:
    return np.array([np.isin(r, row[:depth]).mean() for row, r in zip(rankings, rel)])


# ----------------------------------------------------------------------------- encode
def stage_encode(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t = time.perf_counter()
    doc_ids, texts = load_corpus(Path(args.data) / "corpus.jsonl", args.max_docs)
    qids, qtexts, rel = load_queries(Path(args.data), doc_ids, args.queries, args.seed)
    timings["load_s"] = time.perf_counter() - t
    n = len(texts)
    enc = ColBERTEncoder(EncoderConfig(repo=args.encoder, doc_maxlen=args.doc_maxlen, batch_size=args.batch,
                                       device="cuda:0"))
    dev = enc.device
    queries = enc.encode_queries(qtexts)
    np.save(out / "queries.npy", queries)
    nq, m, dim = queries.shape

    # codebooks: spherical k-means on the tokens of a random passage sample (as in the benchmark)
    rng = np.random.default_rng(args.seed)
    train_docs = np.sort(rng.choice(n, min(n, args.train_docs), replace=False))
    toks = []
    for s in range(0, len(train_docs), args.batch):
        emb, keep, _ = enc.encode_docs_padded([texts[i] for i in train_docs[s:s + args.batch]])
        toks.append(emb[keep].cpu().numpy())
    toks = np.concatenate(toks)
    codebooks = {}
    for k in args.ks:
        t = time.perf_counter()
        sample = training_sample(len(toks), k, 32, 200_000, args.seed)
        codebooks[k] = train_spherical_kmeans(toks[sample], k, 15, args.seed, dev)
        np.save(out / f"codebook_K{k}.npy", codebooks[k])
        timings[f"kmeans_K{k}_s"] = time.perf_counter() - t
    del toks

    # one streaming pass: encode, sketch, exact scores
    exact = np.lib.format.open_memmap(out / "exact.npy", mode="w+", dtype=np.float32, shape=(nq, n))
    lengths = np.zeros(n, np.int32)
    cent_dev = {k: torch.from_numpy(c).to(dev) for k, c in codebooks.items()}
    parts: dict[int, list[np.ndarray]] = {k: [] for k in args.ks}
    sizes = {k: np.zeros(n, np.int64) for k in args.ks}
    q_mat = torch.from_numpy(queries.reshape(nq * m, dim)).to(dev)
    buf = torch.empty((nq, args.chunk), device=dev)
    spent = {"encode": 0.0, "assign": 0.0, "exact": 0.0}
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tokenize")
    t_pass = time.perf_counter()
    for c0 in range(0, n, args.chunk):
        c1 = min(n, c0 + args.chunk)
        order = c0 + np.argsort([-len(texts[i]) for i in range(c0, c1)], kind="stable")   # less padding
        batches = [order[s:s + args.batch] for s in range(0, len(order), args.batch)]
        keys: dict[int, list[torch.Tensor]] = {k: [] for k in args.ks}
        pending = pool.submit(enc.tokenize_docs, [texts[i] for i in batches[0]])
        for bi, idx in enumerate(batches):
            tok_ids, mask = pending.result()
            if bi + 1 < len(batches):
                pending = pool.submit(enc.tokenize_docs, [texts[i] for i in batches[bi + 1]])
            t = time.perf_counter()
            emb, keep, _ = enc.embed_docs(tok_ids, mask)
            counts = keep.sum(1)
            synchronize(dev)
            spent["encode"] += time.perf_counter() - t
            t = time.perf_counter()
            lengths[idx] = counts.cpu().numpy()
            local = torch.from_numpy(idx - c0).to(dev)
            flat = emb[keep]
            seg = torch.repeat_interleave(local, counts)
            for k in args.ks:
                cid, _ = nearest_centroid(flat, cent_dev[k])
                keys[k].append(torch.unique(seg * k + cid))
            synchronize(dev)
            spent["assign"] += time.perf_counter() - t
            t = time.perf_counter()
            nb, length, _ = emb.shape
            group = max(1, (1 << 28) // (nb * length * m))
            for q0 in range(0, nq, group):
                q1 = min(nq, q0 + group)
                sims = emb @ q_mat[q0 * m:q1 * m].T                                   # (B, L, queries x m)
                sims.masked_fill_(~keep[..., None], -torch.inf)
                buf[q0:q1, local] = sims.amax(1).view(nb, q1 - q0, m).sum(-1).T
            synchronize(dev)
            spent["exact"] += time.perf_counter() - t
        for k in args.ks:
            u = torch.unique(torch.cat(keys[k]))                     # sorted by (passage, centroid)
            sizes[k][c0:c1] = torch.bincount(u // k, minlength=c1 - c0).cpu().numpy()
            parts[k].append((u % k).to(torch.int32).cpu().numpy())
        exact[:, c0:c1] = buf[:, :c1 - c0].cpu().numpy()
        el = time.perf_counter() - t_pass
        log.info("%d / %d passages (%.0f/s, ETA %.0f min) | encode %.0fs assign %.0fs exact %.0fs", c1, n, c1 / el,
                 (n - c1) / (c1 / el) / 60, spent["encode"], spent["assign"], spent["exact"])
    timings["pass_s"] = time.perf_counter() - t_pass
    timings.update({f"{name}_s": v for name, v in spent.items()})
    exact.flush()
    del exact
    np.save(out / "lengths.npy", lengths)
    raw = np.fromiter((len(s.encode("utf-8")) for s in texts), np.int64, n)
    sketch_bytes = {}
    for k in args.ks:
        ptr = np.zeros(n + 1, np.int64)
        np.cumsum(sizes[k], out=ptr[1:])
        sk = CentroidBagSketch(k, ptr, np.concatenate(parts[k]), None)
        sk.save(out / f"sketch_K{k}.npz")
        sketch_bytes[str(k)] = sk.storage_bytes()
        log.info("K=%d: %.1f entries/passage, %.1f B/passage", k, sk.sizes.mean(), sketch_bytes[str(k)] / n)
    write_json(out / "meta.json", {
        "encoder": args.encoder, "doc_maxlen": args.doc_maxlen, "n_docs": n, "n_queries": nq, "dim": dim, "m": m,
        "max_docs": args.max_docs, "ks": args.ks, "seed": args.seed, "qids": qids, "query_texts": qtexts, "rel": rel,
        "tokens": int(lengths.sum()), "raw_bytes": int(raw.sum()), "sketch_bytes": sketch_bytes,
        "train_docs": len(train_docs), "timings": timings})
    log.info("encode stage done in %.0f s", sum(v for kk, v in timings.items() if kk in ("load_s", "pass_s")))


# ----------------------------------------------------------------------------- eval
def stable_order(values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """idx sorted by (value desc, index asc): the order of a stable descending sort."""
    idx = idx.sort().values
    return idx[values[idx].sort(descending=True, stable=True).indices]


def candidates(scores: torch.Tensor, top: torch.Tensor, prefix: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-`prefix` passages by `scores`, then the exact top passages outside it and the next ones, all in
    stable decreasing-score order (length `width`); and the global rank of every exact top passage."""
    n = len(scores)
    head = stable_order(scores, torch.topk(scores, min(width, n), sorted=False).indices)
    v = scores[top]
    ar = torch.arange(n, device=scores.device)
    ranks = torch.stack([(scores > x).sum() + ((scores == x) & (ar < d)).sum() for x, d in zip(v, top)])
    pre = head[:prefix]
    missing = top[~torch.isin(top, pre)]
    tail = head[prefix:]
    tail = tail[~torch.isin(tail, missing)][:width - prefix - len(missing)]
    return torch.cat([pre, stable_order(scores, torch.cat([missing, tail]))]), ranks


def run_protocol(name: str, score_fn, exact: np.ndarray, top: np.ndarray, rel: list[list[int]], qids: list[str],
                 args: argparse.Namespace, dev: torch.device) -> dict:
    nq = len(top)
    width = args.prefix + top.shape[1]
    est_p = np.empty((nq, width), np.float32)
    exact_p = np.empty((nq, width), np.float32)
    cand = np.empty((nq, width), np.int64)
    ranks = np.empty((nq, top.shape[1]), np.int64)
    first10 = np.empty((nq, 10), np.int64)
    scan_ms = []
    score_fn(0)                                                      # warm-up (kernel compilation)
    for q in range(nq):
        synchronize(dev)
        t = time.perf_counter()
        s = score_fn(q)
        synchronize(dev)
        scan_ms.append(1000 * (time.perf_counter() - t))
        c, r = candidates(s, torch.from_numpy(top[q]).to(dev), args.prefix, width)
        cand[q], ranks[q] = c.cpu().numpy(), r.cpu().numpy()
        est_p[q] = s[c].cpu().numpy()
        exact_p[q] = np.asarray(exact[q])[cand[q]]
        first10[q] = cand[q, :10]
    pos = np.stack([np.searchsorted(np.sort(cand[q]), top[q, :10]) for q in range(nq)])
    topk_p = np.stack([np.argsort(cand[q])[pos[q]] for q in range(nq)])
    assert all((cand[q, topk_p[q]] == top[q, :10]).all() for q in range(nq))
    splits = kfold_splits(nq, 5, 0.4, args.seed)
    res = evaluate_stopping(est_p, exact_p, topk_p, splits, DELTAS, 10, args.stop_batch, 200, qids,
                            [str(i) for i in range(width)], {}, hazard=HazardSettings(64, 64, 1.0, seed=args.seed))
    rows = np.asarray(res["test_rows"])
    rel_t = [rel[q] for q in rows]
    for d, entry in res["deltas"].items():
        for rule, e in entry.items():
            stops = np.minimum(np.asarray(e["R_list"]), args.prefix)
            ranking = [cand[q, :p][np.argsort(-exact_p[q, :p], kind="stable")[:10]] for q, p in zip(rows, stops)]
            e.pop("ndcg10")                                            # judged against dummy ids, meaningless
            e["mrr10"] = float(mrr_at_10(ranking, rel_t).mean())
            e["ndcg10_qrels"] = float(ndcg_at_10(ranking, rel_t).mean())
            e["recall_vs_exact"] = float(np.mean([np.isin(top[q, :10], rk).mean() for q, rk in zip(rows, ranking)]))
            e["beyond_prefix"] = float((np.asarray(e["R_list"]) > args.prefix).mean())
    res.pop("per_query")
    res.pop("estimate_ndcg10")
    true_r = ranks[:, :10] + 1                                       # passages to recompute to reach each one
    res.update({
        "name": name, "prefix": args.prefix, "stop_batch": args.stop_batch,
        "scan_ms": {"mean": float(np.mean(scan_ms)), "p50": float(np.percentile(scan_ms, 50)),
                    "p90": float(np.percentile(scan_ms, 90))},
        "estimate_mrr10": float(mrr_at_10(list(first10[rows]), rel_t).mean()),
        "true_R_at_recall": {str(t): float(np.quantile(true_r, t)) for t in (0.9, 0.95, 0.99)},
        "true_R_full_recall": {"p50": float(np.percentile(true_r.max(1), 50)),
                               "p90": float(np.percentile(true_r.max(1), 90))},
        "top10_in_prefix": float((true_r <= args.prefix).mean()),
    })
    return res


def suffix(args: argparse.Namespace) -> str:
    return "" if args.stop_batch == 8 else f"_b{args.stop_batch}"


def stage_eval(args: argparse.Namespace) -> None:
    out = Path(args.out)
    meta = read_json(out / "meta.json")
    exact = np.load(out / "exact.npy", mmap_mode="r")
    queries = np.load(out / "queries.npy")
    rel, qids = meta["rel"], meta["qids"]
    nq, n = exact.shape
    dev = torch.device("cuda:0")
    top_path = out / "exact_top.npy"
    if top_path.exists():
        top1k = np.load(top_path)
    else:
        top1k = np.stack([torch.topk(torch.from_numpy(np.asarray(exact[q])).to(dev), 1000).indices.cpu().numpy()
                          for q in range(nq)])
        np.save(top_path, top1k)
    ref = {"mrr10": float(mrr_at_10(list(top1k), rel).mean()), "ndcg10": float(ndcg_at_10(list(top1k), rel).mean()),
           "recall1000": float(recall_at(list(top1k), rel, 1000).mean())}
    log.info("exact ColBERT: MRR@10 %.4f nDCG@10 %.4f R@1k %.4f", ref["mrr10"], ref["ndcg10"], ref["recall1000"])
    top = top1k[:, :args.top_exact]
    tokens, raw, dim = meta["tokens"], meta["raw_bytes"], meta["dim"]
    storage = {"raw_text": raw / n, "colbert_fp16": tokens * dim * 2 / n + 4,
               "colbertv2_2bit": tokens * (dim / 4 + 4) / n + 4, "colbertv2_1bit": tokens * (dim / 8 + 4) / n + 4}
    for k in meta["ks"]:
        if args.ks and k not in args.ks:
            continue
        sketch = CentroidBagSketch.load(out / f"sketch_K{k}.npz")
        est = TorchCentroidBagEstimator(sketch, np.load(out / f"codebook_K{k}.npy"), dev, kernel="triton",
                                        half_table=True)
        q_dev = torch.from_numpy(queries).to(dev)
        res = run_protocol(f"reco-K{k}", lambda q: est.score_tensor(q_dev[q]), exact, top, rel, qids, args, dev)
        res.update({"exact": ref, "storage": dict(storage, sketch=float(sketch.bytes_per_doc().mean()),
                                                  codebook_total=k * dim * 2),
                    "entries_per_doc": float(sketch.sizes.mean())})
        write_json(out / f"eval_K{k}{suffix(args)}.json", res)
        h = res["deltas"]["0.05"]
        log.info("K=%d: scan %.1f ms | estimate MRR %.4f | d=0.05 fixed R %.0f (MRR %.4f) hazard R %.0f (MRR %.4f)",
                 k, res["scan_ms"]["mean"], res["estimate_mrr10"], h["fixed"]["R_mean"], h["fixed"]["mrr10"],
                 h["hazard"]["R_mean"], h["hazard"]["mrr10"])
        del est
        torch.cuda.empty_cache()
    if args.bm25:
        from savemv.baselines.sparse import BM25
        _, texts = load_corpus(Path(args.data) / "corpus.jsonl", meta["max_docs"])
        t = time.perf_counter()
        bm25 = BM25()
        bm25.index(texts)
        index_s = time.perf_counter() - t
        del texts
        toks = bm25._tokenize(meta["query_texts"], ids=False)

        def bm25_scores(q: int) -> torch.Tensor:
            words = [w for w in toks[q] if w in bm25.retriever.vocab_dict]
            s = bm25.retriever.get_scores(words) if words else np.zeros(n, np.float32)
            return torch.from_numpy(np.asarray(s, dtype=np.float32)).to(dev)

        res = run_protocol("bm25", bm25_scores, exact, top, rel, qids, args, dev)
        res.update({"exact": ref, "index_s": index_s, "index_bytes_per_doc": bm25.index_bytes() / n})
        write_json(out / f"eval_bm25{suffix(args)}.json", res)
        h = res["deltas"]["0.05"]
        log.info("BM25 cascade: d=0.05 fixed R %.0f (MRR %.4f) hazard R %.0f (MRR %.4f)",
                 h["fixed"]["R_mean"], h["fixed"]["mrr10"], h["hazard"]["R_mean"], h["hazard"]["mrr10"])


# ----------------------------------------------------------------------------- latency
def stage_latency(args: argparse.Namespace) -> None:
    out = Path(args.out)
    meta = read_json(out / "meta.json")
    k = args.k
    ev = read_json(out / f"eval_K{k}{suffix(args)}.json")
    top = np.load(out / "exact_top.npy")[:, :10]
    _, texts = load_corpus(Path(args.data) / "corpus.jsonl", meta["max_docs"])
    enc = ColBERTEncoder(EncoderConfig(repo=meta["encoder"], doc_maxlen=meta["doc_maxlen"], device="cuda:0"))
    dev = enc.device
    sketch = CentroidBagSketch.load(out / f"sketch_K{k}.npz")
    centroids = np.load(out / f"codebook_K{k}.npy")
    est = TorchCentroidBagEstimator(sketch, centroids, dev, kernel="triton", half_table=True)
    nq = meta["n_queries"]
    picks = kfold_splits(nq, 5, 0.4, meta["seed"])[0].test[:args.num_queries]
    c0 = ev["calibration"][0]
    b = args.stop_batch
    result = {"k": k, "n_docs": meta["n_docs"], "max_candidates": args.prefix, "queries": len(picks),
              "stop_batch": b, "runs": {}, "skipped": {}}
    for rule, delta in (("fixed", "0.1"), ("hazard", "0.1"), ("fixed", "0.05"), ("hazard", "0.05"),
                        ("fixed", "0.01"), ("hazard", "0.01"), ("hazard_doc", "0.01")):
        r_off = ev["deltas"][delta][rule]["R_mean"]
        if r_off > args.max_latency_r:            # e.g. a fixed budget near the whole candidate prefix
            result["skipped"][f"{rule}@{delta}"] = r_off
            log.info("%s@%s skipped (offline R %.0f)", rule, delta, r_off)
            continue
        if rule == "fixed":
            cal = Calibration(c0["a"], c0["b"], float("inf"), 10, b, budget=int(c0["fixed_R"][delta]["fixed"]))
        else:
            cal = Calibration(c0["a"], c0["b"], float(c0["tau"][delta][rule]), 10, b, rule="hazard",
                              hazard=HazardModel.from_dict(c0["hazard"]))
        searcher = ReCoSearcher(enc, est, texts, cal, max_candidates=args.prefix)
        for q in picks[:3]:
            searcher.search(meta["query_texts"][q])                    # warm-up
        stages = {s: [] for s in ("total", "encode_query", "estimate", "recompute")}
        n_rec, recall, ranking = [], [], []
        for q in picks:
            r = searcher.search(meta["query_texts"][q])
            for s in stages:
                stages[s].append(r.timings_ms[s])
            n_rec.append(r.n_recomputed)
            recall.append(np.isin(top[q], r.doc_indices).mean())
            ranking.append(r.doc_indices)
        total = np.asarray(stages["total"])
        offline = np.asarray(ev["deltas"][delta][rule]["R_list"])[:len(picks)]   # fold 0 tests first
        result["runs"][f"{rule}@{delta}"] = {
            "ms_p50": float(np.percentile(total, 50)), "ms_p90": float(np.percentile(total, 90)),
            "ms_p99": float(np.percentile(total, 99)), "ms_mean": float(total.mean()),
            "ms_mean_by_stage": {s: float(np.mean(v)) for s, v in stages.items()},
            "recomputed_mean": float(np.mean(n_rec)), "recomputed_p90": float(np.percentile(n_rec, 90)),
            "recall_vs_exact": float(np.mean(recall)),
            "mrr10": float(mrr_at_10(ranking, [meta["rel"][q] for q in picks]).mean()),
            "recomputed_matches_offline": float(np.mean(np.asarray(n_rec) == offline)),
            "qps_single_stream": float(1000.0 / total.mean())}
        log.info("%s@%s: p50 %.1f p90 %.1f ms | estimate %.1f ms | R %.0f | recall %.3f", rule, delta,
                 result["runs"][f"{rule}@{delta}"]["ms_p50"], result["runs"][f"{rule}@{delta}"]["ms_p90"],
                 np.mean(stages["estimate"]), np.mean(n_rec), np.mean(recall))
    # sketch scan alone on the CPU (numba, all cores and 8 threads)
    import numba
    cpu = CentroidBagEstimator(sketch, centroids)
    queries = np.load(out / "queries.npy")
    cpu.score(queries[0])
    result["cpu_scan_ms"] = {}
    for threads in (numba.config.NUMBA_NUM_THREADS, 8):
        numba.set_num_threads(threads)
        t = time.perf_counter()
        for q in picks[:10]:
            cpu.score(queries[q])
        result["cpu_scan_ms"][str(threads)] = 1000 * (time.perf_counter() - t) / 10
    log.info("CPU sketch scan: %s ms", result["cpu_scan_ms"])
    write_json(out / f"latency_K{k}{suffix(args)}.json", result)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("encode", "eval", "latency"))
    ap.add_argument("--data", default="data/msmarco")
    ap.add_argument("--out", default="outputs/msmarco")
    ap.add_argument("--encoder", default="models/answerai-colbert-small-v1")
    ap.add_argument("--doc-maxlen", type=int, default=300)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--ks", type=int, nargs="*", default=[16384, 65536])
    ap.add_argument("--k", type=int, default=65536, help="latency stage codebook")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=65536, help="passages per streaming chunk")
    ap.add_argument("--train-docs", type=int, default=60000)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--prefix", type=int, default=65536, help="candidate prefix D of the offline protocol")
    ap.add_argument("--top-exact", type=int, default=16)
    ap.add_argument("--bm25", action="store_true")
    ap.add_argument("--num-queries", type=int, default=200)
    ap.add_argument("--stop-batch", type=int, default=8, help="recomputation batch / stopping granularity")
    ap.add_argument("--max-latency-r", type=float, default=4000, help="skip online runs above this offline R")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"encode": stage_encode, "eval": stage_eval, "latency": stage_latency}[args.stage](args)


if __name__ == "__main__":
    main()
