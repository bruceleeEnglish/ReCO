"""Throughput on one GPU (paper Exp-5): ReCo serving C concurrent queries on MS MARCO.

Each step recomputes the next b candidates of every in-flight query in one encoder batch (C x b documents) and
scores them with one batched MaxSim; each query keeps its own running top-k, hazard state and stop, exactly as the
single-query engine (savemv/reco/engine.py), with the calibration of the fold that tests it. New queries are admitted
as others finish (encoded together, then one sketch scan each). Reports queries per second, per-query latency from
admission to answer, recomputations, recall of the exact top-10, MRR@10 and peak GPU memory. The official PLAID
engine's throughput (colbert-ai Searcher.search_all) is measured by scripts/official_baselines/plaid_msmarco.py throughput.

    python scripts/web_scale/throughput.py --out outputs/msmarco --concurrency 1 4 16 64 --deltas 0.1 0.05
Writes <out>/throughput_reco.json.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from msmarco_scale import load_corpus, mrr_at_10  # noqa: E402
from savemv.config import EncoderConfig  # noqa: E402
from savemv.encoders.colbert import ColBERTEncoder  # noqa: E402
from savemv.reco import Calibration, CentroidBagSketch, HazardModel, ReCoSearcher, TorchCentroidBagEstimator  # noqa: E402
from savemv.reco.stopping import kfold_splits  # noqa: E402
from savemv.utils import get_logger, read_json, write_json  # noqa: E402

log = get_logger("savemv.throughput")


class Query:
    __slots__ = ("idx", "q", "searcher", "order", "vals", "n", "p", "best", "last", "resid", "done_ids",
                 "done_scores", "t0", "next_tok")

    def __init__(self, idx, q, searcher, order, vals, t0):
        self.idx, self.q, self.searcher, self.order, self.vals, self.t0 = idx, q, searcher, order, vals, t0
        self.n, self.p, self.best, self.last, self.resid = len(order), 0, np.full(10, -np.inf), -1, 0.0
        self.done_ids, self.done_scores, self.next_tok = [], [], None


def merge(parts: list[tuple[np.ndarray, np.ndarray]], pad_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate separately tokenized segments, padding them to a common length."""
    length = max(ids.shape[1] for ids, _ in parts)
    ids = np.full((sum(len(p[0]) for p in parts), length), pad_id, dtype=np.int64)
    mask = np.zeros_like(ids)
    row = 0
    for a, m in parts:
        ids[row:row + len(a), :a.shape[1]] = a
        mask[row:row + len(a), :a.shape[1]] = m
        row += len(a)
    return ids, mask


def run(args, enc, est, texts, meta, ev, delta: str, concurrency: int, n_queries: int) -> dict:
    dev = enc.device
    nq = meta["n_queries"]
    fold_of = np.empty(nq, np.int64)
    for f, sp in enumerate(kfold_splits(nq, 5, 0.4, meta["seed"])):
        fold_of[sp.test] = f
    searchers = []
    for c in ev["calibration"]:
        cal = Calibration(c["a"], c["b"], float(c["tau"][delta]["hazard"]), 10, args.batch, rule="hazard",
                          hazard=HazardModel.from_dict(c["hazard"]))
        searchers.append(ReCoSearcher(enc, est, texts, cal, max_candidates=args.prefix))
    k, b = 10, args.batch
    order_q = list(range(n_queries))
    pending, active, results = deque(order_q), [], {}
    pool = ThreadPoolExecutor(max_workers=args.tokenizer_threads)

    def prefetch(st: Query) -> None:
        """Tokenize the query's next segment on a worker thread while the GPU runs the current step."""
        seg = st.order[st.p:st.p + b]
        st.next_tok = (seg, pool.submit(enc.tokenize_docs, [texts[d] for d in seg]))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    steps = docs = 0
    spent = {"admit": 0.0, "tok_wait": 0.0, "gpu": 0.0, "cpu_update": 0.0}
    while pending or active:
        new = []
        while pending and len(active) + len(new) < concurrency:
            new.append(pending.popleft())
        if new:
            t_admit = time.perf_counter()
            qv = torch.from_numpy(enc.encode_queries([meta["query_texts"][i] for i in new])).to(dev)
            # one pass over the sketch for the whole group, then a batched selection of each query's candidates
            raw = est.score_batch_tensor(qv)
            a = torch.tensor([searchers[fold_of[i]].cal.a for i in new], device=dev)[:, None]
            bb = torch.tensor([searchers[fold_of[i]].cal.b for i in new], device=dev)[:, None]
            shat = a * raw + bb
            cand = torch.topk(shat, min(args.prefix, shat.shape[1]), dim=1, sorted=False).indices.sort(dim=1).values
            v, pos = shat.gather(1, cand).sort(dim=1, descending=True, stable=True)
            orders, vals = cand.gather(1, pos).cpu().numpy(), v.cpu().numpy()
            for row, (i, q) in enumerate(zip(new, qv)):
                st = Query(i, q, searchers[fold_of[i]], orders[row], vals[row], t_admit)
                prefetch(st)
                active.append(st)
            torch.cuda.synchronize()
            spent["admit"] += time.perf_counter() - t_admit
        # one step: the next b candidates of every in-flight query, one encoder batch
        t1 = time.perf_counter()
        segs, ids, parts = [], [], []
        for st in active:
            seg, fut = st.next_tok
            segs.append(len(seg))
            ids.extend(seg.tolist())
            parts.append(fut.result())
        t2 = time.perf_counter()
        spent["tok_wait"] += t2 - t1
        tok, mask = merge(parts, enc.pad_id)
        qs = torch.repeat_interleave(torch.stack([st.q for st in active]), torch.tensor(segs, device=dev), dim=0)
        # encode in length-sorted chunks so that short documents are not padded to the longest one
        lengths = mask.sum(1)
        perm = np.argsort(lengths, kind="stable")
        scores_t = torch.empty(len(ids), device=dev)
        for c0 in range(0, len(perm), args.chunk):
            sel = perm[c0:c0 + args.chunk]
            width = int(lengths[sel].max())
            emb, keep, _ = enc.embed_docs(tok[sel, :width], mask[sel, :width])
            sel_t = torch.from_numpy(sel).to(dev)
            sims = torch.einsum("bld,bmd->blm", emb, qs[sel_t].to(emb.dtype)).masked_fill(~keep[..., None], -torch.inf)
            scores_t[sel_t] = sims.max(dim=1).values.sum(dim=-1).float()
        scores = scores_t.cpu().numpy()
        t3 = time.perf_counter()
        spent["gpu"] += t3 - t2
        steps += 1
        docs += len(ids)
        off, still = 0, []
        for st, n_seg in zip(active, segs):
            sc = scores[off:off + n_seg].astype(np.float64)
            for j, s in enumerate(sc):
                lo = int(np.argmin(st.best))
                if s > st.best[lo]:
                    st.best[lo], st.last = s, st.p + j
            st.resid += float(sc.sum() - st.vals[st.p:st.p + n_seg].astype(np.float64).sum())
            st.done_ids.extend(st.order[st.p:st.p + n_seg].tolist())
            st.done_scores.extend(sc.tolist())
            st.p += n_seg
            off += n_seg
            stop = st.p >= st.n or (st.p >= k and st.searcher._statistic(
                st.vals, st.p, float(st.best.min()), st.last, st.resid / st.p) >= st.searcher.cal.tau)
            if stop:
                top = np.argsort(-np.asarray(st.done_scores), kind="stable")[:k]
                results[st.idx] = (np.asarray(st.done_ids)[top], st.p, 1000 * (time.perf_counter() - st.t0))
            else:
                prefetch(st)
                still.append(st)
        active = still
        spent["cpu_update"] += time.perf_counter() - t3
    torch.cuda.synchronize()
    wall = time.perf_counter() - t_start
    top10 = np.load(Path(args.out) / "exact_top.npy")[:, :10]
    rank = [results[i][0] for i in order_q]
    lat = np.array([results[i][2] for i in order_q])
    rs = np.array([results[i][1] for i in order_q])
    out = {"delta": delta, "concurrency": concurrency, "queries": n_queries, "batch": b, "wall_s": wall,
           "qps": n_queries / wall, "latency_ms_p50": float(np.percentile(lat, 50)),
           "latency_ms_p90": float(np.percentile(lat, 90)), "latency_ms_p99": float(np.percentile(lat, 99)),
           "R_mean": float(rs.mean()), "docs_per_s": docs / wall, "steps": steps,
           "recall_vs_exact10": float(np.mean([np.isin(top10[i], r).mean() for i, r in zip(order_q, rank)])),
           "mrr10": float(mrr_at_10(rank, [meta["rel"][i] for i in order_q]).mean()),
           "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9, "seconds_by_stage": spent}
    log.info("delta=%s C=%d: %.1f QPS | latency p50 %.0f p90 %.0f ms | R %.0f | %.0f docs/s | recall %.3f | MRR %.4f "
             "| peak %.1f GB | %s", delta, concurrency, out["qps"], out["latency_ms_p50"], out["latency_ms_p90"],
             out["R_mean"], out["docs_per_s"], out["recall_vs_exact10"], out["mrr10"], out["peak_gpu_gb"],
             {k: round(v, 1) for k, v in spent.items()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/msmarco")
    ap.add_argument("--out", default="outputs/msmarco")
    ap.add_argument("--k", type=int, default=65536)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--deltas", nargs="+", default=["0.1", "0.05"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--prefix", type=int, default=65536)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--tokenizer-threads", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=128, help="documents per length-sorted encoder call")
    ap.add_argument("--result", default="throughput_reco.json", help="output file inside --out")
    args = ap.parse_args()
    out = Path(args.out)
    meta, ev = read_json(out / "meta.json"), read_json(out / f"eval_K{args.k}.json")
    _, texts = load_corpus(Path(args.data) / "corpus.jsonl", meta["max_docs"])
    enc = ColBERTEncoder(EncoderConfig(repo=meta["encoder"], doc_maxlen=meta["doc_maxlen"], device="cuda:0"))
    est = TorchCentroidBagEstimator(CentroidBagSketch.load(out / f"sketch_K{args.k}.npz"),
                                    np.load(out / f"codebook_K{args.k}.npy"), enc.device, kernel="triton",
                                    half_table=True)
    n_queries = min(args.queries, meta["n_queries"])
    run(args, enc, est, texts, meta, ev, args.deltas[0], 4, 40)           # warm-up (kernels, allocator)
    res = {"k": args.k, "n_docs": meta["n_docs"], "runs": []}
    for d in args.deltas:
        for c in args.concurrency:
            res["runs"].append(run(args, enc, est, texts, meta, ev, d, c, n_queries))
            write_json(out / args.result, res)


if __name__ == "__main__":
    main()
