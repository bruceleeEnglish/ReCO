"""PLAID at the operating points where some queries exceed the 24 GB of an RTX 3090 (scripts/official_baselines/plaid_msmarco.py sweep stops
such a point at its first failure). Here a query that runs out of GPU memory is counted and skipped, so the others
still give latency and recall on the same GPU as ReCo. Recall is reported on the answered queries and, counting the
skipped queries as misses, on all of them.

    python scripts/official_baselines/plaid_sweep_skip.py
Writes outputs/msmarco/plaid_sweep_skip_nbits2.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plaid_msmarco as pm  # noqa: E402

POINTS = pm.SWEEP[2:4]       # (4, 0.4, 4096), colbert-ai's default for k > 100, and (6, 0.4, 4096)


def run_point(search, queries, sync, oom, release) -> tuple[list, list, list]:
    """One query at a time; returns the latency (ms), ranking and row of each answered query."""
    ms, rankings, rows = [], [], []
    for row, q in enumerate(queries):
        sync()
        t = time.perf_counter()
        try:
            pids = search(q)
        except oom:
            release()
            continue
        sync()
        ms.append(1000 * (time.perf_counter() - t))
        rankings.append(np.asarray(pids))
        rows.append(row)
    return ms, rankings, rows


def main() -> None:
    import torch
    from colbert import Searcher
    from colbert.infra import Run, RunConfig
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/msmarco")
    ap.add_argument("--index-root", default="outputs/plaid")
    ap.add_argument("--checkpoint", default="models/answerai-colbert-small-v1-colbertai")
    ap.add_argument("--nbits", type=int, default=2)
    ap.add_argument("--num-queries", type=int, default=1000)
    ap.add_argument("--result", default="plaid_sweep_skip_nbits2.json")
    args = ap.parse_args()
    meta = json.loads((Path(args.out) / "meta.json").read_text(encoding="utf-8"))
    top = np.load(Path(args.out) / "exact_top.npy")[:, :10]
    qs = meta["query_texts"][:args.num_queries]
    rel = meta["rel"][:len(qs)]
    path = Path(args.out) / args.result
    res = {"engine": "colbert-ai 0.2.22 (PLAID)", "nbits": args.nbits, "queries": len(qs), "device": "gpu", "points": []}
    with Run().context(RunConfig(nranks=1, experiment="msmarco", root=args.index_root, gpus=1)):
        searcher = Searcher(index=pm.index_name(args), config=pm.config(args),
                            index_root=str(Path(pm.index_path(args)).parent))

        def search(q: str) -> list:
            return searcher.search(q, k=10)[0]
        for ncells, threshold, ndocs in POINTS:
            searcher.configure(ncells=ncells, centroid_score_threshold=threshold, ndocs=ndocs)
            run_point(search, qs[:5], lambda: None, torch.cuda.OutOfMemoryError, torch.cuda.empty_cache)   # warm-up
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            ms, rankings, rows = run_point(search, qs, torch.cuda.synchronize, torch.cuda.OutOfMemoryError,
                                           torch.cuda.empty_cache)
            wall = time.perf_counter() - start
            recall = [float(np.isin(top[i], r).mean()) for i, r in zip(rows, rankings)]
            res["points"].append({
                "ncells": ncells, "centroid_score_threshold": threshold, "ndocs": ndocs, "answered": len(rows),
                "skipped": len(qs) - len(rows), "latency_ms": pm.stats(ms), "wall_s": wall, "qps": len(rows) / wall,
                "mrr10": float(pm.mrr_at_10(rankings, [rel[i] for i in rows]).mean()),
                "recall_vs_exact10": float(np.mean(recall)), "recall_with_skipped_as_miss": float(sum(recall) / len(qs)),
                "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9})
            print(json.dumps(res["points"][-1]), flush=True)
            path.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
