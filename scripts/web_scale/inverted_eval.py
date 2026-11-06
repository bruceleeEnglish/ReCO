"""Inverted index over the sketches on CPU (paper Exp-4/5): MS MARCO, 8 threads.

For every test query and recall target, the index produces the head of the estimate order that the query's
calibrated hazard stop needs (stop + 64 look-ahead + one batch, from eval_K*.json), and the result is checked to be
identical to the full CPU scan (same float32 estimator). Reports CPU time per query against the full scan, the probe
depth and candidates touched, and the build time and memory of the posting lists.

    python scripts/web_scale/inverted_eval.py --out outputs/msmarco --queries 200 --threads 8
Writes <out>/inverted_cpu.json.
"""
from __future__ import annotations

import os
import sys

_threads = sys.argv[sys.argv.index("--threads") + 1] if "--threads" in sys.argv else "8"
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_var] = _threads

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from savemv.reco import CentroidBagEstimator, CentroidBagSketch  # noqa: E402
from savemv.reco.inverted import InvertedSketchIndex  # noqa: E402
from savemv.utils import get_logger, read_json, write_json  # noqa: E402

log = get_logger("savemv.inverted_eval")


def stats(ms) -> dict:
    a = np.asarray(ms)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/msmarco")
    ap.add_argument("--k", type=int, default=65536)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--verify", type=int, default=50)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--start", type=int, default=2, help="initial centroids per query token")
    ap.add_argument("--lookahead", type=int, default=72, help="flatness look-ahead (64) + one batch (8)")
    args = ap.parse_args()
    out = Path(args.out)
    sk = CentroidBagSketch.load(out / f"sketch_K{args.k}.npz")
    cents = np.load(out / f"codebook_K{args.k}.npy")
    queries = np.load(out / "queries.npy")
    ev = read_json(out / f"eval_K{args.k}.json")
    rows = np.asarray(ev["test_rows"])[:args.queries]
    index = InvertedSketchIndex(sk, cents)
    log.info("posting lists built in %.1f s, %.2f GB in memory", index.build_s, index.memory_bytes / 1e9)
    full = CentroidBagEstimator(sk, cents)
    full.score(queries[0])
    index.prefix(queries[0], 100)                                         # warm-up (numba compilation)
    res = {"k": args.k, "n_docs": sk.n_docs, "threads": args.threads, "start": args.start, "build_s": index.build_s,
           "memory_gb": index.memory_bytes / 1e9, "deltas": {}}
    full_ms, identical, checked = [], 0, 0
    for d in ("0.1", "0.05", "0.01"):
        r_list = np.asarray(ev["deltas"][d]["hazard"]["R_list"])[:args.queries]
        ms, probes, cands = [], [], []
        for i, (q, r) in enumerate(zip(rows, r_list)):
            need = int(min(r + args.lookahead, sk.n_docs))
            t = time.perf_counter()
            order, _, st = index.prefix(queries[q], need, start=args.start)
            ms.append(1000 * (time.perf_counter() - t))
            probes.append(st["probe"])
            cands.append(st["candidates"])
            if i < args.verify and d == "0.05":
                t = time.perf_counter()
                est = full.score(queries[q])
                full_ms.append(1000 * (time.perf_counter() - t))
                ref = np.argsort(-est, kind="stable")[:need]
                checked += 1
                identical += int(len(order) == need and np.array_equal(order, ref))
        res["deltas"][d] = {"ms": stats(ms), "probe_median": float(np.median(probes)), "probe_max": int(max(probes)),
                            "candidates_mean": float(np.mean(cands)), "need_mean": float(np.mean(r_list) + args.lookahead)}
        log.info("delta=%s: %.1f ms mean (p50 %.1f, p99 %.1f) | probe median %d | %.0f candidates", d,
                 res["deltas"][d]["ms"]["mean"], res["deltas"][d]["ms"]["p50"], res["deltas"][d]["ms"]["p99"],
                 res["deltas"][d]["probe_median"], res["deltas"][d]["candidates_mean"])
    res["full_scan_ms"] = stats(full_ms)
    res["verified"] = {"identical": identical, "checked": checked}
    log.info("full scan %.1f ms mean; identical prefixes %d / %d", res["full_scan_ms"]["mean"], identical, checked)
    write_json(out / f"inverted_cpu_start{args.start}.json", res)


if __name__ == "__main__":
    main()
