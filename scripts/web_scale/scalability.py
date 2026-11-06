"""Scalability: index size and recomputations as one corpus grows (nested random document subsets).

The exact scores and the sketch estimates of the full corpus are restricted to each subset, the exact top-k
is recomputed within it, and every stopping rule is re-evaluated with the benchmark protocol.

    python scripts/web_scale/scalability.py configs/paper_bench.yaml --dataset quora --set max_queries=1000 --k 16384
Writes outputs/scalability/<dataset>_K<k>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from savemv.config import load_config  # noqa: E402
from savemv.experiments.pipeline import Pipeline, hazard_settings, mask_excluded  # noqa: E402
from savemv.reco import TorchCentroidBagEstimator  # noqa: E402
from savemv.reco.stopping import evaluate_stopping, exact_topk  # noqa: E402
from savemv.utils import get_logger  # noqa: E402

log = get_logger("savemv.scalability")
RULES = ("fixed", "crc", "rank1", "patience", "hazard", "hazard_doc", "hazard_u")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--k", type=int, default=16384)
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.02, 0.05, 0.1, 0.25, 0.5, 1.0])
    args = ap.parse_args()
    overrides = {"dataset": args.dataset}
    for pair in args.set:
        key, _, value = pair.partition("=")
        overrides[key] = yaml.safe_load(value)
    cfg = load_config(args.config, overrides)
    pipe = Pipeline(cfg)
    data = pipe.load_data()
    store, queries = pipe.token_store(data), pipe.query_vectors(data)
    pairs = pipe.excluded_pairs(data)
    exact = mask_excluded(pipe.exact_scores(store, queries), pairs)
    sketch, centroids = pipe.sketch(store, args.k)
    scorer = TorchCentroidBagEstimator(sketch, centroids, pipe.device)
    est = mask_excluded(np.stack([scorer.score(q) for q in queries]), pairs)
    sketch_bytes = sketch.bytes_per_doc()
    raw = np.array([len(t.encode("utf-8")) for t in data.doc_texts], dtype=np.float64)
    perm = np.random.default_rng(0).permutation(data.n_docs)       # nested subsets: prefixes of one permutation
    out = {"dataset": args.dataset, "k": args.k, "points": []}
    for f in args.fractions:
        docs = np.sort(perm[:max(cfg.stop.k + 1, int(round(f * data.n_docs)))])
        e, x = est[:, docs], exact[:, docs]
        topk = exact_topk(x, cfg.stop.k)
        ev = evaluate_stopping(e, x, topk, pipe.query_splits(len(queries)), cfg.stop.deltas, cfg.stop.k,
                               cfg.stop.batch, cfg.stop.calib_top, data.query_ids, [data.doc_ids[d] for d in docs],
                               data.qrels, hazard=hazard_settings(cfg.stop, cfg.dataset, args.k))
        point = {"fraction": f, "n_docs": int(len(docs)), "raw_bytes": float(raw[docs].sum()),
                 "sketch_bytes": float(sketch_bytes[docs].sum()) + args.k * store.dim * 2,
                 "colbert_bytes": float(store.lengths[docs].sum() * store.dim * 2 + 4 * len(docs)),
                 "R95": ev["R_at_recall"]["0.95"], "rules": {}}
        for d, entry in ev["deltas"].items():
            point["rules"][d] = {r: {"R_mean": v["R_mean"], "R_p99": float(np.percentile(v["R_list"], 99)),
                                     "recall": v["recall"]} for r, v in entry.items() if r in RULES}
        out["points"].append(point)
        h = point["rules"]["0.05"]
        log.info("%s n=%d: R95 %d | d=0.05 fixed %.1f crc %.1f hazard %.1f", args.dataset, len(docs), point["R95"],
                 h["fixed"]["R_mean"], h["crc"]["R_mean"], h["hazard"]["R_mean"])
    path = Path("outputs/scalability") / f"{args.dataset}_K{args.k}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
