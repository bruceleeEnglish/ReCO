"""Parameter study of the stopping rules on ReCo's sketches (paper Exp-8): the result size k and the size of the
calibration set, with the benchmark protocol (5-fold cross-calibration, batch 8, CRC).

  k       the exact top-k to recover, k in {1, 5, 10, 20, 50}; delta in {0.05, 0.01}
  n_cal   each fold's calibration set subsampled to n queries (the fit and test sets unchanged), repeated with
          several seeds; reports mean recomputations and the achieved recall (mean and worst seed): CRC's guarantee
          holds in expectation for any n with n + 1 >= 1 / delta, at the price of a more conservative threshold

    python scripts/stopping/param_study.py configs/paper_bench.yaml --datasets scidocs fiqa "quora:max_queries=1000"
Writes outputs/param_study/<dataset>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from savemv.config import load_config  # noqa: E402
from savemv.experiments.pipeline import Pipeline, mask_excluded  # noqa: E402
from savemv.reco import CentroidBagEstimator, HazardSettings, TorchCentroidBagEstimator  # noqa: E402
from savemv.reco.stopping import QuerySplit, evaluate_stopping, exact_topk  # noqa: E402
from savemv.utils import get_logger  # noqa: E402

log = get_logger("savemv.param_study")
HEADLINE_K = {"scifact": 4096, "nfcorpus": 4096, "arguana": 4096, "scidocs": 16384, "fiqa": 16384,
              "trec-covid": 65536, "quora": 16384}
RULES = ("fixed", "crc", "patience", "hazard", "rank1", "hazard_doc")


def summary(res: dict, deltas) -> dict:
    return {str(d): {r: {"R_mean": res["deltas"][str(d)][r]["R_mean"], "recall": res["deltas"][str(d)][r]["recall"]}
                     for r in RULES if r in res["deltas"][str(d)]} for d in deltas}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 20, 50])
    ap.add_argument("--k-deltas", nargs="+", type=float, default=[0.05, 0.01])
    ap.add_argument("--ncals", nargs="+", type=int, default=[19, 50, 100, 200])
    ap.add_argument("--ncal-delta", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="outputs/param_study")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for item in args.datasets:
        ds, _, rest = item.partition(":")
        overrides = {"dataset": ds}
        for pair in filter(None, rest.split(",")):
            key, _, value = pair.partition("=")
            overrides[key] = yaml.safe_load(value)
        cfg = load_config(args.config, overrides)
        st = cfg.stop
        pipe = Pipeline(cfg)
        data = pipe.load_data()
        store, queries = pipe.token_store(data), pipe.query_vectors(data)
        pairs = pipe.excluded_pairs(data)
        exact = mask_excluded(pipe.exact_scores(store, queries), pairs)
        sketch, centroids = pipe.sketch(store, HEADLINE_K[ds])
        estimator = (TorchCentroidBagEstimator(sketch, centroids, pipe.device) if pipe.device.type == "cuda"
                     else CentroidBagEstimator(sketch, centroids))
        est = mask_excluded(np.stack([estimator.score(q) for q in queries]), pairs)
        splits = pipe.query_splits(len(queries))
        hz = HazardSettings(st.hazard_horizon, st.hazard_ahead, st.hazard_l2, seed=st.seed)

        def run(topk, k, sp, deltas):
            return evaluate_stopping(est, exact, topk, sp, list(deltas), k, st.batch, st.calib_top, data.query_ids,
                                     data.doc_ids, data.qrels, hazard=hz)

        out = {"dataset": ds, "k_sketch": HEADLINE_K[ds], "n_queries": len(queries),
               "n_cal_full": [len(s.cal) for s in splits], "k": {}, "ncal": {}}
        for k in args.ks:
            t = time.perf_counter()
            out["k"][str(k)] = summary(run(exact_topk(exact, k), k, splits, args.k_deltas), args.k_deltas)
            e = out["k"][str(k)][str(args.k_deltas[0])]
            log.info("%s k=%d (%.0f s): %s", ds, k, time.perf_counter() - t,
                     " | ".join(f"{r} {v['R_mean']:.0f} ({v['recall']:.3f})" for r, v in e.items()))
        topk = exact_topk(exact, st.k)
        for n in args.ncals + [max(len(s.cal) for s in splits)]:
            reps = []
            for seed in range(args.seeds if n < min(len(s.cal) for s in splits) else 1):
                rng = np.random.default_rng(1000 + seed)
                sub = [QuerySplit(s.fit, np.sort(rng.choice(s.cal, min(n, len(s.cal)), replace=False)), s.test)
                       for s in splits]
                reps.append(summary(run(topk, st.k, sub, [args.ncal_delta]), [args.ncal_delta])[str(args.ncal_delta)])
            out["ncal"][str(n)] = {r: {"R_mean": float(np.mean([x[r]["R_mean"] for x in reps])),
                                       "R_std": float(np.std([x[r]["R_mean"] for x in reps])),
                                       "recall": float(np.mean([x[r]["recall"] for x in reps])),
                                       "recall_min": float(np.min([x[r]["recall"] for x in reps])),
                                       "reps": len(reps)} for r in reps[0]}
            log.info("%s n_cal=%d: %s", ds, n, " | ".join(f"{r} {v['R_mean']:.0f} ({v['recall']:.3f}, min "
                                                         f"{v['recall_min']:.3f})" for r, v in out["ncal"][str(n)].items()))
        Path(args.out, f"{ds}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        log.info("wrote %s", Path(args.out, f"{ds}.json"))


if __name__ == "__main__":
    main()
