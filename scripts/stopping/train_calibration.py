"""Exp-2 supplement: every stopping rule fitted and calibrated on BEIR *train* queries, tested on all test queries.

Calibration needs query texts and exact scores only, no relevance labels. SciFact, NFCorpus and FiQA ship train
queries, so query-level CRC applies at delta=0.01 (n_cal >= 4/delta - 1) where the test sets alone are too small for
it and the bench falls back to the pooled correction. The train queries are split R times into fit (40%) and
calibration (60%), as in the bench's folds; each split tests every test query once.

    python scripts/stopping/train_calibration.py configs/paper_bench.yaml --datasets scifact nfcorpus fiqa
Writes outputs/train_calibration/<dataset>.json.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from savemv.config import load_config  # noqa: E402
from savemv.core.exact import maxsim_matrix  # noqa: E402
from savemv.data import RetrievalDataset, load_dataset  # noqa: E402
from savemv.experiments.pipeline import Pipeline, mask_excluded  # noqa: E402
from savemv.reco import CentroidBagEstimator, TorchCentroidBagEstimator  # noqa: E402
from savemv.reco.stopping import QuerySplit, exact_topk  # noqa: E402
from savemv.utils import get_logger  # noqa: E402
from stopping_baselines import (HEADLINE_K, build_paths, evaluate_split, new_results,  # noqa: E402
                                summarize)

log = get_logger("savemv.train_calibration")


def train_queries(cfg, test: RetrievalDataset, n: int) -> RetrievalDataset:
    """Up to n train queries (seeded sample), none of which occurs in the test set by id or text."""
    tr = load_dataset(cfg.dataset, cfg.data_root, "train", cfg.hf_endpoint, cfg.max_docs, None, cfg.stop.seed)
    if tr.doc_ids != test.doc_ids:
        raise ValueError("train and test loads disagree on the corpus")
    ids, texts = set(test.query_ids), set(test.query_texts)
    keep = [i for i, (q, t) in enumerate(zip(tr.query_ids, tr.query_texts)) if q not in ids and t not in texts]
    avail = len(keep)
    if avail > n:
        keep = np.sort(np.random.default_rng(cfg.stop.seed).choice(keep, n, replace=False)).tolist()
    log.info("%s: %d train queries, %d without test overlap, using %d", cfg.dataset, tr.n_queries, avail, len(keep))
    return RetrievalDataset(cfg.dataset, test.doc_ids, test.doc_texts, [tr.query_ids[i] for i in keep],
                            [tr.query_texts[i] for i in keep], {})


def run_dataset(cfg, ds: str, deltas: list[float], p0: int, n_train: int, repeats: int) -> dict:
    pipe = Pipeline(cfg)
    data = pipe.load_data()
    store, queries = pipe.token_store(data), pipe.query_vectors(data)
    train = train_queries(cfg, data, n_train)
    qpath, xpath = pipe.out / f"train_queries_n{n_train}.npy", pipe.out / f"train_exact_n{n_train}.npy"
    if qpath.exists() and xpath.exists():
        tq, tx = np.load(qpath), np.load(xpath)
    else:
        tq = pipe.encoder.encode_queries(train.query_texts)
        tx = maxsim_matrix(store, tq, pipe.device)
        np.save(qpath, tq)
        np.save(xpath, tx)
    k_sketch = HEADLINE_K[ds]
    sketch, centroids = pipe.sketch(store, k_sketch)
    estimator = (TorchCentroidBagEstimator(sketch, centroids, pipe.device) if pipe.device.type == "cuda"
                 else CentroidBagEstimator(sketch, centroids))
    pairs_test, pairs_train = pipe.excluded_pairs(data), pipe.excluded_pairs(train)
    exact = np.concatenate([mask_excluded(pipe.exact_scores(store, queries), pairs_test),
                            mask_excluded(tx, pairs_train)])
    est = np.concatenate([mask_excluded(np.stack([estimator.score(q) for q in queries]), pairs_test),
                          mask_excluded(np.stack([estimator.score(q) for q in tq]), pairs_train)])
    n_test, n_tr = len(queries), len(tq)
    t0 = time.perf_counter()
    paths, darth_x, recall = build_paths(est, exact, exact_topk(exact, cfg.stop.k), cfg)
    log.info("%s: %d test + %d train queries, %d docs, paths in %.0f s", ds, n_test, n_tr, est.shape[1],
             time.perf_counter() - t0)

    corrections = ("query",)
    runs = []
    n_fit = int(round(cfg.stop.fit_fraction * n_tr))
    for r in range(repeats):
        perm = n_test + np.random.default_rng(cfg.stop.seed + r).permutation(n_tr)
        sp = QuerySplit(np.sort(perm[:n_fit]), np.sort(perm[n_fit:]), np.arange(n_test))
        res, declared = new_results(deltas, corrections)
        evaluate_split(sp, est, exact, paths, darth_x, recall, cfg, deltas, corrections, p0,
                       np.random.default_rng(r), res, declared)
        runs.append(summarize(res, declared, deltas, corrections))
        log.info("%s split %d: %s", ds, r, " | ".join(
            f"d={d} hazard {e['hazard']['R_mean']:.0f} ({e['hazard']['recall']:.3f})" for d, e in runs[-1].items()))

    out = {"dataset": ds, "k_sketch": k_sketch, "n_docs": int(est.shape[1]), "n_test": n_test, "n_train": n_tr,
           "n_fit": n_fit, "n_cal": n_tr - n_fit, "repeats": repeats, "p0": p0, "deltas": {}}
    for d in map(str, deltas):
        out["deltas"][d] = {}
        for name in runs[0][d]:
            rs = [run[d][name] for run in runs]
            per_query = np.mean([x["recall_list"] for x in rs], axis=0)   # each test query, averaged over splits
            out["deltas"][d][name] = {"R_mean": float(np.mean([x["R_mean"] for x in rs])),
                                      "R_p99": float(np.mean([x["R_p99"] for x in rs])),
                                      "recall": float(np.mean([x["recall"] for x in rs])),
                                      "recall_se": float(per_query.std(ddof=1) / np.sqrt(len(per_query))),
                                      "recall_min_split": float(min(x["recall"] for x in rs)),
                                      "R_mean_splits": [x["R_mean"] for x in rs],
                                      "recall_splits": [x["recall"] for x in rs],
                                      "recall_query": per_query.tolist()}
        log.info("%s d=%s | %s", ds, d, " | ".join(f"{n} {v['R_mean']:.0f} ({v['recall']:.3f})"
                                                     for n, v in out["deltas"][d].items()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus", "fiqa"])
    ap.add_argument("--deltas", nargs="+", type=float, default=[0.05, 0.01])
    ap.add_argument("--p0", type=int, default=32, help="LAET's fixed initial search (recomputations)")
    ap.add_argument("--n-train", type=int, default=1000, help="train queries per corpus (sampled if more)")
    ap.add_argument("--repeats", type=int, default=5, help="random fit / calibration splits of the train queries")
    ap.add_argument("--out", default="outputs/train_calibration")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for item in args.datasets:
        ds, _, rest = item.partition(":")
        overrides = {"dataset": ds}
        for pair in filter(None, rest.split(",")):
            key, _, value = pair.partition("=")
            overrides[key] = yaml.safe_load(value)
        cfg = load_config(args.config, overrides)
        res = run_dataset(cfg, ds, args.deltas, args.p0, args.n_train, args.repeats)
        Path(args.out, f"{ds}.json").write_text(json.dumps(res), encoding="utf-8")
        log.info("wrote %s", Path(args.out, f"{ds}.json"))


if __name__ == "__main__":
    main()
