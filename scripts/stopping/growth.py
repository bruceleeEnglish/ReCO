"""Does a calibration survive corpus growth? Agent memories and other indexes keep receiving documents after
calibration, which moves the exact top-k and breaks Theorem 2's premise that the corpus is fixed.

A base corpus (a random fraction of the documents) gets its own codebook, score map, hazard model and thresholds, from
its own exact top-k. The other documents then arrive by INSERT: they are assigned to the same codebook and nothing is
retrained. At each size the frozen rules are tested against the exact top-k of the grown corpus, next to rules
recalibrated on that corpus (same codebook). Five-fold protocol of the benchmark, query-level CRC, same document
permutation as scripts/web_scale/scalability.py.

    python scripts/stopping/growth.py configs/paper_bench.yaml --datasets fiqa "quora:max_queries=1000"
Writes outputs/growth/<dataset>.json.
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

from optimality_gap import build_paths, fit_hazard_fold  # noqa: E402
from savemv.config import load_config  # noqa: E402
from savemv.core.codebook import assign_to_codebook, train_spherical_kmeans, training_sample  # noqa: E402
from savemv.experiments.pipeline import Pipeline, mask_excluded  # noqa: E402
from savemv.reco import CentroidBagEstimator, TorchCentroidBagEstimator  # noqa: E402
from savemv.reco.sketch import CentroidBagSketch  # noqa: E402
from savemv.reco.stopping import crc_budget, crc_margin, exact_topk  # noqa: E402
from savemv.utils import get_logger  # noqa: E402
from stopping_baselines import HEADLINE_K  # noqa: E402

log = get_logger("savemv.growth")
RULES = ("hazard", "margin", "fixed")


def base_sketch(pipe, store, k: int, base: np.ndarray) -> tuple[CentroidBagSketch, np.ndarray]:
    """Codebook trained on the base documents only; every document is then assigned to it, as INSERT does."""
    sc = pipe.cfg.sketch
    doc_of_token = store.doc_of_token()
    rows = np.flatnonzero(np.isin(doc_of_token, base))
    pick = np.sort(rows[training_sample(len(rows), k, sc.points_per_centroid, sc.min_train_points, sc.seed)])
    centroids = train_spherical_kmeans(np.asarray(store.vectors[pick]), k, sc.kmeans_iters, sc.seed, pipe.device)
    ids, resid = assign_to_codebook(store.vectors, centroids, pipe.device)
    return CentroidBagSketch.build(doc_of_token, ids, resid, k, store.n_docs), centroids


def calibrate_fold(sp, est, exact, paths, cfg, rng, deltas) -> dict:
    """The three rules of one fold, fitted and calibrated on (est, exact)."""
    ab, raw = fit_hazard_fold(sp, est, exact, paths, cfg, rng)
    cal = [paths[q] for q in sp.cal]
    hazard = [np.maximum.accumulate(raw(p)) for p in cal]
    margin = [p.gmax(*ab) for p in cal]
    ranks = np.stack([p.ranks for p in cal])
    return {"sp": sp, "ab": ab, "raw": raw,
            "hazard": {d: crc_margin(cal, hazard, d, strict=False) for d in deltas},
            "margin": {d: crc_margin(cal, margin, d, strict=False) for d in deltas},
            "fixed": {d: crc_budget(ranks, est.shape[1], d, strict=False) for d in deltas}}


def test_fold(fold: dict, paths, deltas, sink: dict) -> None:
    """Stops of the fold's test queries on `paths` with the fold's rules; sink[rule, delta] collects (R, recall)."""
    for q in fold["sp"].test:
        p = paths[q]
        g = {"hazard": np.maximum.accumulate(fold["raw"](p)), "margin": p.gmax(*fold["ab"])}
        for d in deltas:
            for rule in ("hazard", "margin"):
                s = p.stop(g[rule], fold[rule][d])
                sink[rule, d].append((s, float((p.ranks < s).mean())))
            s = min(fold["fixed"][d], p.n)
            sink["fixed", d].append((s, float((p.ranks < s).mean())))


def summary(sink: dict, deltas) -> dict:
    return {str(d): {r: {"R_mean": float(np.mean([x[0] for x in sink[r, d]])),
                         "recall": float(np.mean([x[1] for x in sink[r, d]]))} for r in RULES} for d in deltas}


def run_dataset(cfg, ds: str, k: int, bases: list[float], grown: list[float], deltas: list[float]) -> dict:
    pipe = Pipeline(cfg)
    data = pipe.load_data()
    store, queries = pipe.token_store(data), pipe.query_vectors(data)
    pairs = pipe.excluded_pairs(data)
    exact = mask_excluded(pipe.exact_scores(store, queries), pairs)
    perm = np.random.default_rng(0).permutation(data.n_docs)       # nested subsets: prefixes of one permutation
    splits = pipe.query_splits(len(queries))
    out = {"dataset": ds, "k_sketch": k, "n_docs": int(data.n_docs), "n_queries": len(queries), "bases": []}
    for f0 in bases:
        base = np.sort(perm[:int(round(f0 * data.n_docs))])
        t0 = time.perf_counter()
        sketch, centroids = base_sketch(pipe, store, k, base)
        scorer = (TorchCentroidBagEstimator(sketch, centroids, pipe.device) if pipe.device.type == "cuda"
                  else CentroidBagEstimator(sketch, centroids))
        est = mask_excluded(np.stack([scorer.score(q) for q in queries]), pairs)
        log.info("%s base %.2f: codebook and estimates in %.0f s", ds, f0, time.perf_counter() - t0)
        e0, x0 = est[:, base], exact[:, base]
        paths0 = build_paths(e0, x0, exact_topk(x0, cfg.stop.k), cfg)
        rng = np.random.default_rng(0)
        folds = [calibrate_fold(sp, e0, x0, paths0, cfg, rng, deltas) for sp in splits]
        entry = {"fraction": f0, "n_docs": int(len(base)), "points": []}
        for f in [f0] + [g for g in grown if g > f0]:
            docs = np.sort(perm[:int(round(f * data.n_docs))])
            e, x = est[:, docs], exact[:, docs]
            paths = paths0 if f == f0 else build_paths(e, x, exact_topk(x, cfg.stop.k), cfg)
            frozen = {(r, d): [] for r in RULES for d in deltas}
            again = {(r, d): [] for r in RULES for d in deltas}
            rng = np.random.default_rng(0)
            for fold, sp in zip(folds, splits):
                test_fold(fold, paths, deltas, frozen)
                test_fold(calibrate_fold(sp, e, x, paths, cfg, rng, deltas), paths, deltas, again)
            if f == f0:
                assert frozen == again, "at the base size the frozen rules are the recalibrated rules"
            point = {"fraction": f, "n_docs": int(len(docs)), "frozen": summary(frozen, deltas),
                     "recalibrated": summary(again, deltas)}
            entry["points"].append(point)
            for d in deltas:
                a, b = point["frozen"][str(d)]["hazard"], point["recalibrated"][str(d)]["hazard"]
                log.info("%s base %.2f -> %.2f d=%s: frozen R %.1f recall %.4f | recalibrated R %.1f recall %.4f", ds,
                         f0, f, d, a["R_mean"], a["recall"], b["R_mean"], b["recall"])
        out["bases"].append(entry)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--datasets", nargs="+", required=True, help="name[:key=value,...] as in run_bench.sh")
    ap.add_argument("--bases", nargs="+", type=float, default=[0.5, 0.8], help="fraction of documents at calibration")
    ap.add_argument("--grown", nargs="+", type=float, default=[0.6, 0.8, 1.0], help="fractions after the insertions")
    ap.add_argument("--deltas", nargs="+", type=float, default=[0.05, 0.01])
    ap.add_argument("--k", type=int, default=None, help="codebook size (default: the corpus's headline size)")
    ap.add_argument("--out", default="outputs/growth")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for item in args.datasets:
        ds, _, rest = item.partition(":")
        overrides = {"dataset": ds}
        for pair in filter(None, rest.split(",")):
            key, _, value = pair.partition("=")
            overrides[key] = yaml.safe_load(value)
        cfg = load_config(args.config, overrides)
        res = run_dataset(cfg, ds, args.k or HEADLINE_K[ds], args.bases, args.grown, args.deltas)
        Path(args.out, f"{ds}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        log.info("wrote %s", Path(args.out, f"{ds}.json"))


if __name__ == "__main__":
    main()
