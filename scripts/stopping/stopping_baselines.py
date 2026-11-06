"""Learned early-termination baselines on ReCo's sketches (paper Exp-2), with the benchmark protocol
(5-fold cross-calibration, batch 8, k 10, same score map and same queries as the bench).

Every method decides, at batch boundaries of the recomputation order, whether to stop; all except
"darth-declared" threshold a monotone statistic whose threshold is set by CRC, so every calibrated row carries the
same recall guarantee, under the query-level and under the pooled per-document correction.

  darth    DARTH (Chatzakis, Papakonstantinou, Palpanas, SIGMOD 2025): a gradient-boosted regressor (LightGBM,
           100 trees) predicts the query's *current recall* from search-state features (steps, distance
           computations, result-set insertions, first / closest / furthest result, mean / variance / median /
           quartiles of the result scores; here exact MaxSim scores of the running top-k). Declared mode: predict at
           adaptive intervals (initial interval ipi = d/2, minimum mpi = d/10 with d the mean depth at which
           training queries reach the target; next interval mpi + (ipi - mpi)(Rt - Rp)) and stop once the predicted
           recall reaches the target Rt = 1 - delta. CRC mode: threshold the running maximum of the prediction,
           calibrated by CRC.
  laet     LAET (Li, Zhang, Andersen, He, SIGMOD 2020): after a fixed initial search (p0 recomputations) a
           gradient-boosted regressor predicts the depth that finds every exact top-k document (log scale) from the
           state at p0; the search stops at multiplier x prediction, the multiplier set by CRC.
  probtopk probabilistic top-k (Theobald, Weikum, Schenkel, VLDB 2004): stop once the estimated probability that
           some exact top-k document is still unseen falls below a threshold; the probability is estimated by the
           same features and logistic model as the hazard, with the residual-risk label.
  hazard / fixed / margin / patience  recomputed here as a consistency check against the bench.

    python scripts/stopping/stopping_baselines.py configs/paper_bench.yaml --datasets scifact fiqa "arguana:encoder.query_maxlen=300,output_root=outputs_q300"
Writes outputs/stopping_baselines/<dataset>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from savemv.config import load_config  # noqa: E402
from savemv.experiments.pipeline import Pipeline, mask_excluded  # noqa: E402
from savemv.reco import CentroidBagEstimator, TorchCentroidBagEstimator  # noqa: E402
from savemv.reco.hazard import logistic_fit, training_rows  # noqa: E402
from savemv.reco.stopping import StopPath, crc_budget, crc_margin, exact_topk, fit_linear_map  # noqa: E402
from savemv.utils import get_logger  # noqa: E402

log = get_logger("savemv.stopping_baselines")
HEADLINE_K = {"scifact": 4096, "nfcorpus": 4096, "arguana": 4096, "scidocs": 16384, "fiqa": 16384,
              "trec-covid": 65536, "quora": 16384}
DARTH_FEATURES = ("p", "log_p", "inserts", "first", "closest", "furthest", "mean", "var", "median", "q25", "q75")
NEG = -1.0   # LAET statistic before its initial search (every multiplier is positive); keeps CRC's bisection range small


@njit(cache=True)
def _darth_state(exact_sorted: np.ndarray, ps: np.ndarray, k: int) -> np.ndarray:
    """Search-state features at every boundary p (documents recomputed so far): DARTH_FEATURES."""
    out = np.empty((len(ps), 11), dtype=np.float32)
    top = np.full(k, -np.inf)
    inserts, best, j = 0, -np.inf, 0
    first = exact_sorted[0] if len(exact_sorted) else 0.0
    for b in range(len(ps)):
        p = ps[b]
        while j < p:
            s = exact_sorted[j]
            if s > best:
                best = s
            lo = 0
            for t in range(1, k):
                if top[t] < top[lo]:
                    lo = t
            if s > top[lo]:
                top[lo] = s
                inserts += 1
            j += 1
        cnt = 0
        for t in range(k):
            if top[t] > -np.inf:
                cnt += 1
        vals = np.empty(cnt)
        c = 0
        for t in range(k):
            if top[t] > -np.inf:
                vals[c] = top[t]
                c += 1
        vals.sort()
        out[b, 0] = p
        out[b, 1] = np.log(max(p, 1))
        out[b, 2] = inserts
        out[b, 3] = first
        out[b, 4] = best
        out[b, 5] = vals[0]
        out[b, 6] = vals.mean()
        out[b, 7] = vals.var()
        out[b, 8] = np.median(vals)
        out[b, 9] = np.percentile(vals, 25.0)
        out[b, 10] = np.percentile(vals, 75.0)
    return out


def recall_at(path: StopPath) -> np.ndarray:
    """Fraction of the exact top-k recomputed at every boundary."""
    return np.searchsorted(np.sort(path.ranks), path.ps, side="left") / len(path.ranks)


def gbdt(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=100, min_child_samples=max(2, min(20, len(y) // 10)), verbose=-1,
                             n_jobs=16, random_state=0).fit(x, y, sample_weight=w)


RULES = ("fixed", "margin", "patience", "hazard", "probtopk", "laet", "darth")


def build_paths(est: np.ndarray, exact: np.ndarray, topk: np.ndarray, cfg) -> tuple[list, list, list]:
    """Stop path, DARTH search state and recall at every boundary, one per query row."""
    k, batch = cfg.stop.k, cfg.stop.batch
    paths, darth_x, recall = [], [], []
    for q in range(len(est)):
        order = np.argsort(-est[q], kind="stable")
        path = StopPath(est[q], exact[q], topk[q], k, batch, order=order, state=True, ahead=cfg.stop.hazard_ahead)
        paths.append(path)
        darth_x.append(_darth_state(exact[q, order].astype(np.float64), path.ps.astype(np.int64), k))
        recall.append(recall_at(path))
    return paths, darth_x, recall


def new_results(deltas: list[float], corrections: tuple[str, ...]) -> tuple[dict, dict]:
    return {(m, d, c): [] for m in RULES for d in deltas for c in corrections}, {d: [] for d in deltas}


def evaluate_split(sp, est, exact, paths, darth_x, recall, cfg, deltas, corrections, p0, rng, res, declared) -> None:
    """Fit every rule on sp.fit, calibrate on sp.cal and append (stop, recall) of each sp.test query to res and
    declared."""
    batch, n_docs = cfg.stop.batch, est.shape[1]
    ab = fit_linear_map(est, exact, sp.fit, cfg.stop.calib_top)
    evalq = np.r_[sp.cal, sp.test]
    rows = {q: training_rows(len(paths[q].ps), 256, 2000, rng) for q in sp.fit}

    # hazard and probabilistic top-k: logistic on the hazard state, two labels
    hx = np.concatenate([paths[q].features(*ab)[rows[q][0]] for q in sp.fit])
    hw = np.concatenate([rows[q][1] for q in sp.fit])
    mean, scale = hx.mean(0), hx.std(0)
    scale[scale < 1e-12] = 1.0
    stats = {"margin": {q: paths[q].gmax(*ab) for q in evalq},
             "patience": {q: paths[q].patience_gmax() for q in evalq}}
    for name, label in (("hazard", lambda p: p.hazard_labels(64)), ("probtopk", lambda p: p.ranks.max() >= p.ps)):
        y = np.concatenate([label(paths[q])[rows[q][0]] for q in sp.fit]).astype(float)
        coef, icpt = logistic_fit((hx - mean) / scale, y, hw, cfg.stop.hazard_l2)
        stats[name] = {q: np.maximum.accumulate(-(((paths[q].features(*ab) - mean) / scale) @ coef + icpt))
                       for q in evalq}

    # DARTH: recall regressor on search-state features
    dx = np.concatenate([darth_x[q][rows[q][0]] for q in sp.fit])
    dy = np.concatenate([recall[q][rows[q][0]] for q in sp.fit])
    reg = gbdt(dx, dy, hw)
    pred = {q: reg.predict(darth_x[q]) for q in evalq}
    stats["darth"] = {q: np.maximum.accumulate(pred[q]) for q in evalq}

    # LAET: one prediction of the depth that finds every exact top-k, after p0 recomputations
    i0 = {q: min(int(np.searchsorted(paths[q].ps, p0, side="left")), len(paths[q].ps) - 1)
          for q in np.r_[sp.fit, evalq]}
    lx = lambda q: np.r_[paths[q].features(*ab)[i0[q]], darth_x[q][i0[q]]]  # noqa: E731
    lreg = gbdt(np.stack([lx(q) for q in sp.fit]), np.array([np.log1p(paths[q].ranks.max()) for q in sp.fit]))
    stats["laet"] = {}
    for q in evalq:
        depth = float(np.expm1(lreg.predict(lx(q)[None])[0]))
        g = paths[q].ps.astype(np.float64) / max(depth, 1.0)
        g[paths[q].ps < paths[q].ps[i0[q]]] = NEG
        stats["laet"][q] = g

    cal = [paths[q] for q in sp.cal]
    ranks = np.stack([p.ranks for p in cal])
    for d in deltas:
        for c in corrections:
            budget = crc_budget(ranks, n_docs, d, strict=False, correction=c)
            res["fixed", d, c] += [(min(budget, n_docs), float((paths[q].ranks < budget).mean()))
                                   for q in sp.test]
            for name in RULES[1:]:
                lam = crc_margin(cal, [stats[name][q] for q in sp.cal], d, strict=False, correction=c)
                for q in sp.test:
                    s = paths[q].stop(stats[name][q], lam)
                    res[name, d, c].append((s, float((paths[q].ranks < s).mean())))
        # DARTH as published: adaptive prediction intervals, stop at predicted recall >= target, no calibration
        rt = 1.0 - d
        reach = [paths[q].ps[np.argmax(recall[q] >= rt)] if (recall[q] >= rt).any() else n_docs for q in sp.fit]
        dist = float(np.mean(reach))
        ipi, mpi = max(batch, dist / 2), max(batch, dist / 10)
        for q in sp.test:
            pos, stop = ipi, n_docs
            while True:
                i = int(np.searchsorted(paths[q].ps, pos, side="left"))
                if i >= len(paths[q].ps):
                    break
                rp = float(pred[q][i])
                if rp >= rt:
                    stop = int(paths[q].ps[i])
                    break
                pos = paths[q].ps[i] + mpi + (ipi - mpi) * max(rt - rp, 0.0)
            declared[d].append((stop, float((paths[q].ranks < stop).mean())))


def summarize(res: dict, declared: dict, deltas: list[float], corrections: tuple[str, ...]) -> dict:
    """Per delta: R mean / p99 and recall of every rule over the appended test queries."""
    def row(r: np.ndarray) -> dict:
        return {"R_mean": float(r[:, 0].mean()), "R_p99": float(np.percentile(r[:, 0], 99)),
                "recall": float(r[:, 1].mean()), "R_list": r[:, 0].astype(int).tolist(),
                "recall_list": r[:, 1].tolist()}
    out = {}
    for d in deltas:
        entry = {}
        for c in corrections:          # every rule under both corrections, so the pooled group is complete
            for name in RULES:
                entry[name if c == "query" else f"{name}_doc"] = row(np.array(res[name, d, c]))
        entry["darth_declared"] = row(np.array(declared[d]))
        out[str(d)] = entry
    return out


def run_dataset(cfg, ds: str, k_sketch: int, deltas: list[float], p0: int) -> dict:
    pipe = Pipeline(cfg)
    data = pipe.load_data()
    store, queries = pipe.token_store(data), pipe.query_vectors(data)
    pairs = pipe.excluded_pairs(data)
    exact = mask_excluded(pipe.exact_scores(store, queries), pairs)
    topk = exact_topk(exact, cfg.stop.k)
    splits = pipe.query_splits(len(queries))
    sketch, centroids = pipe.sketch(store, k_sketch)
    estimator = (TorchCentroidBagEstimator(sketch, centroids, pipe.device) if pipe.device.type == "cuda"
                 else CentroidBagEstimator(sketch, centroids))
    est = mask_excluded(np.stack([estimator.score(q) for q in queries]), pairs)
    n_docs = est.shape[1]
    t0 = time.perf_counter()
    paths, darth_x, recall = build_paths(est, exact, topk, cfg)
    log.info("%s: %d queries, %d docs, paths in %.0f s", ds, len(queries), n_docs, time.perf_counter() - t0)
    corrections = ("query", "doc")
    res, declared = new_results(deltas, corrections)
    rng = np.random.default_rng(0)
    for sp in splits:
        evaluate_split(sp, est, exact, paths, darth_x, recall, cfg, deltas, corrections, p0, rng, res, declared)
    out = {"dataset": ds, "k_sketch": k_sketch, "n_docs": n_docs, "n_queries": len(queries), "p0": p0,
           "deltas": summarize(res, declared, deltas, corrections)}
    for d, entry in out["deltas"].items():
        log.info("%s d=%s | %s", ds, d, " | ".join(f"{n} {v['R_mean']:.0f} ({v['recall']:.3f})"
                                                     for n, v in entry.items()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--datasets", nargs="+", required=True, help="name[:key=value,...] as in run_bench.sh")
    ap.add_argument("--deltas", nargs="+", type=float, default=[0.1, 0.05, 0.01])
    ap.add_argument("--p0", type=int, default=32, help="LAET's fixed initial search (recomputations)")
    ap.add_argument("--out", default="outputs/stopping_baselines")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for item in args.datasets:
        ds, _, rest = item.partition(":")
        overrides = {"dataset": ds}
        for pair in filter(None, rest.split(",")):
            key, _, value = pair.partition("=")
            overrides[key] = yaml.safe_load(value)
        cfg = load_config(args.config, overrides)
        res = run_dataset(cfg, ds, HEADLINE_K[ds], args.deltas, args.p0)
        Path(args.out, f"{ds}.json").write_text(json.dumps(res), encoding="utf-8")
        log.info("wrote %s", Path(args.out, f"{ds}.json"))


if __name__ == "__main__":
    main()
