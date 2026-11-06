"""How close hazard stopping comes to Theorem 1's policy, with the benchmark protocol (5 folds, batch 8, k 10).

Theorem 1 thresholds the true hazard and assumes that it is non-increasing along a path. The learned statistic is a
proxy, and patience resets when the top-k changes, so this script measures on the test queries of every fold:

  monotonicity   how often the statistic g = -logit h_hat falls between consecutive boundaries, that is, how often the
                 model's hazard rises, and how many of these steps follow a change of the top-k
  reliability    the empirical hazard (exact top-k documents in the next batch) per quantile bin of g, and its rank
                 correlation with g: thresholding g thresholds the true hazard if g orders the hazard
  (both on the boundaries where a rule decides: up to each query's stop at the smallest delta, pooled correction)
  bound          a clairvoyant lower bound on the mean recomputations of any stopping rule: it knows where each query's
                 exact top-k lies and spends the recall budget across queries optimally. Solved as the LP over each
                 query's useful stops by Lagrangian duality, so no rule can go below it, and no rule can reach it
                 either, because a stopping time cannot see the future

The hazard model is fitted as in scripts/stopping/stopping_baselines.py, so its stops must equal outputs/stopping_baselines.

    python scripts/stopping/optimality_gap.py configs/paper_bench.yaml --datasets scifact "quora:max_queries=1000"
Writes outputs/optimality_gap/<dataset>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from savemv.config import load_config  # noqa: E402
from savemv.experiments.pipeline import Pipeline, mask_excluded  # noqa: E402
from savemv.reco import CentroidBagEstimator, TorchCentroidBagEstimator  # noqa: E402
from savemv.reco.hazard import logistic_fit, training_rows  # noqa: E402
from savemv.reco.stopping import StopPath, crc_margin, exact_topk, fit_linear_map  # noqa: E402
from savemv.utils import get_logger  # noqa: E402
from stopping_baselines import HEADLINE_K  # noqa: E402

log = get_logger("savemv.optimality_gap")
CAP = 4096      # deepest boundary counted, for queries that the rule would never stop


def build_paths(est: np.ndarray, exact: np.ndarray, topk: np.ndarray, cfg) -> list[StopPath]:
    return [StopPath(est[q], exact[q], topk[q], cfg.stop.k, cfg.stop.batch, order=np.argsort(-est[q], kind="stable"),
                     state=True, ahead=cfg.stop.hazard_ahead) for q in range(len(est))]


def fit_hazard_fold(sp, est: np.ndarray, exact: np.ndarray, paths: list[StopPath], cfg, rng: np.random.Generator):
    """Score map and hazard model of one fold, as evaluate_split of stopping_baselines.py (same draws from rng).

    Returns (a, b) and raw(path), the statistic -logit h_hat at every boundary of a path, before the running max."""
    ab = fit_linear_map(est, exact, sp.fit, cfg.stop.calib_top)
    rows = {q: training_rows(len(paths[q].ps), 256, 2000, rng) for q in sp.fit}
    hx = np.concatenate([paths[q].features(*ab)[rows[q][0]] for q in sp.fit])
    hw = np.concatenate([rows[q][1] for q in sp.fit])
    mean, scale = hx.mean(0), hx.std(0)
    scale[scale < 1e-12] = 1.0
    y = np.concatenate([paths[q].hazard_labels(64)[rows[q][0]] for q in sp.fit]).astype(float)
    coef, icpt = logistic_fit((hx - mean) / scale, y, hw, cfg.stop.hazard_l2)

    def raw(path: StopPath) -> np.ndarray:
        return -(((path.features(*ab) - mean) / scale) @ coef + icpt)
    return ab, raw


def useful_stops(path: StopPath, batch: int) -> tuple[np.ndarray, np.ndarray]:
    """(stop, loss) of the k + 1 stops worth considering: the first boundary, then the boundary that covers each exact
    top-k document. Any other boundary costs more than one of these for the same loss."""
    first = int(path.ps[0]) if len(path.ps) else path.n
    r = np.sort(path.ranks.astype(np.int64))
    stops = np.minimum(np.ceil((r + 1) / batch).astype(np.int64) * batch, path.n)
    stops = np.r_[first, np.maximum(stops, first)]
    return stops.astype(np.float64), np.array([(r >= s).mean() for s in stops])


def clairvoyant_bound(stops: np.ndarray, losses: np.ndarray, delta: float, iters: int = 200) -> float:
    """Lower bound on the mean stop of any rule with mean loss <= delta (stops, losses: one row per query).

    The optimum of the LP that picks a distribution over each query's stops, by bisection on its Lagrange multiplier.
    Every multiplier gives a valid bound (weak duality); the bisection finds the tightest."""
    rows = np.arange(len(stops))

    def solve(mu: float) -> tuple[float, float]:
        pick = (stops + mu * losses).argmin(1)
        return float(stops[rows, pick].mean()), float(losses[rows, pick].mean())

    def dual(mu: float) -> float:
        cost, loss = solve(mu)
        return cost + mu * (loss - delta)

    lo, hi = 0.0, 1.0
    while solve(hi)[1] > delta and hi < 1e15:
        hi *= 2.0
    best = max(dual(lo), dual(hi))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        best = max(best, dual(mid))
        lo, hi = (mid, hi) if solve(mid)[1] > delta else (lo, mid)
    return best


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rho, with ties at their average rank."""
    return float(spearmanr(x, y)[0])


def reliability(g: np.ndarray, hits: np.ndarray, bins: int = 20) -> dict:
    """Mean statistic and empirical hazard per quantile bin of g."""
    edges = np.unique(np.quantile(g, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, g, side="right") - 1, 0, len(edges) - 2)
    keep = [b for b in range(len(edges) - 1) if (idx == b).any()]
    gm = np.array([g[idx == b].mean() for b in keep])
    hm = np.array([hits[idx == b].mean() for b in keep])
    return {"g": gm.tolist(), "hazard": hm.tolist(), "rank_correlation": rank_correlation(gm, hm),
            "inversions": int((np.diff(hm) > 0).sum()), "bins": len(keep)}


def run_dataset(cfg, ds: str, k_sketch: int, deltas: list[float], reference: Path) -> dict:
    pipe = Pipeline(cfg)
    data = pipe.load_data()
    store, queries = pipe.token_store(data), pipe.query_vectors(data)
    pairs = pipe.excluded_pairs(data)
    exact = mask_excluded(pipe.exact_scores(store, queries), pairs)
    topk = exact_topk(exact, cfg.stop.k)
    sketch, centroids = pipe.sketch(store, k_sketch)
    estimator = (TorchCentroidBagEstimator(sketch, centroids, pipe.device) if pipe.device.type == "cuda"
                 else CentroidBagEstimator(sketch, centroids))
    est = mask_excluded(np.stack([estimator.score(q) for q in queries]), pairs)
    batch, n_docs = cfg.stop.batch, est.shape[1]
    t0 = time.perf_counter()
    paths = build_paths(est, exact, topk, cfg)
    log.info("%s: %d queries, %d docs, paths in %.0f s", ds, len(queries), n_docs, time.perf_counter() - t0)

    rng = np.random.default_rng(0)
    stops = {(d, c): [] for d in deltas for c in ("query", "doc")}
    recalls = {(d, c): [] for d in deltas for c in ("query", "doc")}
    steps = falls = falls_after_change = queries_with_fall = 0
    pool_g, pool_hits, fall_sizes, g_std = [], [], [], []
    for sp in pipe.query_splits(len(queries)):
        ab, raw = fit_hazard_fold(sp, est, exact, paths, cfg, rng)
        g = {q: raw(paths[q]) for q in np.r_[sp.cal, sp.test]}
        gmax = {q: np.maximum.accumulate(v) for q, v in g.items()}
        cal = [paths[q] for q in sp.cal]
        for d in deltas:
            for c in ("query", "doc"):
                lam = crc_margin(cal, [gmax[q] for q in sp.cal], d, strict=False, correction=c)
                if (d, c) == (min(deltas), "doc"):
                    deepest = lam
                for q in sp.test:
                    s = paths[q].stop(gmax[q], lam)
                    stops[d, c].append(s)
                    recalls[d, c].append(float((paths[q].ranks < s).mean()))
        for q in sp.test:                   # the boundaries where a rule decides: up to the query's deepest stop
            p = paths[q]
            m = int(np.searchsorted(p.ps, min(p.stop(gmax[q], deepest), CAP), side="right"))
            if m < 2:
                continue
            v, ps = g[q][:m], p.ps[:m].astype(np.int64)
            fell = np.diff(v) < 0
            changed = p.since[1:m] < p.since[:m - 1] + batch     # the top-k changed within the step
            steps += m - 1
            falls += int(fell.sum())
            falls_after_change += int((fell & changed).sum())
            queries_with_fall += int(fell.any())
            fall_sizes.append(-np.diff(v)[fell])
            g_std.append(v)
            r = np.sort(p.ranks.astype(np.int64))
            pool_g.append(v)
            pool_hits.append(np.searchsorted(r, ps + batch, side="left") - np.searchsorted(r, ps, side="left"))
    g_all, hits = np.concatenate(pool_g), np.concatenate(pool_hits).astype(np.float64)
    table = [useful_stops(p, batch) for p in paths]
    s_mat, l_mat = np.stack([t[0] for t in table]), np.stack([t[1] for t in table])

    order = [int(i) for sp in pipe.query_splits(len(queries)) for i in sp.test]
    assert sorted(order) == list(range(len(queries)))
    out = {"dataset": ds, "k_sketch": k_sketch, "n_docs": n_docs, "n_queries": len(queries), "batch": batch,
           "monotonicity": {"steps": steps, "fall_share": falls / max(steps, 1),
                            "fall_after_topk_change_share": falls_after_change / max(falls, 1),
                            "queries_with_fall": queries_with_fall / len(queries),
                            "fall_mean": float(np.concatenate(fall_sizes).mean()) if falls else 0.0,
                            "g_std": float(np.concatenate(g_std).std())},
           "reliability": reliability(g_all, hits), "deltas": {}}
    ref = json.loads(reference.read_text(encoding="utf-8"))["deltas"] if reference.exists() else {}
    for d in deltas:
        entry = {}
        for c, name in (("query", "hazard"), ("doc", "hazard_doc")):
            r_mean, rec = float(np.mean(stops[d, c])), float(np.mean(recalls[d, c]))
            entry[name] = {"R_mean": r_mean, "recall": rec,
                           "bound_at_target": clairvoyant_bound(s_mat, l_mat, d),
                           "bound_at_achieved": clairvoyant_bound(s_mat, l_mat, max(1.0 - rec, 0.0))}
            if str(d) in ref and name in ref[str(d)]:
                entry[name]["reference_R_mean"] = ref[str(d)][name]["R_mean"]
                assert abs(ref[str(d)][name]["R_mean"] - r_mean) < 1e-6, (ds, d, name, ref[str(d)][name]["R_mean"], r_mean)
        out["deltas"][str(d)] = entry
        log.info("%s d=%s: hazard %.1f, bound %.1f (%.2fx)", ds, d, entry["hazard"]["R_mean"],
                 entry["hazard"]["bound_at_achieved"], entry["hazard"]["R_mean"] / entry["hazard"]["bound_at_achieved"])
    log.info("%s: g falls on %.1f%% of steps (%.0f%% after a top-k change), rank correlation %.3f", ds,
             100 * out["monotonicity"]["fall_share"], 100 * out["monotonicity"]["fall_after_topk_change_share"],
             out["reliability"]["rank_correlation"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--datasets", nargs="+", required=True, help="name[:key=value,...] as in run_bench.sh")
    ap.add_argument("--deltas", nargs="+", type=float, default=[0.05, 0.01])
    ap.add_argument("--out", default="outputs/optimality_gap")
    ap.add_argument("--reference", default="outputs/stopping_baselines", help="stops must equal these results")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for item in args.datasets:
        ds, _, rest = item.partition(":")
        overrides = {"dataset": ds}
        for pair in filter(None, rest.split(",")):
            key, _, value = pair.partition("=")
            overrides[key] = yaml.safe_load(value)
        cfg = load_config(args.config, overrides)
        res = run_dataset(cfg, ds, HEADLINE_K[ds], args.deltas, Path(args.reference) / f"{ds}.json")
        Path(args.out, f"{ds}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        log.info("wrote %s", Path(args.out, f"{ds}.json"))


if __name__ == "__main__":
    main()
