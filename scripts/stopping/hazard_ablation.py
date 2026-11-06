"""Hazard rule: theory check and ablations (benchmark protocol: 5-fold cross-calibration, CRC, batch 8, k 10).

Theory (Corollary: equalizing each query's residual risk is not optimal under an average-recall guarantee):
  hazard        label = an exact top-k document within the next H positions (the marginal value of continuing)
  residual      label = any exact top-k document still unseen (each query's residual risk), same features/model
Ablations: feature subsets, label horizon H, model class (logistic vs gradient-boosted trees).
Every variant is a statistic thresholded at lambda with lambda by CRC, so every row has the same guarantee.

    python scripts/stopping/hazard_ablation.py configs/paper_bench.yaml --datasets scifact nfcorpus scidocs fiqa --ks 4096 16384
Writes outputs/ablation/hazard_ablation.json and prints mean recomputations (recall) per variant.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from savemv.config import load_config  # noqa: E402
from savemv.experiments.pipeline import Pipeline, mask_excluded  # noqa: E402
from savemv.reco import CentroidBagEstimator, TorchCentroidBagEstimator  # noqa: E402
from savemv.reco.hazard import FEATURES, logistic_fit, training_rows  # noqa: E402
from savemv.reco.stopping import StopPath, crc_budget, crc_margin, exact_topk, fit_linear_map  # noqa: E402
from savemv.utils import get_logger  # noqa: E402

log = get_logger("savemv.hazard_ablation")
ALL = list(range(len(FEATURES)))
F = {name: i for i, name in enumerate(FEATURES)}
VARIANTS = {   # name -> (feature columns, label target, horizon, model)
    "hazard (all features)": (ALL, "hazard", 64, "logistic"),
    "residual risk (theory)": (ALL, "residual", 64, "logistic"),
    "margin only": ([F["margin"]], "hazard", 64, "logistic"),
    "patience only": ([F["log_patience"]], "hazard", 64, "logistic"),
    "margin + patience": ([F["margin"], F["log_patience"]], "hazard", 64, "logistic"),
    "no online margin": ([i for i in ALL if i != F["margin_online"]], "hazard", 64, "logistic"),
    "no flatness": ([i for i in ALL if i != F["flatness"]], "hazard", 64, "logistic"),
    "H=16": (ALL, "hazard", 16, "logistic"),
    "H=256": (ALL, "hazard", 256, "logistic"),
    "gradient-boosted trees": (ALL, "hazard", 64, "hgb"),
}


def labels(path: StopPath, target: str, horizon: int) -> np.ndarray:
    if target == "residual":
        return path.ranks.max() >= path.ps
    return path.hazard_labels(horizon)


def fit_statistic(paths, ab, cols, target, horizon, model, rng):
    """A function path -> running max of -logit(p_hat) for the variant."""
    xs, ys, ws = [], [], []
    for p in paths:
        rows, w = training_rows(len(p.ps), 256, 2000, rng)
        xs.append(p.features(*ab)[rows][:, cols])
        ys.append(labels(p, target, horizon)[rows])
        ws.append(w)
    x, y, w = np.concatenate(xs), np.concatenate(ys).astype(float), np.concatenate(ws)
    if model == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(x, y, sample_weight=w)

        def raw(f):
            prob = np.clip(clf.predict_proba(f)[:, 1], 1e-9, 1 - 1e-9)
            return -np.log(prob / (1 - prob))
    else:
        mean, scale = x.mean(0), x.std(0)
        scale[scale < 1e-12] = 1.0
        coef, intercept = logistic_fit((x - mean) / scale, y, w, 1.0)

        def raw(f):
            return -(((f - mean) / scale) @ coef + intercept)
    return lambda p: np.maximum.accumulate(raw(p.features(*ab)[:, cols])) if len(p.ps) else np.empty(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[4096, 16384])
    ap.add_argument("--deltas", nargs="+", type=float, default=[0.05, 0.01])
    ap.add_argument("--out", default="outputs/ablation/hazard_ablation.json")
    args = ap.parse_args()
    results = {}
    for ds in args.datasets:
        cfg = load_config(args.config, {"dataset": ds})
        pipe = Pipeline(cfg)
        data = pipe.load_data()
        store, queries = pipe.token_store(data), pipe.query_vectors(data)
        pairs = pipe.excluded_pairs(data)
        exact = mask_excluded(pipe.exact_scores(store, queries), pairs)
        topk = exact_topk(exact, cfg.stop.k)
        splits = pipe.query_splits(len(queries))
        for k in args.ks:
            sketch, centroids = pipe.sketch(store, k)
            estimator = (TorchCentroidBagEstimator(sketch, centroids, pipe.device) if pipe.device.type == "cuda"
                         else CentroidBagEstimator(sketch, centroids))
            est = mask_excluded(np.stack([estimator.score(q) for q in queries]), pairs)
            paths = [StopPath(est[q], exact[q], topk[q], cfg.stop.k, cfg.stop.batch, state=True,
                              ahead=cfg.stop.hazard_ahead) for q in range(len(queries))]
            res = {(name, d): [] for name in ["fixed", "crc margin"] + list(VARIANTS) for d in args.deltas}
            for sp in splits:
                ab = fit_linear_map(est, exact, sp.fit, cfg.stop.calib_top)
                fit, cal = [paths[q] for q in sp.fit], [paths[q] for q in sp.cal]
                stats = {"crc margin": lambda p: p.gmax(*ab)}
                for name, (cols, target, horizon, model) in VARIANTS.items():
                    stats[name] = fit_statistic(fit, ab, cols, target, horizon, model, np.random.default_rng(0))
                g = {name: {q: f(paths[q]) for q in np.r_[sp.cal, sp.test]} for name, f in stats.items()}
                ranks = np.stack([p.ranks for p in cal])
                for d in args.deltas:
                    budget = crc_budget(ranks, est.shape[1], d, strict=False)
                    res["fixed", d] += [(budget, float((paths[q].ranks < budget).mean())) for q in sp.test]
                    for name in stats:
                        lam = crc_margin(cal, [g[name][q] for q in sp.cal], d, strict=False)
                        for q in sp.test:
                            s = paths[q].stop(g[name][q], lam)
                            res[name, d].append((s, float((paths[q].ranks < s).mean())))
            for d in args.deltas:
                for name in ["fixed", "crc margin"] + list(VARIANTS):
                    r = np.array(res[name, d])
                    results.setdefault(f"{ds}/K{k}/{d}", {})[name] = {"R_mean": float(r[:, 0].mean()),
                                                                      "R_p99": float(np.percentile(r[:, 0], 99)),
                                                                      "recall": float(r[:, 1].mean())}
                row = results[f"{ds}/K{k}/{d}"]
                log.info("%s K=%d d=%s | %s", ds, k, d, " | ".join(f"{n} {v['R_mean']:.1f} ({v['recall']:.3f})"
                                                                for n, v in row.items()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
