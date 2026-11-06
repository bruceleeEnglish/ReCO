"""Does a calibration across users hold as memories grow? Exp-6 calibrates on other users' questions, so a memory that
grows moves within the calibrated population as long as it stays in its size range. Two measurements, K = 64k, delta 0.05:

  by_size         the cross-user folds of Exp-6, with recall and recomputations per quartile of memory size
  small_to_large  score map, hazard model and threshold fitted on the smaller half of the memories only and tested on the
                  larger half: memories that grew after calibration, with no recalibration

The folds must reproduce Table XII of the paper, which is asserted against outputs/memory/<bench>/memory.json.

    python scripts/stopping/memory_by_size.py --bench longmemeval --data data/memory/longmemeval_s_cleaned.json
Writes outputs/memory/<bench>/by_size.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from savemv.config import EncoderConfig  # noqa: E402
from savemv.core.codebook import assign_to_codebook  # noqa: E402
from savemv.core.exact import maxsim_matrix  # noqa: E402
from savemv.core.store import TokenStore  # noqa: E402
from savemv.data.memory import LOADERS  # noqa: E402
from savemv.encoders import ColBERTEncoder  # noqa: E402
from savemv.experiments.memory import TOP, MemoryConfig, _fit_map, calibrated_stops, subsketch  # noqa: E402
from savemv.reco.estimate import CentroidBagEstimator  # noqa: E402
from savemv.reco.hazard import HazardSettings, fit_hazard, hazard_gmax  # noqa: E402
from savemv.reco.sketch import CentroidBagSketch  # noqa: E402
from savemv.reco.stopping import StopPath, crc_margin  # noqa: E402
from savemv.utils import get_logger  # noqa: E402

log = get_logger("savemv.memory_by_size")


def frozen_stops(est: list, exact: list, paths: list, train: np.ndarray, test: np.ndarray, delta: float, settings,
                 seed: int) -> np.ndarray:
    """Fit on 40% of `train`, calibrate on the other 60%, stop the `test` tasks; no task of `test` is seen."""
    perm = np.random.default_rng(seed).permutation(train)
    n_fit = int(round(0.4 * len(perm)))
    fit, cal = np.sort(perm[:n_fit]), np.sort(perm[n_fit:])
    a, b = _fit_map(est, exact, fit)
    model = fit_hazard([paths[i] for i in fit], [(a, b)] * len(fit), settings)
    lam = crc_margin([paths[i] for i in cal], [hazard_gmax(model, paths[i], a, b) for i in cal], delta, strict=False)
    return np.array([paths[i].stop(hazard_gmax(model, paths[i], a, b), lam) for i in test])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--repeats", type=int, default=5, help="random fit / calibration splits of the smaller memories")
    args = ap.parse_args()
    cfg = MemoryConfig(args.bench, args.data)
    out = Path(f"outputs/memory/{cfg.bench}")
    device = torch.device("cuda:0")
    bench = LOADERS[cfg.bench](cfg.data)
    tasks = bench.tasks
    enc = ColBERTEncoder(EncoderConfig(repo=cfg.colbert, doc_maxlen=cfg.doc_maxlen, batch_size=256, device=str(device)))
    store = TokenStore.load(out / f"tokens_d{cfg.doc_maxlen}")
    queries = enc.encode_queries([t.query for t in tasks])
    exact_all = maxsim_matrix(store, queries, device)
    exact = [exact_all[i, t.units] for i, t in enumerate(tasks)]
    centroids = np.load(cfg.codebooks[-1]).astype(np.float32)
    assert len(centroids) == 65536, cfg.codebooks[-1]
    ids, resid = assign_to_codebook(store.vectors, centroids, device)
    sketch = CentroidBagSketch.build(store.doc_of_token(), ids, resid, len(centroids), store.n_docs)
    est = [CentroidBagEstimator(subsketch(sketch, t.units), centroids).score(queries[i]) for i, t in enumerate(tasks)]

    settings = HazardSettings(horizon=32, ahead=32)            # as evaluate() of savemv/experiments/memory.py
    sizes = np.array([len(t.units) for t in tasks])
    paths = [StopPath(e, x, np.argsort(-x, kind="stable")[:TOP], TOP, 8, state=True, ahead=settings.ahead)
             for e, x in zip(est, exact)]
    stops = calibrated_stops(est, exact, args.delta, settings)["hazard"]
    recall = np.array([(p.ranks < s).mean() for p, s in zip(paths, stops)])
    ref = json.loads((out / "memory.json").read_text(encoding="utf-8"))["methods"]["reco-K64k"]["hazard_stop"]
    assert abs(ref["R_mean"] - stops.mean()) < 1e-9, (ref["R_mean"], stops.mean())
    assert abs(ref["recall_of_colbert_top10"] - recall.mean()) < 1e-9, (ref["recall_of_colbert_top10"], recall.mean())

    res = {"benchmark": cfg.bench, "tasks": len(tasks), "delta": args.delta, "R_mean": float(stops.mean()),
           "recall": float(recall.mean()),
           "sizes": {"min": int(sizes.min()), "p25": float(np.percentile(sizes, 25)), "median": float(np.median(sizes)),
                     "p75": float(np.percentile(sizes, 75)), "max": int(sizes.max())}, "by_size": []}
    edges = np.percentile(sizes, [25, 50, 75])
    group = np.searchsorted(edges, sizes, side="left")
    for g in range(4):
        idx = np.flatnonzero(group == g)
        if len(idx):
            res["by_size"].append({"quartile": g + 1, "tasks": len(idx), "size_min": int(sizes[idx].min()),
                                   "size_max": int(sizes[idx].max()), "recall": float(recall[idx].mean()),
                                   "R_mean": float(stops[idx].mean())})
    small, large = np.flatnonzero(sizes <= np.median(sizes)), np.flatnonzero(sizes > np.median(sizes))
    rows, r_means = [], []
    for seed in range(args.repeats):
        st = frozen_stops(est, exact, paths, small, large, args.delta, settings, seed)
        rows.append(np.mean([(paths[i].ranks < s).mean() for i, s in zip(large, st)]))
        r_means.append(float(st.mean()))
    same = np.mean([np.mean([(paths[i].ranks < s).mean() for i, s in zip(small, frozen_stops(
        est, exact, paths, small, small, args.delta, settings, seed))]) for seed in range(args.repeats)])
    res["small_to_large"] = {"small_tasks": len(small), "large_tasks": len(large), "small_mean_size": float(sizes[small].mean()),
                             "large_mean_size": float(sizes[large].mean()), "recall": float(np.mean(rows)),
                             "recall_splits": [float(x) for x in rows], "R_mean": float(np.mean(r_means)),
                             "recall_on_small_itself": float(same)}
    (out / "by_size.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log.info("%s: sizes %s, recall %.4f, R %.1f", cfg.bench, res["sizes"], res["recall"], res["R_mean"])
    for row in res["by_size"]:
        log.info("%s quartile %d (%d-%d units, %d tasks): recall %.4f, R %.1f", cfg.bench, row["quartile"], row["size_min"],
                 row["size_max"], row["tasks"], row["recall"], row["R_mean"])
    s = res["small_to_large"]
    log.info("%s calibrated on smaller memories (%.0f units), tested on larger (%.0f units): recall %.4f, R %.1f, recall on "
             "the smaller ones %.4f", cfg.bench, s["small_mean_size"], s["large_mean_size"], s["recall"], s["R_mean"],
             s["recall_on_small_itself"])


if __name__ == "__main__":
    main()
