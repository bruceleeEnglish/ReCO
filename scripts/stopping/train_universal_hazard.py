"""Train universal hazard models for ReCo: on the stop paths of other corpora, per codebook size.

The hazard model only has to rank recomputation states (savemv/reco/hazard.py), so it can ship with the
encoder like the universal codebook; a deployment then only calibrates the threshold lambda by CRC. For
every target dataset a logistic model is fitted on the paths of all other datasets (each dataset weighs
the same); `all_K<k>.json` uses every dataset (for new corpora such as agent memories).

    .venv/bin/python scripts/stopping/train_universal_hazard.py configs/beir_bench.yaml --ks 4096 16384 65536 \
        --datasets scifact nfcorpus scidocs fiqa trec-covid quora:max_queries=1000 \
                   arguana:encoder.query_maxlen=300,output_root=outputs_q300
Writes models/hazard/loo-<dataset>_K<k>.json and models/hazard/all_K<k>.json; the benchmark uses them with
--set stop.hazard_model=models/hazard/loo-{dataset}_K{k}.json (rule hazard_u). Needs the cached exact
scores and sketches of every dataset (run the benchmark on it first).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from savemv.config import load_config  # noqa: E402
from savemv.experiments.pipeline import Pipeline, hazard_settings, mask_excluded  # noqa: E402
from savemv.reco import CentroidBagEstimator, TorchCentroidBagEstimator  # noqa: E402
from savemv.reco.hazard import fit_hazard  # noqa: E402
from savemv.reco.stopping import StopPath, exact_topk, fit_linear_map  # noqa: E402
from savemv.utils import get_logger  # noqa: E402

log = get_logger("savemv.universal_hazard")


def parse_spec(spec: str) -> tuple[str, dict]:
    """'name[:key=value,...]' -> (name, overrides)."""
    name, _, rest = spec.partition(":")
    overrides = {"dataset": name}
    for pair in filter(None, rest.split(",")):
        key, _, value = pair.partition("=")
        overrides[key.strip()] = yaml.safe_load(value)
    return name, overrides


def dataset_paths(config: str, overrides: dict, k: int, max_queries: int, seed: int):
    """Stop paths (state=True) of a sample of the dataset's queries under the ReCo sketch with K=k."""
    cfg = load_config(config, overrides)
    pipe = Pipeline(cfg)
    data = pipe.load_data()
    store, queries = pipe.token_store(data), pipe.query_vectors(data)
    pairs = pipe.excluded_pairs(data)
    exact = mask_excluded(pipe.exact_scores(store, queries), pairs)
    topk = exact_topk(exact, cfg.stop.k)
    sketch, centroids = pipe.sketch(store, k)
    estimator = (TorchCentroidBagEstimator(sketch, centroids, pipe.device) if pipe.device.type == "cuda"
                 else CentroidBagEstimator(sketch, centroids))
    est = mask_excluded(np.stack([estimator.score(q) for q in queries]), pairs)
    rows = np.arange(len(queries))
    if len(rows) > max_queries:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_queries, replace=False))
    a, b = fit_linear_map(est, exact, rows, cfg.stop.calib_top)
    paths = [StopPath(est[q], exact[q], topk[q], cfg.stop.k, cfg.stop.batch, state=True, ahead=cfg.stop.hazard_ahead)
             for q in rows]
    return paths, (a, b), hazard_settings(cfg.stop, cfg.dataset)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--datasets", nargs="+", required=True, help="name[:key=value,...] (config overrides)")
    ap.add_argument("--ks", nargs="+", type=int, default=[4096, 16384, 65536])
    ap.add_argument("--max-queries", type=int, default=500, help="queries sampled per dataset")
    ap.add_argument("--out", default="models/hazard")
    args = ap.parse_args()
    out = Path(args.out)
    specs = [parse_spec(s) for s in args.datasets]
    for k in args.ks:
        per = {}
        for name, overrides in specs:
            paths, ab, settings = dataset_paths(args.config, overrides, k, args.max_queries, 0)
            per[name] = (paths, ab)
            log.info("K=%d %s: %d paths, a=%.3f b=%.3f", k, name, len(paths), *ab)

        def fit(names):
            paths = [p for n in names for p in per[n][0]]
            maps = [per[n][1] for n in names for _ in per[n][0]]
            weights = [1.0 / len(per[n][0]) for n in names for _ in per[n][0]]
            return fit_hazard(paths, maps, settings, weights)

        for target in per:
            fit([n for n in per if n != target]).save(out / f"loo-{target}_K{k}.json")
        fit(list(per)).save(out / f"all_K{k}.json")
        log.info("wrote %d leave-one-out models and all_K%d.json to %s", len(per), k, out)


if __name__ == "__main__":
    main()
