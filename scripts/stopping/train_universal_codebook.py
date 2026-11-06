"""Train leave-one-dataset-out ("universal") codebooks for ReCo from other corpora's token stores.

A codebook trained without the target corpus is a model artifact (like the encoder weights), so the
index of a new corpus is just its centroid bags. For every target, tokens are sampled evenly from
all other datasets' stores and spherical k-means is run with the bench settings.

    .venv/bin/python scripts/stopping/train_universal_codebook.py configs/beir_bench.yaml --sizes 4096 16384 65536 \
        --datasets scifact nfcorpus arguana scidocs fiqa trec-covid quora --points 3000000
Writes models/codebooks/loo-<target>_K<k>.npy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from savemv.config import load_config  # noqa: E402
from savemv.core import TokenStore, train_spherical_kmeans  # noqa: E402
from savemv.utils import get_logger, resolve_device  # noqa: E402

log = get_logger("savemv.universal")


def sample_tokens(stores: dict[str, TokenStore], total: int, seed: int) -> np.ndarray:
    per = total // len(stores)
    rng = np.random.default_rng(seed)
    parts = []
    for name, store in stores.items():
        rows = np.sort(rng.choice(store.n_tokens, min(per, store.n_tokens), replace=False))
        parts.append(np.asarray(store.vectors[rows], dtype=np.float32))
        log.info("sampled %d tokens from %s", len(rows), name)
    return np.concatenate(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--sizes", nargs="+", type=int, default=[4096, 16384, 65536])
    ap.add_argument("--points", type=int, default=3_000_000)
    ap.add_argument("--out", default="models/codebooks")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.encoder.device)
    stores = {}
    for ds in args.datasets:
        path = Path(cfg.output_root) / ds / "tokens"
        if not TokenStore.exists(path):
            raise SystemExit(f"missing token store {path}; run the benchmark on {ds} first")
        stores[ds] = TokenStore.load(path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for target in args.datasets:
        sources = {k: v for k, v in stores.items() if k != target}
        points = sample_tokens(sources, args.points, cfg.sketch.seed)
        for k in args.sizes:
            path = out / f"loo-{target}_K{k}.npy"
            if path.exists():
                continue
            centroids = train_spherical_kmeans(points, k, cfg.sketch.kmeans_iters, cfg.sketch.seed, device)
            np.save(path, centroids.astype(np.float32))
            log.info("wrote %s (trained on %s)", path, ", ".join(sources))


if __name__ == "__main__":
    main()
