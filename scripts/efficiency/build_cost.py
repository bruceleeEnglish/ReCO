"""Index construction and insertion cost: uncompressed ColBERT (fp16), ColBERTv2/PLAID residual compression (2-bit),
ReCo with an in-domain codebook, and ReCo with a universal (leave-one-out) codebook.

Encoding is shared by all four and is timed once on a document sample; the rest is timed on the full token store.
Insertion of one document = encoding + its share of the per-token work (assignment / compression).

    python scripts/efficiency/build_cost.py configs/paper_bench.yaml --dataset fiqa --k 16384
Writes outputs/build_cost/<dataset>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from savemv.baselines.residual import ResidualCodec, plaid_num_centroids  # noqa: E402
from savemv.config import load_config  # noqa: E402
from savemv.core.codebook import assign_to_codebook, train_spherical_kmeans, training_sample  # noqa: E402
from savemv.experiments.pipeline import Pipeline  # noqa: E402
from savemv.reco.sketch import CentroidBagSketch  # noqa: E402
from savemv.utils import get_logger  # noqa: E402

log = get_logger("savemv.build_cost")


def timed(device: torch.device, fn, *args, **kwargs):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--k", type=int, default=16384)
    ap.add_argument("--sample", type=int, default=2000, help="documents used to time encoding")
    args = ap.parse_args()
    overrides = {"dataset": args.dataset}
    for pair in args.set:
        key, _, value = pair.partition("=")
        overrides[key] = yaml.safe_load(value)
    cfg = load_config(args.config, overrides)
    pipe = Pipeline(cfg)
    data = pipe.load_data()
    store = pipe.token_store(data)
    dev, sc = pipe.device, cfg.sketch
    n_docs, n_tok, dim = store.n_docs, store.n_tokens, store.dim
    out = {"dataset": args.dataset, "n_docs": n_docs, "n_tokens": n_tok, "dim": dim, "device": str(dev),
           "raw_bytes_per_doc": float(np.mean([len(t.encode("utf-8")) for t in data.doc_texts])), "methods": {}}

    sample = np.sort(np.random.default_rng(0).choice(n_docs, min(args.sample, n_docs), replace=False))
    texts = [data.doc_texts[i] for i in sample]
    pipe.encoder.encode_corpus(texts[:64])                                   # warm-up
    _, t_enc = timed(dev, pipe.encoder.encode_corpus, texts)
    enc_ms = 1000 * t_enc / len(texts)
    out["encode_ms_per_doc"] = enc_ms
    log.info("encoding: %.2f ms / doc", enc_ms)
    vectors = store.vectors

    # uncompressed ColBERT: nothing beyond encoding
    out["methods"]["colbert-fp16"] = {"train_s": 0.0, "per_token_s": 0.0, "insert_ms_per_doc": enc_ms,
                                      "bytes_per_doc": n_tok / n_docs * dim * 2 + 4}
    # ColBERTv2 / PLAID residual compression, 2-bit
    kp = plaid_num_centroids(n_tok)
    codecs, t_fit = timed(dev, ResidualCodec.fit, vectors, [2], kp, sc.kmeans_iters, sc.seed, dev,
                          sc.points_per_centroid, sc.min_train_points)
    _, t_cmp = timed(dev, codecs[0].reconstruct, vectors, dev)
    out["methods"]["colbertv2-2bit"] = {"k": kp, "train_s": t_fit, "per_token_s": t_cmp,
                                        "insert_ms_per_doc": enc_ms + 1000 * t_cmp / n_docs,
                                        "bytes_per_doc": n_tok / n_docs * codecs[0].bytes_per_vector(dim) + 4}
    # ReCo, in-domain codebook
    rows = training_sample(n_tok, args.k, sc.points_per_centroid, sc.min_train_points, sc.seed)
    centroids, t_train = timed(dev, train_spherical_kmeans, np.asarray(vectors[rows]), args.k, sc.kmeans_iters,
                               sc.seed, dev)

    def sketch_with(c):
        ids, resid = assign_to_codebook(vectors, c, dev)
        return CentroidBagSketch.build(store.doc_of_token(), ids, resid, len(c), n_docs)

    sk, t_assign = timed(dev, sketch_with, centroids)
    out["methods"][f"reco-K{args.k}"] = {"train_s": t_train, "per_token_s": t_assign,
                                        "insert_ms_per_doc": enc_ms + 1000 * t_assign / n_docs,
                                        "bytes_per_doc": float(sk.bytes_per_doc().mean())}
    # ReCo, universal codebook (trained on other corpora): no training at all
    uni = Path(f"models/codebooks/loo-{args.dataset}_K{args.k}.npy")
    if uni.exists():
        sk_u, t_u = timed(dev, sketch_with, np.load(uni).astype(np.float32))
        out["methods"][f"reco-K{args.k}-U"] = {"train_s": 0.0, "per_token_s": t_u,
                                              "insert_ms_per_doc": enc_ms + 1000 * t_u / n_docs,
                                              "bytes_per_doc": float(sk_u.bytes_per_doc().mean())}
    for name, m in out["methods"].items():
        log.info("%-16s train %.1fs | per-token pass %.1fs | insert %.2f ms/doc | %.0f B/doc", name, m["train_s"],
                 m["per_token_s"], m["insert_ms_per_doc"], m["bytes_per_doc"])
    path = Path("outputs/build_cost") / f"{args.dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
