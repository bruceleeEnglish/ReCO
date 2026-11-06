"""XTR/WARP baseline on BEIR (paper Exp-1): the official WARP engine (Scheerer et al., SIGIR 2025;
github.com/jlscheerer/xtr-warp) with the XTR-base checkpoint it is built for, run as published (repository defaults:
nprobe 16, t' from the collection size, fused CPU kernels), on the same corpora, test queries, qrels and identical-id
exclusion as every other method. WARP's kernels are specific to XTR's 128-dimensional vectors, so it cannot serve the
other ColBERT checkpoints; like ConstBERT it is a separately trained model.

  prepare  (.venv)       <root>/beir/<ds>/collection.tsv, collection_map.json, questions.test.tsv, eval.json
  index    (.venv_warp)  WARP indexer on the GPU, then the conversion to WARP's compacted layout
  search   (.venv_warp)  CPU search with T threads, k = 10 (+1 when identical ids are excluded); nDCG@10 with this
                         repository's metric code, index size (WARP's files) and latency -> results/warp/<ds>.json

    .venv/bin/python scripts/official_baselines/warp_beir.py prepare --datasets scifact ...
    .venv_warp/bin/python scripts/official_baselines/warp_beir.py index --datasets scifact --nbits 4
    .venv_warp/bin/python scripts/official_baselines/warp_beir.py search --datasets scifact --nbits 4 --threads 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WARP = ROOT / "third_party" / "xtr-warp"
WARP_FILES = ("bucket_cutoffs.npy", "bucket_weights.npy", "centroids.npy", "sizes.compacted.pt",
              "codes.compacted.pt", "residuals.repacked.compacted.pt")


def env(args) -> None:
    root = Path(args.root).resolve()
    os.environ.update({"INDEX_ROOT": str(root / "indexes"), "EXPERIMENT_ROOT": str(root / "experiments"),
                       "BEIR_COLLECTION_PATH": str(root / "beir"), "LOTTE_COLLECTION_PATH": str(root / "lotte")})


def clean(s: str) -> str:
    return s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def stage_prepare(args) -> None:
    sys.path.insert(0, str(ROOT))
    import yaml
    from savemv.config import load_config
    from savemv.experiments.pipeline import Pipeline
    for item in args.datasets:
        ds, _, rest = item.partition(":")
        overrides = {"dataset": ds}
        for pair in filter(None, rest.split(",")):
            key, _, value = pair.partition("=")
            overrides[key] = yaml.safe_load(value)
        cfg = load_config(args.config, overrides)
        pipe = Pipeline(cfg)
        data = pipe.load_data()
        out = Path(args.root) / "beir" / ds
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "collection.tsv", "w", encoding="utf-8") as fh:
            for i, text in enumerate(data.doc_texts):        # title + " " + text, as the bench and XTR do
                fh.write(f"{i}\t{clean(text)}\n")
        (out / "collection_map.json").write_text(json.dumps({i: d for i, d in enumerate(data.doc_ids)}))
        with open(out / "questions.test.tsv", "w", encoding="utf-8") as fh:
            for qid, q in zip(data.query_ids, data.query_texts):
                fh.write(f"{qid}\t{clean(q)}\n")
        (out / "eval.json").write_text(json.dumps({
            "qids": data.query_ids, "qrels": {q: data.qrels.get(q, {}) for q in data.query_ids},
            "exclude_identical": bool(cfg.exclude_identical_ids),
            "raw_bytes": int(sum(len(t.encode("utf-8")) for t in data.doc_texts)), "n_docs": data.n_docs}))
        print(f"{ds}: {data.n_docs} docs, {len(data.query_ids)} queries -> {out}")


def warp_config(args, ds: str):
    sys.path.insert(0, str(WARP))
    from warp.engine.config import WARPRunConfig
    return WARPRunConfig(nranks=1, collection="beir", dataset=ds, datasplit="test", nbits=args.nbits, k=10)


def stage_index(args) -> None:
    env(args)
    os.chdir(WARP)
    sys.path.insert(0, str(WARP))
    from warp.engine.utils.collection_indexer import index
    from warp.engine.utils.index_converter import convert_index
    for ds in args.datasets:
        config = warp_config(args, ds)
        t = time.perf_counter()
        index(config)
        convert_index(os.path.join(config.index_root, config.index_name))
        Path(args.results).mkdir(parents=True, exist_ok=True)
        (Path(args.results) / f"{ds}.nbits{args.nbits}.build.json").write_text(
            json.dumps({"build_s": time.perf_counter() - t}))


def stage_search(args) -> None:
    env(args)
    os.chdir(WARP)
    sys.path.insert(0, str(WARP))
    sys.path.insert(1, str(ROOT))
    import numpy as np
    import torch
    from savemv.core.metrics import ndcg_per_query
    from warp.engine.searcher import WARPSearcher
    torch.set_num_threads(args.threads)
    for ds in args.datasets:
        config = warp_config(args, ds)
        ev = json.loads((Path(args.root) / "beir" / ds / "eval.json").read_text())
        cmap = json.loads((Path(args.root) / "beir" / ds / "collection_map.json").read_text())
        queries = [line.rstrip("\n").split("\t", 1) for line in
                   open(Path(args.root) / "beir" / ds / "questions.test.tsv", encoding="utf-8")]
        searcher = WARPSearcher(config)
        k = 10 + int(ev["exclude_identical"])
        for _, q in queries[:5]:
            searcher.search(q, k=k)                                    # warm-up
        rankings, ms = [], []
        for qid, q in queries:
            t = time.perf_counter()
            pids, *_ = searcher.search(q, k=k)
            ms.append(1000 * (time.perf_counter() - t))
            docs = [cmap[str(int(p))] for p in pids]
            if ev["exclude_identical"]:
                docs = [d for d in docs if d != qid]
            rankings.append(docs[:10])
        doc_ids = sorted({d for r in rankings for d in r})
        pos = {d: i for i, d in enumerate(doc_ids)}
        ndcg = ndcg_per_query([np.array([pos[d] for d in r], dtype=np.int64) for r in rankings],
                              [qid for qid, _ in queries], doc_ids, ev["qrels"], 10)
        index_dir = Path(config.index_root) / config.index_name
        size = sum((index_dir / f).stat().st_size for f in WARP_FILES if (index_dir / f).exists())
        build = Path(args.results) / f"{ds}.nbits{args.nbits}.build.json"
        res = {"dataset": ds, "engine": "WARP (xtr-warp, XTR-base)", "nbits": args.nbits, "nprobe": config.nprobe,
               "t_prime": config.t_prime, "threads": args.threads, "queries": len(queries),
               "ndcg10": float(ndcg.mean()), "per_query_ndcg10": ndcg.tolist(), "index_bytes": size,
               "storage_ratio": size / ev["raw_bytes"], "bytes_per_doc": size / ev["n_docs"],
               "latency_ms_mean": float(np.mean(ms)), "latency_ms_p50": float(np.percentile(ms, 50)),
               "latency_ms_p99": float(np.percentile(ms, 99)),
               "build_s": json.loads(build.read_text())["build_s"] if build.exists() else None}
        Path(args.results).mkdir(parents=True, exist_ok=True)
        (Path(args.results) / f"{ds}.nbits{args.nbits}.json").write_text(json.dumps(res))
        print(f"{ds}: nDCG@10 {res['ndcg10']:.4f} | index {size / 1e6:.1f} MB = {res['storage_ratio']:.2f}x raw | "
              f"latency {res['latency_ms_p50']:.1f} ms p50 ({args.threads} threads)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("prepare", "index", "search"))
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--config", default="configs/paper_bench.yaml")
    ap.add_argument("--root", default="outputs/warp")
    ap.add_argument("--results", default="outputs/warp/results")
    ap.add_argument("--nbits", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    args.root = str((ROOT / args.root).resolve()) if not Path(args.root).is_absolute() else args.root
    args.results = str((ROOT / args.results).resolve()) if not Path(args.results).is_absolute() else args.results
    {"prepare": stage_prepare, "index": stage_index, "search": stage_search}[args.stage](args)


if __name__ == "__main__":
    main()
