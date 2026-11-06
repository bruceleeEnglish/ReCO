"""The official PLAID engine (colbert-ai 0.2.22, ColBERTv2 residual compression) on MS MARCO, as a baseline for
scripts/web_scale/msmarco_scale.py: same passages in the same order (pid = line index), same 1,000 dev queries, same
encoder checkpoint, same exact top-10 and qrels. Runs in its own venv (.venv_plaid; see scripts/official_baselines/run_plaid.sh).

  prepare  -> <index_root>/collection.tsv, queries.tsv   (tabs/newlines become spaces: BERT treats both as whitespace)
  check    element-wise check of colbert-ai's encoder against vectors written by our encoder (--ref, .npz)
  index    colbert-ai Indexer (nbits, nranks GPUs); records wall time and on-disk size
  search   colbert-ai Searcher with its defaults for k = 10 (ncells 1, centroid threshold 0.5, ndocs 256),
           one query at a time; latency includes query encoding. --device cpu for the CPU engine.
           -> <out>/plaid_nbits<b>.json (merged per device)
  sweep    the same search at several operating points -> <out>/plaid_sweep_nbits2.json

    python scripts/official_baselines/plaid_msmarco.py search --device gpu
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np


def _raise_nccl_timeout() -> None:
    """With nranks > 1, rank 0 trains k-means alone (about 10 min on MS MARCO) while the other ranks wait in an NCCL
    all-reduce, longer than torch 2.2's 10-minute collective timeout. Module level, so it also runs in the ranks
    that multiprocessing spawns (they re-import this script); indexing itself is unchanged."""
    import datetime
    import torch.distributed as dist
    if getattr(dist.init_process_group, "_long_timeout", False):
        return
    init = dist.init_process_group

    def init_with_timeout(*args, **kwargs):
        kwargs.setdefault("timeout", datetime.timedelta(hours=3))
        print(f"[pid {os.getpid()}] init_process_group timeout {kwargs['timeout']}", flush=True)
        return init(*args, **kwargs)
    init_with_timeout._long_timeout = True
    dist.init_process_group = init_with_timeout


_raise_nccl_timeout()

# (ncells, centroid_score_threshold, ndocs): colbert-ai's defaults for k <= 10, k <= 100 and k > 100, then deeper points.
# From ncells 8 on, single queries exceed the 24 GB of an RTX 3090, so those points run on the CPU for their recall.
SWEEP = ((1, 0.5, 256), (2, 0.45, 1024), (4, 0.4, 4096), (6, 0.4, 4096), (8, 0.4, 8192), (16, 0.4, 16384))
# --grid low: lower centroid-score thresholds at the deepest probe, which prune fewer candidates (CPU only)
SWEEP_LOW = ((16, 0.3, 16384), (16, 0.2, 16384), (32, 0.2, 32768))


def mrr_at_10(rankings, rel) -> np.ndarray:
    out = np.zeros(len(rankings))
    for i, (row, r) in enumerate(zip(rankings, rel)):
        hits = np.flatnonzero(np.isin(np.asarray(row)[:10], r))
        out[i] = 1.0 / (hits[0] + 1) if len(hits) else 0.0
    return out


def ndcg_at_10(rankings, rel) -> np.ndarray:
    disc = 1.0 / np.log2(np.arange(2, 12))
    out = np.zeros(len(rankings))
    for i, (row, r) in enumerate(zip(rankings, rel)):
        gains = np.isin(np.asarray(row)[:10], r).astype(np.float64)
        out[i] = float((gains * disc[:len(gains)]).sum()) / float(disc[:min(10, len(r))].sum())
    return out


def stats(ms) -> dict:
    a = np.asarray(ms)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)), "n": len(a)}


def clean(text: str) -> str:
    return " ".join(text.replace("\t", " ").replace("\r", " ").replace("\n", " ").split(" "))


def stage_prepare(args) -> None:
    root = Path(args.index_root)
    root.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(Path(args.data) / "corpus.jsonl", encoding="utf-8") as src, \
            open(root / "collection.tsv", "w", encoding="utf-8") as dst:
        for line in src:
            d = json.loads(line)
            text = f"{d.get('title') or ''} {d.get('text') or ''}".strip()   # as msmarco_scale.load_corpus
            dst.write(f"{n}\t{clean(text)}\n")
            n += 1
    meta = json.loads((Path(args.out) / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_docs"] == n, (meta["n_docs"], n)
    with open(root / "queries.tsv", "w", encoding="utf-8") as fh:
        for i, q in enumerate(meta["query_texts"]):
            fh.write(f"{i}\t{clean(q)}\n")
    print(f"wrote {n} passages and {len(meta['query_texts'])} queries to {root}")


def config(args, **extra):
    from colbert.infra import ColBERTConfig
    # model_name: the checkpoint's metadata names its hub base model; the local copy holds the same weights
    return ColBERTConfig(nbits=args.nbits, doc_maxlen=300, query_maxlen=32, root=args.index_root,
                         experiment="msmarco", model_name=args.checkpoint, **extra)


def stage_check(args) -> None:
    """Our encoder's vectors (--ref: npz with docs, doc_vecs, doc_lens, queries, query_vecs) vs colbert-ai's."""
    import torch
    from colbert.modeling.checkpoint import Checkpoint
    ref = np.load(args.ref, allow_pickle=True)
    ckpt = Checkpoint(args.checkpoint, colbert_config=config(args)).cuda().eval()
    with torch.inference_mode():
        d, lens = ckpt.docFromText(list(ref["docs"]), bsize=32, keep_dims="flatten")
        q = ckpt.queryFromText(list(ref["queries"]), bsize=32)
    d, q = d.float().cpu().numpy(), q.float().cpu().numpy()
    same_lens = bool(np.array_equal(np.asarray(lens), ref["doc_lens"]))
    out = {"same_doc_lengths": same_lens, "max_abs_doc_vec_diff": float(np.abs(d - ref["doc_vecs"]).max()) if same_lens
           else None, "max_abs_query_vec_diff": float(np.abs(q - ref["query_vecs"]).max())}
    if same_lens:
        ptr = np.r_[0, np.cumsum(lens)]
        ours = ref["doc_vecs"]

        def maxsim(qv, dv):
            return np.array([(dv[ptr[j]:ptr[j + 1]] @ qv.T).max(0).sum() for j in range(len(lens))])
        out["max_abs_maxsim_diff"] = float(max(np.abs(maxsim(q[i], d) - maxsim(ref["query_vecs"][i], ours)).max()
                                               for i in range(len(q))))
    print(json.dumps(out))
    Path(args.out, "plaid_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def stage_index(args) -> None:
    from colbert import Indexer
    from colbert.infra import Run, RunConfig
    t = time.perf_counter()
    with Run().context(RunConfig(nranks=args.nranks, experiment="msmarco", root=args.index_root)):
        Indexer(checkpoint=args.checkpoint, config=config(args)).index(
            name=index_name(args), collection=str(Path(args.index_root) / "collection.tsv"), overwrite=True)
    build_s = time.perf_counter() - t
    path = index_path(args)
    size = int(subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True).stdout.split()[0])
    res = load_result(args)
    res.update({"index_path": str(path), "index_bytes": size, "build_s": build_s, "nranks": args.nranks})
    save_result(args, res)
    print(f"indexed in {build_s:.0f} s, {size / 1e9:.2f} GB")


def stage_search(args) -> None:
    import torch
    from colbert import Searcher
    from colbert.infra import Run, RunConfig
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    meta = json.loads((Path(args.out) / "meta.json").read_text(encoding="utf-8"))
    top = np.load(Path(args.out) / "exact_top.npy")[:, :10]
    qs = meta["query_texts"][:args.num_queries]
    with Run().context(RunConfig(nranks=1, experiment="msmarco", root=args.index_root,
                                 gpus=1 if args.device == "gpu" else 0)):
        searcher = Searcher(index=index_name(args), config=config(args), index_root=str(Path(index_path(args)).parent))
        for q in qs[:5]:
            searcher.search(q, k=10)                                   # warm-up (extension loading, caches)
        ms, rankings = [], []
        for q in qs:
            if args.device == "gpu":
                torch.cuda.synchronize()
            t = time.perf_counter()
            pids, _, _ = searcher.search(q, k=10)
            if args.device == "gpu":
                torch.cuda.synchronize()
            ms.append(1000 * (time.perf_counter() - t))
            rankings.append(np.asarray(pids))
    rel = meta["rel"][:len(qs)]
    res = load_result(args)
    res[args.device] = {
        "latency_ms": stats(ms), "mrr10": float(mrr_at_10(rankings, rel).mean()),
        "ndcg10": float(ndcg_at_10(rankings, rel).mean()),
        "recall_vs_exact10": float(np.mean([np.isin(top[i], r).mean() for i, r in enumerate(rankings)])),
        "queries": len(qs), "threads": args.threads if args.device == "cpu" else None,
        "affinity": sorted(os.sched_getaffinity(0)), "settings": "colbert-ai defaults for k=10"}
    save_result(args, res)
    print(json.dumps({k: v for k, v in res[args.device].items() if k != "affinity"}))


def stage_sweep(args) -> None:
    """Recall-latency curve of the official engine: one query at a time on one GPU, as stage_search, at each operating
    point of SWEEP. Starts with colbert-ai's defaults, so the first point must reproduce stage_search."""
    import torch
    from colbert import Searcher
    from colbert.infra import Run, RunConfig
    gpu = args.device == "gpu"
    sync = torch.cuda.synchronize if gpu else (lambda: None)
    if not gpu:
        torch.set_num_threads(args.threads)
    meta = json.loads((Path(args.out) / "meta.json").read_text(encoding="utf-8"))
    top = np.load(Path(args.out) / "exact_top.npy")[:, :10]
    qs = meta["query_texts"][:args.num_queries]
    rel = meta["rel"][:len(qs)]
    path = Path(args.out) / args.result
    res = {"engine": "colbert-ai 0.2.22 (PLAID)", "nbits": args.nbits, "queries": len(qs), "device": args.device,
           "points": []}
    with Run().context(RunConfig(nranks=1, experiment="msmarco", root=args.index_root, gpus=1 if gpu else 0)):
        searcher = Searcher(index=index_name(args), config=config(args), index_root=str(Path(index_path(args)).parent))
        grid = SWEEP if args.grid == "default" else SWEEP_LOW
        for ncells, threshold, ndocs in grid[args.start:args.points]:
            searcher.configure(ncells=ncells, centroid_score_threshold=threshold, ndocs=ndocs)
            try:
                for q in qs[:5]:
                    searcher.search(q, k=10)                           # warm-up (extension loading, caches)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
            if gpu:
                torch.cuda.reset_peak_memory_stats()
            ms, rankings = [], []
            start = time.perf_counter()
            try:
                for q in qs:
                    sync()
                    t = time.perf_counter()
                    pids, _, _ = searcher.search(q, k=10)
                    sync()
                    ms.append(1000 * (time.perf_counter() - t))
                    rankings.append(np.asarray(pids))
            except torch.cuda.OutOfMemoryError as err:                 # a point the GPU cannot hold, not a failed run
                res["points"].append({"ncells": ncells, "centroid_score_threshold": threshold, "ndocs": ndocs,
                                      "out_of_memory": True, "queries_done": len(ms), "error": str(err)[:200]})
                print(json.dumps(res["points"][-1]), flush=True)
                path.write_text(json.dumps(res, indent=1), encoding="utf-8")
                torch.cuda.empty_cache()
                continue
            wall = time.perf_counter() - start
            assert (searcher.config.ncells, searcher.config.centroid_score_threshold, searcher.config.ndocs) == \
                   (ncells, threshold, ndocs)
            res["points"].append({
                "ncells": ncells, "centroid_score_threshold": threshold, "ndocs": ndocs, "latency_ms": stats(ms),
                "wall_s": wall, "qps": len(qs) / wall, "mrr10": float(mrr_at_10(rankings, rel).mean()),
                "ndcg10": float(ndcg_at_10(rankings, rel).mean()),
                "recall_vs_exact10": float(np.mean([np.isin(top[i], r).mean() for i, r in enumerate(rankings)])),
                "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if gpu else None})
            print(json.dumps(res["points"][-1]), flush=True)
            path.write_text(json.dumps(res, indent=1), encoding="utf-8")


def stage_throughput(args) -> None:
    """Queries per second of the official engine's batch API (search_all) on one GPU, and its peak GPU memory."""
    import torch
    from colbert import Searcher
    from colbert.infra import Run, RunConfig
    meta = json.loads((Path(args.out) / "meta.json").read_text(encoding="utf-8"))
    qs = {i: t for i, t in enumerate(meta["query_texts"][:args.num_queries])}
    with Run().context(RunConfig(nranks=1, experiment="msmarco", root=args.index_root, gpus=1)):
        searcher = Searcher(index=index_name(args), config=config(args), index_root=str(Path(index_path(args)).parent))
        searcher.search_all({i: qs[i] for i in range(20)}, k=10)                  # warm-up
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        ranking = searcher.search_all(qs, k=10)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t
    flat = ranking.todict()
    rankings = [np.asarray([pid for pid, *_ in flat[i]]) for i in range(len(qs))]
    top = np.load(Path(args.out) / "exact_top.npy")[:, :10]
    res = load_result(args)
    res["throughput"] = {"queries": len(qs), "wall_s": wall, "qps": len(qs) / wall,
                         "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
                         "mrr10": float(mrr_at_10(rankings, meta["rel"][:len(qs)]).mean()),
                         "recall_vs_exact10": float(np.mean([np.isin(top[i], r).mean()
                                                             for i, r in enumerate(rankings)]))}
    save_result(args, res)
    print(json.dumps(res["throughput"]))


def index_name(args) -> str:
    return f"msmarco.nbits{args.nbits}"


def index_path(args) -> Path:
    return Path(args.index_root) / "msmarco" / "indexes" / index_name(args)


def load_result(args) -> dict:
    p = Path(args.out) / f"plaid_nbits{args.nbits}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"engine": "colbert-ai 0.2.22 (PLAID)",
                                                                       "nbits": args.nbits}


def save_result(args, res: dict) -> None:
    (Path(args.out) / f"plaid_nbits{args.nbits}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("prepare", "check", "index", "search", "sweep", "throughput"))
    ap.add_argument("--data", default="data/msmarco")
    ap.add_argument("--out", default="outputs/msmarco", help="msmarco_scale.py output (meta.json, exact_top.npy)")
    ap.add_argument("--index-root", default="outputs/plaid")
    ap.add_argument("--checkpoint", default="models/answerai-colbert-small-v1-colbertai")
    ap.add_argument("--nbits", type=int, default=2)
    ap.add_argument("--nranks", type=int, default=2)
    ap.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--num-queries", type=int, default=1000)
    ap.add_argument("--ref", default="outputs/plaid/encoder_ref.npz")
    ap.add_argument("--grid", choices=("default", "low"), default="default", help="sweep: SWEEP or SWEEP_LOW")
    ap.add_argument("--start", type=int, default=0, help="sweep: first operating point")
    ap.add_argument("--points", type=int, default=len(SWEEP), help="sweep: stop before this operating point")
    ap.add_argument("--result", default="plaid_sweep_nbits2.json", help="sweep: output file inside --out")
    args = ap.parse_args()
    {"prepare": stage_prepare, "check": stage_check, "index": stage_index, "search": stage_search,
     "sweep": stage_sweep, "throughput": stage_throughput}[args.stage](args)


if __name__ == "__main__":
    main()
