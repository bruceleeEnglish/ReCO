"""On-device (CPU) latency of agent-memory retrieval: one question against one user's memory.

Two stages, so that the replay can run on the device itself (a laptop) from a small pack:

  prepare (server, GPU for speed): samples questions, builds every sampled user's sketches with the universal
          codebooks, and writes a replay pack: questions, memory texts, sketches, codebooks, token counts, the units
          ReCo's calibrated hazard rule recomputed offline (memory.json -> hazard_stop.R_list) and index sizes.
  run     (any machine, CPU threads only; pin cores with taskset on a shared server): replays every question.
          ReCo encodes the query, scans the user's sketch (a per-query table over only the centroids that occur in
          this memory) and recomputes from raw text exactly the units its hazard rule recomputed offline, through
          the online engine in batches of 8 (the engine reproduces offline stops; the test suite checks it).
          Baselines over a per-user index: exact ColBERT on stored token vectors (fp32, the fastest CPU form; the
          vectors are random unit vectors with each unit's true token count, since the scan time does not depend on
          their values), re-encoding the whole memory from text (no index), BM25, and bge-small (stored unit vectors
          likewise random). Index construction is not timed. --gpu adds the same ReCo replay on a GPU; --reader times
          the prefill of the top-10 context by a 4B reader (GPU, and CPU for a few questions), for scale.

    python scripts/efficiency/cpu_latency.py prepare --bench longmemeval --data data/memory/longmemeval_s_cleaned.json \
        --memory outputs/memory/longmemeval/memory.json --pack outputs/cpu_latency/pack_longmemeval
    taskset -c 0-7 python scripts/efficiency/cpu_latency.py run --pack outputs/cpu_latency/pack_longmemeval --threads 8
Writes <out>/<bench>_<machine>.json.
"""
from __future__ import annotations

import os
import sys

_threads = sys.argv[sys.argv.index("--threads") + 1] if "--threads" in sys.argv else "8"
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, _threads)

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from savemv.config import EncoderConfig  # noqa: E402
from savemv.encoders import ColBERTEncoder  # noqa: E402
from savemv.reco import Calibration, CentroidBagEstimator, CentroidBagSketch, ReCoSearcher  # noqa: E402
from savemv.utils import get_logger, write_json  # noqa: E402

log = get_logger("savemv.cpu_latency")
BGE_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalBagEstimator:
    """The centroid-bag estimate for a small memory: the (centroids x query tokens) table is computed only for
    the centroids that occur in it, so a query costs O(|memory entries|) instead of O(K)."""

    def __init__(self, sketch: CentroidBagSketch, centroids: np.ndarray) -> None:
        self.sketch, self.centroids = sketch, centroids

    def score(self, query: np.ndarray) -> np.ndarray:
        sk = self.sketch
        uniq, local = np.unique(sk.cent, return_inverse=True)
        table = self.centroids[uniq] @ np.asarray(query, dtype=np.float32).T            # (|U|, m)
        out = np.full(sk.n_docs, -np.inf, dtype=np.float32)
        full = sk.sizes > 0
        out[full] = np.maximum.reduceat(table[local], sk.ptr[:-1][full], axis=0).sum(1)
        return out


def user_sketch(sketch: CentroidBagSketch, units: np.ndarray) -> CentroidBagSketch:
    ptr = np.zeros(len(units) + 1, dtype=np.int64)
    np.cumsum(sketch.sizes[units], out=ptr[1:])
    cent = np.concatenate([sketch.cent[sketch.ptr[u]:sketch.ptr[u + 1]] for u in units])
    return CentroidBagSketch(sketch.k, ptr, cent, None)


def stats(ms: list[float]) -> dict[str, float]:
    a = np.asarray(ms)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)), "n": len(a)}


def timed(fn) -> float:
    t = time.perf_counter()
    fn()
    return 1000 * (time.perf_counter() - t)


def cpu_name() -> str:
    if sys.platform.startswith("linux"):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if sys.platform == "win32":
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True)
        return out.stdout.strip()
    return platform.processor()


# ----------------------------------------------------------------------------- prepare
def stage_prepare(args: argparse.Namespace) -> None:
    from savemv.core.codebook import assign_to_codebook
    from savemv.core.store import TokenStore
    from savemv.data.memory import LOADERS
    from savemv.utils import read_json

    pack = Path(args.pack)
    pack.mkdir(parents=True, exist_ok=True)
    bench = LOADERS[args.bench](args.data)
    mem = read_json(Path(args.memory))
    store = TokenStore.load(Path(args.tokens) if args.tokens else Path(args.memory).parent / "tokens_d300")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    picks = np.sort(np.random.default_rng(0).choice(len(bench.tasks), min(args.tasks, len(bench.tasks)),
                                                    replace=False))
    tasks = [bench.tasks[i] for i in picks]
    used = np.unique(np.concatenate([t.units for t in tasks]))          # pack-local unit index = position in used
    local = {int(u): i for i, u in enumerate(used)}
    meta = {"bench": bench.name, "storage_bytes_per_unit": mem["storage_bytes_per_unit"],
            "tokens": store.lengths[used].tolist(), "texts": [bench.texts[u] for u in used], "codebooks": [],
            "tasks": [{"qid": t.qid, "query": t.query, "units": [local[int(u)] for u in t.units]} for t in tasks]}
    for path in args.codebooks:
        centroids = np.load(path).astype(np.float32)
        k = len(centroids)
        name = f"reco-K{k // 1024}k"
        ids, resid = assign_to_codebook(store.vectors, centroids, device)
        sk = CentroidBagSketch.build(store.doc_of_token(), ids, resid, k, store.n_docs)
        user_sketch(sk, used).save(pack / f"sketch_{name}.npz")
        np.save(pack / f"codebook_{name}.npy", centroids)
        r_list = np.asarray(mem["methods"][name]["hazard_stop"]["R_list"])[picks]
        meta["codebooks"].append(name)
        for t, r in zip(meta["tasks"], r_list):
            t[f"R_{name}"] = int(r)
    (pack / "pack.json").write_text(json.dumps(meta), encoding="utf-8")
    log.info("pack %s: %d questions, %d units", pack, len(tasks), len(used))


# ----------------------------------------------------------------------------- run
def stage_run(args: argparse.Namespace) -> None:
    from savemv.baselines.dense import DenseEncoder
    from savemv.baselines.sparse import BM25

    torch.set_num_threads(args.threads)
    pack = Path(args.pack)
    meta = json.loads((pack / "pack.json").read_text(encoding="utf-8"))
    texts, tokens, tasks = meta["texts"], np.asarray(meta["tokens"]), meta["tasks"]
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    machine = args.machine or platform.node()
    report: dict = {"bench": meta["bench"], "machine": machine, "cpu": cpu_name(), "threads": args.threads,
                    "affinity": affinity, "torch": torch.__version__, "tasks": len(tasks),
                    "units_per_memory": float(np.mean([len(t["units"]) for t in tasks])),
                    "storage_bytes_per_unit": meta["storage_bytes_per_unit"], "methods": {}}
    log.info("%s on %s (%s), %d threads: %d questions, %.0f units per memory", meta["bench"], machine, report["cpu"],
             args.threads, len(tasks), report["units_per_memory"])
    sketches = {name: (CentroidBagSketch.load(pack / f"sketch_{name}.npz"), np.load(pack / f"codebook_{name}.npy"))
                for name in meta["codebooks"]}
    rng = np.random.default_rng(0)
    enc = ColBERTEncoder(EncoderConfig(repo=args.colbert, doc_maxlen=300, device="cpu", fp16=False))
    dim = enc.dim
    for t in tasks[:3]:                                                   # warm-up
        enc.encode_query_tensor(t["query"])

    # ---- ReCo through the online engine
    for name, (sketch, centroids) in sketches.items():
        stages = {s: [] for s in ("total", "encode_query", "estimate", "recompute")}
        rs = []
        for i, t in enumerate(tasks):
            units = np.asarray(t["units"])
            est = LocalBagEstimator(user_sketch(sketch, units), centroids)
            if i < 3:
                q = enc.encode_queries([t["query"]])[0]
                ref = CentroidBagEstimator(est.sketch, centroids).score(q)
                assert np.allclose(est.score(q), ref, atol=1e-4), "local estimator differs from the reference"
            r = max(t[f"R_{name}"], 10)
            res = ReCoSearcher(enc, est, [texts[u] for u in units],
                               Calibration(1.0, 0.0, float("inf"), 10, 8, budget=r)).search(t["query"])
            for s in stages:
                stages[s].append(res.timings_ms[s])
            rs.append(t[f"R_{name}"])
        report["methods"][name] = {"R_mean": float(np.mean(rs)), **{s: stats(v) for s, v in stages.items()}}
        m = report["methods"][name]
        log.info("%s: p50 %.0f p90 %.0f ms (encode %.1f, estimate %.2f, recompute %.0f; R %.1f)", name,
                 m["total"]["p50"], m["total"]["p90"], m["encode_query"]["mean"], m["estimate"]["mean"],
                 m["recompute"]["mean"], m["R_mean"])

    # ---- exact ColBERT over stored token vectors (random unit vectors with the true token counts)
    total = []
    for t in tasks:
        counts = tokens[t["units"]]
        flat = rng.standard_normal((int(counts.sum()), dim)).astype(np.float32)
        flat /= np.linalg.norm(flat, axis=1, keepdims=True)
        ptr = np.r_[0, np.cumsum(counts)[:-1]]

        def scan() -> None:
            q = enc.encode_query_tensor(t["query"])
            sims = (torch.from_numpy(flat) @ q.T).numpy()
            np.argsort(-np.maximum.reduceat(sims, ptr, axis=0).sum(1))[:10]
        total.append(timed(scan))
    report["methods"]["colbert-stored"] = {"total": stats(total)}

    # ---- no index: re-encode the whole memory from text for every question
    name0 = meta["codebooks"][0]
    total = []
    for t in tasks[:args.reencode_tasks]:
        units = np.asarray(t["units"])
        res = ReCoSearcher(enc, LocalBagEstimator(user_sketch(sketches[name0][0], units), sketches[name0][1]),
                           [texts[u] for u in units],
                           Calibration(1.0, 0.0, float("inf"), 10, 8, budget=len(units))).search(t["query"])
        total.append(res.timings_ms["total"])
    report["methods"]["reencode-all"] = {"total": stats(total)}

    # ---- BM25 and bge-small over per-user indexes
    bms = []
    for t in tasks:
        bm = BM25()
        bm.index([texts[u] for u in t["units"]])
        bms.append(bm)
    report["methods"]["bm25"] = {"total": stats([timed(lambda: np.argsort(-bm.scores([t["query"]])[0])[:10])
                                                 for t, bm in zip(tasks, bms)])}
    dense = DenseEncoder(args.dense, torch.device("cpu"), fp16=False, query_prefix=BGE_PREFIX)
    for t in tasks[:3]:
        dense.encode([t["query"]], query=True)
    total = []
    for t in tasks:
        vecs = rng.standard_normal((len(t["units"]), 384)).astype(np.float32)
        total.append(timed(lambda: np.argsort(-(vecs @ dense.encode([t["query"]], query=True)[0]))[:10]))
    report["methods"]["bge-small"] = {"total": stats(total)}
    for name, m in report["methods"].items():
        log.info("%-14s p50 %8.1f ms  p90 %8.1f ms", name, m["total"]["p50"], m["total"]["p90"])

    # ---- GPU replay of ReCo (same budgets) and the reader's prefill, for scale
    if args.gpu and torch.cuda.is_available():
        enc_gpu = ColBERTEncoder(EncoderConfig(repo=args.colbert, doc_maxlen=300, device="cuda:0"))
        for name, (sketch, centroids) in sketches.items():
            total = []
            for t in tasks:
                units = np.asarray(t["units"])
                res = ReCoSearcher(enc_gpu, LocalBagEstimator(user_sketch(sketch, units), centroids),
                                   [texts[u] for u in units],
                                   Calibration(1.0, 0.0, float("inf"), 10, 8, budget=max(t[f"R_{name}"], 10))
                                   ).search(t["query"])
                total.append(res.timings_ms["total"])
            report.setdefault("gpu", {})[name] = {"total": stats(total[3:])}
        del enc_gpu
    if args.reader:
        report["reader"] = reader_prefill(args, meta, enc)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{meta['bench']}_{machine}.json"
    write_json(path, report)
    log.info("wrote %s", path)


def reader_prefill(args: argparse.Namespace, meta: dict, enc: ColBERTEncoder) -> dict:
    """Prefill time of the 4B reader on the top-10 units (by ColBERT recomputed from text) of each question."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.reader)
    texts, prompts = meta["texts"], []
    for t in meta["tasks"][:args.reader_tasks]:
        units = np.asarray(t["units"])
        q = enc.encode_query_tensor(t["query"])
        emb, keep, _ = enc.encode_docs_padded([texts[u] for u in units])
        from savemv.core.exact import maxsim_padded
        order = np.argsort(-maxsim_padded(q, emb, keep).numpy())[:10]
        context = "\n\n".join(texts[units[j]] for j in order)
        msgs = [{"role": "user", "content": f"Memory:\n{context}\n\nQuestion: {t['query']}\nAnswer briefly."}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    lengths = [len(tok(p)["input_ids"]) for p in prompts]
    out = {"model": args.reader, "prompt_tokens_mean": float(np.mean(lengths))}
    for dev, n in (("cuda:0", len(prompts)), ("cpu", args.reader_cpu_tasks)):
        if (dev.startswith("cuda") and not torch.cuda.is_available()) or n == 0:
            continue
        dtype = torch.bfloat16 if dev.startswith("cuda") or args.reader_cpu_bf16 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.reader, torch_dtype=dtype).to(dev).eval()
        ms = []
        with torch.inference_mode():
            for p in [prompts[0]] + prompts[:n]:                         # first call warms up
                ids = tok(p, return_tensors="pt").to(dev)
                if dev.startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(**ids, use_cache=False, logits_to_keep=1)
                if dev.startswith("cuda"):
                    torch.cuda.synchronize()
                ms.append(1000 * (time.perf_counter() - t0))
        out[f"prefill_ms_{dev.split(':')[0]}"] = stats(ms[1:])
        out[f"prefill_dtype_{dev.split(':')[0]}"] = str(dtype).replace("torch.", "")
        log.info("reader prefill on %s (%s): %.0f ms for %.0f tokens", dev, dtype, np.mean(ms[1:]),
                 np.mean(lengths[:n]))
        del model
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("prepare", "run"))
    ap.add_argument("--pack", required=True)
    ap.add_argument("--bench")
    ap.add_argument("--data")
    ap.add_argument("--memory", help="memory.json of `python -m savemv memory`")
    ap.add_argument("--tokens", default=None, help="token store of the memory run (default: next to memory.json)")
    ap.add_argument("--codebooks", nargs="+", default=["models/codebooks/loo-scifact_K16384.npy",
                                                       "models/codebooks/loo-scifact_K65536.npy"])
    ap.add_argument("--tasks", type=int, default=100)
    ap.add_argument("--colbert", default="models/answerai-colbert-small-v1")
    ap.add_argument("--dense", default="models/bge-small-en-v1.5")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--machine", default=None)
    ap.add_argument("--reencode-tasks", type=int, default=20)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--reader", default=None)
    ap.add_argument("--reader-tasks", type=int, default=20)
    ap.add_argument("--reader-cpu-tasks", type=int, default=3)
    ap.add_argument("--reader-cpu-bf16", action="store_true")
    ap.add_argument("--out", default="outputs/cpu_latency")
    args = ap.parse_args()
    (stage_prepare if args.stage == "prepare" else stage_run)(args)


if __name__ == "__main__":
    main()
