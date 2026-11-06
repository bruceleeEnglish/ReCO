"""Storage-versus-recomputation cost model (in the spirit of Gray's five-minute rule).

ReCo drops the token vectors (index bytes) and pays for it at query time (re-encoding the candidates it
verifies). Keeping the vectors is worth it only for collections queried often enough:

    break-even rate  lambda* = (bytes saved x storage price per byte-month) / (ReCo's compute cost per query)

Below lambda* queries per month the sketch is cheaper; the break-even interval 1 / lambda* is the "rule" form.
We charge ReCo its whole measured online time (query encoding, sketch scan and recomputation) and the vector
index nothing at query time, which favours the baselines. All prices are inputs; results scale linearly.

    python scripts/efficiency/cost_model.py --results results --out results/cost/cost_model.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MONTH_S = 30 * 86400


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def break_even(saved_bytes: float, price_gb_month: float, usd_per_query: float) -> dict:
    saved_usd = saved_bytes / 1e9 * price_gb_month
    per_month = saved_usd / usd_per_query
    return {"saved_usd_month": saved_usd, "queries_per_month": per_month, "qps": per_month / MONTH_S,
            "interval_s": MONTH_S / per_month if per_month > 0 else float("inf")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/cost/cost_model.json")
    ap.add_argument("--dram", type=float, default=3.0, help="USD per GB-month, memory-resident index")
    ap.add_argument("--ssd", type=float, default=0.08, help="USD per GB-month, SSD block storage")
    ap.add_argument("--gpu", type=float, default=0.50, help="USD per GPU-hour")
    ap.add_argument("--vcpu", type=float, default=0.04, help="USD per vCPU-hour")
    ap.add_argument("--msmarco-k", type=int, default=65536)
    ap.add_argument("--msmarco-run", default="hazard@0.05")
    args = ap.parse_args()
    root = Path(args.results)
    prices = {"dram_usd_gb_month": args.dram, "ssd_usd_gb_month": args.ssd, "gpu_usd_hour": args.gpu,
              "vcpu_usd_hour": args.vcpu}
    report: dict = {"prices": prices, "settings": {}}

    # web-scale collection: MS MARCO passages, GPU serving
    ev = load(root / "msmarco" / f"eval_K{args.msmarco_k}.json")
    lat = {b: load(root / "msmarco" / f"latency_K{args.msmarco_k}{s}.json") for b, s in ((8, ""), (32, "_b32"))}
    meta = load(root / "msmarco" / "meta.json")
    if ev and meta:
        n, st = meta["n_docs"], ev["storage"]
        reco = st["sketch"] * n + st["codebook_total"]
        plaid = load(root / "msmarco" / "plaid_nbits2.json")       # measured official index, else the analytic size
        v2 = plaid["index_bytes"] if plaid and "index_bytes" in plaid else st["colbertv2_2bit"] * n
        entry = {"n_docs": n, "index_gb": {"reco_sketch": reco / 1e9, "colbert_fp16": st["colbert_fp16"] * n / 1e9,
                                           "colbertv2_2bit": v2 / 1e9,
                                           "raw_text": st["raw_text"] * n / 1e9}, "per_query": {}, "break_even": {}}
        for b, run in lat.items():
            if run and args.msmarco_run in run["runs"]:
                ms = run["runs"][args.msmarco_run]["ms_mean"]
                entry["per_query"][f"gpu_b{b}"] = {"ms": ms, "usd": ms / 3.6e6 * args.gpu}
        for qname, q in entry["per_query"].items():
            for base in ("colbert_fp16", "colbertv2_2bit"):
                saved = entry["index_gb"][base] * 1e9 - reco
                for tier in ("dram", "ssd"):
                    entry["break_even"][f"{base}|{tier}|{qname}"] = break_even(saved, getattr(args, tier), q["usd"])
        if plaid and "gpu" in plaid:          # charge ReCo only its extra GPU time over the measured PLAID query
            p_ms = plaid["gpu"]["latency_ms"]["mean"]
            entry["per_query"]["plaid_gpu"] = {"ms": p_ms, "usd": p_ms / 3.6e6 * args.gpu}
            for qname, q in list(entry["per_query"].items()):
                extra = q["usd"] - entry["per_query"]["plaid_gpu"]["usd"]
                if qname.startswith("gpu_b") and extra > 0:
                    for tier in ("dram", "ssd"):
                        entry["break_even"][f"colbertv2_2bit_net|{tier}|{qname}"] = break_even(
                            v2 - reco, getattr(args, tier), extra)
        report["settings"]["msmarco"] = entry

    # per-user agent memory: one user's memory is the collection
    for bench in ("longmemeval", "locomo"):
        mem = load(root / "memory" / f"memory_{bench}.json")
        cpu = load(root / "cpu_latency" / f"{bench}_server.json") or load(root / "cpu_latency" / f"{bench}_laptop.json")
        if not mem:
            continue
        units = mem["units_per_task"]
        st = mem["storage_bytes_per_unit"]
        sketch_key = next(k for k in st if k.startswith("ReCo sketch K=64k"))
        entry = {"units_per_memory": units, "latency_source": cpu["machine"] if cpu else None,
                 "memory_mb": {k: v * units / 1e6 for k, v in st.items()}, "per_query": {}, "break_even": {}}
        if cpu:
            m = cpu["methods"]["reco-K64k"]["total"]["mean"]
            entry["per_query"]["cpu8"] = {"ms": m, "usd": m / 3.6e6 * cpu["threads"] * args.vcpu}
            if "gpu" in cpu:
                g = cpu["gpu"]["reco-K64k"]["total"]["mean"]
                entry["per_query"]["gpu"] = {"ms": g, "usd": g / 3.6e6 * args.gpu}
        for qname, q in entry["per_query"].items():
            for base in ("ColBERT fp16", "ColBERTv2 1-bit (ref.)"):
                saved = (st[base] - st[sketch_key]) * units
                for tier in ("dram", "ssd"):
                    entry["break_even"][f"{base}|{tier}|{qname}"] = break_even(saved, getattr(args, tier), q["usd"])
        report["settings"][bench] = entry

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for name, e in report["settings"].items():
        print(f"## {name}")
        for key, v in e["per_query"].items():
            print(f"  per query {key}: {v['ms']:.1f} ms, ${v['usd']:.2e}")
        for key, v in e["break_even"].items():
            print(f"  {key:45s} saves ${v['saved_usd_month']:.4g}/month; break-even {v['queries_per_month']:.3g} "
                  f"queries/month ({v['qps']:.3g} QPS, one query every {v['interval_s']:.3g} s)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
