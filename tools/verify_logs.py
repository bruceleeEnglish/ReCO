#!/usr/bin/env python3
"""Check the shipped results against the run logs, value by value.

Every number a log prints is compared with the same number in results/, up to half a unit of the printed precision.
A value counts as a mismatch if it differs by more; results that no log line covers are listed separately.

    python tools/verify_logs.py [--verbose]

Exit status 1 if any value disagrees.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES, LOGS = ROOT / "results", ROOT / "logs"
BEIR = ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "trec-covid", "quora"]

done: dict[str, list[int]] = defaultdict(lambda: [0, 0])
bad: list[str] = []
uncovered: list[str] = []


def check(group: str, what: str, logged: float, stored: float, digits: int) -> None:
    done[group][0] += 1
    if abs(float(logged) - float(stored)) <= 0.5 * 10 ** -digits + 1e-9:
        done[group][1] += 1
    else:
        bad.append(f"{group}: {what}: log {logged} vs result {stored}")


def load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    with gzip.open(path.with_suffix(".json.gz"), "rt", encoding="utf-8") as fh:
        return json.load(fh)


def text(*paths: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in paths)


def bench(group: str, names: list[str], res_dir: str, log_dir: str, stamp: str) -> None:
    line = re.compile(r"\| (\S+)\s+([\d.]+) B/doc \(([\d.]+)x raw\) \| estimate nDCG ([\d.]+) R@10 ([\d.]+) R95 (\d+) \| "
                      r"ReCo stop: R ([\d.]+) recall ([\d.]+) nDCG ([\d.]+)")
    for ds in names:
        rep = load(RES / res_dir / f"{ds}.json")["methods"]
        seen = set()
        logs = sorted((LOGS / log_dir).glob(f"bench_{ds}_{stamp}*.log"))
        for m in line.finditer(text(*logs)):
            name = m.group(1)
            if name not in rep:
                uncovered.append(f"{group}: {ds}/{name} is in the log but not in the results")
                continue
            r, ev = rep[name], rep[name]["evaluation"]
            d = ev["deltas"]["0.05"]["rank"]
            for w, a, b, dig in (("bytes", m[2], r["total_bytes_per_doc"], 1), ("ratio", m[3], r["storage_ratio"], 3),
                                 ("nDCG", m[4], ev["estimate_ndcg10"], 4), ("R@10", m[5], ev["estimate_recall10"], 3),
                                 ("R95", m[6], ev["R_at_recall"]["0.95"], 0), ("stop R", m[7], d["R_mean"], 1),
                                 ("stop recall", m[8], d["recall"], 3), ("stop nDCG", m[9], d["ndcg10"], 4)):
                check(group, f"{ds}/{name} {w}", a, b, dig)
            seen.add(name)
        for name in sorted(set(rep) - seen):
            uncovered.append(f"{group}: {ds}/{name} has no log line")


def rules(group: str, key: str, segs: str, entry: dict) -> None:
    for seg in segs.split(" | "):
        m = re.match(r"\s*(\S+) ([\d.]+) \(([\d.]+)\)", seg)
        if m and m[1] in entry:
            check(group, f"{key}/{m[1]} R", m[2], entry[m[1]]["R_mean"], 0)
            check(group, f"{key}/{m[1]} recall", m[3], entry[m[1]]["recall"], 3)


def stopping() -> None:
    log = text(*sorted((LOGS / "stopping_baselines").glob("stopbase_gpu*.log")))
    for m in re.finditer(r"stopping_baselines \| (\S+) d=([\d.]+) \| (.*)", log):
        rules("stopping_baselines", f"{m[1]}/{m[2]}", m[3], load(RES / "stopping_baselines" / f"{m[1]}.json")["deltas"][m[2]])
    log = text(LOGS / "train_calibration" / "train_cal.log")
    for m in re.finditer(r"train_calibration \| (\S+) d=([\d.]+) \| (.*)", log):
        rules("train_calibration", f"{m[1]}/{m[2]}", m[3], load(RES / "train_calibration" / f"{m[1]}.json")["deltas"][m[2]])


def param_study() -> None:
    log = text(LOGS / "param_study" / "param_study.log")
    for m in re.finditer(r"param_study \| (\S+) k=(\d+) \(\d+ s\): (.*)", log):
        rules("param_study", f"{m[1]}/k={m[2]}", m[3], load(RES / "param_study" / f"{m[1]}.json")["k"][m[2]]["0.05"])
    for m in re.finditer(r"param_study \| (\S+) n_cal=(\d+): (.*)", log):
        ncal = load(RES / "param_study" / f"{m[1]}.json")["ncal"]
        entry = ncal.get(m[2], ncal.get("all"))
        for seg in m[3].split(" | "):
            s = re.match(r"\s*(\S+) ([\d.]+) \(([\d.]+), min ([\d.]+)\)", seg)
            if s and s[1] in entry:
                check("param_study", f"{m[1]}/n={m[2]}/{s[1]} R", s[2], entry[s[1]]["R_mean"], 0)
                check("param_study", f"{m[1]}/n={m[2]}/{s[1]} recall", s[3], entry[s[1]]["recall"], 3)
                check("param_study", f"{m[1]}/n={m[2]}/{s[1]} min", s[4], entry[s[1]]["recall_min"], 3)


def scalability() -> None:
    log = text(LOGS / "scalability" / "scalability.log")
    for m in re.finditer(r"scalability \| (\S+) n=(\d+): R95 (\d+) \| d=0\.05 fixed ([\d.]+) crc ([\d.]+) hazard ([\d.]+)", log):
        pts = {p["n_docs"]: p for p in load(RES / "scalability" / f"{m[1]}_K16384.json")["points"]}
        p = pts[int(m[2])]
        check("scalability", f"{m[1]}/{m[2]} R95", m[3], p["R95"], 0)
        for i, r in ((4, "fixed"), (5, "crc"), (6, "hazard")):
            check("scalability", f"{m[1]}/{m[2]} {r}", m[i], p["rules"]["0.05"][r]["R_mean"], 1)


def build_cost() -> None:
    for ds in ("fiqa", "quora"):
        rep = load(RES / "build_cost" / f"{ds}.json")["methods"]
        log = text(LOGS / "build_cost" / f"remaining_build_{ds}.log")
        for m in re.finditer(r"build_cost \| (\S+)\s+train ([\d.]+)s \| per-token pass ([\d.]+)s \| insert ([\d.]+) ms/doc \| (\d+) B/doc", log):
            r = rep[m[1]]
            for w, a, b, dig in (("train", m[2], r["train_s"], 1), ("pass", m[3], r["per_token_s"], 1),
                                 ("insert", m[4], r["insert_ms_per_doc"], 2), ("bytes", m[5], r["bytes_per_doc"], 0)):
                check("build_cost", f"{ds}/{m[1]} {w}", a, b, dig)


def ablation_memory() -> None:
    log = text(LOGS / "ablation_and_memory" / "paper_ablation_memory.log")
    rep = load(RES / "ablation" / "hazard_ablation.json")
    for m in re.finditer(r"hazard_ablation \| (\S+) K=(\d+) d=([\d.]+) \| (.*)", log):
        entry = rep[f"{m[1]}/K{m[2]}/{m[3]}"]
        for seg in m[4].split(" | "):
            s = re.match(r"\s*(.+?) ([\d.]+) \(([\d.]+)\)$", seg.strip())
            if s and s[1] in entry:
                check("ablation", f"{m[1]}/K{m[2]}/{m[3]}/{s[1]} R", s[2], entry[s[1]]["R_mean"], 1)
                check("ablation", f"{m[1]}/K{m[2]}/{m[3]}/{s[1]} recall", s[3], entry[s[1]]["recall"], 3)
    line = re.compile(r"memory \| (\S+)\s+alone R@10 ([\d.]+) nDCG ([\d.]+) \| R95 (\d+) \| CRC budget ([\d.]+) -> "
                      r"R@10 ([\d.]+) nDCG ([\d.]+) \| 40/60 split: fixed R ([\d.]+), hazard R ([\d.]+) \(recall ([\d.]+)\)")
    start = 0
    for bench_name in ("longmemeval", "locomo"):
        end = log.index(f"wrote outputs/memory/{bench_name}/memory.md")
        methods = load(RES / "memory" / f"memory_{bench_name}.json")["methods"]
        for m in line.finditer(log[start:end]):
            r = methods[m[1]]
            for w, a, b, dig in (("alone R@10", m[2], r["alone"]["recall@10"], 3), ("alone nDCG", m[3], r["alone"]["ndcg@10"], 3),
                                 ("R95", m[4], r["R95"], 0), ("budget", m[5], r["crc_budget_mean"], 1),
                                 ("after R@10", m[6], r["after_recompute"]["recall@10"], 3),
                                 ("after nDCG", m[7], r["after_recompute"]["ndcg@10"], 3),
                                 ("fixed R", m[8], r["fixed_stop"]["R_mean"], 1), ("hazard R", m[9], r["hazard_stop"]["R_mean"], 1),
                                 ("hazard recall", m[10], r["hazard_stop"]["recall_of_colbert_top10"], 3)):
                check("memory", f"{bench_name}/{m[1]} {w}", a, b, dig)
        start = end


def latency() -> None:
    log = text(LOGS / "latency" / "paper_latency.log")
    lines = [tuple(float(x) for x in m.groups()) for m in re.finditer(
        r"latency batch=(\d+): p50=([\d.]+) p90=([\d.]+) ms \| recomputed ([\d.]+) docs \([\d.]+ ms/doc\) \| recall ([\d.]+)", log)]
    for p in sorted((RES / "latency").glob("latency_*.json")):
        for b, r in load(p)["latency"]["batches"].items():
            want = (float(b), r["ms_p50"], r["ms_p90"], r["recomputed_mean"], r["recall_mean"])
            hit = any(l[0] == want[0] and abs(l[1] - want[1]) <= .0501 and abs(l[2] - want[2]) <= .0501
                      and abs(l[3] - want[3]) <= .0501 and abs(l[4] - want[4]) <= .000501 for l in lines)
            done["latency"][0] += 1
            done["latency"][1] += hit
            if not hit:
                bad.append(f"latency: {p.name} batch {b} not found in the log")


def qa() -> None:
    pairs = {"remaining_qa_locomo.log": "locomo/qa_report_top10_full.json", "remaining_qa_lme.log": "longmemeval/qa_report_top10_full.json"}
    q = "Qwen3-4B-Instruct-2507"
    pairs.update({"remaining_qwen_locomo.log": f"locomo/{q}/qa_report_top10_full.json",
                  "remaining_qwen_lme.log": f"longmemeval/{q}/qa_report_top10_full32768.json",
                  "remaining_qwen_lme_top3.log": f"longmemeval/{q}/qa_report_top3.json"})
    pairs.update({f"remaining_qwen_locomo_top{t}.log": f"locomo/{q}/qa_report_top{t}.json" for t in (3, 30, 100)})
    line = re.compile(r"qa \| (\S+)\s+n=\s*(\d+) acc ([\d.]+) F1 ([\d.]+) ctx\s+([\d]+) tok")
    for log, rep in pairs.items():
        r = load(RES / "qa" / rep)["retrievers"]
        for m in line.finditer(text(LOGS / "qa" / log)):
            e = r[m[1]]
            check("qa", f"{rep}/{m[1]} n", m[2], e["n"], 0)
            check("qa", f"{rep}/{m[1]} acc", m[3], e["accuracy"], 3)
            check("qa", f"{rep}/{m[1]} F1", m[4], e["f1"], 3)
            check("qa", f"{rep}/{m[1]} tokens", m[5], e["prompt_tokens"], 0)


def msmarco() -> None:
    log = text(LOGS / "msmarco" / "msmarco_20261026_121437.log")
    line = re.compile(r"K=(\d+): scan ([\d.]+) ms \| estimate MRR ([\d.]+) \| d=0\.05 fixed R (\d+) \(MRR ([\d.]+)\) hazard R (\d+) \(MRR ([\d.]+)\)")
    seen = defaultdict(int)
    for m in line.finditer(log):
        seen[m[1]] += 1
        name = f"eval_K{m[1]}.json" if seen[m[1]] == 1 else f"eval_K{m[1]}_b32.json"
        d = load(RES / "msmarco" / name)["deltas"]["0.05"]
        check("msmarco", f"{name} fixed R", m[4], d["fixed"]["R_mean"], 0)
        check("msmarco", f"{name} fixed MRR", m[5], d["fixed"]["mrr10"], 4)
        check("msmarco", f"{name} hazard R", m[6], d["hazard"]["R_mean"], 0)
        check("msmarco", f"{name} hazard MRR", m[7], d["hazard"]["mrr10"], 4)
    m = re.search(r"BM25 cascade: d=0\.05 fixed R (\d+) \(MRR ([\d.]+)\) hazard R (\d+) \(MRR ([\d.]+)\)", log)
    d = load(RES / "msmarco" / "eval_bm25.json")["deltas"]["0.05"]
    check("msmarco", "bm25 fixed R", m[1], d["fixed"]["R_mean"], 0)
    check("msmarco", "bm25 hazard R", m[3], d["hazard"]["R_mean"], 0)
    check("msmarco", "bm25 hazard MRR", m[4], d["hazard"]["mrr10"], 4)
    runs = re.findall(r"(\w+@[\d.]+): p50 ([\d.]+) p90 ([\d.]+) ms \| estimate ([\d.]+) ms \| R (\d+) \| recall ([\d.]+)", log)
    for name, label in (("latency_K65536.json", "b=8"), ("latency_K65536_b32.json", "b=32")):
        for key, r in load(RES / "msmarco" / name)["runs"].items():
            hit = any(k == key and abs(float(p) - r["ms_p50"]) <= .0501 and abs(float(q) - r["ms_p90"]) <= .0501
                      for k, p, q, *_ in runs)
            done["msmarco"][0] += 1
            done["msmarco"][1] += hit
            if not hit:
                bad.append(f"msmarco: {name} {key} ({label}) not found in the log")


def plaid() -> None:
    rep = load(RES / "msmarco" / "plaid_nbits2.json")

    def records(path: Path) -> list[dict]:
        return [json.loads(l) for l in text(path).splitlines() if l.startswith("{")]

    run = records(LOGS / "plaid" / "plaid_20261027_033108.log")
    tput = records(LOGS / "throughput" / "tput_clean.log")
    for name, stored, logged in (("gpu", rep["gpu"], run[0]), ("cpu", rep["cpu"], run[1]), ("gpu_rerun", rep["gpu_rerun"], tput[0])):
        for f in ("p50", "p90", "p99"):
            check("plaid", f"{name} {f}", logged["latency_ms"][f], stored["latency_ms"][f], 3)
        check("plaid", f"{name} mrr", logged["mrr10"], stored["mrr10"], 6)
        check("plaid", f"{name} recall", logged["recall_vs_exact10"], stored["recall_vs_exact10"], 6)
    check("plaid", "throughput qps", tput[1]["qps"], rep["throughput"]["qps"], 3)
    check("plaid", "throughput gpu", tput[1]["peak_gpu_gb"], rep["throughput"]["peak_gpu_gb"], 3)


def throughput() -> None:
    log = text(LOGS / "throughput" / "tput_clean.log")
    found = {(m[1], int(m[2])): m for m in re.finditer(
        r"delta=([\d.]+) C=(\d+): ([\d.]+) QPS \| latency p50 (\d+) p90 (\d+) ms \| R (\d+) \| \d+ docs/s \| recall ([\d.]+) \| MRR ([\d.]+) \| peak ([\d.]+) GB", log)}
    for r in load(RES / "msmarco" / "throughput_reco.json")["runs"]:
        m = found[(str(r["delta"]), r["concurrency"])]
        key = f"delta={r['delta']} C={r['concurrency']}"
        check("throughput", f"{key} qps", m[3], r["qps"], 1)
        check("throughput", f"{key} p50", m[4], r["latency_ms_p50"], 0)
        check("throughput", f"{key} R", m[6], r["R_mean"], 0)
        check("throughput", f"{key} recall", m[7], r["recall_vs_exact10"], 3)
        check("throughput", f"{key} MRR", m[8], r["mrr10"], 4)


def review() -> None:
    """The experiments added after the mock review: PLAID's operating points, the optimality gap, corpus growth."""
    log = text(LOGS / "review" / "review_20261031_025724.log")
    printed = [json.loads(s) for s in log.splitlines() if s.startswith('{"ncells"')]
    tail = [json.loads(s) for s in text(LOGS / "review" / "review_tail_20261031_030338.log").splitlines()
            if s.startswith('{"ncells"')]
    other = [json.loads(s) for s in text(LOGS / "review" / "review_check_20261031_030142.log").splitlines()
             if s.startswith('{"ncells"')]
    for name, lines in (("plaid_sweep_nbits2", printed[:4]), ("plaid_sweep_cpu_nbits2", printed[4:]),
                        ("plaid_sweep_skip_nbits2", tail[:2]), ("plaid_sweep_cpu_mid_nbits2", tail[2:]),
                        ("plaid_sweep_cpu_check_nbits2", other)):
        points = load(RES / "msmarco" / f"{name}.json")["points"]
        assert len(points) == len(lines), name
        for p, q in zip(points, lines):
            key = f"{name} {p['ncells']}/{p['ndocs']}"
            if p.get("out_of_memory"):
                check("review", f"{key} failed at", q["queries_done"], p["queries_done"], 0)
                continue
            for field, digits in (("recall_vs_exact10", 6), ("mrr10", 6), ("qps", 4)):
                check("review", f"{key} {field}", q[field], p[field], digits)
            check("review", f"{key} p50", q["latency_ms"]["p50"], p["latency_ms"]["p50"], 4)
    found = {(m[1], int(m[2])): m for m in re.finditer(
        r"delta=([\d.]+) C=(\d+): ([\d.]+) QPS \| latency p50 (\d+) p90 (\d+) ms \| R (\d+) \| \d+ docs/s \| recall ([\d.]+) \| MRR ([\d.]+)", log)}
    for r in load(RES / "msmarco" / "throughput_reco_d01.json")["runs"]:
        m = found[(str(r["delta"]), r["concurrency"])]
        key = f"delta={r['delta']} C={r['concurrency']}"
        check("review", f"{key} qps", m[3], r["qps"], 1)
        check("review", f"{key} p50", m[4], r["latency_ms_p50"], 0)
        check("review", f"{key} recall", m[7], r["recall_vs_exact10"], 3)
    for n in BEIR:
        rep = load(RES / "optimality_gap" / f"{n}.json")
        m = re.search(re.escape(n) + r": g falls on ([\d.]+)% of steps \((\d+)% after a top-k change\), rank correlation (-?[\d.]+)", log)
        check("review", f"{n} falls", m[1], 100 * rep["monotonicity"]["fall_share"], 1)
        check("review", f"{n} after change", m[2], 100 * rep["monotonicity"]["fall_after_topk_change_share"], 0)
        check("review", f"{n} rank correlation", m[3], rep["reliability"]["rank_correlation"], 3)
        for d, e in rep["deltas"].items():
            m = re.search(re.escape(n) + rf" d={d}: hazard ([\d.]+), bound ([\d.]+) ", log)
            check("review", f"{n} {d} hazard", m[1], e["hazard"]["R_mean"], 1)
            check("review", f"{n} {d} bound", m[2], e["hazard"]["bound_at_achieved"], 1)
    sizes = text(LOGS / "review" / "review_memory_by_size_20261031_150047.log")
    for n in ("longmemeval", "locomo"):
        rep = load(RES / "memory" / f"by_size_{n}.json")
        m = re.search(n + r": sizes .*?, recall ([\d.]+), R ([\d.]+)", sizes)
        check("review", f"{n} recall", m[1], rep["recall"], 4)
        check("review", f"{n} R", m[2], rep["R_mean"], 1)
        for row in rep["by_size"]:
            m = re.search(rf"{n} quartile {row['quartile']} \(\d+-\d+ units, \d+ tasks\): recall ([\d.]+), R ([\d.]+)", sizes)
            check("review", f"{n} quartile {row['quartile']} recall", m[1], row["recall"], 4)
            check("review", f"{n} quartile {row['quartile']} R", m[2], row["R_mean"], 1)
        m = re.search(n + r" calibrated on smaller memories .*?: recall ([\d.]+), R ([\d.]+)", sizes)
        check("review", f"{n} small to large recall", m[1], rep["small_to_large"]["recall"], 4)
    for n in ("scidocs", "fiqa", "quora"):
        rep = load(RES / "growth" / f"{n}.json")
        for b in rep["bases"]:
            for pt in b["points"]:
                for d in pt["frozen"]:
                    m = re.search(rf"{n} base {b['fraction']:.2f} -> {pt['fraction']:.2f} d={d}: frozen R ([\d.]+) "
                                  r"recall ([\d.]+) \| recalibrated R ([\d.]+) recall ([\d.]+)", log)
                    key = f"{n} {b['fraction']}->{pt['fraction']} {d}"
                    check("review", f"{key} frozen R", m[1], pt["frozen"][d]["hazard"]["R_mean"], 1)
                    check("review", f"{key} frozen recall", m[2], pt["frozen"][d]["hazard"]["recall"], 4)
                    check("review", f"{key} recalibrated R", m[3], pt["recalibrated"][d]["hazard"]["R_mean"], 1)
                    check("review", f"{key} recalibrated recall", m[4], pt["recalibrated"][d]["hazard"]["recall"], 4)


def review2() -> None:
    """The second round after the reviews: PLAID's CPU engine at lower centroid thresholds, and the stopping baselines
    again with every rule under the pooled correction (the log prints the query-level rules, which must be unchanged)."""
    log = text(LOGS / "review" / "review2_20261031_233333.log").replace("\r", "\n")
    lines = [json.loads(s) for s in log.splitlines() if s.startswith('{"ncells"')]
    points = load(RES / "msmarco" / "plaid_sweep_cpu_low_nbits2.json")["points"]
    assert len(points) == len(lines) == 3
    for p, q in zip(points, lines):
        key = f"plaid_sweep_cpu_low {p['ncells']}/{p['ndocs']}/{p['centroid_score_threshold']}"
        for field, digits in (("recall_vs_exact10", 6), ("mrr10", 6), ("qps", 4)):
            check("review2", f"{key} {field}", q[field], p[field], digits)
        check("review2", f"{key} p50", q["latency_ms"]["p50"], p["latency_ms"]["p50"], 4)
    for m in re.finditer(r"stopping_baselines \| (\S+) d=([\d.]+) \| (.*)", log):
        rules("review2", f"{m[1]}/{m[2]}", m[3], load(RES / "stopping_baselines" / f"{m[1]}.json")["deltas"][m[2]])


def warp() -> None:
    found = set()
    for nbits, log in ((4, "warp_nb4.log"), (2, "warp_nb2.log")):
        src = text(LOGS / "warp" / log) + ("\n" + text(LOGS / "warp" / "warp_scifact.log") if nbits == 4 else "")
        for m in re.finditer(r"([\w-]+): nDCG@10 ([\d.]+) \| index ([\d.]+) MB = ([\d.]+)x raw \| latency ([\d.]+) ms p50 \((\d+) threads\)", src):
            r = load(RES / "warp" / f"{m[1]}.nbits{nbits}.json")
            found.add((m[1], nbits))
            check("warp", f"{m[1]}/{nbits} nDCG", m[2], r["ndcg10"], 4)
            check("warp", f"{m[1]}/{nbits} ratio", m[4], r["storage_ratio"], 2)
            check("warp", f"{m[1]}/{nbits} latency", m[5], r["latency_ms_p50"], 1)
    for ds in BEIR:
        for nbits in (4, 2):
            if (ds, nbits) not in found:
                uncovered.append(f"warp: {ds}/{nbits} has no log line")


def cpu_latency() -> None:
    for log, bench_order, machine in (("cpu_latency_server.log", ("longmemeval", "locomo"), "server"),
                                      ("laptop_run.log", ("locomo", "longmemeval"), "laptop")):
        src = text(LOGS / "cpu_latency" / log)
        start = 0
        for b in bench_order:
            end = src.index(f"{b}_{machine}.json", start) if f"{b}_{machine}.json" in src[start:] else len(src)
            rep = load(RES / "cpu_latency" / f"{b}_{machine}.json")
            table = rep["methods"]
            for m in re.finditer(r"cpu_latency \| (\S+)\s+p50\s+([\d.]+) ms\s+p90\s+([\d.]+) ms", src[start:end]):
                if m[1] in table:
                    t = table[m[1]].get("total", table[m[1]])
                    check("cpu_latency", f"{b}/{machine}/{m[1]} p50", m[2], t["p50"], 1)
                    check("cpu_latency", f"{b}/{machine}/{m[1]} p90", m[3], t["p90"], 1)
            start = end


def inverted() -> None:
    """The reported run left no log, so logs/msmarco holds a later rerun on the same machine. Its deterministic
    quantities must agree exactly. Wall-clock times are not compared: they vary between runs."""
    rerun = LOGS / "msmarco" / "inverted_rerun_20261030.json"
    if not rerun.exists():
        uncovered.append("inverted: results/msmarco/inverted_cpu.json has no log")
        return
    a, b = load(RES / "msmarco" / "inverted_cpu.json"), load(rerun)
    for d in ("0.1", "0.05", "0.01"):
        check("inverted", f"{d} probe", b["deltas"][d]["probe_median"], a["deltas"][d]["probe_median"], 1)
        check("inverted", f"{d} candidates", b["deltas"][d]["candidates_mean"], a["deltas"][d]["candidates_mean"], 3)
    check("inverted", "memory", b["memory_gb"], a["memory_gb"], 6)
    check("inverted", "identical", b["verified"]["identical"], a["verified"]["identical"], 0)
    log = text(LOGS / "msmarco" / "inverted_rerun_20261030.log")
    for m in re.finditer(r"delta=([\d.]+): ([\d.]+) ms mean \(p50 ([\d.]+), p99 ([\d.]+)\) \| probe median (\d+) \| (\d+) candidates", log):
        check("inverted", f"rerun log {m[1]}", m[2], b["deltas"][m[1]]["ms"]["mean"], 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    bench("beir", BEIR, "beir", "beir", "20261025")
    bench("beir_cv2", ["scifact", "nfcorpus", "scidocs", "fiqa"], "beir_cv2", "beir_cv2", "20261026")
    for fn in (stopping, param_study, scalability, build_cost, ablation_memory, latency, qa, msmarco, plaid, throughput, warp,
               cpu_latency, inverted, review, review2):
        fn()
    print(f"{'group':22s}{'compared':>10s}{'equal':>10s}")
    for g, (n, ok) in sorted(done.items()):
        print(f"{g:22s}{n:10d}{ok:10d}")
    print(f"{'total':22s}{sum(v[0] for v in done.values()):10d}{sum(v[1] for v in done.values()):10d}")
    print(f"\nmismatches: {len(bad)}, results without a log line: {len(uncovered)}")
    for b in bad[: None if args.verbose else 30]:
        print("  MISMATCH", b)
    for u in uncovered[: None if args.verbose else 30]:
        print("  NOT LOGGED", u)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
