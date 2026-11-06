#!/usr/bin/env python3
"""Check the numbers the paper states in its prose, which are not macros of paper_artifacts/numbers.tex.

Each line recomputes one statement from results/ and prints the value next to what the paper says.

    python tools/verify_literals.py

Exit status 1 if a statement does not hold.
"""
from __future__ import annotations

import gzip
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
failed = 0


def load(rel: str) -> dict:
    p = RES / rel
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    with gzip.open(p.with_suffix(".json.gz"), "rt", encoding="utf-8") as fh:
        return json.load(fh)


def state(claim: str, value, holds: bool) -> None:
    global failed
    failed += not holds
    print(f"{'ok  ' if holds else 'FAIL'} {claim:66s} {value}")


BEIR = ("scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "trec-covid", "quora")
beir = {n: load(f"beir/{n}.json") for n in BEIR}
sb = {n: load(f"stopping_baselines/{n}.json") for n in BEIR}
K = {"scifact": 4096, "nfcorpus": 4096, "arguana": 4096, "scidocs": 16384, "fiqa": 16384, "trec-covid": 65536, "quora": 16384}

# ---- agent memory, introduction and Exp-6
mem = load("memory/memory_longmemeval.json")
st, units = mem["storage_bytes_per_unit"], mem["units_per_task"]
text_mb, col_mb = st["raw text"] * units / 1e6, st["ColBERT fp16"] * units / 1e6
state("about half a megabyte of text per LongMemEval user (0.49 MB)", f"{text_mb:.3f} MB", abs(text_mb - 0.49) < 0.01)
tasks = load("memory/memory_locomo.json")
state("about 250 and 600 units per memory", f"{mem['units_per_task']:.0f}, {tasks['units_per_task']:.0f}",
      abs(mem["units_per_task"] - 250) < 5 and abs(tasks["units_per_task"] - 600) < 5)
m = {k: mem["methods"][k]["hazard_stop"]["R_mean"] for k in ("reco-K64k", "bge-base", "bge-small", "bm25")}
ratio = [m[k] / m["reco-K64k"] for k in ("bge-base", "bge-small", "bm25")]
state("about 20 recomputed rounds", f"{m['reco-K64k']:.1f}", abs(m["reco-K64k"] - 20) < 1)
state("2 to 6 times more for dense and lexical first stages", f"{min(ratio):.1f} to {max(ratio):.1f}", min(ratio) >= 2 and max(ratio) < 6.5)
for bench, n, full in (("locomo", 400, 200), ("longmemeval", 470, 60)):
    r = load(f"qa/{bench}/qa_report_top10_full.json")
    state(f"{bench}: {n} questions, {full} with the whole memory", f"{r['n_questions']}, {r['retrievers']['full']['n']}",
          r["n_questions"] == n and r["retrievers"]["full"]["n"] == full)
r = load("qa/locomo/qa_report_top10_full.json")["retrievers"]
x = r["full"]["prompt_tokens"] / r["colbert"]["prompt_tokens"]
state("the whole memory takes 43 times the input tokens of the top-10", f"{x:.1f}", round(x) == 43)

# ---- models, introduction, scope and encoder paragraph


def ndcg(res: dict, method: str = "full-fp16") -> float:
    return res["methods"][method]["evaluation"]["estimate_ndcg10"]


ahead = [n for n in BEIR if ndcg(beir[n]) > ndcg(beir[n], "dense-fp16")]
state("multi-vector often ranks above single-vector: bge-small, of 7 corpora", len(ahead), len(ahead) == 4)
cv2 = {n: load(f"beir_cv2/{n}.json") for n in ("scifact", "nfcorpus", "scidocs", "fiqa")}
state("the small encoder outperforms ColBERTv2 on every shared corpus", [round(ndcg(beir[n]) - ndcg(r), 3) for n, r in cv2.items()],
      all(ndcg(beir[n]) > ndcg(r) for n, r in cv2.items()))

# ---- protocol and methods
p = beir["scifact"]["protocol"]
state("k=10, b=8, five folds, 40% fit", (p["k"], p["batch"], p["folds"], p["fit_fraction"]), (p["k"], p["batch"], p["folds"], p["fit_fraction"]) == (10, 8, 5, 0.4))
state("hazard horizon 64", p["hazard"]["horizon"], p["hazard"]["horizon"] == 64)
state("queries of 32 tokens, 300 on ArguAna, documents of 300",
      (p["encoder"]["query_maxlen"], beir["arguana"]["protocol"]["encoder"]["query_maxlen"], p["encoder"]["doc_maxlen"]),
      (p["encoder"]["query_maxlen"], beir["arguana"]["protocol"]["encoder"]["query_maxlen"], p["encoder"]["doc_maxlen"]) == (32, 300, 300))
state("1,000 sampled queries on Quora", beir["quora"]["protocol"]["max_queries"], beir["quora"]["protocol"]["max_queries"] == 1000)
state("Quora documents are 62 bytes long", f"{beir['quora']['dataset']['raw_bytes_per_doc']:.1f}", round(beir["quora"]["dataset"]["raw_bytes_per_doc"]) == 62)
names = list(beir["scifact"]["methods"])
n_conf = sum(1 for m_ in names if not m_.startswith(("reco", "dense", "bm25", "splade"))) + len(list((RES / "warp").glob("scifact.nbits?.json")))
state("the indexes (1) to (7) run at 33 configurations", n_conf, n_conf == 33)


def calib(n: str) -> int:
    """Calibration queries of the smallest fold: of the queries outside the fold, 40% fit and 60% calibrate."""
    n_q = beir[n]["dataset"]["n_queries"]
    rest = n_q - math.ceil(n_q / 5)
    return rest - round(0.4 * rest)


pooled = [(n, d) for n in BEIR for d in (0.05, 0.01) if calib(n) + 1 < 4 / d]
state("5 of the 14 headline settings use the pooled correction", sorted(pooled), len(pooled) == 5)
state("TREC-COVID folds hold 24 calibration queries", calib("trec-covid"), calib("trec-covid") == 24)
fit = {n: round(0.4 * (beir[n]["dataset"]["n_queries"] - math.ceil(beir[n]["dataset"]["n_queries"] / 5))) for n in BEIR}
state("learned models fit on 16 to 448 queries per fold", f"{min(fit.values())} to {max(fit.values())}",
      (min(fit.values()), max(fit.values())) == (16, 448))
level = {(n, d): d + (1 - d) * (1 - 1 / 10) / (calib(n) + 1) for n, d in pooled}
state("pooled correction: 0.084 and 0.046 on TREC-COVID at 0.05 and 0.01",
      f"{level['trec-covid', 0.05]:.4f}, {level['trec-covid', 0.01]:.4f}",
      (round(level["trec-covid", 0.05], 3), round(level["trec-covid", 0.01], 3)) == (0.084, 0.046))
rest = [v for (n, d), v in level.items() if n != "trec-covid"]
state("pooled correction: at most 0.016 on the other corpora", f"{max(rest):.4f}", round(max(rest), 3) == 0.016)
state("a universal 64k codebook takes 12.6 MB", f"{65536 * 96 * 2 / 1e6:.2f} MB", round(65536 * 96 * 2 / 1e6, 1) == 12.6)
state("the shared codebook outweighs one LongMemEval memory of vectors", f"12.6 vs {col_mb:.1f} MB", 65536 * 96 * 2 / 1e6 > col_mb)
state("a codebook of 16k centroids in fp16 adds 3 MB", f"{16384 * 96 * 2 / 1e6:.2f} MB", abs(16384 * 96 * 2 / 1e6 - 3) < 0.2)

# ---- stopping rules, Exp-2


def row(n: str, d: str) -> dict:
    ev = beir[n]["methods"][f"reco-K{K[n]}"]["evaluation"]["deltas"][d]
    out = {k: ev[k]["R_mean"] for k in ("fixed", "crc", "patience", "hazard")}
    out.update({k: sb[n]["deltas"][d][k]["R_mean"] for k in ("probtopk", "laet", "darth")})
    return out


r = row("trec-covid", "0.05")
state("TREC-COVID, query-level CRC: four rules are cheaper", [k for k, v in r.items() if v < r["hazard"]], sum(v < r["hazard"] for v in r.values()) == 4)
r = row("scifact", "0.01")
state("SciFact at 0.01, query-level CRC: LAET is cheaper", [k for k, v in r.items() if v < r["hazard"]], [k for k, v in r.items() if v < r["hazard"]] == ["laet"])

wins = total = 0
for n in BEIR:
    for d in ("0.05", "0.01"):
        r = row(n, d)
        if r["hazard"] < beir[n]["dataset"]["n_docs"]:     # infeasible with query-level CRC otherwise
            total += 1
            wins += all(r["hazard"] < v for key, v in r.items() if key != "hazard")
state("fewest recomputations in 11 of 13 query-level settings", f"{wins} of {total}", (wins, total) == (11, 13))
hd = beir["trec-covid"]["methods"]["reco-K65536"]["evaluation"]["deltas"]["0.05"]["hazard_doc"]["R_mean"]
pa = sb["trec-covid"]["deltas"]["0.05"]["patience_doc"]["R_mean"]
state("TREC-COVID at 0.05, pooled: Patience 3% cheaper than ReCo-D", f"{pa:.2f} vs {hd:.2f}", round(100 * (1 - pa / hd)) == 3)
hd = beir["scifact"]["methods"]["reco-K4096"]["evaluation"]["deltas"]["0.01"]["hazard_doc"]["R_mean"]
other = {k: v["R_mean"] for k, v in sb["scifact"]["deltas"]["0.01"].items() if k.endswith("_doc") and k != "hazard_doc"}
other["rank1"] = beir["scifact"]["methods"]["reco-K4096"]["evaluation"]["deltas"]["0.01"]["rank1"]["R_mean"]
state("SciFact at 0.01, pooled: ReCo-D is cheapest", f"{hd:.1f} vs {min(other.values()):.1f}", hd < min(other.values()))

ev05 = {n: beir[n]["methods"][f"reco-K{K[n]}"]["evaluation"]["deltas"]["0.05"]["hazard"] for n in BEIR if n != "trec-covid"}
whole = [e["full_recall_rate"] for e in ev05.values()]
lost = [(1 - e["recall"]) * 10 / (1 - e["full_recall_rate"]) for e in ev05.values()]
state("65 to 72% of queries recover the whole top-10 at 0.05", f"{min(whole):.3f} to {max(whole):.3f}", round(min(whole), 2) == 0.65 and round(max(whole), 2) == 0.72)
state("the others miss 1.4 to 1.8 documents on average", f"{min(lost):.2f} to {max(lost):.2f}", round(min(lost), 1) == 1.4 and round(max(lost), 1) == 1.8)


def headline_ndcg(n: str, d: float) -> float:
    """ReCo's nDCG@10 at the corpus's headline correction, as in Table IV."""
    rule = "hazard_doc" if (n, d) in pooled else "hazard"
    return beir[n]["methods"][f"reco-K{K[n]}"]["evaluation"]["deltas"][str(d)][rule]["ndcg10"]


ratio = [headline_ndcg(n, 0.05) / ndcg(beir[n]) for n in BEIR]
state("nDCG@10 stays at 99.0% of exact ColBERT's at 0.05", f"{100 * sum(ratio) / 7:.1f}", round(100 * sum(ratio) / 7, 1) == 99.0)
gaps = [100 * (headline_ndcg(n, 0.01) - ndcg(beir[n])) for n in BEIR]
state("within 0.4 nDCG@10 points of ColBERT at 0.01", f"{min(gaps):.2f} to {max(gaps):.2f}", round(min(gaps), 1) == -0.4)

# ---- memory cost, introduction and Exp-5
gb = col_mb * 1e6 / 1e3
state("a million users: about $31,000 per month in memory", f"{gb * 3:,.0f}", 30500 < gb * 3 < 31500)
state("a million users: about $800 per month on SSD", f"{gb * 0.08:,.0f}", 780 < gb * 0.08 < 850)

# ---- latency, Exp-5
lat = {}
for f in sorted((RES / "latency").glob("latency_*.json")):
    m_ = re.match(r"latency_(.+)_K\d+_(\w+)", f.stem)
    lat[m_[1], m_[2]] = json.loads(f.read_text(encoding="utf-8"))["latency"]["batches"]
p50 = [lat[d, "hazard"]["32"]["ms_p50"] for d in ("scifact", "fiqa", "trec-covid")]
state("median latency is 45 to 108 ms", f"{min(p50):.0f} to {max(p50):.0f}", round(min(p50)) == 45 and round(max(p50)) == 108)
half = [lat[d, "hazard"]["32"]["ms_p99"] / lat[d, "crc"]["32"]["ms_p99"] for d in ("scifact", "fiqa")]
state("hazard stopping roughly halves the p99 of TA-Margin", [round(x, 2) for x in half], all(0.45 < x < 0.6 for x in half))
state("slower than FixedBudget on TREC-COVID", f"{lat['trec-covid', 'hazard']['32']['ms_p50']:.0f} vs {lat['trec-covid', 'fixed']['32']['ms_p50']:.0f}",
      lat["trec-covid", "hazard"]["32"]["ms_p50"] > lat["trec-covid", "fixed"]["32"]["ms_p50"])
same = [lat[d, "hazard"]["8"]["recomputed_matches_offline"] for d in ("scifact", "fiqa", "trec-covid")]
diff = max(lat[d, r_]["8"]["max_abs_score_diff_vs_stored"] for d in ("scifact", "fiqa", "trec-covid") for r_ in ("hazard", "crc", "fixed"))
state("online and offline stops coincide on 96.5 to 98% of the queries", same, min(same) == 0.965 and max(same) == 0.98)
state("re-encoded scores deviate by up to 0.004", f"{diff:.4f}", diff <= 0.004)
n_lat = load("msmarco/latency_K65536_b32.json")
state("MS MARCO latency of ReCo: batches of 32, 200 queries", (n_lat["stop_batch"], n_lat["queries"]), (n_lat["stop_batch"], n_lat["queries"]) == (32, 200))
r32 = {d: (lat[d, "fixed"]["32"]["recomputed_mean"], lat[d, "hazard"]["32"]["recomputed_mean"]) for d in ("scifact", "fiqa", "trec-covid")}
state("at batches of 32 FixedBudget recomputes fewer documents", {d: (round(a), round(b)) for d, (a, b) in r32.items()},
      all(a < b for a, b in r32.values()))
# run_latency.sh asks for 200 queries (latency.num_queries=200), which a corpus with fewer test queries caps
nq = {d: min(200, beir[d]["dataset"]["n_queries"]) for d in ("scifact", "fiqa", "trec-covid")}
state("Table VIII: 200 queries, 50 on TREC-COVID", nq, nq == {"scifact": 200, "fiqa": 200, "trec-covid": 50})
tp = {(r_["delta"], r_["concurrency"]): r_ for r_ in load("msmarco/throughput_reco.json")["runs"]}
state("0.90 and 0.95 of the exact top-10 in Table IX", (round(tp["0.1", 64]["recall_vs_exact10"], 3), round(tp["0.05", 64]["recall_vs_exact10"], 3)),
      round(tp["0.1", 64]["recall_vs_exact10"], 2) == 0.90 and round(tp["0.05", 64]["recall_vs_exact10"], 2) == 0.95)

# ---- PLAID's operating points, Exp-4 and Exp-5
sweep = load("msmarco/plaid_sweep_nbits2.json")["points"]
skip = load("msmarco/plaid_sweep_skip_nbits2.json")["points"]
check_cpu = load("msmarco/plaid_sweep_cpu_check_nbits2.json")["points"]
cpu = check_cpu + load("msmarco/plaid_sweep_cpu_mid_nbits2.json")["points"] + load("msmarco/plaid_sweep_cpu_nbits2.json")["points"]
gpu = sweep[:2] + skip
single = load("msmarco/latency_K65536_b32.json")["runs"]
one_q = {d: 1000 / single[f"hazard@{d}"]["ms_mean"] for d in ("0.1", "0.05")}
state("the first GPU point reproduces the earlier run", (sweep[0]["recall_vs_exact10"], round(sweep[0]["mrr10"], 4)),
      sweep[0]["recall_vs_exact10"] == load("msmarco/plaid_nbits2.json")["gpu"]["recall_vs_exact10"])
state("two settings cannot finish on the 24 GB GPU", [p.get("queries_done") for p in sweep[2:]], all(p.get("out_of_memory") for p in sweep[2:]))
state("one of 1,000 queries exceeds the GPU at the defaults for k > 100", skip[0]["skipped"], skip[0]["skipped"] == 1)
same = [p for p in cpu if any(p["ndocs"] == g["ndocs"] and p["ncells"] == g["ncells"] for g in gpu)]
state("the CPU engine returns more at equal settings", [round(p["recall_vs_exact10"], 3) for p in same],
      all(p["recall_vs_exact10"] > g["recall_vs_exact10"] for p in same for g in gpu if p["ndocs"] == g["ndocs"] and p["ncells"] == g["ncells"]))
ceiling = max(p["recall_vs_exact10"] for p in cpu)
state("PLAID stops at 0.873, ReCo's lowest target is above it", f"{ceiling:.4f}", round(ceiling, 3) == 0.873 and ceiling < tp["0.1", 64]["recall_vs_exact10"])
state("ReCo one at a time leads every PLAID point but the shallowest", f"{one_q['0.1']:.1f} QPS vs {gpu[1]['qps']:.1f}",
      all(one_q["0.1"] > p["qps"] for p in gpu[1:]) and one_q["0.1"] < sweep[0]["qps"])
state("PLAID's shallowest setting is the faster one", f"{sweep[0]['qps']:.1f} vs {one_q['0.1']:.1f}", sweep[0]["qps"] > one_q["0.1"])
low = load("msmarco/plaid_sweep_cpu_low_nbits2.json")["points"]
deep = max((p for p in cpu + low if round(p["recall_vs_exact10"], 3) == 0.873), key=lambda p: (p["ncells"], p["ndocs"]))
state("PLAID stays at 0.873 up to 32 cells, 32,768 documents, threshold 0.2",
      (deep["ncells"], deep["ndocs"], min(p["centroid_score_threshold"] for p in low), max(p["recall_vs_exact10"] for p in cpu + low)),
      (deep["ncells"], deep["ndocs"]) == (32, 32768) and min(p["centroid_score_threshold"] for p in low) == 0.2
      and round(max(p["recall_vs_exact10"] for p in cpu + low), 3) == 0.873)
state("PLAID's deepest GPU setting reaches exact ColBERT's MRR@10 too", f"{skip[0]['mrr10']:.4f}", round(skip[0]["mrr10"], 3) == 0.369)
state("C=64 answers at a median of 2 to 3 seconds", (round(tp["0.1", 64]["latency_ms_p50"]), round(tp["0.05", 64]["latency_ms_p50"])),
      round(tp["0.1", 64]["latency_ms_p50"] / 1000) == 2 and round(tp["0.05", 64]["latency_ms_p50"] / 1000) == 3)

# ---- agent memories by size, Exp-11
for bench_name in ("longmemeval", "locomo"):
    by = load(f"memory/by_size_{bench_name}.json")
    ref = load(f"memory/memory_{bench_name}.json")["methods"]["reco-K64k"]["hazard_stop"]
    state(f"{bench_name}: the size check reproduces Table XII", f"{by['R_mean']:.1f}", abs(by["R_mean"] - ref["R_mean"]) < 1e-9)
    q = [x["recall"] for x in by["by_size"]]
    rising = sum(b > a for a, b in zip(q, q[1:]))
    state(f"{bench_name}: no trend of recall with memory size", [round(x, 3) for x in q], 0 < rising < len(q) - 1)

# ---- parameter study
ps = {n: load(f"param_study/{n}.json") for n in ("scidocs", "fiqa", "quora")}
one = ps["fiqa"]["k"]["1"]["0.05"], ps["quora"]["k"]["1"]["0.05"]
sci = ps["scidocs"]["k"]["1"]["0.05"]
state("k=1: on SCIDOCS TA-Margin needs 8.9 against 9.1", f"{sci['crc']['R_mean']:.2f} vs {sci['hazard']['R_mean']:.2f}",
      (round(sci["crc"]["R_mean"], 1), round(sci["hazard"]["R_mean"], 1)) == (8.9, 9.1))
state("k=1: on Quora FixedBudget needs 6.2 against 8.0", f"{one[1]['fixed']['R_mean']:.2f} vs {one[1]['hazard']['R_mean']:.2f}",
      (round(one[1]["fixed"]["R_mean"], 1), round(one[1]["hazard"]["R_mean"], 1)) == (6.2, 8.0))
state("k=1: hazard stopping wins on FiQA", f"{one[0]['hazard']['R_mean']:.2f} vs {one[0]['crc']['R_mean']:.2f}",
      one[0]["hazard"]["R_mean"] < min(one[0][r]["R_mean"] for r in ("fixed", "crc", "patience")))
grow = {}
for n in ("scidocs", "quora"):                      # Fig. 7: targets 0.90 to 0.99, LAET from the stopping baselines
    ev = beir[n]["methods"][f"reco-K{K[n]}"]["evaluation"]["deltas"]
    grow[n] = {r: (sb[n] if r == "laet" else beir[n]["methods"][f"reco-K{K[n]}"]["evaluation"])["deltas"]["0.01"][r]["R_mean"]
               / (sb[n] if r == "laet" else beir[n]["methods"][f"reco-K{K[n]}"]["evaluation"])["deltas"]["0.1"][r]["R_mean"]
               for r in ("fixed", "crc", "laet", "hazard")}
state("as the target tightens TA-Margin and FixedBudget grow faster than ReCo", {n: {r: round(v, 2) for r, v in g.items()} for n, g in grow.items()},
      all(g["crc"] > g["hazard"] and g["fixed"] > g["hazard"] for g in grow.values()))
over = []
for n, r in ps.items():
    keys = list(r["ncal"])
    over.append(r["ncal"]["50"]["hazard"]["R_mean"] / r["ncal"][keys[-1]]["hazard"]["R_mean"])
state("50 calibration queries need 1.2 to 1.7 times the full set", [round(x, 2) for x in over], round(min(over), 1) == 1.2 and round(max(over), 1) == 1.7)
state("Exp-9 covers 15 (corpus, k) settings on three corpora", {n: len(r["k"]) for n, r in ps.items()},
      len(ps) == 3 and sum(len(r["k"]) for r in ps.values()) == 15)

# ---- third review: statements added to the text
loss = [100 * (ndcg(beir[n]) - ndcg(beir[n], f"reco-K{K[n]}")) for n in BEIR]
state("scoring with the sketch alone loses 2.8 to 8.1 nDCG@10 points", f"{min(loss):.2f} to {max(loss):.2f}",
      (round(min(loss), 1), round(max(loss), 1)) == (2.8, 8.1))
arg_bin = ndcg(beir["arguana"], "dense-binary")
state("ArguAna: binary bge-small leads ReCo and exact ColBERT", f"{arg_bin:.3f} vs {headline_ndcg('arguana', 0.05):.3f}, {ndcg(beir['arguana']):.3f}",
      arg_bin > headline_ndcg("arguana", 0.05) and arg_bin > ndcg(beir["arguana"]))
cpu_mrr = max(p["mrr10"] for p in cpu + low)
exact_mrr = load("msmarco/eval_K65536.json")["exact"]["mrr10"]
state("PLAID's CPU engine reaches an MRR@10 of up to 0.373, above exact ColBERT", f"{cpu_mrr:.4f} vs {exact_mrr:.4f}",
      round(cpu_mrr, 3) == 0.373 and cpu_mrr > exact_mrr)
over32 = [lat[d, r_]["32"]["recall_mean"] for d in ("scifact", "fiqa", "trec-covid") for r_ in ("hazard", "crc")]
state("at batches of 32 the adaptive rules overshoot to 0.96 to 0.98 recall", [round(x, 3) for x in over32],
      (round(min(over32), 2), round(max(over32), 2)) == (0.96, 0.98))
tc = {k: (lat["trec-covid", "hazard"]["32"][k], lat["trec-covid", "fixed"]["32"][k]) for k in ("ms_p50", "ms_p90", "ms_p99", "recomputed_mean")}
state("TREC-COVID: ReCo trails FixedBudget in latency and recomputations", {k: (round(a), round(b)) for k, (a, b) in tc.items()},
      all(a > b for a, b in tc.values()))
ms_meta, ms_ev = load("msmarco/meta.json"), load("msmarco/eval_K65536.json")
sk_gb = (ms_ev["storage"]["sketch"] * ms_meta["n_docs"] + ms_ev["storage"]["codebook_total"]) / 1e9
inv_gb = load("msmarco/inverted_cpu.json")["memory_gb"]
state("the CPU path holds 2.5 GB with the sketch and its posting lists", f"{sk_gb:.3f} + {inv_gb:.3f} GB", round(sk_gb + inv_gb, 1) == 2.5)
one_bit = load("cost/cost_model.json")["settings"]["longmemeval"]["memory_mb"]["ColBERTv2 1-bit (ref.)"]
state("1-bit ColBERTv2 vectors take 0.75 MB per LongMemEval memory", f"{one_bit:.3f} MB", round(one_bit, 2) == 0.75)
enc = (ROOT / "paper_artifacts" / "tables" / "encoder.tex").read_text(encoding="utf-8")
state("Exp-12: three corpora use the pooled correction at 0.01", enc.count(r"\ddagger"), enc.count(r"\ddagger") == 3)

print(f"\n{failed} statement(s) failed")
sys.exit(1 if failed else 0)
