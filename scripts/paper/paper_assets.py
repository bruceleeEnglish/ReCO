"""Every number, table and figure of the ICDE paper, generated from the collected results.

    python scripts/paper/paper_assets.py results paper_artifacts

Inputs (missing ones are skipped): results/beir/<dataset>.json (configs/paper_bench.yaml, protocol v3),
results/ablation/hazard_ablation.json, results/latency/latency_*.json, results/memory/memory_<bench>.json,
results/qa/<bench>/[qwen3-4b/]qa_rows*.json, results/scalability/<dataset>_K<k>.json.
Outputs: <paper>/numbers.tex (macros), <paper>/tables/*.tex, <paper>/figures/*.pdf (scripts/paper/paper_figures.py).
Tables shade our rows with \\ours (defined in main.tex) and set the best value of a column in bold.

Headline stopping rule ("auto", as in the G1 report): the query-level CRC rule when its +1/(n+1) correction uses
under a quarter of the risk budget (n + 1 >= 4 / delta calibration queries), else the pooled per-document
correction. Old rules: crc | rank1; ours: hazard | hazard_doc.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_report as mr  # noqa: E402
import paper_figures as pf  # noqa: E402

ORDER = mr.ORDER
TITLE = mr.TITLES
SHORT = {"scifact": "SciFact", "nfcorpus": "NFCorp.", "arguana": "ArguAna", "scidocs": "SCIDOCS", "fiqa": "FiQA",
         "trec-covid": "TREC-C.", "quora": "Quora"}
DELTAS = ("0.05", "0.01")


def query_level(report: dict, delta: str) -> bool:
    return mr.min_calibration_queries(report) + 1 >= 4 / float(delta)


def headline_rules(report: dict, delta: str) -> tuple[str, str]:
    """(old rule, ours) under the same finite-sample correction."""
    return ("crc", "hazard") if query_level(report, delta) else ("rank1", "hazard_doc")


def entry(report: dict, method: str, delta: str, rule: str) -> dict:
    return report["methods"][method]["evaluation"]["deltas"][delta][rule]


def pick(report: dict) -> str:
    return mr.choose_reco(report, 0.3)


def p99(e: dict) -> float:
    return float(np.percentile(e["R_list"], 99))


def fmt_r(v: float) -> str:
    return f"{v:.0f}" if v >= 10 else f"{v:.1f}"


# ------------------------------------------------------------------------------------------ main table
def main_table(data: dict, macros: dict) -> str:
    """Quality and storage: ColBERT, the best lossy index of ReCo's size, and ReCo."""
    rows, gaps5, gaps1, lossy, catch, fullx, ratios, saving = [], [], [], [], [], [], [], []
    for name in ORDER:
        if name not in data:
            continue
        rep = data[name]
        ps = mr.points(rep, "crc")
        full = ps["full-fp16"]
        m = pick(rep)
        r = ps[m]
        bl = mr.best_within(mr.family(ps, mr.LOSSY_FAMILIES), r.ratio, lambda p: p.est_ndcg)
        close = [p.ratio for p in mr.family(ps, mr.LOSSY_FAMILIES) if p.est_ndcg >= full.est_ndcg - 0.005]
        cells = [TITLE[name], f"{full.ratio:.0f}", f"{full.est_ndcg:.3f}"]
        if bl:
            cells += [mr.method_tex(bl.name), f"{100 * (bl.est_ndcg - full.est_ndcg):+.1f}"]
            lossy.append(100 * (bl.est_ndcg - full.est_ndcg))
        else:
            cells += [r"\multicolumn{2}{c}{none fits}"]
        g = []
        for d, gaps in (("0.05", gaps5), ("0.01", gaps1)):
            e = entry(rep, m, d, headline_rules(rep, d)[1])
            g.append(100 * (e["ndcg10"] - full.est_ndcg))
            gaps.append(g[-1])
        sv = full.ratio / r.ratio
        cells += [rf"\ours {int(m[6:]) // 1024}k", rf"\ours \textbf{{{r.ratio:.2f}}}", rf"\ours {g[0]:+.1f}",
                  rf"\ours {g[1]:+.1f}", rf"\ours \textbf{{{sv:.0f}}}$\times$"]
        rows.append(" & ".join(cells) + r" \\")
        if close:
            catch.append(min(close) / r.ratio)
        fullx.append(full.ratio)
        saving.append(sv)
        if name != "quora":
            ratios.append(r.ratio)
    macros.update({
        "FullRatioMin": f"{min(fullx):.0f}", "FullRatioMax": f"{max(fullx):.0f}",
        "RecoRatioMin": f"{min(ratios):.2f}", "RecoRatioMax": f"{max(ratios):.2f}",
        "GapFiveMin": signed(min(gaps5)), "GapFiveMax": signed(max(gaps5)),
        "GapOneMin": signed(min(gaps1)), "GapOneMax": signed(max(gaps1)),
        "LossyGapMin": signed(min(lossy)), "LossyGapMax": signed(max(lossy)),
        "LossyDropMin": f"{-max(lossy):.1f}", "LossyDropMax": f"{-min(lossy):.1f}",
        "CatchUpMin": f"{min(catch):.0f}", "CatchUpMax": f"{max(catch):.0f}",
        "SavingMin": f"{min(saving):.0f}", "SavingMax": f"{max(saving):.0f}"})
    head = [r"\begin{tabular}{@{}l rr lr rrrrr@{}}", r"\toprule",
            r" & \multicolumn{2}{c}{ColBERT (uncompressed)} & \multicolumn{2}{c}{best lossy MV of ReCo's size} "
            r"& \multicolumn{5}{c}{\ours ReCo} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-10}",
            r"Dataset & $\times$raw & nDCG@10 & method & $\Delta$nDCG & \ours $K$ & \ours $\times$raw "
            r"& \ours $\Delta$ ($\delta$=.05) & \ours $\Delta$ ($\delta$=.01) & \ours smaller \\", r"\midrule"]
    return "\n".join(head + rows + [r"\bottomrule", r"\end{tabular}"]) + "\n"


# ------------------------------------------------------------------------------------------ second encoder
def encoder_table(data: dict, macros: dict) -> str:
    """ColBERTv2 checkpoint (128-d): quality at ReCo's size and the verification saving of hazard stopping."""
    names = [n for n in ORDER if n in data]
    if not names:
        return ""
    lines = [r"\begin{tabular}{@{}l rr r rrr rr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{ColBERTv2} & lossy & \multicolumn{3}{c}{\ours ReCo} "
             r"& \multicolumn{2}{c}{\ours fewer recomp.} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-7}\cmidrule(l){8-9}",
             r"Dataset & $\times$raw & nDCG@10 & $\Delta$ & \ours $\times$raw & \ours $\Delta_{.05}$ & \ours $\Delta_{.01}$ "
             r"& \ours $\delta{=}.05$ & \ours $\delta{=}.01$ \\", r"\midrule"]
    gaps, lossy, cuts5, cuts1, saving = [], [], [], [], []
    for name in names:
        rep = data[name]
        ps = mr.points(rep, "crc")
        full, m = ps["full-fp16"], pick(rep)
        r = ps[m]
        bl = mr.best_within(mr.family(ps, mr.LOSSY_FAMILIES), r.ratio, lambda p: p.est_ndcg)
        cells = [TITLE[name], f"{full.ratio:.0f}", f"{full.est_ndcg:.3f}",
                 signed(100 * (bl.est_ndcg - full.est_ndcg)) if bl else "--", rf"\ours \textbf{{{r.ratio:.2f}}}"]
        if bl:
            lossy.append(100 * (bl.est_ndcg - full.est_ndcg))
        for d in ("0.05", "0.01"):
            g = 100 * (entry(rep, m, d, headline_rules(rep, d)[1])["ndcg10"] - full.est_ndcg)
            gaps.append(g)
            cells.append(r"\ours " + signed(g))
        for d, cuts in (("0.05", cuts5), ("0.01", cuts1)):
            old, new = headline_rules(rep, d)
            c = 1 - entry(rep, m, d, new)["R_mean"] / entry(rep, m, d, old)["R_mean"]
            cuts.append(100 * c)
            cells.append(rf"\ours {100 * c:.0f}\%" + ("$^\\ddagger$" if old == "rank1" else ""))
        saving.append(full.ratio / r.ratio)
        lines.append(" & ".join(cells) + r" \\")
    macros.update({"EncGapMin": signed(min(gaps)), "EncGapMax": signed(max(gaps)),
                   "EncLossyDropMin": f"{-max(lossy):.1f}", "EncLossyDropMax": f"{-min(lossy):.1f}",
                   "EncCutFiveMin": f"{min(cuts5):.0f}", "EncCutFiveMax": f"{max(cuts5):.0f}",
                   "EncCutOneMin": f"{min(cuts1):.0f}", "EncCutOneMax": f"{max(cuts1):.0f}",
                   "EncSavingMin": f"{min(saving):.0f}", "EncSavingMax": f"{max(saving):.0f}",
                   "EncNumDatasets": str(len(names))})
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


# ------------------------------------------------------------------------------------------ web scale
def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def msmarco_table(d: Path, macros: dict) -> str:
    """MS MARCO passages: index size, first-stage and verified MRR@10, recomputations (hazard stopping)."""
    meta = load_json(d / "meta.json")
    evs = {k: load_json(d / f"eval_K{k}.json") for k in (16384, 65536)}
    evs = {k: v for k, v in evs.items() if v}
    if not meta or not evs:
        return ""
    bm = load_json(d / "eval_bm25.json")
    n, nq = meta["n_docs"], meta["n_queries"]
    n_cal = round(0.6 * (nq - nq / 5))
    rules = {dl: ("crc", "hazard") if n_cal + 1 >= 4 / float(dl) else ("rank1", "hazard_doc") for dl in DELTAS}
    any_ev = next(iter(evs.values()))
    st, exact = any_ev["storage"], any_ev["exact"]
    raw = st["raw_text"]

    def gb(bytes_per_doc: float) -> str:
        return f"{bytes_per_doc * n / 1e9:,.1f}" if bytes_per_doc * n >= 1e9 else f"{bytes_per_doc * n / 1e9:.2f}"

    def r_cell(e: dict, o: str) -> str:
        if e["beyond_prefix"] > 0.005:
            return o + rf"$>${e['R_mean'] / 1000:.0f}k$^\S$"
        return o + (f"{e['R_mean']:,.0f}")

    plaid = load_json(d / "plaid_nbits2.json")
    plaid_bpd = plaid["index_bytes"] / n if plaid and "index_bytes" in plaid else st["colbertv2_2bit"]
    lines = [r"\begin{tabular}{@{}l rr r rr rr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{index} & alone & \multicolumn{2}{c}{$\delta{=}0.05$} & \multicolumn{2}{c}{$\delta{=}0.01$} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}",
             r"Method & GB & $\times$raw & MRR & MRR & $\bar R$ & MRR & $\bar R$ \\", r"\midrule",
             rf"ColBERT, exact (fp16) & {gb(st['colbert_fp16'])} & {st['colbert_fp16'] / raw:.0f} & "
             rf"{exact['mrr10']:.3f} & \multicolumn{{4}}{{c}}{{(every passage scored)}} \\"]
    if plaid and "gpu" in plaid:
        lines.append(rf"PLAID, 2-bit (official) & {gb(plaid_bpd)} & {plaid_bpd / raw:.1f} & {plaid['gpu']['mrr10']:.3f} & "
                     r"\multicolumn{4}{c}{(fixed pruning, no recall target)} \\")
    else:
        lines.append(rf"ColBERTv2, 2-bit (size) & {gb(plaid_bpd)} & {plaid_bpd / raw:.1f} & \multicolumn{{5}}{{c}}{{--}} \\")
    rows = []
    if bm:
        rows.append(("BM25-Rerank", bm["index_bytes_per_doc"], bm, False))
    for k, ev in sorted(evs.items()):
        rows.append((f"ReCo, $K$={k // 1024}k", ev["storage"]["sketch"], ev, True))
    for lab, size, ev, ours in rows:
        o = r"\ours " if ours else ""
        cells = [o + lab, o + gb(size), o + f"{size / raw:.2f}", o + f"{ev['estimate_mrr10']:.3f}"]
        for dl in DELTAS:
            e = ev["deltas"][dl][rules[dl][1]]
            cells += [o + f"{e['mrr10']:.3f}", r_cell(e, o)]
        lines.append(" & ".join(cells) + r" \\")
    head = evs.get(65536) or any_ev
    k_head = 65536 if 65536 in evs else next(iter(evs))
    h5, h1 = (head["deltas"][dl][rules[dl][1]] for dl in DELTAS)
    o5, o1 = (head["deltas"][dl][rules[dl][0]] for dl in DELTAS)
    reco_bpd = head["storage"]["sketch"] + head["storage"]["codebook_total"] / n      # sketches + shared codebook
    macros.update({
        "MsDocs": f"{n / 1e6:.1f}", "MsQueries": f"{nq:,}", "MsRaw": f"{raw:.0f}", "MsRawGB": gb(raw),
        "MsColbertGB": gb(st["colbert_fp16"]), "MsPlaidGB": gb(plaid_bpd),
        "MsRecoGB": gb(reco_bpd), "MsRecoRatio": f"{reco_bpd / raw:.2f}",
        "MsRecoK": f"{k_head // 1024}k", "MsCodebookMB": f"{head['storage']['codebook_total'] / 1e6:.0f}",
        "MsSavingColbert": f"{st['colbert_fp16'] / reco_bpd:.0f}",
        "MsSavingPlaid": f"{plaid_bpd / reco_bpd:.0f}",
        "MsExactMrr": f"{exact['mrr10']:.3f}", "MsExactRecall": f"{exact['recall1000']:.3f}",
        "MsSketchMrr": f"{head['estimate_mrr10']:.3f}",
        "MsMrrFive": f"{h5['mrr10']:.3f}", "MsMrrOne": f"{h1['mrr10']:.3f}",
        "MsRFive": f"{h5['R_mean']:,.0f}", "MsROne": f"{h1['R_mean']:,.0f}",
        "MsRecallFive": f"{h5['recall']:.3f}", "MsRecallOne": f"{h1['recall']:.3f}",
        "MsRFivePct": f"{100 * h5['R_mean'] / n:.4f}", "MsROnePct": f"{100 * h1['R_mean'] / n:.3f}",
        "MsMarginRFive": f"{o5['R_mean']:,.0f}", "MsMarginROne": f"{o1['R_mean']:,.0f}",
        "MsCutFive": f"{100 * (1 - h5['R_mean'] / o5['R_mean']):.0f}",
        "MsCutOne": f"{100 * (1 - h1['R_mean'] / o1['R_mean']):.0f}",
        "MsFixedRFive": f"{head['deltas']['0.05']['fixed']['R_mean']:,.0f}",
        "MsFixedROne": f"{head['deltas']['0.01']['fixed']['R_mean']:,.0f}",
        "MsPatienceRFive": f"{head['deltas']['0.05']['patience']['R_mean']:,.0f}",
        "MsPatienceROne": f"{head['deltas']['0.01']['patience']['R_mean']:,.0f}",
        "MsCutFixedFive": f"{100 * (1 - h5['R_mean'] / head['deltas']['0.05']['fixed']['R_mean']):.0f}",
        "MsCutFixedOne": f"{100 * (1 - h1['R_mean'] / head['deltas']['0.01']['fixed']['R_mean']):.0f}",
        "MsMarginBeyond": f"{100 * o1['beyond_prefix']:.0f}",
        "MsScanMs": f"{head['scan_ms']['p50']:.1f}", "MsPrefix": f"{head['prefix']:,}",
        "MsTrueRNinetyNine": f"{head['true_R_at_recall']['0.99']:,.0f}"})
    if 16384 in evs:
        macros["MsScanMsSmall"] = f"{evs[16384]['scan_ms']['p50']:.1f}"
        macros["MsMrrFiveSmall"] = f"{evs[16384]['deltas']['0.05'][rules['0.05'][1]]['mrr10']:.3f}"
        macros["MsRFiveSmall"] = f"{evs[16384]['deltas']['0.05'][rules['0.05'][1]]['R_mean']:,.0f}"
    if bm:
        b5 = bm["deltas"]["0.05"][rules["0.05"][1]]
        macros.update({"MsBmRFive": f"{b5['R_mean']:,.0f}", "MsBmGainFive": f"{b5['R_mean'] / h5['R_mean']:.0f}",
                       "MsBmMrr": f"{bm['estimate_mrr10']:.3f}",
                       "MsBmTrueRNinetyNine": f"{bm['true_R_at_recall']['0.99'] / 1e6:.1f}",
                       "MsBmTrueRNinetyNinePct": f"{100 * bm['true_R_at_recall']['0.99'] / n:.0f}"})
    if plaid and "gpu" in plaid:
        g = plaid["gpu"]
        macros.update({"MsPlaidMrr": f"{g['mrr10']:.3f}", "MsPlaidRecall": f"{g['recall_vs_exact10']:.2f}",
                       "MsPlaidGpu": f"{g['latency_ms']['p50']:.0f}", "MsPlaidGpuNinety": f"{g['latency_ms']['p90']:.0f}",
                       "MsPlaidGpuNinetyNine": f"{g['latency_ms']['p99']:.0f}", "MsPlaidRatio": f"{plaid_bpd / raw:.1f}",
                       "MsRecoRecallTen": f"{head['deltas']['0.1']['hazard']['recall']:.2f}",
                       "MsRecoMrrTen": f"{head['deltas']['0.1']['hazard']['mrr10']:.3f}"})
        if "build_s" in plaid and meta.get("timings"):
            t0 = meta["timings"]
            reco_min = (t0["encode_s"] + t0["assign_s"] + t0.get(f"kmeans_K{k_head}_s", 0)) / 60
            plaid_gpu_min = plaid["build_s"] / 60 * plaid.get("nranks", 1)
            macros.update({"MsPlaidBuildMin": f"{plaid['build_s'] / 60:.0f}", "MsPlaidGpus": str(plaid.get("nranks", 1)),
                           "MsPlaidBuildGpuMin": f"{plaid_gpu_min:.0f}", "MsBuildX": f"{plaid_gpu_min / reco_min:.1f}"})
        if "cpu" in plaid:
            macros["MsPlaidCpu"] = f"{plaid['cpu']['latency_ms']['p50']:.0f}"
            macros["MsPlaidCpuThreads"] = str(plaid["cpu"]["threads"])
    t = meta.get("timings", {})
    if t:
        macros["MsRecoBuildMin"] = f"{(t['encode_s'] + t['assign_s'] + t.get(f'kmeans_K{k_head}_s', 0)) / 60:.0f}"
        macros["MsPassMin"] = f"{t['pass_s'] / 60:.0f}"
        macros["MsEncodeRate"] = f"{n / t['pass_s']:,.0f}"
        macros["MsAssignShare"] = f"{100 * t['assign_s'] / t['pass_s']:.0f}"
    for b, suf in ((8, ""), (32, "B")):
        lat = load_json(d / f"latency_K{k_head}{'' if b == 8 else '_b32'}.json")
        if not lat:
            continue
        for key, tag in (("hazard@0.05", "Five"), ("hazard@0.1", "Ten"), ("fixed@0.05", "FixFive")):
            if key in lat["runs"]:
                r = lat["runs"][key]
                macros[f"MsLat{tag}{suf}"] = f"{r['ms_p50']:.0f}"
                macros[f"MsLatNinety{tag}{suf}"] = f"{r['ms_p90']:.0f}"
                macros[f"MsLatEst{tag}{suf}"] = f"{r['ms_mean_by_stage']['estimate']:.1f}"
                macros[f"MsLatMatch{tag}{suf}"] = f"{100 * r['recomputed_matches_offline']:.0f}"
                macros[f"MsLatNinetyNine{tag}{suf}"] = f"{r['ms_p99']:.0f}"
                macros[f"MsLatRecall{tag}{suf}"] = f"{r['recall_vs_exact']:.2f}"
        if "cpu_scan_ms" in lat:
            macros[f"MsCpuScan{suf}"] = f"{lat['cpu_scan_ms']['8']:.0f}"
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


# ------------------------------------------------------------------------------------------ on-device latency, cost
def sig(x: float) -> str:
    """Two significant digits, with thousands separators for large values."""
    if x >= 100:
        return f"{float(f'{x:.2g}'):,.0f}"
    return f"{x:.2g}" if x >= 1 else f"{x:.1g}"


def ms_fmt(v: float) -> str:
    return f"{v:.1f}" if v < 10 else f"{v:,.0f}"


CPU_ROWS = [("colbert-stored", "ColBERT, stored vectors", "ColBERT fp16", False),
            ("reco-K64k", "ReCo, $K$=64k", "ReCo sketch K=64k", True),
            ("reco-K16k", "ReCo, $K$=16k", "ReCo sketch K=16k", True),
            ("reencode-all", "Re-encode memory, no index", None, False),
            ("bge-small", "bge-small", "bge-small fp16", False), ("bm25", "BM25", "BM25 index", False)]


def cpu_table(d: Path, macros: dict) -> str:
    """One question against one user's memory on a laptop CPU (8 threads): index bytes per unit, latency (ms).
    The server run (same replay on 8 pinned cores) adds the GPU replay and the reader's prefill, for scale."""
    reps = {b: load_json(d / f"{b}_laptop.json") for b in ("longmemeval", "locomo")}
    reps = {b: r for b, r in reps.items() if r}
    if not reps:
        return ""
    lines = [r"\begin{tabular}{@{}l r rr r rr@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{LongMemEval-S} & \multicolumn{3}{c}{LoCoMo} \\",
             r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
             r"Retriever & B/unit & p50 (ms) & p90 (ms) & B/unit & p50 (ms) & p90 (ms) \\", r"\midrule"]
    for key, lab, size_key, ours in CPU_ROWS:
        o = r"\ours " if ours else ""
        cells = [o + lab]
        for b in ("longmemeval", "locomo"):
            r = reps.get(b)
            m = r["methods"].get(key) if r else None
            if not m:
                cells += ["--"] * 3
                continue
            size = r["storage_bytes_per_unit"].get(size_key, 0) if size_key else 0
            cells += [o + f"{size:,.0f}", o + ms_fmt(m["total"]["p50"]), o + ms_fmt(m["total"]["p90"])]
        lines.append(" & ".join(cells) + r" \\")
    for b, tag in (("longmemeval", "Lme"), ("locomo", "Lc")):
        r = reps.get(b)
        if not r:
            continue
        m = r["methods"]
        rk, rs = m["reco-K64k"], m["reco-K16k"]
        macros.update({f"Cpu{tag}Reco": f"{rk['total']['p50']:,.0f}", f"Cpu{tag}RecoNinety": f"{rk['total']['p90']:,.0f}",
                       f"Cpu{tag}RecoEnc": f"{rk['encode_query']['mean']:.0f}",
                       f"Cpu{tag}RecoEst": f"{rk['estimate']['mean']:.0f}",
                       f"Cpu{tag}RecoRec": f"{rk['recompute']['mean']:,.0f}", f"Cpu{tag}R": f"{rk['R_mean']:.0f}",
                       f"Cpu{tag}RecoShare": f"{100 * rk['recompute']['mean'] / rk['total']['mean']:.0f}",
                       f"Cpu{tag}RecoMsPerUnit": f"{rk['recompute']['mean'] / rk['R_mean']:.0f}",
                       f"Cpu{tag}RecoSmall": f"{rs['total']['p50']:,.0f}",
                       f"Cpu{tag}Colbert": f"{m['colbert-stored']['total']['p50']:,.0f}",
                       f"Cpu{tag}Reencode": f"{m['reencode-all']['total']['p50'] / 1000:.1f}",
                       f"Cpu{tag}ReencodeX": f"{m['reencode-all']['total']['p50'] / rk['total']['p50']:.0f}",
                       f"Cpu{tag}Bge": f"{m['bge-small']['total']['p50']:,.0f}",
                       f"Cpu{tag}Units": f"{r['units_per_memory']:.0f}", "CpuThreads": str(r["threads"]),
                       "CpuName": r["cpu"].split(" w/")[0]})
        srv = load_json(d / f"{b}_server.json")
        if not srv:
            continue
        sk = srv["methods"]["reco-K64k"]["total"]["p50"]
        macros[f"Srv{tag}Reco"] = f"{sk:,.0f}"
        if "gpu" in srv:
            macros[f"Gpu{tag}Reco"] = f"{srv['gpu']['reco-K64k']['total']['p50']:.0f}"
        rd = srv.get("reader", {})
        if "prefill_ms_cpu" in rd:
            p = rd["prefill_ms_cpu"]["p50"]
            macros[f"Reader{tag}Cpu"] = f"{p / 1000:.0f}"
            macros[f"Reader{tag}CpuX"] = f"{p / sk:.0f}"
        if "prefill_ms_cuda" in rd:
            macros[f"Reader{tag}Gpu"] = f"{rd['prefill_ms_cuda']['p50']:.0f}"
            macros[f"Reader{tag}Tok"] = f"{rd['prompt_tokens_mean']:,.0f}"
            if "gpu" in srv:
                macros[f"Reader{tag}GpuX"] = f"{rd['prefill_ms_cuda']['p50'] / srv['gpu']['reco-K64k']['total']['p50']:.1f}"
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def cost_macros(path: Path, macros: dict) -> None:
    rep = load_json(path)
    if not rep:
        return
    p = rep["prices"]
    macros.update({"PriceDram": f"{p['dram_usd_gb_month']:g}", "PriceSsd": f"{p['ssd_usd_gb_month']:g}",
                   "PriceGpu": f"{p['gpu_usd_hour']:.2f}", "PriceVcpu": f"{p['vcpu_usd_hour']:.2f}"})
    ms = rep["settings"].get("msmarco")
    if ms:
        q = "gpu_b32" if "gpu_b32" in ms["per_query"] else "gpu_b8"
        for base, btag in (("colbert_fp16", "Colbert"), ("colbertv2_2bit", "Plaid"), ("colbertv2_2bit_net", "PlaidNet")):
            for tier, ttag in (("dram", "Dram"), ("ssd", "Ssd")):
                e = ms["break_even"].get(f"{base}|{tier}|{q}")
                if e:
                    macros[f"CostMs{btag}{ttag}"] = sig(e["qps"])
                    macros[f"CostMs{btag}{ttag}Usd"] = sig(e["saved_usd_month"])
        if q in ms["per_query"]:
            macros["CostMsQueryUsd"] = f"{ms['per_query'][q]['usd'] * 1e6:.1f}"
    for bench, tag in (("longmemeval", "Lme"), ("locomo", "Lc")):
        e = rep["settings"].get(bench)
        if not e:
            continue
        for base, btag in (("ColBERT fp16", "Colbert"), ("ColBERTv2 1-bit (ref.)", "Plaid")):
            for tier, ttag in (("dram", "Dram"), ("ssd", "Ssd")):
                for q, qtag in (("cpu8", "Cpu"), ("gpu", "Gpu")):
                    v = e["break_even"].get(f"{base}|{tier}|{q}")
                    if v:   # queries per user per day
                        macros[f"Cost{tag}{btag}{ttag}{qtag}"] = sig(v["queries_per_month"] / 30)
        macros[f"Cost{tag}ColbertMb"] = f"{e['memory_mb']['ColBERT fp16']:.1f}"
        sk = next(k for k in e["memory_mb"] if k.startswith("ReCo sketch K=64k"))
        macros[f"Cost{tag}SketchKb"] = f"{1000 * e['memory_mb'][sk]:.0f}"


# ------------------------------------------------------------------------------------------ verification table
RULE_ROWS = [("fixed", "Fixed budget", False), ("crc", "Margin", False), ("patience", "Patience", False),
             ("hazard_u", "Hazard-U (ours)", True), ("hazard", "Hazard (ours)", True),
             ("rank1", r"Rank+1$^\ddagger$", False), ("hazard_doc", r"Hazard-D$^\ddagger$ (ours)", True)]


def verification_table(data: dict, macros: dict) -> str:
    """Methods as rows, datasets as columns: mean recomputations at the headline configuration."""
    names = [n for n in ORDER if n in data]
    lines = [r"\begin{tabular}{@{}l " + "r" * len(names) + r"@{}}", r"\toprule",
             "Rule & " + " & ".join(SHORT[n] for n in names) + r" \\",
             " & " + " & ".join(f"{int(pick(data[n])[6:]) // 1024}k" for n in names) + r" \\"]
    red5, red1, tail = [], [], []
    for d in DELTAS:
        lines += [r"\midrule", rf"\multicolumn{{{len(names) + 1}}}{{@{{}}l}}{{\textit{{recall target {1 - float(d):.2f}"
                  rf" ($\delta={d}$)}}}} \\"]
        vals = {}
        for name in names:
            rep = data[name]
            ev = rep["methods"][pick(rep)]["evaluation"]["deltas"][d]
            n_docs = rep["dataset"]["n_docs"]
            vals[name] = {r: ev[r]["R_mean"] for r, _, _ in RULE_ROWS if r in ev and ev[r]["R_mean"] < n_docs}
            old, new = headline_rules(rep, d)
            if old in ev and new in ev:
                (red5 if d == "0.05" else red1).append(1 - ev[new]["R_mean"] / ev[old]["R_mean"])
                tail.append(p99(ev[old]) / max(p99(ev[new]), 1))
        for group in (RULE_ROWS[:5], RULE_ROWS[5:]):
            best = {n: min((vals[n][r] for r, _, _ in group if r in vals[n]), default=None) for n in names}
            for r, lab, ours in group:
                cells = [rf"\ours {lab}" if ours else lab]
                for n in names:
                    v = vals[n].get(r)
                    s = "--" if v is None else (rf"\textbf{{{fmt_r(v)}}}" if v == best[n] else fmt_r(v))
                    cells.append(rf"\ours {s}" if ours else s)
                lines.append(" & ".join(cells) + r" \\")
            if group is RULE_ROWS[:5]:
                lines.append(r"\addlinespace[1pt]")
    macros.update({"RedFiveMin": f"{100 * min(red5):.0f}", "RedFiveMax": f"{100 * max(red5):.0f}",
                   "RedOneMin": f"{100 * min(red1):.0f}", "RedOneMax": f"{100 * max(red1):.0f}",
                   "RedFiveMean": f"{100 * np.mean(red5):.0f}", "RedOneMean": f"{100 * np.mean(red1):.0f}",
                   "TailRatioMedian": f"{np.median(tail):.1f}", "TailRatioMax": f"{max(tail):.1f}"})
    # tail factor over every (dataset, K, delta) setting with both rules feasible
    allt = []
    for name in names:
        rep = data[name]
        for k in (1024, 4096, 16384, 65536):
            m = rep["methods"].get(f"reco-K{k}")
            if not m:
                continue
            for d in DELTAS:
                ev = m["evaluation"]["deltas"][d]
                if ev["crc"]["R_mean"] < rep["dataset"]["n_docs"] and ev["hazard"]["R_mean"] < rep["dataset"]["n_docs"]:
                    allt.append(p99(ev["crc"]) / max(p99(ev["hazard"]), 1))
    macros["TailAllMax"] = f"{max(allt):.0f}"
    macros["TailAllMedian"] = f"{np.median(allt):.1f}"
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def cascade_macros(data: dict, macros: dict) -> None:
    cut, gain, sizes = [], [], {m: [] for m in ("bm25", "splade", "dense-fp16")}
    for name in ORDER:
        if name not in data:
            continue
        rep = data[name]
        reco = pick(rep)
        for m in sizes:
            sizes[m].append(rep["methods"][m]["storage_ratio"] / rep["methods"][reco]["storage_ratio"])
        for d in DELTAS:
            er = rep["methods"][reco]["evaluation"]["deltas"][d]
            if er["hazard"]["R_mean"] >= rep["dataset"]["n_docs"]:
                continue
            best = math.inf
            for m in sizes:
                e = rep["methods"][m]["evaluation"]["deltas"][d]
                cut.append(1 - e["hazard"]["R_mean"] / e["fixed"]["R_mean"])
                best = min(best, e["hazard"]["R_mean"])
            gain.append(best / er["hazard"]["R_mean"])
    above = [g for g in gain if g >= 1]
    macros.update({"CascadeCutMin": f"{100 * min(cut):.0f}", "CascadeCutMax": f"{100 * max(cut):.0f}",
                   "CascadeCutMedian": f"{100 * np.median(cut):.0f}", "CascadeGainMin": f"{min(above):.1f}",
                   "CascadeGainMax": f"{max(above):.0f}", "CascadeGainMedian": f"{np.median(gain):.1f}"})
    for m, tag in (("bm25", "Bm"), ("splade", "Splade"), ("dense-fp16", "Dense")):
        lo, hi = min(sizes[m]), max(sizes[m])
        macros[f"Size{tag}Min"] = f"{lo:.1f}"
        macros[f"Size{tag}Max"] = f"{hi:.1f}" if hi < 10 else f"{hi:.0f}"


# ------------------------------------------------------------------------------------------ ablation
ABL_ROWS = [("hazard (all features)", "ReCo (full hazard model)", True), ("residual risk (theory)", "Residual-risk target", False),
            ("crc margin", "Margin only (= TA-Margin)", False), ("patience only", "Patience only (= Patience)", False),
            ("margin + patience", "Margin + patience", False), ("no online margin", "w/o online margin", False),
            ("no flatness", "w/o flatness", False), ("H=16", "horizon $H=16$", False), ("H=256", "horizon $H=256$", False),
            ("gradient-boosted trees", "Gradient-boosted trees", False), ("fixed", "FixedBudget", False)]


def ablation_table(abl: dict, macros: dict) -> str:
    """Geometric mean over (dataset, K) of R / R(hazard) per delta."""
    lines = [r"\begin{tabular}{@{}l rr rr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{$K=4$k} & \multicolumn{2}{c}{$K=16$k} \\", r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
             r"Variant & $\delta{=}0.05$ & $\delta{=}0.01$ & $\delta{=}0.05$ & $\delta{=}0.01$ \\", r"\midrule"]
    for key, lab, ours in ABL_ROWS:
        cells = [rf"\ours {lab}" if ours else lab]
        for k in ("4096", "16384"):
            for d in ("0.05", "0.01"):
                ratios = [v[key]["R_mean"] / v["hazard (all features)"]["R_mean"]
                          for s, v in abl.items() if s.split("/")[1] == f"K{k}" and s.split("/")[2] == d and key in v]
                g = float(np.exp(np.mean(np.log(ratios)))) if ratios else math.nan
                s = (f"{g:.2f}" if g < 10 else f"{g:.0f}") + ("$\\times$" if not ours else "")
                cells.append(r"\ours 1.00" if ours else s)
                if key == "residual risk (theory)":
                    macros[f"ResidualRatio{'Four' if k == '4096' else 'Sixteen'}{'Five' if d == '0.05' else 'One'}"] = \
                        f"{100 * (g - 1):.0f}"
        lines.append(" & ".join(cells) + r" \\")
        if key in ("hazard (all features)", "residual risk (theory)", "patience only", "no flatness", "H=256"):
            lines.append(r"\addlinespace[2pt]")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


# ------------------------------------------------------------------------------------------ latency
def latency_table(lat_dir: Path, macros: dict) -> str:
    lines = [r"\begin{tabular}{@{}l l rrr r@{}}", r"\toprule",
             r"Dataset ($K$) & Rule & p50 (ms) & p90 (ms) & p99 (ms) & $\bar R$ \\", r"\midrule"]
    found = False
    for ds, k in (("scifact", 4096), ("fiqa", 16384), ("trec-covid", 65536)):
        group = []
        for rule, lab in (("fixed", "FixedBudget"), ("crc", "TA-Margin"), ("hazard", "ReCo")):
            path = lat_dir / f"latency_{ds}_K{k}_{rule}.json"
            if path.exists():
                group.append((rule, lab, json.loads(path.read_text(encoding="utf-8"))["latency"]["batches"]["32"]))
        if not group:
            continue
        found = True
        for i, (rule, lab, b) in enumerate(group):
            o = r"\ours " if rule == "hazard" else ""
            first = f"{TITLE[ds]} ({k // 1024}k)" if i == 0 else ""
            lines.append(f"{first} & {o}{lab} & {o}{b['ms_p50']:.0f} & {o}{b['ms_p90']:.0f} & {o}{b['ms_p99']:.0f} "
                         f"& {o}{b['recomputed_mean']:.0f} \\\\")
            macros["MsPerDoc"] = f"{b['ms_per_recomputed_doc']:.2f}"
        lines.append(r"\addlinespace[2pt]")
    return ("\n".join(lines[:-1] + [r"\bottomrule", r"\end{tabular}"]) + "\n") if found else ""


# ------------------------------------------------------------------------------------------ memory / QA
def qa_rows(path: Path) -> dict[str, dict[str, tuple[bool, int]]]:
    out: dict[str, dict[str, tuple[bool, int]]] = {}
    if path.exists():
        for r in json.loads(path.read_text(encoding="utf-8")):
            out.setdefault(r["retriever"], {})[r["qid"]] = (r["correct"], r["prompt_tokens"])
    return out


def acc(rows: dict, name: str) -> tuple[float, float, int]:
    v = list(rows.get(name, {}).values())
    return (float(np.mean([c for c, _ in v])), float(np.mean([t for _, t in v])), len(v)) if v else (math.nan, math.nan, 0)


def paired(rows: dict, a: str, b: str = "colbert") -> tuple[float, float]:
    common = sorted(set(rows.get(a, {})) & set(rows.get(b, {})))
    if len(common) < 2:
        return math.nan, math.nan
    diff = np.array([rows[a][q][0] - rows[b][q][0] for q in common], dtype=float)
    return float(diff.mean()), float(1.96 * diff.std(ddof=1) / math.sqrt(len(diff)))


READER = "Qwen3-4B-Instruct-2507"

BUILD_ROWS = [("colbert-fp16", "ColBERT (fp16)", False), ("colbertv2-2bit", "ColBERTv2 (2-bit)", False),
              ("reco-K16384", "ReCo", True), ("reco-K16384-U", "ReCo, universal codebook", True)]


def build_table(build_dir: Path, macros: dict) -> str:
    """Index size and construction / insertion cost (encoding is shared and timed separately)."""
    reps = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(build_dir.glob("*.json"))} \
        if build_dir.is_dir() else {}
    names = [n for n in ("fiqa", "quora") if n in reps]
    if not names:
        return ""
    lines = [r"\begin{tabular}{@{}l " + "rrr " * len(names) + r"@{}}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{TITLE[n]} ({reps[n]['n_docs'] / 1000:.0f}k docs)}}"
                                for n in names) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(names))),
             "Index & " + " & ".join(["B/doc", "Build (s)", "Insert (ms)"] * len(names)) + r" \\", r"\midrule"]
    best_b = {n: min(reps[n]["methods"][k]["bytes_per_doc"] for k, _, _ in BUILD_ROWS if k in reps[n]["methods"])
              for n in names}
    for key, lab, ours in BUILD_ROWS:
        o = r"\ours " if ours else ""
        cells = [o + lab]
        for n in names:
            m = reps[n]["methods"].get(key)
            if not m:
                cells += ["--"] * 3
                continue
            b = m["bytes_per_doc"]
            cells += [o + (rf"\textbf{{{b:,.0f}}}" if b == best_b[n] else f"{b:,.0f}"), o + f"{m['train_s'] + m['per_token_s']:.1f}",
                      o + f"{m['insert_ms_per_doc']:.2f}"]
        lines.append(" & ".join(cells) + r" \\")
    for n in names:
        m = reps[n]["methods"]
        tag = n.title()
        macros[f"BuildEnc{tag}"] = f"{reps[n]['encode_ms_per_doc']:.2f}"
        macros[f"BuildSizeColbert{tag}"] = f"{m['colbert-fp16']['bytes_per_doc'] / m['reco-K16384']['bytes_per_doc']:.0f}"
        macros[f"BuildSizePlaid{tag}"] = f"{m['colbertv2-2bit']['bytes_per_doc'] / m['reco-K16384']['bytes_per_doc']:.0f}"
        macros[f"BuildTimePlaid{tag}"] = f"{m['colbertv2-2bit']['train_s'] + m['colbertv2-2bit']['per_token_s']:.1f}"
        macros[f"BuildTimeReco{tag}"] = f"{m['reco-K16384']['train_s'] + m['reco-K16384']['per_token_s']:.1f}"
        if "reco-K16384-U" in m:
            macros[f"BuildTimeUni{tag}"] = f"{m['reco-K16384-U']['per_token_s']:.1f}"
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


QA_COLS = [("oracle", "Oracle", False), ("colbert", "ColBERT", False), ("reco-K64k+R", "ReCo", True),
           ("reco-K64k", "ReCo w/o verification", True), ("bm25", "BM25", False), ("bge-base", "bge-base", False)]


def qa_table(qa: Path, mem: Path, macros: dict) -> str:
    sets = {"lc_ds": qa_rows(qa / "locomo" / "qa_rows_top10_full.json"),
            "lme_ds": qa_rows(qa / "longmemeval" / "qa_rows_top10_full.json"),
            "lc_qw": qa_rows(qa / "locomo" / READER / "qa_rows_top10_full.json"),
            "lme_qw": qa_rows(qa / "longmemeval" / READER / "qa_rows_top10_full32768.json")}
    for bench, tag in (("locomo", "Lc"), ("longmemeval", "Lme")):
        path = qa / bench / "qa_report_top10_full.json"
        jc = json.loads(path.read_text(encoding="utf-8")).get("judge_check") if path.exists() else None
        if jc:
            macros[f"JudgeAgree{tag}"] = f"{100 * jc['agreement']:.1f}"
            macros[f"JudgeKappa{tag}"] = f"{jc['kappa']:.2f}"
            macros[f"JudgeN{tag}"] = str(jc["n"])
    storage = {b: json.loads((mem / f"memory_{b}.json").read_text(encoding="utf-8"))["storage_bytes_per_unit"]
               for b in ("longmemeval", "locomo") if (mem / f"memory_{b}.json").exists()}
    lines = [r"\begin{tabular}{@{}l rr rr rr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Index (B/unit)} & \multicolumn{2}{c}{4B reader (on-device)} "
             r"& \multicolumn{2}{c}{cloud reader} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
             r"Retriever (top-10) & LME & LoCoMo & LME & LoCoMo & LME & LoCoMo \\", r"\midrule"]
    size_key = {"colbert": "ColBERT fp16", "reco-K64k+R": "ReCo sketch K=64k", "reco-K64k": "ReCo sketch K=64k",
                "bm25": "BM25 index", "bge-base": "bge-base fp16"}
    best = {s: max(acc(sets[s], n)[0] for n, _, _ in QA_COLS if n != "oracle" and acc(sets[s], n)[2]) for s in sets
            if sets[s]}
    for name, lab, ours in QA_COLS:
        o = r"\ours " if ours else ""
        cells = [o + lab]
        for b in ("longmemeval", "locomo"):
            key = size_key.get(name)
            cells.append(o + (f"{storage[b][key]:,.0f}" if key and b in storage else "--"))
        for s in ("lme_qw", "lc_qw", "lme_ds", "lc_ds"):
            a, _, n = acc(sets[s], name)
            txt = "--" if not n else (rf"\textbf{{{a:.3f}}}" if name != "oracle" and abs(a - best.get(s, -1)) < 1e-9
                                      else f"{a:.3f}")
            cells.append(o + txt)
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    cells = ["FullContext", "--", "--"]
    for s in ("lme_qw", "lc_qw", "lme_ds", "lc_ds"):
        a, _, n = acc(sets[s], "full")
        cells.append("--" if not n else f"{a:.3f}")
    lines.append(" & ".join(cells) + r" \\")
    for s, tag in (("lc_ds", "LcDs"), ("lme_ds", "LmeDs"), ("lc_qw", "LcQw"), ("lme_qw", "LmeQw")):
        for name, nm in (("colbert", "Colbert"), ("reco-K64k+R", "Reco"), ("bm25", "Bm"), ("bge-base", "Bge"),
                         ("full", "Full"), ("oracle", "Oracle")):
            a, tok, n = acc(sets[s], name)
            if n:
                macros[f"Qa{tag}{nm}"] = f"{a:.3f}"
                macros[f"Tok{tag}{nm}"] = f"{tok:,.0f}"
        for name, nm in (("reco-K64k+R", "Reco"), ("bm25", "Bm"), ("bge-base", "Bge"), ("full", "Full")):
            d, ci = paired(sets[s], name)
            if not math.isnan(d):
                macros[f"Diff{tag}{nm}"] = signed(100 * d)
                macros[f"Ci{tag}{nm}"] = f"{100 * ci:.1f}"
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


MEM_ROWS = [("reco-K64k", "ReCo, $K$=64k", True), ("reco-K16k", "ReCo, $K$=16k", True),
            ("bge-base", "bge-base", False), ("bge-small", "bge-small", False), ("bm25", "BM25", False)]


def memory_table(mem: Path, macros: dict) -> str:
    reps = {b: json.loads((mem / f"memory_{b}.json").read_text(encoding="utf-8"))
            for b in ("longmemeval", "locomo") if (mem / f"memory_{b}.json").exists()}
    if not reps or not all("hazard_stop" in r["methods"]["reco-K64k"] for r in reps.values()):
        return ""
    lines = [r"\begin{tabular}{@{}l rr rr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{LongMemEval-S} & \multicolumn{2}{c}{LoCoMo} \\",
             r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
             r"First stage & FixedBudget & Hazard & FixedBudget & Hazard \\", r"\midrule"]
    for key, lab, ours in MEM_ROWS:
        o = r"\ours " if ours else ""
        cells = [o + lab]
        for b in ("longmemeval", "locomo"):
            m = reps[b]["methods"].get(key)
            if not m or "hazard_stop" not in m:
                cells += ["--", "--"]
                continue
            f, h = m["fixed_stop"], m["hazard_stop"]
            cells += [o + f"{f['R_mean']:.1f}", o + rf"\textbf{{{h['R_mean']:.1f}}}"]
        lines.append(" & ".join(cells) + r" \\")
    cut = [1 - m["hazard_stop"]["R_mean"] / m["fixed_stop"]["R_mean"] for r in reps.values()
           for m in r["methods"].values() if "hazard_stop" in m]
    macros["MemCutMin"], macros["MemCutMax"] = f"{100 * min(cut):.0f}", f"{100 * max(cut):.0f}"
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def memory_macros(mem: Path, macros: dict) -> None:
    for bench, tag in (("longmemeval", "Lme"), ("locomo", "Lc")):
        path = mem / f"memory_{bench}.json"
        if not path.exists():
            continue
        rep = json.loads(path.read_text(encoding="utf-8"))
        st = rep["storage_bytes_per_unit"]
        macros[f"Raw{tag}"] = f"{st['raw text']:,.0f}"
        macros[f"Colbert{tag}"] = f"{st['ColBERT fp16']:,.0f}"
        macros[f"Sketch{tag}"] = f"{st['ReCo sketch K=64k']:,.0f}"
        macros[f"ColbertRatio{tag}"] = f"{st['ColBERT fp16'] / st['raw text']:.0f}"
        macros[f"SketchRatio{tag}"] = f"{st['ReCo sketch K=64k'] / st['raw text']:.2f}"
        macros[f"MemSaving{tag}"] = f"{st['ColBERT fp16'] / st['ReCo sketch K=64k']:.0f}"


def r95_macros(data: dict, macros: dict) -> None:
    ratios = []
    for name in ORDER:
        if name not in data:
            continue
        ps = mr.points(data[name], "crc")
        r = ps[pick(data[name])]
        others = [p.r95 for p in mr.family(ps, mr.LOSSY_FAMILIES + mr.FIRST_STAGE_FAMILIES) if p.ratio <= r.ratio * (1 + 1e-9)]
        if others and r.r95:
            ratios.append(min(others) / r.r95)
    macros["RninetyfiveMin"], macros["RninetyfiveMax"] = f"{min(ratios):.1f}", f"{max(ratios):.0f}"


def muvera_macros(data: dict, macros: dict) -> None:
    """MUVERA used as intended (FDE candidates, exact re-scoring): recomputations of its best FDE
    configuration under ReCo's verifier and rule at delta=0.05, relative to ReCo's sketch."""
    ratios, sizes = [], []
    for name in ORDER:
        if name not in data:
            continue
        rep, m = data[name], pick(data[name])
        rule = headline_rules(rep, "0.05")[1]
        fde = [k for k in rep["methods"] if k.startswith("fde")]
        if not fde:
            continue
        best = min(fde, key=lambda k: entry(rep, k, "0.05", rule)["R_mean"])
        ratios.append(entry(rep, best, "0.05", rule)["R_mean"] / entry(rep, m, "0.05", rule)["R_mean"])
        sizes.append(rep["methods"][best]["storage_ratio"] / rep["methods"][m]["storage_ratio"])
    if ratios:
        macros.update({"MuveraRMin": f"{min(ratios):.0f}", "MuveraRMax": f"{max(ratios):.0f}",
                       "MuveraSizeMin": f"{min(sizes):.1f}", "MuveraSizeMax": f"{max(sizes):.0f}"})


def tradeoff_macros(data: dict, macros: dict) -> None:
    sizes, rs = [], []
    for name in ORDER:
        if name not in data:
            continue
        rep = data[name]
        rule = headline_rules(rep, "0.05")[1]
        pts = [(rep["methods"][f"reco-K{k}"]["storage_ratio"], rep["methods"][f"reco-K{k}"]["evaluation"]["deltas"]["0.05"][rule]["R_mean"])
               for k in (1024, 4096, 16384, 65536) if f"reco-K{k}" in rep["methods"]]
        for a, b in zip(pts, pts[1:]):
            sizes.append(b[0] / a[0])
            rs.append(a[1] / b[1])
    macros.update({"StepSizeMin": f"{min(sizes):.1f}", "StepSizeMax": f"{max(sizes):.1f}",
                   "StepSizeMedian": f"{np.median(sizes):.1f}", "StepRMin": f"{min(rs):.1f}",
                   "StepRMax": f"{max(rs):.1f}", "StepRMedian": f"{np.median(rs):.1f}"})


# ------------------------------------------------------------------------------------------ comparison tables
def tex_minus(s: str) -> str:
    """Typographic minus (\\minus, defined in main.tex) for negative numbers: a hyphen is not a minus sign."""
    return re.sub(r"(?m)(^|[\s(${&])-(?=\d)", lambda m: m.group(1) + r"\minus{}", s)


def edge_shading(s: str) -> str:
    """\\ours in the first / last cell of a row -> \\oursL / \\oursR (no colour past the table edge)."""
    out = []
    for line in s.split("\n"):
        body = line.rstrip()
        if " & " in body and body.endswith(r"\\"):
            cells = body[:-2].rstrip().split(" & ")
            if cells[0].lstrip().startswith(r"\ours "):
                cells[0] = cells[0].replace(r"\ours ", r"\oursL ", 1)
            if cells[-1].lstrip().startswith(r"\ours "):
                cells[-1] = cells[-1].replace(r"\ours ", r"\oursR ", 1)
            line = " & ".join(cells) + r" \\"
        out.append(line)
    return "\n".join(out)


def signed(v: float, nd: int = 1) -> str:
    """+x / -x with nd decimals; values that round to zero print without a sign."""
    return f"{0.0:.{nd}f}" if round(abs(v), nd) == 0 else f"{v:+.{nd}f}"


def count(n: float) -> str:
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.1f}k" if n >= 1e4 else f"{n:,.0f}"


def datasets_table(data: dict, res: Path) -> str:
    """Corpus statistics: documents, queries, tokens and raw bytes per document, size of ColBERT's index."""
    rows = []
    for name in [n for n in ORDER if n in data]:
        ds, full = data[name]["dataset"], mr.points(data[name], "crc")["full-fp16"]
        rows.append((TITLE[name], ds["n_docs"], ds["n_queries"], ds["tokens_per_doc"], ds["raw_bytes_per_doc"],
                     full.ratio))
    ms, ev = load_json(res / "msmarco" / "meta.json"), load_json(res / "msmarco" / "eval_K65536.json")
    if ms and ev:
        rows.append(("MS MARCO", ms["n_docs"], ms["n_queries"], ms["tokens"] / ms["n_docs"], ms["raw_bytes"] / ms["n_docs"],
                     ev["storage"]["colbert_fp16"] / ev["storage"]["raw_text"]))
    for bench, title in (("longmemeval", "LongMemEval-S"), ("locomo", "LoCoMo")):
        m = load_json(res / "memory" / f"memory_{bench}.json")
        if m:
            st = m["storage_bytes_per_unit"]
            rows.append((title + r"$^\dagger$", m["n_units"], m["n_tasks"], m["tokens_per_unit"], st["raw text"],
                         st["ColBERT fp16"] / st["raw text"]))
    lines = [r"\begin{tabular}{@{}l rrrrr@{}}", r"\toprule",
             r" & & & Tokens & Bytes & ColBERT \\",
             r"Dataset & \#Docs & \#Queries & /doc & /doc & ($\times$raw) \\", r"\midrule"]
    for i, (t, nd, nq, tok, raw, ratio) in enumerate(rows):
        if i in (7, 8):
            lines.append(r"\midrule")
        lines.append(f"{t} & {count(nd)} & {nq:,} & {tok:.0f} & {raw:,.0f} & {ratio:.0f} \\\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


CMP_ROWS = [("Multi-vector compression", [("PLAID (ColBERTv2)~\\cite{colbertv2,plaid}", ("residual-",)),
                                          ("Token pooling~\\cite{pooling}", ("pool",)),
                                          ("ConstBERT~\\cite{constbert}", ("constbert",)),
                                          ("MUVERA~\\cite{muvera}", ("fde-",))]),
            ("Single-vector and sparse", [("Mean pooling", ("meanpool-",)), ("bge-small~\\cite{bge}", ("dense-",)),
                                          ("BM25~\\cite{bm25}", ("bm25",)), ("SPLADE++~\\cite{splade}", ("splade",))])]


def comparison_table(data: dict, macros: dict) -> str:
    """nDCG@10 of every method at an equal storage budget (ReCo's own index size, per dataset): the best
    configuration of each baseline family that fits, ReCo after verification at both recall targets."""
    names = [n for n in ORDER if n in data]
    reps = {n: data[n] for n in names}
    budget = {n: mr.points(reps[n], "crc")[pick(reps[n])].ratio for n in names}
    reco = {d: {n: entry(reps[n], pick(reps[n]), d, headline_rules(reps[n], d)[1])["ndcg10"] for n in names}
            for d in DELTAS}
    base: dict[str, dict[str, float | None]] = {}
    for _, group in CMP_ROWS:
        for lab, prefixes in group:
            base[lab] = {}
            for n in names:
                fits = [p for k, p in mr.points(reps[n], "crc").items() if k.startswith(prefixes)
                        and p.ratio <= budget[n] * (1 + 1e-9)]
                base[lab][n] = max(p.est_ndcg for p in fits) if fits else None
    best_base = {n: max(v[n] for v in base.values() if v[n] is not None) for n in names}
    best_all = {n: max([best_base[n]] + [reco[d][n] for d in DELTAS]) for n in names}
    cols = len(names) + 1
    lines = [r"\begin{tabular}{@{}l " + "r" * len(names) + r"@{}}", r"\toprule",
             "Method & " + " & ".join(TITLE[n] for n in names) + r" \\", r"\midrule",
             r"\textit{Storage budget ($\times$raw)} & " + " & ".join(rf"\textit{{{budget[n]:.2f}}}" for n in names) + r" \\",
             r"\textit{ColBERT, uncompressed} & " + " & ".join(
                 rf"\textit{{{mr.points(reps[n], 'crc')['full-fp16'].est_ndcg:.3f}}}" for n in names) + r" \\"]

    def cell(v: float | None, n: str) -> str:
        if v is None:
            return "--"
        return rf"\textbf{{{v:.3f}}}" if f"{v:.3f}" == f"{best_all[n]:.3f}" else f"{v:.3f}"

    for title, group in CMP_ROWS:
        lines += [r"\midrule", rf"\multicolumn{{{cols}}}{{@{{}}l}}{{\textit{{{title}}}}} \\"]
        for lab, _ in group:
            lines.append(lab + " & " + " & ".join(cell(base[lab][n], n) for n in names) + r" \\")
    lines += [r"\midrule", rf"\multicolumn{{{cols}}}{{@{{}}l}}{{\textit{{Ours}}}} \\"]
    for d in DELTAS:
        lines.append(rf"\ours ReCo ($\delta{{=}}{d}$) & " + " & ".join(rf"\ours {cell(reco[d][n], n)}" for n in names)
                     + r" \\")
    gain = {n: 100 * (reco["0.05"][n] - best_base[n]) for n in names}
    lines.append(r"\midrule")
    lines.append(r"Gain over best baseline & " + " & ".join(signed(gain[n]) for n in names) + r" \\")
    wins = [n for n in names if gain[n] > 0]
    smallest = {key: min(p.ratio for n in names for k, p in mr.points(reps[n], "crc").items() if k.startswith(key))
                for key in ("residual-", "constbert")}
    macros.update({"CmpPlaidMin": f"{smallest['residual-']:.1f}", "CmpConstMin": f"{smallest['constbert']:.2f}"})
    macros.update({"CmpWins": str(len(wins)), "CmpN": str(len(names)),
                   "CmpGainMin": f"{min(gain[n] for n in wins):.1f}", "CmpGainMax": f"{max(gain[n] for n in wins):.1f}",
                   "CmpBudgetMin": f"{min(budget.values()):.2f}", "CmpBudgetMax": f"{max(budget.values()):.2f}"})
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


OVERALL = [("", [            # multi-vector indexes only (the paper's scope)
                ("PLAID (2-bit)~\\cite{colbertv2,plaid}", "residual-2bit"),
                ("WARP (XTR, 4-bit)$^\\dagger$~\\cite{warp}", "warp:4"),
                ("WARP (XTR, 2-bit)$^\\dagger$~\\cite{warp}", "warp:2"),
                ("TokenPool ($\\times$2, 2-bit)~\\cite{pooling}", "pool2-2bit"),
                ("ConstBERT (32 vectors)$^\\dagger$~\\cite{constbert}", "constbert-fp16"),
                ("MUVERA (FDE only, $R$=20)$^\\S$~\\cite{muvera}", "fde-r20-pq")])]


def size_range(lo: float, hi: float) -> str:
    f = lambda v: f"{v:.2f}" if v < 1 else f"{v:.1f}" if v < 10 else f"{v:.0f}"  # noqa: E731
    return f"{f(lo)}--{f(hi)}"


def overall_table(data: dict, macros: dict, warp: dict | None = None) -> str:
    """Every method at the configuration its authors recommend: index size and nDCG@10 per corpus. Bold marks the
    best late-interaction index of each column (the uncompressed reference and the other models excluded)."""
    names = [n for n in ORDER if n in data]
    pts = {n: mr.points(data[n], "crc") for n in names}
    full = {n: pts[n]["full-fp16"].est_ndcg for n in names}

    def row_of(key: str) -> tuple[list[float], list[float]] | None:
        if key.startswith("warp:"):          # official engine, own model: results/warp/<ds>.nbits<b>.json
            got = [(warp or {}).get((n, key[5:])) for n in names]
            if all(g is None for g in got):
                return None
            return ([g["ndcg10"] if g else math.nan for g in got], [g["storage_ratio"] if g else math.nan for g in got])
        return [pts[n][key].est_ndcg for n in names], [pts[n][key].ratio for n in names]

    rows = {}
    for _, group in OVERALL:
        for lab, key in group:
            r = row_of(key)
            if r is not None:
                rows[lab] = r
    reco = {}
    for d in DELTAS:
        reco[d] = ([entry(data[n], pick(data[n]), d, headline_rules(data[n], d)[1])["ndcg10"] for n in names],
                   [pts[n][pick(data[n])].ratio for n in names])
    # a row may miss corpora (a baseline still running): NaN cells print "--" and are left out of averages and bests
    avg = lambda v: 100 * float(np.nanmean([x / full[n] for x, n in zip(v, names)]))  # noqa: E731
    compared = [rows[lab] for lab, _ in OVERALL[0][1] if lab in rows] + [reco[d] for d in DELTAS]
    best_col = [f"{np.nanmax([r[0][j] for r in compared]):.3f}" for j in range(len(names))]
    best_avg = max(f"{avg(r[0]):.1f}" for r in compared)
    best_size = min(np.nanmax(r[1]) for r in compared)

    def cells(v, s, o: str = "", bold: bool = True) -> str:
        lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
        out = [o + (rf"\textbf{{{size_range(lo, hi)}}}" if bold and hi == best_size else size_range(lo, hi))]
        for j, x in enumerate(v):
            t = "--" if math.isnan(x) else f"{x:.3f}"
            out.append(o + (rf"\textbf{{{t}}}" if bold and t == best_col[j] else t))
        a = f"{avg(v):.1f}"
        out.append(o + (rf"\textbf{{{a}}}" if bold and a == best_avg else a))
        return " & ".join(out)

    cols = len(names) + 3
    lines = [r"\begin{tabular}{@{}l r " + "r" * len(names) + r" r@{}}", r"\toprule",
             r" & Index size & \multicolumn{" + str(len(names)) + r"}{c}{nDCG@10} & Avg. \\",
             r"\cmidrule(lr){3-" + str(len(names) + 2) + r"}",
             r"Method & ($\times$raw text) & " + " & ".join(TITLE[n] for n in names) + r" & (\%) \\", r"\midrule",
             r"\textit{ColBERT, uncompressed}~\cite{colbert} & \textit{" + size_range(
                 min(pts[n]["full-fp16"].ratio for n in names), max(pts[n]["full-fp16"].ratio for n in names))
             + "} & " + " & ".join(rf"\textit{{{full[n]:.3f}}}" for n in names) + r" & \textit{100.0} \\"]
    for gi, (title, group) in enumerate(OVERALL):
        lines += [r"\midrule"] + ([rf"\multicolumn{{{cols}}}{{@{{}}l}}{{\textit{{{title}}}}} \\"] if title else [])
        for lab, _ in group:
            if lab in rows:
                lines.append(lab + " & " + cells(*rows[lab], bold=gi == 0) + r" \\")
        if gi == 0:
            for d in DELTAS:
                lines.append(rf"\ours ReCo, $\delta{{=}}{d}$$^\ddagger$ & " + cells(*reco[d], o=r"\ours ") + r" \\")
    macros.update({"OvAvgReco": f"{avg(reco['0.01'][0]):.1f}", "OvAvgRecoFive": f"{avg(reco['0.05'][0]):.1f}",
                   "OvAvgPlaid": f"{avg(rows[OVERALL[0][1][0][0]][0]):.1f}",
                   "OvAvgPool": f"{avg(rows[OVERALL[0][1][3][0]][0]):.1f}",
                   "OvAvgConst": f"{avg(rows[OVERALL[0][1][4][0]][0]):.1f}",
                   "OvAvgMuvera": f"{avg(rows[OVERALL[0][1][5][0]][0]):.0f}",
                   "OvPlaidSize": size_range(min(rows[OVERALL[0][1][0][0]][1]), max(rows[OVERALL[0][1][0][0]][1])),
                   "OvRecoSize": size_range(min(reco["0.01"][1]), max(reco["0.01"][1])),
                   "OvPlaidX": f"{min(p / r for p, r in zip(rows[OVERALL[0][1][0][0]][1], reco['0.01'][1])):.0f}",
                   "OvPlaidXMax": f"{max(p / r for p, r in zip(rows[OVERALL[0][1][0][0]][1], reco['0.01'][1])):.0f}",
                   "OvRecoBest": str(sum(f"{x:.3f}" == best_col[j] for j, x in enumerate(reco['0.01'][0])))})
    # Fig. 1(a): one point per index, geometric mean of the size and mean relative nDCG@10 over the corpora
    geo = lambda s: float(np.exp(np.nanmean(np.log(s))))  # noqa: E731
    size_full = geo([pts[n]["full-fp16"].ratio for n in names])
    macros.update({"IntroSizeColbert": f"{size_full:.2f}", "IntroSizeReco": f"{geo(reco['0.01'][1]):.3f}",
                   "IntroAccReco": f"{avg(reco['0.01'][0]):.1f}",
                   "IntroGain": f"{size_full / geo(reco['0.01'][1]):.0f}",
                   "IntroColbertTimes": f"{size_full:.0f}", "IntroRecoTimes": f"{geo(reco['0.01'][1]):.2f}"})
    for tag, i in (("Plaid", 0), ("WarpFour", 1), ("Pool", 3), ("Const", 4)):
        lab = OVERALL[0][1][i][0]
        if lab in rows:
            macros.update({f"IntroSize{tag}": f"{geo(rows[lab][1]):.2f}", f"IntroAcc{tag}": f"{avg(rows[lab][0]):.1f}"})
    for nb, word in (("4", "Four"), ("2", "Two")):          # LaTeX macro names cannot contain digits
        lab = next(l for l, k in OVERALL[0][1] if k == f"warp:{nb}")
        if lab in rows:
            v, sz = rows[lab]
            macros.update({f"OvAvgWarp{word}": f"{avg(v):.1f}", f"OvWarpSize{word}": size_range(min(sz), max(sz)),
                           f"OvWarpX{word}": f"{np.nanmin([a / r for a, r in zip(sz, reco['0.01'][1])]):.0f}",
                           f"OvWarpXMax{word}": f"{np.nanmax([a / r for a, r in zip(sz, reco['0.01'][1])]):.0f}"})
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


STOP_Q = [("fixed", "FixedBudget"), ("crc", r"TA-Margin~\cite{fagin}"), ("patience", r"Patience~\cite{pip}"),
          ("probtopk", r"ProbTopK~\cite{theobald}"), ("laet", r"LAET~\cite{learnedterm}"),
          ("darth", r"DARTH~\cite{darth}"), ("hazard", "ReCo")]
STOP_D = [("fixed_doc", "FixedBudget"), ("margin_doc", r"TA-Margin~\cite{fagin}"), ("patience_doc", r"Patience~\cite{pip}"),
          ("rank1", "ConfMargin"), ("probtopk_doc", "ProbTopK"), ("laet_doc", "LAET"), ("darth_doc", "DARTH"),
          ("hazard_doc", "ReCo-D")]
BENCH_RULES = ("fixed", "crc", "patience", "hazard", "hazard_u", "rank1", "hazard_doc")
STOP_SHORT = {"scifact": "SciF.", "nfcorpus": "NFC.", "arguana": "Arg.", "scidocs": "SCID.", "fiqa": "FiQA",
              "trec-covid": "TREC", "quora": "Quora"}


def fmt_big(v: float) -> str:
    return f"{v:,.0f}" if v >= 1000 else fmt_r(v)


def stopping_table(data: dict, sb: dict, macros: dict) -> str:
    """Exp-2: mean recomputations of every stopping rule on ReCo's sketch, both recall targets side by side (a
    table*), grouped by finite-sample correction, with the saving of hazard stopping over the best other rule of its
    group and DARTH as published (uncalibrated) with its achieved recall."""
    names = [n for n in ORDER if n in data and n in sb]

    def ev(n: str, d: str) -> dict:
        return data[n]["methods"][pick(data[n])]["evaluation"]["deltas"][d]

    def val(n: str, d: str, r: str) -> float | None:
        e = ev(n, d) if r in BENCH_RULES else sb[n]["deltas"][d]
        if r not in e or e[r]["R_mean"] >= data[n]["dataset"]["n_docs"]:
            return None
        return e[r]["R_mean"]

    cols = 1 + 2 * len(names)
    lines = [r"\begin{tabular}{@{}l " + " ".join("r" * len(names) for _ in DELTAS) + r"@{}}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{{len(names)}}}{{c}}{{Recall target {1 - float(d):.2f} "
                                rf"($\delta{{=}}{d}$)}}" for d in DELTAS) + r" \\",
             "".join(rf"\cmidrule({'lr' if i == 0 else 'l'}){{{2 + i * len(names)}-{1 + (i + 1) * len(names)}}}"
                     for i in range(len(DELTAS))),
             "Rule & " + " & ".join(STOP_SHORT[n] for _ in DELTAS for n in names) + r" \\", r"\midrule"]
    for title, group, ours in ((r"Query-level CRC", STOP_Q, "hazard"),
                               (r"Pooled per-document correction$^\ddagger$", STOP_D, "hazard_doc")):
        lines.append(rf"\multicolumn{{{cols}}}{{@{{}}l}}{{\textit{{{title}}}}} \\")
        best = {(n, d): min((v for r, _ in group if (v := val(n, d, r)) is not None), default=None)
                for d in DELTAS for n in names}
        for r, lab in group:
            o = r == ours
            cells = [(r"\oursL " if o else "") + lab]
            for d in DELTAS:
                for n in names:
                    v = val(n, d, r)
                    tie = v is not None and fmt_big(v) == fmt_big(best[n, d])     # equal as printed: bold both
                    s = "--" if v is None else (rf"\textbf{{{fmt_big(v)}}}" if tie else fmt_big(v))
                    cells.append((r"\ours " if o else "") + s)
            if o:
                cells[-1] = cells[-1].replace(r"\ours ", r"\oursR ", 1)
            lines.append(" & ".join(cells) + r" \\")
        cells = [r"\quad\textit{Saving over best other}"]
        for d in DELTAS:
            for n in names:
                mine = val(n, d, ours)
                prev = [v for r, _ in group if r != ours and (v := val(n, d, r)) is not None]
                cells.append(rf"\textit{{{signed_pct(1 - mine / min(prev))}}}" if mine is not None and prev else "--")
        lines.append(" & ".join(cells) + r" \\")
        if ours == "hazard":         # transferred hazard model, for reference (not compared)
            cells = [r"\quad\textit{ReCo-U}"]
            for d in DELTAS:
                for n in names:
                    v = val(n, d, "hazard_u")
                    cells.append("--" if v is None else rf"\textit{{{fmt_big(v)}}}")
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines.append(rf"\multicolumn{{{cols}}}{{@{{}}l}}{{\textit{{Uncalibrated}}}} \\")
    r_cells, rec_cells = [r"DARTH as published~\cite{darth}"], [r"\quad\textit{achieved recall}"]
    misses = []
    for d in DELTAS:
        for n in names:
            e = sb[n]["deltas"][d]["darth_declared"]
            miss = e["recall"] < 1 - float(d) - 1e-9
            r_cells.append(fmt_big(e["R_mean"]))
            shown = math.floor(e["recall"] * 1000) / 1000          # round down: never shows a miss as on target
            rec_cells.append(rf"\textit{{{shown:.3f}}}" + (r"$^\times$" if miss else ""))
            if miss:
                misses.append((n, d, e["recall"]))
    lines += [" & ".join(r_cells) + r" \\", " & ".join(rec_cells) + r" \\"]

    # summary numbers at each corpus's headline correction
    red, tail, best_other, ratio = {d: [] for d in DELTAS}, [], {d: [] for d in DELTAS}, {}
    for d in DELTAS:
        for n in names:
            old, new = headline_rules(data[n], d)
            group, doc = (STOP_Q, False) if new == "hazard" else (STOP_D, True)
            mine = val(n, d, new)
            if mine is None:
                continue
            ref = "crc" if new == "hazard" else "margin_doc"          # TA-Margin under the same correction
            o = val(n, d, ref)
            if o is not None:
                red[d].append(1 - mine / o)
                src = ev(n, d) if ref in BENCH_RULES else sb[n]["deltas"][d]
                tail.append(p99(src[ref]) / max(p99(ev(n, d)[new]), 1))
            prev = [v for r, _ in group if r != new and (v := val(n, d, r)) is not None]
            best_other[d].append(1 - mine / min(prev))
            for base in ("laet", "probtopk", "darth"):
                v = val(n, d, base + ("_doc" if doc else ""))
                if v is not None:
                    ratio.setdefault(base, []).append(v / mine)
            dec = sb[n]["deltas"][d]["darth_declared"]["R_mean"]
            ratio.setdefault("declared", []).append(dec / mine)
    for d, tag in (("0.05", "Five"), ("0.01", "One")):
        v = best_other[d]
        macros.update({f"Best{tag}Min": f"{100 * min(v):.0f}", f"Best{tag}Max": f"{100 * max(v):.0f}",
                       f"Best{tag}Mean": f"{100 * np.mean(v):.0f}", f"Best{tag}Pos": str(sum(x > 0 for x in v)),
                       f"Best{tag}N": str(len(v)), f"Red{tag}Min": f"{100 * min(red[d]):.0f}",
                       f"Red{tag}Max": f"{100 * max(red[d]):.0f}", f"Red{tag}Mean": f"{100 * np.mean(red[d]):.0f}"})
    geo = lambda xs: float(np.exp(np.mean(np.log(xs))))  # noqa: E731
    macros.update({"TailRatioMedian": f"{np.median(tail):.1f}", "TailRatioMax": f"{max(tail):.1f}",
                   "StopLaetX": f"{geo(ratio['laet']):.2f}", "StopProbX": f"{geo(ratio['probtopk']):.2f}",
                   "StopDarthX": f"{geo(ratio['darth']):.1f}", "StopDeclaredX": f"{geo(ratio['declared']):.1f}",
                   "DarthMisses": str(len(misses)), "DarthMissesOne": str(sum(1 for _, d, _ in misses if d == "0.01")),
                   "DarthMissMin": f"{math.floor(min(r for *_, r in misses) * 1000) / 1000:.3f}" if misses else "--",
                   "DarthMissMax": f"{math.floor(max(r for *_, r in misses) * 1000) / 1000:.3f}" if misses else "--",
                   "StopCells": str(sum(len(v) for v in best_other.values())),
                   "StopWins": str(sum(sum(x > 0 for x in v) for v in best_other.values())),
                   "StopExceptPct": f"{-100 * min(min(min(v) for v in best_other.values()), 0):.0f}"})
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


TC_RULES = [("fixed", "FixedBudget"), ("margin", r"TA-Margin~\cite{fagin}"), ("patience", r"Patience~\cite{pip}"),
            ("probtopk", r"ProbTopK~\cite{theobald}"), ("laet", r"LAET~\cite{learnedterm}"),
            ("darth", r"DARTH~\cite{darth}"), ("hazard", "ReCo")]


def train_calibration_table(data: dict, tc: dict, macros: dict) -> str:
    """Exp-2 supplement: every rule fitted and calibrated on the train split's queries (query-level CRC, which the
    test sets alone cannot support at delta=0.01), tested on all test queries; mean over random fit / calibration
    splits. The pooled Hazard-D of the cross-calibrated bench is listed for reference."""
    names = [n for n in ORDER if n in tc]
    if not names:
        return ""

    def val(n: str, d: str, r: str) -> float | None:
        e = tc[n]["deltas"][d][r]
        return None if e["R_mean"] >= tc[n]["n_docs"] else e["R_mean"]

    cols = 1 + len(names) * len(DELTAS)
    lines = [r"\begin{tabular}{@{}l " + "r" * (cols - 1) + r"@{}}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{{len(names)}}}{{c}}{{$\delta{{=}}{d}$}}" for d in DELTAS) + r" \\",
             "".join(rf"\cmidrule({'lr' if i == 0 else 'l'}){{{2 + i * len(names)}-{1 + (i + 1) * len(names)}}}"
                     for i in range(len(DELTAS))),
             "Rule & " + " & ".join(STOP_SHORT[n] for _ in DELTAS for n in names) + r" \\", r"\midrule"]
    best = {(n, d): min(v for r, _ in TC_RULES if (v := val(n, d, r)) is not None) for d in DELTAS for n in names}
    for r, lab in TC_RULES:
        o = r == "hazard"
        cells = [(r"\oursL " if o else "") + lab]
        for d in DELTAS:
            for n in names:
                v = val(n, d, r)
                s = "--" if v is None else (rf"\textbf{{{fmt_big(v)}}}" if v == best[n, d] else fmt_big(v))
                cells.append((r"\ours " if o else "") + s)
        if o:
            cells[-1] = cells[-1].replace(r"\ours ", r"\oursR ", 1)
        lines.append(" & ".join(cells) + r" \\")
    save, recall, wins, pooled = [], [], 0, []
    s_cells, r_cells, p_cells = ([r"\quad\textit{Saving over best other}"], [r"\quad\textit{achieved recall}"],
                                 [r"\quad\textit{ReCo-D, test folds}$^\ddagger$"])
    for d in DELTAS:
        for n in names:
            mine = val(n, d, "hazard")
            prev = min(v for r, _ in TC_RULES if r != "hazard" and (v := val(n, d, r)) is not None)
            save.append(1 - mine / prev)
            wins += mine <= prev
            s_cells.append(rf"\textit{{{signed_pct(1 - mine / prev)}}}")
            rec = tc[n]["deltas"][d]["hazard"]["recall"]
            recall.append(rec)
            shown = math.floor(rec * 1000) / 1000
            r_cells.append(rf"\textit{{{shown:.3f}}}" + (r"$^\times$" if rec < 1 - float(d) - 1e-9 else ""))
            ref = data[n]["methods"][pick(data[n])]["evaluation"]["deltas"][d]["hazard_doc"]["R_mean"]
            pooled.append(mine / ref)
            p_cells.append(rf"\textit{{{fmt_big(ref)}}}")
    lines += [" & ".join(c) + r" \\" for c in (s_cells, r_cells)]
    lines += [r"\midrule", " & ".join(p_cells) + r" \\"]
    # shortfalls below the target, in standard errors of the test-set mean, over every calibrated rule
    se = [tc[n]["deltas"][d][r]["recall_se"] for d in DELTAS for n in names for r, _ in TC_RULES]
    z = [(1 - float(d) - tc[n]["deltas"][d][r]["recall"]) / max(tc[n]["deltas"][d][r]["recall_se"], 1e-9)
         for d in DELTAS for n in names for r, _ in TC_RULES]
    n_cal = [tc[n]["n_cal"] for n in names]
    macros.update({"TcSeMax": f"{max(se):.3f}", "TcZMax": f"{max(z):.1f}", "TcBelow": str(sum(x > 0 for x in z)),
                   "TcRuleCells": str(len(z)),
                   "TcCalMin": str(min(n_cal)), "TcCalMax": str(max(n_cal)), "TcRepeats": str(tc[names[0]]["repeats"]),
                   "TcWins": str(wins), "TcCells": str(len(save)), "TcSaveMin": f"{100 * min(save):.0f}",
                   "TcSaveMax": f"{100 * max(save):.0f}", "TcSaveMean": f"{100 * np.mean(save):.0f}",
                   "TcRecallMin": f"{math.floor(min(recall) * 1000) / 1000:.3f}",
                   "TcPooledMin": f"{min(pooled):.2f}", "TcPooledMax": f"{max(pooled):.2f}"})
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def throughput_table(d: Path, macros: dict) -> str:
    """Exp-5: queries per second on one GPU (MS MARCO, 1,000 queries), latency, recall of the exact top-10 and peak
    GPU memory: the official PLAID engine (per-query search and its batch API) and ReCo with C concurrent queries."""
    plaid, reco = load_json(d / "plaid_nbits2.json"), load_json(d / "throughput_reco.json")
    sweep, skip, tail = (load_json(d / f) for f in ("plaid_sweep_nbits2.json", "plaid_sweep_skip_nbits2.json",
                                                    "throughput_reco_d01.json"))
    lat, ev32 = load_json(d / "latency_K65536_b32.json"), load_json(d / "eval_K65536_b32.json")
    if not (plaid and reco and "throughput" in plaid and sweep and skip and tail and lat and ev32):
        return ""
    runs = reco["runs"] + tail["runs"]

    def row(lab: str, p: dict) -> tuple:
        return (lab, p["qps"], p["latency_ms"]["p50"], p["latency_ms"]["p99"], p["recall_vs_exact10"], p["peak_gpu_gb"],
                False)
    rows = [row(r"PLAID, $k{\le}10$ defaults", sweep["points"][0]),
            ("PLAID, batch API", plaid["throughput"]["qps"], None, None, plaid["throughput"]["recall_vs_exact10"],
             plaid["throughput"]["peak_gpu_gb"], False),
            row(r"PLAID, $k{\le}100$ defaults", sweep["points"][1]),
            row(r"PLAID, $k{>}100$ defaults$^\dagger$", skip["points"][0])]
    for dl in ("0.1", "0.05", "0.01"):
        one = lat["runs"][f"hazard@{dl}"]
        rows.append((rf"ReCo, $\delta{{=}}{dl}$, $C{{=}}1$", 1000 / one["ms_mean"], one["ms_p50"], one["ms_p99"],
                     ev32["deltas"][dl]["hazard"]["recall"], None, True))
        r = next(x for x in runs if x["delta"] == dl and x["concurrency"] == 64)
        rows.append((rf"ReCo, $\delta{{=}}{dl}$, $C{{=}}64$", r["qps"], r["latency_ms_p50"], r["latency_ms_p99"],
                     r["recall_vs_exact10"], r["peak_gpu_gb"], True))
    best_qps = max(r[1] for r in rows)
    lines = [r"\begin{tabular}{@{}l rrrrr@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{Latency (ms)} & Recall & GPU mem. \\", r"\cmidrule(lr){3-4}",
             r"System & QPS & p50 & p99 & @10 & (GB) \\", r"\midrule"]
    for lab, qps, p50, p99_, rec, mem, ours in rows:
        o = r"\ours " if ours else ""
        cells = [(r"\oursL " if ours else "") + lab, o + (rf"\textbf{{{qps:.1f}}}" if qps == best_qps else f"{qps:.1f}"),
                 o + ("--" if p50 is None else f"{p50:,.0f}"), o + ("--" if p99_ is None else f"{p99_:,.0f}"),
                 o + f"{rec:.3f}", (r"\oursR " if ours else "") + ("--" if mem is None else f"{mem:.1f}")]
        lines.append(" & ".join(cells) + r" \\")
    plaid_best = max(rows[0][1], rows[1][1])
    r10 = max((r for r in reco["runs"] if r["delta"] == "0.1"), key=lambda r: r["qps"])
    r05 = max((r for r in reco["runs"] if r["delta"] == "0.05"), key=lambda r: r["qps"])
    macros.update({"TpPlaidLoop": f"{rows[0][1]:.1f}", "TpPlaidBatch": f"{rows[1][1]:.1f}",
                   "TpPlaidMem": f"{plaid['throughput']['peak_gpu_gb']:.1f}",
                   "TpRecoTen": f"{r10['qps']:.1f}", "TpRecoFive": f"{r05['qps']:.1f}",
                   "TpRecoTenC": str(r10["concurrency"]), "TpRecoMem": f"{r10['peak_gpu_gb']:.1f}",
                   "TpRecoTenX": f"{r10['qps'] / plaid_best:.1f}", "TpRecoFiveX": f"{r05['qps'] / plaid_best:.1f}",
                   "TpRecoDocs": f"{max(r['docs_per_s'] for r in runs):,.0f}",
                   "TpRecoOneQps": f"{next(r['qps'] for r in reco['runs'] if r['delta'] == '0.1' and r['concurrency'] == 1):.1f}",
                   "TpPlaidMid": f"{rows[2][1]:.1f}", "TpPlaidDeep": f"{rows[3][1]:.1f}",
                   "TpRecoSingleTen": f"{rows[4][1]:.1f}", "TpRecoSingleFive": f"{rows[6][1]:.1f}",
                   "TpRecoOneC": f"{rows[9][1]:.1f}"})
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def plaid_sweep_macros(d: Path, macros: dict) -> dict:
    """PLAID's operating points on the GPU and its recall limit on the CPU, which holds every setting. Replaces the
    macros of the single default run, so that every PLAID number of the paper comes from one sweep."""
    sweep, skip = load_json(d / "plaid_sweep_nbits2.json"), load_json(d / "plaid_sweep_skip_nbits2.json")
    check, mid, deep = (load_json(d / f"plaid_sweep_{n}_nbits2.json") for n in ("cpu_check", "cpu_mid", "cpu"))
    if not (sweep and skip and check and mid and deep):
        return {}
    first, second, third = sweep["points"][0], sweep["points"][1], skip["points"][0]
    low = load_json(d / "plaid_sweep_cpu_low_nbits2.json")             # lower centroid thresholds, run after the review
    cpu = check["points"] + mid["points"] + deep["points"] + (low["points"] if low else [])
    best = max(p["recall_vs_exact10"] for p in cpu)
    # the deepest setting that reaches the limit, to show that more search does not raise it
    top = max((p for p in cpu if p["recall_vs_exact10"] >= best - 1e-9), key=lambda p: (p["ncells"], p["ndocs"]))
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    macros.update({"MsPlaidMrr": f"{first['mrr10']:.3f}", "MsPlaidRecall": f"{first['recall_vs_exact10']:.2f}",
                   "MsPlaidGpu": f"{first['latency_ms']['p50']:.0f}",
                   "MsPlaidGpuNinety": f"{first['latency_ms']['p90']:.0f}",
                   "MsPlaidGpuNinetyNine": f"{first['latency_ms']['p99']:.0f}",
                   "MsPlaidRecallFine": f"{first['recall_vs_exact10']:.3f}",
                   "MsPlaidRecallMid": f"{second['recall_vs_exact10']:.3f}",
                   "MsPlaidRecallDeep": f"{third['recall_vs_exact10']:.3f}", "MsPlaidMrrDeep": f"{third['mrr10']:.3f}",
                   "MsPlaidSkippedWord": words[third["skipped"]],
                   "MsPlaidCpuRecallFirst": f"{check['points'][0]['recall_vs_exact10']:.3f}",
                   "MsPlaidCeiling": f"{top['recall_vs_exact10']:.2f}", "MsPlaidCeilingCells": str(top["ncells"]),
                   "MsPlaidCeilingFine": f"{top['recall_vs_exact10']:.3f}",
                   "MsPlaidCeilingDocs": f"{top['ndocs']:,}"})
    return {"gpu": sweep["points"][:2] + skip["points"], "ceiling": top["recall_vs_exact10"]}


def optimality_macros(data: dict, sb: dict, d: Path, macros: dict) -> None:
    """Section VI-D: how close hazard stopping comes to Theorem 1 (scripts/stopping/optimality_gap.py). The bound is the
    clairvoyant one at the target; both rules are taken at the corpus's headline correction, as in Table V."""
    reps = {n: r for n in ORDER if n in data and (r := load_json(d / f"{n}.json"))}
    if not reps:
        return

    def val(n: str, dl: str, r: str) -> float | None:
        e = data[n]["methods"][pick(data[n])]["evaluation"]["deltas"][dl] if r in BENCH_RULES else sb[n]["deltas"][dl]
        return e[r]["R_mean"] if r in e and e[r]["R_mean"] < data[n]["dataset"]["n_docs"] else None
    falls = [100 * r["monotonicity"]["fall_share"] for r in reps.values()]
    rho = [r["reliability"]["rank_correlation"] for r in reps.values()]
    macros.update({"OptFallMin": f"{min(falls):.0f}", "OptFallMax": f"{max(falls):.0f}",
                   "OptKeepMin": f"{100 - max(falls):.0f}", "OptKeepMax": f"{100 - min(falls):.0f}",
                   "OptNeverMin": f"{100 * (1 - max(r['monotonicity']['queries_with_fall'] for r in reps.values())):.0f}",
                   "OptNeverMax": f"{100 * (1 - min(r['monotonicity']['queries_with_fall'] for r in reps.values())):.0f}",
                   "OptAfterChange": f"{100 * min(r['monotonicity']['fall_after_topk_change_share'] for r in reps.values()):.0f}",
                   "OptRhoStrong": f"{min(rho):.2f}", "OptRhoWeak": f"{max(rho):.2f}"})
    closer = 0
    for dl, tag in (("0.05", "Five"), ("0.01", "One")):
        mine, other = [], []
        for n, r in reps.items():
            new = headline_rules(data[n], dl)[1]
            group = STOP_Q if new == "hazard" else STOP_D
            bound, m = r["deltas"][dl]["hazard"]["bound_at_target"], val(n, dl, new)
            rest = [v for rule, _ in group if rule != new and (v := val(n, dl, rule)) is not None]
            if m is not None and rest:
                mine.append(m / bound)
                other.append(min(rest) / bound)
                closer += m < min(rest)
        macros.update({f"OptGap{tag}Min": f"{min(mine):.1f}", f"OptGap{tag}Max": f"{max(mine):.1f}",
                       f"OptOther{tag}Min": f"{min(other):.1f}", f"OptOther{tag}Max": f"{max(other):.1f}",
                       f"OptSave{tag}Min": f"{100 * (1 - 1 / min(mine)):.0f}",
                       f"OptSave{tag}Max": f"{100 * (1 - 1 / max(mine)):.0f}"})
    macros["OptCloser"] = str(closer)


def growth_macros(d: Path, macros: dict) -> None:
    """Exp-11: recall of rules calibrated on part of a corpus and tested on the grown corpus (scripts/stopping/growth.py)."""
    reps = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))] if d.is_dir() else []
    if not reps:
        return

    def recalls(base: float, dl: str, how: str = "frozen", rule: str = "hazard") -> list[float]:
        out = []
        for r in reps:
            for b in r["bases"]:
                if b["fraction"] == base:
                    out.append(b["points"][-1][how][dl][rule]["recall"])
        return out
    macros["GrowQuarterPct"] = f"{100 * (1 / 0.8 - 1):.0f}"
    for base, name in ((0.8, "Quarter"), (0.5, "Double")):
        for dl, tag in (("0.05", "Five"), ("0.01", "One")):
            v = recalls(base, dl)
            macros.update({f"Grow{name}{tag}Min": f"{min(v):.3f}", f"Grow{name}{tag}Max": f"{max(v):.3f}"})
    for dl, tag in (("0.05", "Five"), ("0.01", "One")):
        v = [p["recalibrated"][dl]["hazard"]["recall"] for r in reps for b in r["bases"] for p in b["points"]]
        macros[f"GrowRecal{tag}"] = f"{min(v):.3f}"
        cost, save = [], []
        for r in reps:                   # the corpus doubled, recalibrated: against the base corpus and the other rules
            b = next(x for x in r["bases"] if x["fraction"] == 0.5)
            base, last = b["points"][0]["recalibrated"][dl], b["points"][-1]["recalibrated"][dl]
            cost.append(100 * (last["hazard"]["R_mean"] / base["hazard"]["R_mean"] - 1))
            save.append(100 * (1 - last["hazard"]["R_mean"] / min(last["margin"]["R_mean"], last["fixed"]["R_mean"])))
        macros.update({f"GrowCost{tag}Min": f"{min(cost):.0f}", f"GrowCost{tag}Max": f"{max(cost):.0f}",
                       f"GrowSave{tag}Min": f"{min(save):.0f}", f"GrowSave{tag}Max": f"{max(save):.0f}"})
    short = [100 * (0.95 - x) for x in recalls(0.8, "0.05")]
    macros["GrowQuarterLoss"] = f"{math.ceil(10 * max(short)) / 10:.1f}"
    short = [100 * (0.95 - x) for x in recalls(0.5, "0.05")]
    macros.update({"GrowDoubleLossMin": f"{min(short):.1f}", "GrowDoubleLossMax": f"{max(short):.1f}"})


def memory_size_macros(d: Path, macros: dict) -> None:
    """Exp-11: agent memories by size, calibrated across users (scripts/stopping/memory_by_size.py)."""
    for bench, tag in (("longmemeval", "Lme"), ("locomo", "Lc")):
        r = load_json(d / f"by_size_{bench}.json")
        if r:
            q = [x["recall"] for x in r["by_size"]]
            macros.update({f"MemSize{tag}Min": f"{min(q):.3f}", f"MemSize{tag}Max": f"{max(q):.3f}",
                           f"MemSize{tag}Large": f"{r['small_to_large']['recall']:.3f}"})


def inverted_macros(d: Path, macros: dict) -> None:
    """Exact estimate prefixes from the inverted index on CPU (MS MARCO, 8 threads) against the full scan."""
    r = load_json(d / "inverted_cpu.json")
    if not r:
        return
    full = r["full_scan_ms"]["mean"]
    macros.update({"InvFullMs": f"{full:.0f}", "InvBuildS": f"{r['build_s']:.0f}", "InvMemGB": f"{r['memory_gb']:.1f}",
                   "InvIdentical": str(r["verified"]["identical"]), "InvChecked": str(r["verified"]["checked"]),
                   "InvThreads": str(r["threads"])})
    for dl, tag in (("0.1", "Ten"), ("0.05", "Five"), ("0.01", "One")):
        e = r["deltas"][dl]
        macros.update({f"Inv{tag}Ms": f"{e['ms']['mean']:.0f}", f"Inv{tag}Median": f"{e['ms']['p50']:.0f}",
                       f"Inv{tag}X": f"{full / e['ms']['mean']:.1f}",
                       f"Inv{tag}Cand": f"{100 * e['candidates_mean'] / r['n_docs']:.0f}"})


def significance_macros(data: dict, macros: dict) -> None:
    """Paired t-tests on per-query nDCG@10 (same test queries): ReCo at delta = 0.01 against uncompressed ColBERT and
    against PLAID 2-bit (Exp-1)."""
    from scipy import stats
    same, exceptions, above, sig_above = 0, [], 0, []
    names = [n for n in ORDER if n in data]
    for n in names:
        rep, m = data[n], pick(data[n])
        full = np.asarray(rep["methods"]["full-fp16"]["evaluation"]["per_query"]["estimate_ndcg10"])
        reco = np.asarray(rep["methods"][m]["evaluation"]["per_query"]["ndcg10"]["0.01"][headline_rules(rep, "0.01")[1]])
        p = stats.ttest_rel(reco, full).pvalue if np.any(reco != full) else 1.0
        if p > 0.05:
            same += 1
        else:
            exceptions.append(f"{TITLE[n]}: ${100 * (reco.mean() - full.mean()):+.2f}$ points, $p={p:.3f}$")
        plaid = np.asarray(rep["methods"]["residual-2bit"]["evaluation"]["per_query"]["estimate_ndcg10"])
        above += reco.mean() > plaid.mean()
        if stats.ttest_rel(reco, plaid).pvalue < 0.05 and reco.mean() > plaid.mean():
            sig_above.append(TITLE[n])
    macros.update({"SigSame": str(same), "SigN": str(len(names)), "SigExceptions": "; ".join(exceptions) or "none",
                   "SigAbovePlaid": str(above), "SigAbovePlaidSig": ", ".join(sig_above) or "none"})


def param_macros(params: dict, macros: dict) -> None:
    """Exp-9 (result size k) and Exp-10 (calibration-set size) numbers quoted in the text."""
    cut = []
    for p in params.values():
        for k, e in p["k"].items():
            best = min(e["0.05"][r]["R_mean"] for r in ("fixed", "crc", "patience"))
            cut.append(1 - e["0.05"]["hazard"]["R_mean"] / best)
    macros.update({"TopkCutMin": f"{100 * min(cut):.0f}", "TopkCutMax": f"{100 * max(cut):.0f}",
                   "TopkPos": str(sum(c > 0 for c in cut)), "TopkN": str(len(cut))})
    q = params.get("quora") or next(iter(params.values()))
    ns = sorted(q["ncal"], key=int)
    small, full = q["ncal"][ns[0]], q["ncal"][ns[-1]]
    macros.update({"NcalSmall": ns[0], "NcalFull": ns[-1],
                   "NcalHazSmall": f"{small['hazard']['R_mean']:,.0f}", "NcalHazFull": f"{full['hazard']['R_mean']:,.0f}",
                   "NcalDocSmall": f"{small['hazard_doc']['R_mean']:,.0f}",
                   "NcalDocFull": f"{full['hazard_doc']['R_mean']:,.0f}",
                   "NcalBlowup": f"{small['hazard']['R_mean'] / full['hazard']['R_mean']:.0f}"})


def signed_pct(x: float) -> str:
    """A saving as a percentage; a negative saving (we recompute more) gets a typographic minus."""
    v = round(100 * x)
    return f"{v}\\%" if v >= 0 else f"\\minus{{}}{-v}\\%"


def msmarco_comparison(d: Path) -> str:
    """MS MARCO: exact ColBERT, the official PLAID engine, a BM25 cascade and ReCo at three recall targets."""
    ev, plaid, bm = load_json(d / "eval_K65536.json"), load_json(d / "plaid_nbits2.json"), load_json(d / "eval_bm25.json")
    lat = load_json(d / "latency_K65536_b32.json")
    meta = load_json(d / "meta.json")
    sweep, skip = load_json(d / "plaid_sweep_nbits2.json"), load_json(d / "plaid_sweep_skip_nbits2.json")
    if not (ev and plaid and lat and meta and sweep and skip):
        return ""
    n, st = meta["n_docs"], ev["storage"]
    gb = lambda bpd: bpd * n / 1e9  # noqa: E731
    rule = "hazard"
    rows = [("ColBERT (exact)", gb(st["colbert_fp16"]), ev["exact"]["mrr10"], 1.0, None, None, None, True)]
    for lab, p in ((r"PLAID, $k{\le}10$ defaults", sweep["points"][0]),
                   (r"PLAID, $k{>}100$ defaults$^\dagger$", skip["points"][0])):
        rows.append((lab, plaid["index_bytes"] / 1e9, p["mrr10"], p["recall_vs_exact10"], None, p["latency_ms"]["p50"],
                     p["latency_ms"]["p99"], False))
    if bm:
        b = bm["deltas"]["0.05"][rule]
        rows.append((r"BM25-Rerank", bm["index_bytes_per_doc"] * n / 1e9, b["mrr10"], b["recall"],
                     b["R_mean"], None, None, False))
    for dl in ("0.1", "0.05", "0.01"):
        e, r = ev["deltas"][dl][rule], lat["runs"].get(f"{rule}@{dl}")
        rows.append((rf"\ours ReCo, $\delta{{=}}{dl}$", gb(st["sketch"]) + st["codebook_total"] / 1e9, e["mrr10"],
                     e["recall"], e["R_mean"], r["ms_p50"] if r else None, r["ms_p99"] if r else None, False))
    cand = [r for r in rows if not r[7]]
    fmts = {1: lambda v: f"{v:.2f}" if v < 10 else f"{v:.1f}", 2: lambda v: f"{v:.3f}", 3: lambda v: f"{v:.3f}",
            4: lambda v: f"{v:,.0f}", 5: lambda v: f"{v:.0f}", 6: lambda v: f"{v:,.0f}"}
    # best of each column among non-reference rows, compared as printed (ties are all bold)
    best = {i: fmts[i]((min if i in (1, 4, 5, 6) else max)(r[i] for r in cand if r[i] is not None)) for i in range(1, 7)}
    lines = [r"\begin{tabular}{@{}l rrrrrr@{}}", r"\toprule",
             r" & Index & & Recall & & \multicolumn{2}{c}{Latency (ms)} \\", r"\cmidrule(l){6-7}",
             r"Method & (GB) & MRR@10 & @10 & $\bar R$ & p50 & p99 \\", r"\midrule"]
    for row in rows:
        lab, ours, ref = row[0], row[0].startswith(r"\ours"), row[7]
        o = r"\ours " if ours else ""
        cells = [lab]
        for i in range(1, 7):
            v = row[i]
            if v is None:
                cells.append(o + ("all" if ref and i == 4 else "--"))
                continue
            s = fmts[i](v)
            cells.append(o + (rf"\textit{{{s}}}" if ref else rf"\textbf{{{s}}}" if s == best[i] else s))
        lines.append(" & ".join(cells) + r" \\")
        if ref:
            lines.append(r"\midrule")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("paper", type=Path)
    args = ap.parse_args()
    tables, figures = args.paper / "tables", args.paper / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for stale in ("rules.tex", "tail.tex", "cascade.tex"):
        if (tables / stale).exists():
            (tables / stale).unlink()
    pf.setup()
    res = args.results
    data = {k: v for k, v in mr.load(res / "beir").items() if v["protocol"]["version"] >= 3}
    macros: dict[str, str] = {"NumDatasets": str(len(data))}
    main_table(data, macros)                       # macros only (abstract, introduction)
    verification_table(data, macros)               # macros only (tail ratios)
    (tables / "datasets.tex").write_text(datasets_table(data, res), encoding="utf-8")
    comparison_table(data, macros)                 # macros only: equal-storage comparison (text, abstract)
    warp = {}
    for p in sorted((res / "warp").glob("*.nbits*.json")) if (res / "warp").is_dir() else []:
        if not p.name.endswith(".build.json"):
            r = json.loads(p.read_text(encoding="utf-8"))
            warp[r["dataset"], str(r["nbits"])] = r
    (tables / "main.tex").write_text(overall_table(data, macros, warp), encoding="utf-8")
    if warp:
        lat = [r["latency_ms_p50"] for (_, nb), r in warp.items() if nb == "4"]
        macros.update({"WarpLatMin": f"{min(lat):.0f}", "WarpLatMax": f"{max(lat):.0f}",
                       "WarpThreads": str(next(iter(warp.values()))["threads"])})
    sb = {n: load_json(res / "stopping_baselines" / f"{n}.json") for n in data}
    (tables / "stopping.tex").write_text(stopping_table(data, {n: v for n, v in sb.items() if v}, macros),
                                         encoding="utf-8")
    tc = {n: v for n in data if (v := load_json(res / "train_calibration" / f"{n}.json"))}
    if tc:
        (tables / "traincal.tex").write_text(train_calibration_table(data, tc, macros), encoding="utf-8")
    if (res / "beir_cv2").is_dir():
        cv2 = {k: v for k, v in mr.load(res / "beir_cv2").items() if v["protocol"]["version"] >= 3}
        enc = encoder_table(cv2, macros)
        if enc:
            (tables / "encoder.tex").write_text(enc, encoding="utf-8")
    cascade_macros(data, macros)
    significance_macros(data, macros)
    r95_macros(data, macros)
    muvera_macros(data, macros)
    tradeoff_macros(data, macros)
    if (res / "ablation" / "hazard_ablation.json").exists():
        abl = json.loads((res / "ablation" / "hazard_ablation.json").read_text(encoding="utf-8"))
        (tables / "ablation.tex").write_text(ablation_table(abl, macros), encoding="utf-8")
    lat = latency_table(res / "latency", macros)
    if lat:
        (tables / "latency.tex").write_text(lat, encoding="utf-8")
    msmarco_table(res / "msmarco", macros)          # macros only
    ms_tab = msmarco_comparison(res / "msmarco")
    if ms_tab:
        (tables / "msmarco.tex").write_text(ms_tab, encoding="utf-8")
    inverted_macros(res / "msmarco", macros)
    tp_tab = throughput_table(res / "msmarco", macros)
    if tp_tab:
        (tables / "throughput.tex").write_text(tp_tab, encoding="utf-8")
    sweep = plaid_sweep_macros(res / "msmarco", macros)
    if sweep:
        ev32 = load_json(res / "msmarco" / "eval_K65536_b32.json")
        lat = load_json(res / "msmarco" / "latency_K65536_b32.json")
        tail = load_json(res / "msmarco" / "throughput_reco_d01.json")
        runs = load_json(res / "msmarco" / "throughput_reco.json")["runs"] + tail["runs"]
        wide = [next(r for r in runs if r["delta"] == dl and r["concurrency"] == 64) for dl in ("0.1", "0.05", "0.01")]
        one = [(ev32["deltas"][dl]["hazard"]["recall"], 1000 / lat["runs"][f"hazard@{dl}"]["ms_mean"],
                lat["runs"][f"hazard@{dl}"]["ms_p50"]) for dl in ("0.1", "0.05", "0.01")]
        pf.recall_throughput([(p["recall_vs_exact10"], p["qps"], p["latency_ms"]["p50"]) for p in sweep["gpu"]], one,
                             [(r["recall_vs_exact10"], r["qps"], r["latency_ms_p50"]) for r in wide], sweep["ceiling"],
                             figures)
    optimality_macros(data, sb, res / "optimality_gap", macros)
    growth_macros(res / "growth", macros)
    memory_size_macros(res / "memory", macros)
    cpu_tab = cpu_table(res / "cpu_latency", macros)
    if cpu_tab:
        (tables / "cpu.tex").write_text(cpu_tab, encoding="utf-8")
    cost_macros(res / "cost" / "cost_model.json", macros)
    memory_macros(res / "memory", macros)
    mem_tab = memory_table(res / "memory", macros)
    if mem_tab:
        (tables / "memory.tex").write_text(mem_tab, encoding="utf-8")
    (tables / "qa.tex").write_text(qa_table(res / "qa", res / "memory", macros), encoding="utf-8")

    macros.update(pf.storage_to_tie(data))
    pf.frontier(data, figures, lambda rep, m: entry(rep, m, "0.05", headline_rules(rep, "0.05")[1])["ndcg10"],
                warp)
    cas = pf.cascades(data, figures, pick, headline_rules)
    above = [g for g in cas.values() if g >= 1]
    macros.update({"CasGainMin": f"{min(above):.1f}", "CasGainMax": f"{max(above):.1f}",
                   "CasWins": str(len(above)), "CasQuora": f"{cas.get('quora', math.nan):.1f}"})
    pf.codebook(data, figures, headline_rules)
    pf.recall_target(data, figures, pick, {n: v for n, v in sb.items() if v})
    params = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((res / "param_study").glob("*.json"))} \
        if (res / "param_study").is_dir() else {}
    if params:
        pf.result_size(params, figures)
        macros.update(pf.calibration_size(params, figures))
        param_macros(params, macros)
    scal = {p.stem.split("_K")[0]: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((res / "scalability").glob("*_K*.json"))} if (res / "scalability").is_dir() else {}
    if scal:
        macros.update(pf.scalability(scal, figures))
    build = build_table(res / "build_cost", macros)
    if build:
        (tables / "build.tex").write_text(build, encoding="utf-8")
    base = res / "qa" / "locomo" / READER
    rows_by_k = {k: qa_rows(base / f) for f, k in (("qa_rows_top3.json", 3), ("qa_rows_top10_full.json", 10),
                                                   ("qa_rows_top30.json", 30), ("qa_rows_top100.json", 100))}
    lme_base = res / "qa" / "longmemeval" / READER
    lme_by_k = {k: qa_rows(lme_base / f) for f, k in (("qa_rows_top3.json", 3), ("qa_rows_top10_full32768.json", 10))}
    if rows_by_k[10]:
        pf.budget({"LoCoMo": rows_by_k, "LongMemEval-S": lme_by_k}, figures, acc)
        for name, nm in (("colbert", "Colbert"), ("reco-K64k+R", "Reco"), ("bm25", "Bm"), ("bge-base", "Bge")):
            a, tok, n = acc(lme_by_k.get(3, {}), name)
            if n:
                macros[f"SweepLme{nm}Three"] = f"{a:.3f}"
        for k, tag in ((3, "Three"), (30, "Thirty"), (100, "Hundred")):
            for name, nm in (("colbert", "Colbert"), ("reco-K64k+R", "Reco"), ("bm25", "Bm"), ("bge-base", "Bge")):
                a, tok, n = acc(rows_by_k[k], name)
                if n:
                    macros[f"Sweep{nm}{tag}"] = f"{a:.3f}"
                    macros[f"SweepTok{nm}{tag}"] = f"{tok:,.0f}"
    for stale in ("pareto.pdf", "rules.pdf", "storage.pdf", "verification.pdf", "tail.pdf", "tradeoff.pdf"):
        if (figures / stale).exists():
            (figures / stale).unlink()
    if (tables / "verification.tex").exists():
        (tables / "verification.tex").unlink()
    for t in tables.glob("*.tex"):                  # typographic minus signs; every table spans its float's width
        s = edge_shading(tex_minus(t.read_text(encoding="utf-8"))).rstrip().replace("ReCo", r"\reco{}")
        spec = re.match(r"\\begin\{tabular\}\{([^\n]*)\}\n", s).group(1)
        n_cols = sum(ch in "lrc" for ch in spec.replace("@{}", ""))
        t.write_text(f"\\filltable{{{n_cols}}}{{%\n{s}}}\n", encoding="utf-8")
    bad = [k for k in macros if not k.isalpha()]
    if bad:
        raise ValueError(f"LaTeX macro names must be letters only: {bad}")
    (args.paper / "numbers.tex").write_text("% generated by SaveMvSearch/scripts/paper/paper_assets.py\n" + "".join(
        f"\\newcommand{{\\{k}}}{{{tex_minus(v)}}}\n" for k, v in sorted(macros.items())), encoding="utf-8")
    print(f"{len(macros)} macros; tables: {sorted(p.name for p in tables.iterdir())}; "
          f"figures: {sorted(p.name for p in figures.iterdir())}")


if __name__ == "__main__":
    main()
