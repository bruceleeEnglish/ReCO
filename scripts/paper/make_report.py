"""Aggregate the BEIR benchmark into the G1 report's tables (Markdown + LaTeX), figures and number macros.

    python scripts/paper/make_report.py results/beir reports/g1 --latency results/latency

`results/beir` holds <dataset>.json copies of the server's outputs/<dataset>/bench/bench_results.json.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ORDER = ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "trec-covid", "quora"]
TITLES = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "arguana": "ArguAna", "scidocs": "SCIDOCS",
          "fiqa": "FiQA", "trec-covid": "TREC-COVID", "quora": "Quora"}
DELTA = "0.05"
# reference palette (dataviz skill), fixed slot order; validated all-pairs for these three slots
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LOSSY_FAMILIES = ("lossy-mv", "learned-mv")
FIRST_STAGE_FAMILIES = ("single-vector", "sparse")


@dataclass
class Point:
    name: str
    family: str
    ratio: float          # total index bytes (incl. amortized codebooks) / raw text bytes
    ratio_doc: float      # per-document bytes only / raw text bytes
    est_ndcg: float       # ranking by the index alone
    est_recall: float     # |top-10 by index ∩ exact top-10| / 10
    r95: int              # recomputations for 95% recall of the exact top-10 (oracle, descriptive)
    fixed_r: float        # calibrated fixed budget (delta = DELTA) and its recall
    fixed_recall: float
    stop_r: float         # rank-aware conformal stop (delta = DELTA)
    stop_p90: float
    stop_recall: float
    stop_ndcg: float
    per_query_est: np.ndarray
    per_query_stop: np.ndarray


def load(results_dir: Path) -> dict[str, dict]:
    out = {}
    for name in ORDER:
        path = results_dir / f"{name}.json"
        if path.exists():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
        elif path.with_suffix(".json.gz").exists():
            with gzip.open(path.with_suffix(".json.gz"), "rt", encoding="utf-8") as fh:
                out[name] = json.load(fh)
    return out


def points(report: dict, rule: str = "auto") -> dict[str, Point]:
    """Per-method summary; the stop_* fields use `rule` (auto: CRC or rank1 by calibration-set size)."""
    raw = report["dataset"]["raw_bytes_per_doc"]
    rule = resolve_rule(report, rule)
    out = {}
    for name, r in report["methods"].items():
        ev = r["evaluation"]
        d = ev["deltas"][DELTA]
        out[name] = Point(name, r["family"], r["storage_ratio"], r["bytes_per_doc"] / raw, ev["estimate_ndcg10"],
                          ev["estimate_recall10"], ev["R_at_recall"]["0.95"], d["fixed"]["R_mean"],
                          d["fixed"]["recall"], d[rule]["R_mean"], d[rule]["R_p90"], d[rule]["recall"],
                          d[rule]["ndcg10"], np.asarray(ev["per_query"]["estimate_ndcg10"]),
                          np.asarray(ev["per_query"]["ndcg10"][DELTA][rule]))
    return out


def upper_envelope(pts: list[Point], value) -> list[Point]:
    """Points not dominated by a cheaper point (higher value is better)."""
    best, out = -math.inf, []
    for p in sorted(pts, key=lambda p: (p.ratio, -value(p))):
        if value(p) > best:
            out.append(p)
            best = value(p)
    return out


def lower_envelope(pts: list[Point], value) -> list[Point]:
    best, out = math.inf, []
    for p in sorted(pts, key=lambda p: (p.ratio, value(p))):
        if value(p) < best:
            out.append(p)
            best = value(p)
    return out


def best_within(pts: list[Point], budget: float, value) -> Point | None:
    cands = [p for p in pts if p.ratio <= budget * (1 + 1e-9)]
    return max(cands, key=value) if cands else None


def reco_points(ps: dict[str, Point]) -> list[Point]:
    """ReCo with an in-domain codebook (counted in the index size)."""
    return sorted((p for p in ps.values() if p.family == "ours" and not p.name.endswith("-U")), key=lambda p: p.ratio)


def universal_points(ps: dict[str, Point]) -> list[Point]:
    """ReCo with a leave-one-dataset-out codebook (a model artifact, not counted)."""
    return sorted((p for p in ps.values() if p.family == "ours" and p.name.endswith("-U")), key=lambda p: p.ratio)


def family(ps: dict[str, Point], families: tuple[str, ...]) -> list[Point]:
    return [p for p in ps.values() if p.family in families]


def paired_p(a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(a, b):
        return 1.0
    return float(stats.ttest_rel(a, b).pvalue)


# ----------------------------------------------------------------------------------------- figures
def _style(ax, xlabel: str | None, ylabel: str | None) -> None:
    ax.set_xscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=7, length=2.5)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=7.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=7.5)


def figure_pareto(data: dict[str, dict], out: Path) -> None:
    """nDCG@10 vs storage: ReCo after recomputation vs the best lossy index / first stage at each budget."""
    names = [n for n in ORDER if n in data]
    cols = 4
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 1.95 * rows + 0.35), squeeze=False)
    for i, name in enumerate(names):
        ax = axes[i // cols][i % cols]
        ps = points(data[name])
        full = ps["full-fp16"].est_ndcg
        reco = reco_points(ps)
        lossy = upper_envelope(family(ps, LOSSY_FAMILIES), lambda p: p.est_ndcg)
        first = upper_envelope(family(ps, FIRST_STAGE_FAMILIES), lambda p: p.est_ndcg)
        ax.axhline(full, color=MUTED, linewidth=0.9, zorder=1)
        ax.plot([p.ratio for p in first], [p.est_ndcg for p in first], color=AQUA, linewidth=1.4, marker="^",
                markersize=4.2, markeredgecolor="white", markeredgewidth=0.6, drawstyle="steps-post", zorder=2,
                label="single-vector / sparse\nindex alone")
        ax.plot([p.ratio for p in lossy], [p.est_ndcg for p in lossy], color=ORANGE, linewidth=1.4, marker="s",
                markersize=3.8, markeredgecolor="white", markeredgewidth=0.6, drawstyle="steps-post", zorder=3,
                label="best lossy\nmulti-vector index")
        ax.plot([p.ratio for p in reco], [p.stop_ndcg for p in reco], color=BLUE, linewidth=1.8, marker="o",
                markersize=4.6, markeredgecolor="white", markeredgewidth=0.7, zorder=4,
                label=f"ReCo (sketch +\nrecompute, $\\delta$={DELTA})")
        uni = universal_points(ps)
        if uni:
            ax.plot([p.ratio for p in uni], [p.stop_ndcg for p in uni], color=BLUE, linewidth=1.2, linestyle="--",
                    marker="o", markersize=4.2, markerfacecolor="white", markeredgecolor=BLUE, markeredgewidth=1.0,
                    zorder=4, label="ReCo, universal\ncodebook")
        ax.set_ylim(0.6 * full, 1.04 * full)        # same relative window everywhere; worse points fall below it
        ax.set_xlim(0.02, 40)
        ax.text(0.022, full, "full ColBERT", ha="left", va="bottom", fontsize=6, color=MUTED)
        ax.set_title(TITLES[name], fontsize=8, color=INK, loc="left", pad=3)
        _style(ax, "index size / raw text" if i // cols == rows - 1 else None, "nDCG@10" if i % cols == 0 else None)
    _finish(fig, axes, len(names), out / "fig_pareto")


def _finish(fig, axes, used: int, stem: Path) -> None:
    """Legend (ReCo first) in the first unused panel, or below the grid if all panels are used; save."""
    cols = axes.shape[1]
    spare = [axes[j // cols][j % cols] for j in range(used, axes.size)]
    for ax in spare:
        ax.axis("off")
    seen: dict[str, object] = {}
    for ax in axes.flat:
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lbl, h)
    labels = sorted(seen, key=lambda lbl: (not lbl.startswith("ReCo"), "universal" in lbl, not lbl.startswith("best"),
                                           not lbl.startswith("lossy")))
    handles = [seen[lbl] for lbl in labels]
    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.8)
    if spare:
        spare[0].legend(handles, labels, loc="center left", fontsize=7, frameon=False, borderaxespad=0.0)
    else:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=7, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=220)
    plt.close(fig)


def figure_r95(data: dict[str, dict], out: Path) -> None:
    """Recomputations needed for 95% recall of the exact top-10 vs index size (candidate-generator quality)."""
    names = [n for n in ORDER if n in data]
    cols = 4
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 1.95 * rows + 0.35), squeeze=False)
    for i, name in enumerate(names):
        ax = axes[i // cols][i % cols]
        ps = points(data[name])
        reco = reco_points(ps)
        lossy = lower_envelope(family(ps, LOSSY_FAMILIES), lambda p: p.r95)
        first = lower_envelope(family(ps, FIRST_STAGE_FAMILIES), lambda p: p.r95)
        ax.plot([p.ratio for p in first], [p.r95 for p in first], color=AQUA, linewidth=1.4, marker="^",
                markersize=4.2, markeredgecolor="white", markeredgewidth=0.6, drawstyle="steps-post", zorder=2,
                label="single-vector /\nsparse index")
        ax.plot([p.ratio for p in lossy], [p.r95 for p in lossy], color=ORANGE, linewidth=1.4, marker="s",
                markersize=3.8, markeredgecolor="white", markeredgewidth=0.6, drawstyle="steps-post", zorder=3,
                label="lossy multi-\nvector index")
        ax.plot([p.ratio for p in reco], [p.r95 for p in reco], color=BLUE, linewidth=1.8, marker="o",
                markersize=4.6, markeredgecolor="white", markeredgewidth=0.7, zorder=4, label="ReCo sketch")
        uni = universal_points(ps)
        if uni:
            ax.plot([p.ratio for p in uni], [p.r95 for p in uni], color=BLUE, linewidth=1.2, linestyle="--",
                    marker="o", markersize=4.2, markerfacecolor="white", markeredgecolor=BLUE, markeredgewidth=1.0,
                    zorder=4, label="ReCo sketch,\nuniversal codebook")
        ax.set_yscale("log")
        ax.set_xlim(0.02, 40)
        ax.set_title(TITLES[name], fontsize=8, color=INK, loc="left", pad=3)
        _style(ax, "index size / raw text" if i // cols == rows - 1 else None,
               "recomputations for 95% recall" if i % cols == 0 else None)
    _finish(fig, axes, len(names), out / "fig_r95")


# ------------------------------------------------------------------------------------------ tables
def table_datasets(data: dict[str, dict]) -> list[dict]:
    rows = []
    for name in ORDER:
        if name not in data:
            continue
        ds, ps = data[name]["dataset"], points(data[name])
        full = ps["full-fp16"]
        rows.append({"dataset": TITLES[name], "docs": ds["n_docs"], "queries": ds["n_queries"],
                     "raw_B": ds["raw_bytes_per_doc"], "tokens": ds["tokens_per_doc"],
                     "excluded": ds["excluded_pairs"], "full_ratio": full.ratio, "full_ndcg": full.est_ndcg})
    return rows


def table_g1(data: dict[str, dict]) -> list[dict]:
    """For every ReCo configuration: its quality vs the best baseline that fits in the same storage."""
    rows = []
    for name in ORDER:
        if name not in data:
            continue
        ps = points(data[name])
        full = ps["full-fp16"]
        lossy, first = family(ps, LOSSY_FAMILIES), family(ps, FIRST_STAGE_FAMILIES)
        for r in reco_points(ps):
            bl = best_within(lossy, r.ratio, lambda p: p.est_ndcg)
            bl2 = best_within(lossy, 2 * r.ratio, lambda p: p.est_ndcg)
            bf = best_within(first, r.ratio, lambda p: p.est_ndcg)
            rows.append({
                "dataset": TITLES[name], "reco": r.name, "ratio": r.ratio, "full": full.est_ndcg,
                "reco_ndcg": r.stop_ndcg, "reco_gap": 100 * (r.stop_ndcg - full.est_ndcg),
                "reco_p_vs_full": paired_p(r.per_query_stop, full.per_query_est),
                "R_mean": r.stop_r, "R_p90": r.stop_p90, "recall": r.stop_recall,
                "lossy": bl.name if bl else "-", "lossy_ratio": bl.ratio if bl else math.nan,
                "lossy_gap": 100 * (bl.est_ndcg - full.est_ndcg) if bl else math.nan,
                "lossy_p_vs_reco": paired_p(r.per_query_stop, bl.per_query_est) if bl else math.nan,
                "lossy2": bl2.name if bl2 else "-", "lossy2_gap": 100 * (bl2.est_ndcg - full.est_ndcg) if bl2 else math.nan,
                "first": bf.name if bf else "-", "first_gap": 100 * (bf.est_ndcg - full.est_ndcg) if bf else math.nan,
            })
    return rows


def table_r95(data: dict[str, dict]) -> list[dict]:
    """Candidate generators: R95 of every ReCo configuration vs the best baseline estimator of no larger size."""
    rows = []
    for name in ORDER:
        if name not in data:
            continue
        ps = points(data[name])
        others = family(ps, LOSSY_FAMILIES + FIRST_STAGE_FAMILIES)
        for r in reco_points(ps):
            b = min((p for p in others if p.ratio <= r.ratio * (1 + 1e-9)), key=lambda p: p.r95, default=None)
            named = {n: ps[n].r95 for n in ("bm25", "splade", "dense-int8", "dense-binary") if n in ps}
            rows.append({"dataset": TITLES[name], "reco": r.name, "ratio": r.ratio, "r95": r.r95,
                         "fixed_r": r.fixed_r, "stop_r": r.stop_r, "best": b.name if b else "-",
                         "best_ratio": b.ratio if b else math.nan, "best_r95": b.r95 if b else math.nan,
                         **{f"r95_{k}": v for k, v in named.items()},
                         **{f"ratio_{k}": ps[k].ratio for k in named}})
    return rows


def table_coverage(data: dict[str, dict]) -> list[dict]:
    rows = []
    for name in ORDER:
        if name not in data:
            continue
        for mname, m in data[name]["methods"].items():
            if m["family"] != "ours":
                continue
            for delta, d in m["evaluation"]["deltas"].items():
                row = {"dataset": TITLES[name], "reco": mname, "delta": float(delta)}
                for rule in ("fixed", "naive", "rank", "rank1", "crc"):
                    row[f"{rule}_R"], row[f"{rule}_recall"] = d[rule]["R_mean"], d[rule]["recall"]
                for rule in ("fixed_strict", "strict", "strict1", "crc_strict"):
                    row[f"{rule}_R"], row[f"{rule}_full"] = d[rule]["R_mean"], d[rule]["full_recall_rate"]
                row["rank_ndcg"] = d["rank"]["ndcg10"]
                rows.append(row)
    return rows


def table_methods(data: dict[str, dict], name: str) -> list[dict]:
    ps = points(data[name])
    return [{"method": p.name, "family": p.family, "ratio": p.ratio, "ratio_doc": p.ratio_doc, "est_ndcg": p.est_ndcg,
             "est_recall": p.est_recall, "r95": p.r95, "fixed_r": p.fixed_r, "fixed_recall": p.fixed_recall,
             "stop_r": p.stop_r, "stop_recall": p.stop_recall, "stop_ndcg": p.stop_ndcg}
            for p in sorted(ps.values(), key=lambda p: p.ratio)]


# ----------------------------------------------------------------------------------- LaTeX output
METHOD_TEX = {"bm25": "BM25", "splade": "SPLADE++", "full-fp16": "ColBERT fp16"}


def method_tex(name: str) -> str:
    """Readable LaTeX label of a method variant name."""
    if name in METHOD_TEX:
        return METHOD_TEX[name]
    if name.startswith("reco-K"):
        return f"ReCo $K$={int(name[6:]) // 1024}k"
    if name.startswith("pool"):
        f, rest = name[4:].split("-", 1)
        return f"Pool$\\times${f} " + rest.replace("bit", "b").replace("-raw", " raw")
    if name.startswith("residual-"):
        return "ColBERTv2 " + name.split("-")[1].replace("bit", "b")
    if name.startswith("fde-"):
        parts = name.split("-")
        return f"MUVERA r={parts[1][1:]}" + (" fill" if "fill" in parts else "")
    if name.startswith("constbert"):
        return "ConstBERT " + name.split("-")[-1].replace("bit", "b") + (" (HF)" if "-hf-" in name else "")
    if name.startswith("dense-"):
        return "bge-small " + name.split("-")[1]
    if name.startswith("meanpool-"):
        return "MeanPool " + name.split("-")[1]
    return name


def _gap(v: float | None) -> str:
    return "--" if v is None or math.isnan(v) else f"{v:+.1f}"


def min_calibration_queries(report: dict) -> int:
    """Smallest calibration set over the folds (reconstructed from the protocol and the fold test sizes)."""
    ev = next(iter(report["methods"].values()))["evaluation"]
    n, fit = report["dataset"]["n_queries"], report["protocol"]["fit_fraction"]
    folds = len(ev["calibration"])
    sizes = [len(part) for part in np.array_split(np.arange(n), folds)] if folds > 1 else [0]
    return min((n - t) - int(round(fit * (n - t))) for t in sizes)


def resolve_rule(report: dict, rule: str, delta: str = DELTA) -> str:
    """`auto`: CRC when its +1/(n+1) correction uses under a quarter of the risk budget, else the k+1 pooled rule."""
    if rule != "auto":
        return rule
    return "crc" if min_calibration_queries(report) + 1 >= 4 / float(delta) else "rank1"


def rule_entry(report: dict, method: str, rule: str, delta: str = DELTA) -> tuple[dict, np.ndarray]:
    ev = report["methods"][method]["evaluation"]
    r = resolve_rule(report, rule, delta)
    return ev["deltas"][delta][r], np.asarray(ev["per_query"]["ndcg10"][delta][r])


# Quora documents are 62 bytes (15 tokens) long: even the smallest sketch is 0.31x raw text
CAP_OVERRIDES = {"quora": 0.6}


def choose_reco(report: dict, max_ratio: float) -> str:
    """The ReCo configuration (in-domain codebook) with the largest codebook whose total index fits in the
    storage budget (the smallest one if none fits): G1 compares methods at equal storage."""
    max_ratio = CAP_OVERRIDES.get(report["dataset"]["name"], max_ratio)
    reco = sorted((m for m in report["methods"].values() if m["family"] == "ours" and not m["name"].endswith("-U")),
                  key=lambda m: m["storage_ratio"])
    fits = [m for m in reco if m["storage_ratio"] <= max_ratio]
    return max(fits, key=lambda m: m["variant"]["codebook_size"])["name"] if fits else reco[0]["name"]


def latex_main_table(data: dict[str, dict], rule: str, max_ratio: float) -> str:
    """G1 at a storage budget: ReCo (two recall targets) vs. the best baselines that fit in the same storage."""
    lines = [r"\begin{tabular}{l rr rr rr rr l r l r}", r"\toprule",
             r" & \multicolumn{2}{c}{ColBERT fp16} & \multicolumn{2}{c}{ReCo (ours)} & "
             r"\multicolumn{2}{c}{$\delta=0.05$} & \multicolumn{2}{c}{$\delta=0.01$} & "
             r"\multicolumn{2}{c}{best lossy MV $\le$ size} & \multicolumn{2}{c}{best 1st stage $\le$ size} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}"
             r"\cmidrule(lr){12-13}",
             r"dataset & $\times$raw & nDCG & $K$ & $\times$raw & $\Delta$ & $\bar R$ & $\Delta$ & $\bar R$ "
             r"& method & $\Delta$ & method & $\Delta$ \\", r"\midrule"]
    for name in ORDER:
        if name not in data:
            continue
        report = data[name]
        ps = points(report)
        full = ps["full-fp16"]
        pick = choose_reco(report, max_ratio)
        d5, _ = rule_entry(report, pick, rule, "0.05")
        d1, _ = rule_entry(report, pick, rule, "0.01")
        r = ps[pick]
        bl = best_within(family(ps, LOSSY_FAMILIES), r.ratio, lambda p: p.est_ndcg)
        bf = best_within(family(ps, FIRST_STAGE_FAMILIES), r.ratio, lambda p: p.est_ndcg)
        lines.append(
            f"{TITLES[name]} & {full.ratio:.1f} & {full.est_ndcg:.3f} & {int(pick[6:]) // 1024}k & {r.ratio:.2f} "
            f"& {_gap(100 * (d5['ndcg10'] - full.est_ndcg))} & {d5['R_mean']:.0f} "
            f"& {_gap(100 * (d1['ndcg10'] - full.est_ndcg))} & {d1['R_mean']:.0f} "
            f"& {method_tex(bl.name) if bl else '--'} & {_gap(100 * (bl.est_ndcg - full.est_ndcg)) if bl else '--'} "
            f"& {method_tex(bf.name) if bf else '--'} & {_gap(100 * (bf.est_ndcg - full.est_ndcg)) if bf else '--'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_sweep_table(data: dict[str, dict], rule: str) -> str:
    """Every ReCo configuration: storage, quality and recomputation, next to the best lossy index at <= storage."""
    lines = [r"\begin{tabular}{l r rr rrrr r l r}", r"\toprule",
             r"dataset & $K$ & B/doc & $\times$raw & est.\ nDCG & R95 & $\bar R$ (p90) & nDCG ($\Delta$) & $p$ "
             r"& best lossy MV $\le$ storage & $\Delta$ \\", r"\midrule"]
    for name in ORDER:
        if name not in data:
            continue
        ps = points(data[name])
        full = ps["full-fp16"]
        for j, r in enumerate(reco_points(ps)):
            m = data[name]["methods"][r.name]
            d, pq = rule_entry(data[name], r.name, rule)
            pval = paired_p(pq, full.per_query_est)
            bl = best_within(family(ps, LOSSY_FAMILIES), r.ratio, lambda p: p.est_ndcg)
            lines.append(
                f"{TITLES[name] if j == 0 else ''} & {int(r.name[6:]) // 1024}k & {m['total_bytes_per_doc']:.0f} "
                f"& {r.ratio:.2f} & {r.est_ndcg:.3f} & {r.r95} & {d['R_mean']:.0f} ({d['R_p90']:.0f}) "
                f"& {d['ndcg10']:.3f} ({_gap(100 * (d['ndcg10'] - full.est_ndcg))}) & {pval:.2f} "
                f"& {method_tex(bl.name) if bl else '--'} & {_gap(100 * (bl.est_ndcg - full.est_ndcg)) if bl else '--'} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def latex_r95_table(data: dict[str, dict]) -> str:
    """Candidate generators: R95 (and index size / raw text) of ReCo sketches and of the baselines."""
    cols = [("reco-K4096", "ReCo 4k"), ("reco-K16384", "ReCo 16k"), ("bm25", "BM25"), ("splade", "SPLADE++"),
            ("dense-binary", "bge 1b"), ("dense-int8", "bge int8"), ("meanpool-fp16", "MeanPool"),
            ("fde-r20-pq", "MUVERA r20"), ("pool16-1bit", "Pool16 1b"), ("constbert-1bit", "ConstBERT 1b")]
    lines = [r"\begin{tabular}{l " + "r" * len(cols) + "}", r"\toprule",
             "dataset & " + " & ".join(lbl for _, lbl in cols) + r" \\", r"\midrule"]
    for name in ORDER:
        if name not in data:
            continue
        ps = points(data[name])
        vals = [f"{ps[c].r95} ({ps[c].ratio:.2f})" if c in ps else "--" for c, _ in cols]
        lines.append(f"{TITLES[name]} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


STRICT = ("fixed_strict", "strict", "strict1", "crc_strict")


def latex_rules_table(data: dict[str, dict], reco_name: str, deltas=("0.05", "0.01")) -> str:
    """Stopping rules for one ReCo configuration: mean recomputations / achieved recall (or full-recall rate)."""
    rules = [("fixed", "fixed"), ("rank", "rank"), ("rank1", "rank+1"), ("crc", "CRC"),
             ("fixed_strict", "fixed$^s$"), ("strict", "strict"), ("strict1", "strict+1"), ("crc_strict", "CRC$^s$")]
    lines = [r"\begin{tabular}{l l " + "r" * len(rules) + "}", r"\toprule",
             r"dataset & $\delta$ & " + " & ".join(lbl for _, lbl in rules) + r" \\", r"\midrule"]
    for name in ORDER:
        if name not in data or reco_name not in data[name]["methods"]:
            continue
        ev = data[name]["methods"][reco_name]["evaluation"]
        for j, delta in enumerate(deltas):
            d = ev["deltas"][delta]
            cells = [f"{d[r]['R_mean']:.0f} / {d[r]['full_recall_rate' if r in STRICT else 'recall']:.3f}"
                     for r, _ in rules]
            lines.append(f"{TITLES[name] if j == 0 else ''} & {delta} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_datasets_table(data: dict[str, dict]) -> str:
    lines = [r"\begin{tabular}{l rrrrr r}", r"\toprule",
             r"dataset & docs & queries & raw B/doc & tokens/doc & excluded & ColBERT nDCG@10 \\", r"\midrule"]
    for row in table_datasets(data):
        lines.append(f"{row['dataset']} & {row['docs']:,} & {row['queries']:,} & {row['raw_B']:.0f} & "
                     f"{row['tokens']:.0f} & {row['excluded']} & {row['full_ndcg']:.4f} \\\\".replace(",", "{,}"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def choose_universal(report: dict, max_ratio: float) -> str | None:
    """Largest-codebook ReCo-U configuration within the storage budget (its codebook is not counted)."""
    max_ratio = CAP_OVERRIDES.get(report["dataset"]["name"], max_ratio)
    uni = [m for m in report["methods"].values() if m["family"] == "ours" and m["name"].endswith("-U")]
    fits = [m for m in uni if m["storage_ratio"] <= max_ratio]
    if not fits:
        return min(uni, key=lambda m: m["storage_ratio"])["name"] if uni else None
    return max(fits, key=lambda m: m["variant"]["codebook_size"])["name"]


def latex_universal_table(data: dict[str, dict], rule: str, max_ratio: float) -> str:
    """In-domain vs leave-one-dataset-out codebook at the same storage budget."""
    lines = [r"\begin{tabular}{l rrrr rrrr rr}", r"\toprule",
             r" & \multicolumn{4}{c}{in-domain codebook (counted)} & \multicolumn{4}{c}{universal codebook (not counted)}"
             r" & \multicolumn{2}{c}{$\Delta$ nDCG (universal)} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-11}",
             r"dataset & $K$ & $\times$raw & R95 & $\bar R$ & $K$ & $\times$raw & R95 & $\bar R$ & $\delta$=0.05 & $\delta$=0.01 \\",
             r"\midrule"]
    for name in ORDER:
        if name not in data:
            continue
        report = data[name]
        u = choose_universal(report, max_ratio)
        if u is None:
            continue
        ps = points(report)
        full = ps["full-fp16"]
        i = choose_reco(report, max_ratio)
        di, _ = rule_entry(report, i, rule)
        du, _ = rule_entry(report, u, rule)
        du1, _ = rule_entry(report, u, rule, "0.01")
        lines.append(
            f"{TITLES[name]} & {int(i.split('-')[1][1:]) // 1024}k & {ps[i].ratio:.2f} & {ps[i].r95} & {di['R_mean']:.0f} "
            f"& {int(u.split('-')[1][1:]) // 1024}k & {ps[u].ratio:.2f} & {ps[u].r95} & {du['R_mean']:.0f} "
            f"& {_gap(100 * (du['ndcg10'] - full.est_ndcg))} & {_gap(100 * (du1['ndcg10'] - full.est_ndcg))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_latency_table(latency_dir: Path) -> str:
    """Online search with real recomputation from raw text (outputs of scripts/pipeline/run_latency.sh)."""
    rule_tex = {"crc": "CRC", "fixed": "fixed", "rank1": "rank+1", "rank": "rank"}
    lines = [r"\begin{tabular}{l r l r rr rr r rrr}", r"\toprule",
             r"dataset & $K$ & rule & batch & p50 & p90 & $\bar R$ & ms/doc & recall & query & estimate & recompute \\",
             r"\midrule"]
    for path in sorted(latency_dir.glob("latency_*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        lat, name = report["latency"], report["dataset"]["name"]
        for b, v in lat["batches"].items():
            st = v["ms_mean_by_stage"]
            lines.append(f"{TITLES.get(name, name)} & {lat['codebook_size'] // 1024}k & {rule_tex.get(lat['rule'], lat['rule'])} "
                         f"& {b} & {v['ms_p50']:.0f} & {v['ms_p90']:.0f} & {v['recomputed_mean']:.0f} "
                         f"& {v['ms_per_recomputed_doc']:.2f} & {v['recall_mean']:.3f} & {st['encode_query']:.1f} "
                         f"& {st['estimate']:.1f} & {st['recompute']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_ablation_table(data: dict[str, dict]) -> str:
    """Baseline-variant choices (nDCG@10 of the index alone) that we resolved in the baselines' favour."""
    pairs = [("pool4-fp16", "pool4-fp16-raw", "Pool$\\times$4: re-normalized / PyLate raw"),
             ("fde-r20-pq", "fde-r20-fill-pq", "MUVERA r=20: no fill / fill"),
             ("constbert-fp16", "constbert-hf-fp16", "ConstBERT: trained / HF marker")]
    lines = [r"\begin{tabular}{l " + "r" * len(ORDER) + "}", r"\toprule",
             "variant (ours / alternative) & " + " & ".join(TITLES[n] for n in ORDER if n in data) + r" \\",
             r"\midrule"]
    for a, b, label in pairs:
        cells = []
        for name in ORDER:
            if name not in data:
                continue
            m = data[name]["methods"]
            if a in m and b in m:
                va, vb = m[a]["evaluation"]["estimate_ndcg10"], m[b]["evaluation"]["estimate_ndcg10"]
                cells.append(f"{va:.3f} / {vb:.3f}")
            else:
                cells.append("--")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def rule_statistics(data: dict[str, dict]) -> dict[str, str]:
    """Macros on the stopping rules over all ReCo configurations and datasets."""
    saving, crc_cov, crc_beats_fixed, n = [], [], 0, 0
    for name in ORDER:
        if name not in data:
            continue
        n_cal = min_calibration_queries(data[name])
        for m in data[name]["methods"].values():
            if m["family"] != "ours" or m["name"].endswith("-U"):
                continue
            for delta, d in m["evaluation"]["deltas"].items():
                if d["rank"]["R_mean"] > 0:
                    saving.append(1 - d["rank1"]["R_mean"] / d["rank"]["R_mean"])
                if n_cal + 1 >= 4 / float(delta):
                    crc_cov.append(d["crc"]["recall"] - (1 - float(delta)))
                    n += 1
                    crc_beats_fixed += d["crc"]["R_mean"] < d["fixed"]["R_mean"]
    return {"RankOneSavingMean": f"{100 * np.mean(saving):.0f}", "RankOneSavingMax": f"{100 * np.max(saving):.0f}",
            "CrcExcessMin": f"{np.min(crc_cov):+.3f}", "CrcExcessMax": f"{np.max(crc_cov):+.3f}",
            "CrcBeatsFixed": str(crc_beats_fixed), "CrcCases": str(n)}


MACRO = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "arguana": "ArguAna", "scidocs": "Scidocs", "fiqa": "FiQA",
         "trec-covid": "TrecCovid", "quora": "Quora"}


def summary_numbers(data: dict[str, dict], rule: str, max_ratio: float) -> tuple[str, dict]:
    """LaTeX macros for the headline numbers (per dataset and min/max over datasets)."""
    macros, rows = {}, {}
    for name in ORDER:
        if name not in data:
            continue
        report = data[name]
        ps = points(report)
        full = ps["full-fp16"]
        pick = choose_reco(report, max_ratio)
        d, pq = rule_entry(report, pick, rule)
        d1, pq1 = rule_entry(report, pick, rule, "0.01")
        r = ps[pick]
        bl = best_within(family(ps, LOSSY_FAMILIES), r.ratio, lambda p: p.est_ndcg)
        bf = best_within(family(ps, FIRST_STAGE_FAMILIES), r.ratio, lambda p: p.est_ndcg)
        b5 = best_within(family(ps, LOSSY_FAMILIES), 5 * r.ratio, lambda p: p.est_ndcg)
        pval = paired_p(pq, full.per_query_est)
        others = family(ps, LOSSY_FAMILIES + FIRST_STAGE_FAMILIES)
        best_r95 = min((p.r95 for p in others if p.ratio <= r.ratio * (1 + 1e-9)), default=math.nan)
        # storage at which the best lossy multi-vector index first gets within 0.5 points of full ColBERT
        close = [p.ratio for p in family(ps, LOSSY_FAMILIES) if p.est_ndcg >= full.est_ndcg - 0.005]
        rows[name] = {"ratio": r.ratio, "K": int(pick[6:]), "rule": resolve_rule(report, rule),
                      "catch_up": (min(close) / r.ratio) if close else math.nan,
                      "catch_up_abs": min(close) if close else math.nan,
                      "gap": 100 * (d["ndcg10"] - full.est_ndcg),
                      "R": d["R_mean"], "recall": d["recall"], "p": pval, "r95": r.r95,
                      "gap01": 100 * (d1["ndcg10"] - full.est_ndcg), "R01": d1["R_mean"],
                      "p01": paired_p(pq1, full.per_query_est),
                      "lossy_gap": 100 * (bl.est_ndcg - full.est_ndcg) if bl else math.nan,
                      "lossy5_gap": 100 * (b5.est_ndcg - full.est_ndcg) if b5 else math.nan,
                      "first_gap": 100 * (bf.est_ndcg - full.est_ndcg) if bf else math.nan,
                      "r95_ratio": best_r95 / r.r95 if r.r95 else math.nan}
        tag = MACRO[name]
        macros[f"RecoRatio{tag}"] = f"{r.ratio:.2f}"
        macros[f"RecoGap{tag}"] = f"{rows[name]['gap']:+.1f}"
        macros[f"LossyGap{tag}"] = _gap(rows[name]["lossy_gap"])
        macros[f"FirstGap{tag}"] = _gap(rows[name]["first_gap"])
        macros[f"RecoR{tag}"] = f"{d['R_mean']:.0f}"
    vals = list(rows.values())

    def rng(key: str, fmt: str = "{:+.1f}") -> tuple[str, str]:
        xs = [v[key] for v in vals if not math.isnan(v[key])]
        return (fmt.format(min(xs)), fmt.format(max(xs))) if xs else ("--", "--")

    for key, fmt, stem in (("gap", "{:+.1f}", "RecoGap"), ("lossy_gap", "{:+.1f}", "LossyGap"),
                           ("lossy5_gap", "{:+.1f}", "LossyFiveGap"), ("first_gap", "{:+.1f}", "FirstGap"),
                           ("ratio", "{:.2f}", "RecoRatio"), ("R", "{:.0f}", "RecoR"), ("p", "{:.2f}", "RecoP"),
                           ("r95_ratio", "{:.1f}", "RNinetyFiveRatio"), ("catch_up", "{:.0f}", "CatchUp"),
                           ("catch_up_abs", "{:.1f}", "CatchUpAbs")):
        lo, hi = rng(key, fmt)
        macros[f"{stem}Min"], macros[f"{stem}Max"] = lo, hi
    macros["NumDatasets"] = str(len(rows))
    macros["NumSignificant"] = str(sum(v["p"] < 0.05 for v in vals))
    for key, fmt, stem in (("gap01", "{:+.1f}", "RecoGapStrict"), ("R01", "{:.0f}", "RecoRStrict"),
                           ("p01", "{:.2f}", "RecoPStrict")):
        lo, hi = rng(key, fmt)
        macros[f"{stem}Min"], macros[f"{stem}Max"] = lo, hi
    macros["NumSignificantStrict"] = str(sum(v["p01"] < 0.05 for v in vals))
    macros["HeadlineRule"] = rule
    macros["HeadlineMaxRatio"] = f"{max_ratio:.1f}"
    text = "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
    return text, rows


def to_markdown(rows: list[dict]) -> str:
    if not rows:
        return ""
    keys = list(rows[0])

    def fmt(v):
        if isinstance(v, float):
            return "nan" if math.isnan(v) else (f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}")
        return str(v)
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--rule", default="auto", help="stopping rule of the headline ReCo numbers (auto: crc or rank1)")
    ap.add_argument("--max-ratio", type=float, default=0.3, help="storage budget (x raw text) of the headline config")
    ap.add_argument("--latency", type=Path, default=None, help="directory with latency_*.json (online search runs)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})
    data = load(args.results)
    if not data:
        raise SystemExit(f"no results in {args.results}")
    figure_pareto(data, args.out)
    figure_r95(data, args.out)
    tables = {"datasets": table_datasets(data), "g1": table_g1(data), "r95": table_r95(data),
              "coverage": table_coverage(data)}
    for name in data:
        tables[f"methods_{name}"] = table_methods(data, name)
    (args.out / "tables.json").write_text(json.dumps(tables, indent=1, default=float), encoding="utf-8")
    md = "\n".join(f"## {k}\n\n{to_markdown(v)}" for k, v in tables.items())
    (args.out / "tables.md").write_text(md, encoding="utf-8")
    tex = args.out / "tables"
    tex.mkdir(exist_ok=True)
    (tex / "datasets.tex").write_text(latex_datasets_table(data), encoding="utf-8")
    (tex / "main.tex").write_text(latex_main_table(data, args.rule, args.max_ratio), encoding="utf-8")
    (tex / "sweep.tex").write_text(latex_sweep_table(data, args.rule), encoding="utf-8")
    (tex / "r95.tex").write_text(latex_r95_table(data), encoding="utf-8")
    for k in ("reco-K4096", "reco-K16384"):
        (tex / f"rules_{k[5:]}.tex").write_text(latex_rules_table(data, k), encoding="utf-8")
    numbers, headline = summary_numbers(data, args.rule, args.max_ratio)
    numbers += "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in rule_statistics(data).items())
    (tex / "ablation.tex").write_text(latex_ablation_table(data), encoding="utf-8")
    (tex / "universal.tex").write_text(latex_universal_table(data, args.rule, args.max_ratio), encoding="utf-8")
    if args.latency and args.latency.is_dir():
        (tex / "latency.tex").write_text(latex_latency_table(args.latency), encoding="utf-8")
    (args.out / "numbers.tex").write_text(numbers, encoding="utf-8")
    (args.out / "headline.json").write_text(json.dumps(headline, indent=1, default=float), encoding="utf-8")
    print(f"datasets: {', '.join(data)} -> {args.out}")


if __name__ == "__main__":
    main()
