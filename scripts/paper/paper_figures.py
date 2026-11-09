"""Figures of the ICDE paper, in one database-paper style.

Every method keeps one identity in every figure (STYLE): ReCo / hazard stopping in blue, the two main competitors of
a figure in orange and aqua (the three hues that stay distinguishable under every colour-vision deficiency even when
lines cross; checked with the dataviz skill's validator), all further baselines in greys with their own marker or
hatch, exact ColBERT as a thin black reference. Panels carry "(a) ..." captions below them, one shared legend sits
above, axes are boxed with inward ticks and a hairline grid, text is Times at the body size. Sizes fit the IEEE
two-column layout (column 3.45 in, page 7.16 in). Called by scripts/paper/paper_assets.py.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

import make_report as mr  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
G1, G2, G3 = "#4a4946", "#7d7b75", "#aaa8a0"
INK, INK2, GRID, SHADE = "#0b0b0b", "#52514e", "#e1e0d9", "#f2f1ec"
COL, FULL = 3.45, 7.16
FLAT, TOP, BOTTOM = 1.28, 0.865, 0.185     # height of a one-row figure in inches, axes box as a fraction of it
TITLE = mr.TITLES
ORDER = mr.ORDER

# method / rule -> (label, colour, marker, hatch); identical in every figure
STYLE = {
    "reco": ("ReCo", BLUE, "o", None),
    "colbert": ("ColBERT (exact)", INK, "o", None),
    "plaid": ("PLAID", ORANGE, "s", None),
    "warp": ("WARP (XTR)", ORANGE, "D", None),
    "pool": ("TokenPool", AQUA, "^", None),
    "constbert": ("ConstBERT", G1, "P", None),
    "dense": ("bge (single-vector)", G2, "D", "...."),
    "sparse": ("BM25 / SPLADE++", G3, "v", "////"),
    "bm25": ("BM25", G3, "v", "////"),
    "splade": ("SPLADE++", G2, "<", "xxxx"),
    "hazard": ("ReCo", BLUE, "o", None),
    "fixed": ("FixedBudget", G1, "D", None),
    "crc": ("TA-Margin", ORANGE, "s", None),
    "patience": ("Patience", AQUA, "^", None),
    "laet": ("LAET", G2, "v", None),
    "hazard_doc": ("ReCo-D", "#86b6ef", "o", None),
}


def setup() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.8, "ytick.major.size": 2.8, "xtick.major.pad": 1.8, "ytick.major.pad": 1.8,
        "axes.labelpad": 1.2, "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5, "hatch.linewidth": 0.5, "hatch.color": "white",
        "lines.linewidth": 1.2, "lines.markersize": 4, "legend.frameon": False, "legend.handlelength": 2.0,
        "legend.columnspacing": 1.2, "legend.handletextpad": 0.4, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.015})


def frame(ax, grid: str = "y") -> None:
    """Boxed axes, inward ticks, a hairline solid grid behind the data."""
    for side in ax.spines.values():
        side.set_color(INK2)
        side.set_linewidth(0.6)
    if grid:
        ax.grid(True, axis=grid, which="major", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def caption(ax, text: str, below: float = 18.5) -> None:
    """Sub-figure caption "(a) ..." under a panel, `below` points under the axes."""
    ax.annotate(text, xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -below), textcoords="offset points",
                ha="center", va="top", fontsize=8)


def top_legend(fig, *args, top: float, x: float = 0.55, **kw) -> None:
    """Shared legend right above the axes; `top` is the top of the axes as a fraction of the figure."""
    fig.legend(*args, loc="lower center", bbox_to_anchor=(x, top), borderaxespad=0, borderpad=0.15, **kw)


HOLLOW = ("colbert", "warp")       # drawn with white-filled markers


def _mark_style(key: str) -> dict:
    lab, color, marker, _ = STYLE[key]
    ours, hollow = key in ("reco", "hazard"), key in HOLLOW
    return dict(color=color, marker=marker, markersize=4.6 if ours else 4,
                markerfacecolor="white" if hollow else color, markeredgecolor=color if hollow else "white",
                markeredgewidth=0.9 if hollow else 0.6, linestyle="--" if key == "warp" else "-")


def handle(key: str, line: bool = True, label: str | None = None, **kw) -> Line2D:
    ours = key in ("reco", "hazard")
    st = {**_mark_style(key), **kw}
    return Line2D([], [], linewidth=(1.6 if ours else 1.1) if line else 0, label=label or STYLE[key][0], **st)


def plot(ax, key: str, xs, ys, label: str | None = None, **kw) -> None:
    ours = key in ("reco", "hazard")
    st = {**_mark_style(key), **kw}
    ax.plot(xs, ys, linewidth=1.6 if ours else 1.1, zorder=5 if ours else 3, label=label or STYLE[key][0], **st)


def log_ticks(ax, axis: str, ticks) -> None:
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(fmt)
    a.set_minor_formatter(NullFormatter())


# ------------------------------------------------------------------------------------------ storage (macros only)
def storage_to_tie(data: dict) -> dict:
    """How many times smaller ReCo's index is than the smallest lossy multi-vector index within 0.5 nDCG@10 points
    of uncompressed ColBERT (the figure that showed it was replaced by the comparison table)."""
    gains = []
    for name in [n for n in ORDER if n in data]:
        ps = mr.points(data[name], "crc")
        target = ps["full-fp16"].est_ndcg - 0.005
        reco = [p.ratio for p in mr.reco_points(ps) if p.stop_ndcg >= target]
        lossy = [p.ratio for p in mr.family(ps, mr.LOSSY_FAMILIES) if p.est_ndcg >= target]
        if reco and lossy:
            gains.append(min(lossy) / min(reco))
    return {"TieGainMin": f"{min(gains):.0f}", "TieGainMax": f"{max(gains):.0f}",
            "TieGainMedian": f"{np.median(gains):.0f}"}


# ------------------------------------------------------------------------------------------ Exp-1 frontier
FRONTIER_FAMILIES = [("plaid", ("residual-",)), ("pool", ("pool",)), ("constbert", ("constbert",))]   # multi-vector only


def frontier(data: dict, out: Path, reco_ndcg, warp: dict | None = None) -> None:
    """nDCG@10 (% of exact ColBERT) versus index size: each family's best configuration at every size."""
    names = [n for n in ORDER if n in data]
    fig, axes = plt.subplots(2, 4, figsize=(FULL, 2.4))
    fig.subplots_adjust(left=0.05, right=0.995, top=0.985, bottom=0.1, wspace=0.22, hspace=0.27)
    for i, name in enumerate(names):
        ax = axes[i // 4][i % 4]
        rep = data[name]
        ps = mr.points(rep, "crc")
        full = ps["full-fp16"]
        rel = lambda v: 100 * v / full.est_ndcg  # noqa: E731
        ax.axvspan(1e-3, 1.0, color=SHADE, zorder=0, linewidth=0)
        ax.axhline(100, color=INK, linewidth=0.6, zorder=1)
        top = 100.0
        for key, prefixes in FRONTIER_FAMILIES:
            pts = mr.upper_envelope([p for k, p in ps.items() if k.startswith(prefixes)], lambda p: p.est_ndcg)
            if pts:
                plot(ax, key, [p.ratio for p in pts], [rel(p.est_ndcg) for p in pts])
                top = max(top, max(rel(p.est_ndcg) for p in pts))
        wp = sorted(((r["storage_ratio"], 100 * r["ndcg10"] / full.est_ndcg) for (n, _), r in (warp or {}).items()
                     if n == name))
        if wp:
            plot(ax, "warp", [w[0] for w in wp], [w[1] for w in wp])
        plot(ax, "colbert", [full.ratio], [100.0])
        reco = mr.reco_points(ps)
        plot(ax, "reco", [p.ratio for p in reco], [rel(reco_ndcg(rep, p.name)) for p in reco])
        ax.set_xscale("log")
        ax.set_xlim(0.03, 80)
        ax.set_ylim(60, 104 if top <= 101 else 10 * math.ceil((top + 3) / 10))
        log_ticks(ax, "x", [0.1, 1, 10])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
        frame(ax, "both")
        caption(ax, f"({chr(97 + i)}) {TITLE[name]}", below=9.5)
    fig.supylabel("nDCG@10 (% of exact ColBERT)", x=0.004, fontsize=8)
    key_ax = axes[1][3]
    key_ax.axis("off")
    handles = [handle("reco"), handle("plaid")] + ([handle("warp")] if warp else []) + [handle("pool"), handle("constbert"),
               handle("colbert"), Patch(facecolor=SHADE, edgecolor="none", label="Smaller than raw text")]
    key_ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.1), ncol=1, labelspacing=0.2,
                  handlelength=2.2, title="x: index size (× raw text)", title_fontsize=7.5, alignment="left")
    fig.savefig(out / "frontier.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------------------ Exp-3 cascades
def cascades(data: dict, out: Path, pick_reco, headline_rules) -> dict:
    """Mean recomputations of each retrieve-then-verify cascade at delta = 0.05, every system with the same
    (hazard) stopping rule, so the first stage is the only difference."""
    names = [n for n in ORDER if n in data]
    stages = [("bm25", "bm25"), ("splade", "splade"), ("dense", "dense-fp16")]
    d = "0.05"
    fig, ax = plt.subplots(figsize=(COL, 1.42))
    fig.subplots_adjust(left=0.115, right=0.995, top=0.875, bottom=0.085)
    w, x = 0.2, np.arange(len(names))
    gains = {}
    for xi, name in zip(x, names):
        rep = data[name]
        rule = headline_rules(rep, d)[1]
        cols = stages + [("reco", pick_reco(rep))]
        vals = [rep["methods"][m]["evaluation"]["deltas"][d][rule]["R_mean"] for _, m in cols]
        for j, ((key, _), v) in enumerate(zip(cols, vals)):
            _, color, _, hatch = STYLE[key]
            ax.bar(xi + (j - 1.5) * w, v, w * 0.9, color=color, hatch=hatch, edgecolor="white", linewidth=0.4,
                   zorder=3)
        gain = min(vals[:3]) / vals[3]
        gains[name] = gain
        ax.text(xi, 0.965, f"{gain:.1f}×" if gain < 10 else f"{gain:.0f}×", ha="center", va="top", fontsize=6.6,
                color=BLUE if gain >= 1 else INK2, fontweight="bold", transform=ax.get_xaxis_transform(), zorder=6)
    ax.set_yscale("log")
    ax.set_ylim(10, ax.get_ylim()[1] * 4.0)
    ax.set_xlim(-0.55, len(names) - 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([{"trec-covid": "TREC"}.get(n, TITLE[n]) for n in names], fontsize=6.6)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Recomputations")
    frame(ax)
    handles = [Patch(facecolor=STYLE[k][1], hatch=STYLE[k][3], edgecolor="white", linewidth=0.4,
                     label={"bm25": "BM25-Rerank", "splade": "SPLADE-Rerank", "dense": "BGE-Rerank"}.get(k, "ReCo"))
               for k in ("bm25", "splade", "dense", "reco")]
    ax.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), handlelength=1.4,
              columnspacing=1.0, borderaxespad=0.15, borderpad=0.15)
    fig.savefig(out / "cascade.pdf")
    plt.close(fig)
    return gains


# ------------------------------------------------------------------------------------------ Exp-7 codebook size
def codebook(data: dict, out: Path, headline_rules) -> None:
    """(a) index size and (b) recomputations (delta = 0.05) as the codebook grows; one line per corpus."""
    names = [n for n in ("scifact", "nfcorpus", "scidocs", "fiqa") if n in data]
    markers = {"scifact": "o", "nfcorpus": "s", "scidocs": "^", "fiqa": "D"}
    ks = (1024, 4096, 16384, 65536)
    fig, axes = plt.subplots(1, 2, figsize=(COL, FLAT))
    fig.subplots_adjust(left=0.115, right=0.99, top=TOP, bottom=BOTTOM, wspace=0.36)
    for name in names:
        rep = data[name]
        rule = headline_rules(rep, "0.05")[1]
        ms = [rep["methods"][f"reco-K{k}"] for k in ks]
        x = np.arange(len(ks))
        kw = dict(color=BLUE, marker=markers[name], linewidth=1.2, markersize=4, markeredgecolor="white",
                  markeredgewidth=0.6, label=TITLE[name])
        axes[0].plot(x, [m["storage_ratio"] for m in ms], **kw)
        axes[1].plot(x, [m["evaluation"]["deltas"]["0.05"][rule]["R_mean"] for m in ms], **kw)
    for ax, ylab, cap in ((axes[0], "Size (× raw text)", "(a) Index size versus $K$"),
                          (axes[1], "Recomputations", "(b) Verification cost versus $K$")):
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(["1k", "4k", "16k", "64k"])
        ax.set_yscale("log")
        ax.set_ylabel(ylab)
        frame(ax)
        caption(ax, cap, below=9.5)
    log_ticks(axes[0], "y", [0.1, 0.3, 1, 3])
    log_ticks(axes[1], "y", [30, 100, 300])
    handles = [Line2D([], [], color=BLUE, marker=markers[n], markeredgecolor="white", markeredgewidth=0.6,
                      linewidth=1.2, label=TITLE[n]) for n in names]
    top_legend(fig, handles=handles, top=TOP, ncol=4, columnspacing=0.9, handlelength=1.6)
    fig.savefig(out / "codebook.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------------------ Exp-8 recall target
def recall_target(data: dict, out: Path, pick_reco, sb: dict | None = None) -> None:
    """Mean recomputations versus the recall target for every stopping rule (query-level CRC throughout; the two
    corpora whose calibration sets support it at every target); LAET from the learned-termination run."""
    names = [n for n in ("scidocs", "quora") if n in data]
    deltas = ("0.1", "0.05", "0.01")
    rules = ("fixed", "crc", "patience") + (("laet",) if sb else ()) + ("hazard",)
    fig, axes = plt.subplots(1, len(names), figsize=(COL, FLAT))
    fig.subplots_adjust(left=0.115, right=0.99, top=TOP, bottom=BOTTOM, wspace=0.22)
    for i, (ax, name) in enumerate(zip(np.atleast_1d(axes), names)):
        rep = data[name]
        ev = rep["methods"][pick_reco(rep)]["evaluation"]["deltas"]
        x = np.arange(len(deltas))
        for rule in rules:
            src = sb[name]["deltas"] if rule == "laet" else ev
            plot(ax, rule, x, [src[d][rule]["R_mean"] for d in deltas])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{1 - float(d):.2f}" for d in deltas])
        ax.set_yscale("log")
        frame(ax)
        caption(ax, f"({chr(97 + i)}) {TITLE[name]}: recall target $1-\\delta$", below=9.5)
    np.atleast_1d(axes)[0].set_ylabel("Recomputations")
    log_ticks(np.atleast_1d(axes)[0], "y", [30, 100, 300])
    if len(names) > 1:
        log_ticks(axes[1], "y", [30, 100, 300, 1000])
    top_legend(fig, handles=[handle(r) for r in ("hazard",) + rules[:-1]], top=TOP, ncol=5, columnspacing=0.7,
               handlelength=1.5)
    fig.savefig(out / "delta.pdf")
    plt.close(fig)


def result_size(params: dict, out: Path) -> None:
    """Exp-9: mean recomputations versus the result size k (delta = 0.05) for every stopping rule."""
    names = [n for n in ("fiqa", "quora") if n in params]
    ks = sorted(int(k) for k in params[names[0]]["k"])
    fig, axes = plt.subplots(1, len(names), figsize=(COL, FLAT))
    fig.subplots_adjust(left=0.115, right=0.99, top=TOP, bottom=BOTTOM, wspace=0.22)
    for i, (ax, name) in enumerate(zip(np.atleast_1d(axes), names)):
        e = params[name]["k"]
        x = np.arange(len(ks))
        for rule in ("fixed", "crc", "patience", "hazard"):
            plot(ax, rule, x, [e[str(k)]["0.05"][rule]["R_mean"] for k in ks])
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_yscale("log")
        frame(ax)
        caption(ax, f"({chr(97 + i)}) {TITLE[name]}: result size $k$", below=9.5)
        log_ticks(ax, "y", [10, 30, 100, 300])
    np.atleast_1d(axes)[0].set_ylabel("Recomputations")
    top_legend(fig, handles=[handle(r) for r in ("hazard", "fixed", "crc", "patience")], top=TOP, ncol=4,
               columnspacing=0.9, handlelength=1.6)
    fig.savefig(out / "topk.pdf")
    plt.close(fig)


def calibration_size(params: dict, out: Path) -> dict:
    """Exp-10: (a) mean recomputations on Quora and (b) the worst achieved recall over corpora and repetitions,
    versus the number of calibration queries per fold (delta = 0.05)."""
    names = [n for n in ("scidocs", "fiqa", "quora") if n in params]
    head = "quora" if "quora" in names else names[-1]
    sizes = [n for n in params[head]["ncal"]]
    order = sorted(sizes, key=int)
    labels = [s for s in order[:-1]] + ["all"]
    fig, axes = plt.subplots(1, 2, figsize=(COL, FLAT))
    fig.subplots_adjust(left=0.115, right=0.99, top=TOP, bottom=BOTTOM, wspace=0.36)
    x = np.arange(len(order))
    ax = axes[0]
    for rule in ("fixed", "crc", "patience", "hazard"):
        plot(ax, rule, x, [params[head]["ncal"][n][rule]["R_mean"] for n in order])
    plot(ax, "hazard_doc", x, [params[head]["ncal"][n]["hazard_doc"]["R_mean"] for n in order], linestyle="--")
    ax.set_yscale("log")
    ax.set_ylabel("Recomputations")
    log_ticks(ax, "y", [100, 300, 1000, 3000])
    ax = axes[1]
    low, worst = {}, {}
    for rule in ("hazard", "hazard_doc"):
        low[rule], worst[rule] = [], []
        for j in range(len(order)):
            means, mins = [], []
            for nm in names:
                ks = sorted(params[nm]["ncal"], key=int)
                means.append(params[nm]["ncal"][ks[j]][rule]["recall"])
                mins.append(params[nm]["ncal"][ks[j]][rule]["recall_min"])
            low[rule].append(min(means))
            worst[rule].append(min(mins))
        plot(ax, rule, x, low[rule], linestyle="--" if rule == "hazard_doc" else "-")
    ax.axhline(0.95, color=INK, linewidth=0.7, zorder=1)
    ax.text(len(order) - 1, 0.951, "target", ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax.set_ylabel("Lowest recall")
    for a, cap in zip(axes, ("(a) Quora: calibration queries", "(b) 3 corpora: calibration queries")):
        a.set_xticks(x)
        a.set_xticklabels(labels)
        frame(a)
        caption(a, cap, below=9.5)
    top_legend(fig, handles=[handle(r) for r in ("hazard", "fixed", "crc", "patience")] +
               [handle("hazard_doc", linestyle="--")], top=TOP, ncol=5, columnspacing=0.6, handlelength=1.5,
               fontsize=6.6)
    fig.savefig(out / "ncal.pdf")
    plt.close(fig)
    return {"NcalQueryLow": f"{min(low['hazard']):.3f}", "NcalDocLow": f"{min(low['hazard_doc']):.3f}",
            "NcalQueryWorst": f"{min(worst['hazard']):.3f}", "NcalDocWorst": f"{min(worst['hazard_doc']):.3f}"}


# ------------------------------------------------------------------------------------------ Exp-9 corpus size
def scalability(scal: dict, out: Path) -> dict:
    """(a) index size and (b) recomputations as one corpus grows (nested random subsets)."""
    macros = {}
    for name, s in scal.items():
        pts, d = s["points"], "0.05"
        n = np.array([p["n_docs"] for p in pts])
        r_last = pts[-1]["rules"][d]
        macros[f"Scal{name.title()}N"] = f"{n[-1] / n[0]:.0f}"
        macros[f"Scal{name.title()}R"] = f"{r_last['hazard']['R_mean'] / pts[0]['rules'][d]['hazard']['R_mean']:.1f}"
        macros[f"Scal{name.title()}Margin"] = f"{r_last['crc']['R_mean'] / pts[0]['rules'][d]['crc']['R_mean']:.1f}"
        macros[f"Scal{name.title()}Frac"] = f"{100 * r_last['hazard']['R_mean'] / n[-1]:.2f}"
        macros[f"Scal{name.title()}Size"] = f"{pts[-1]['colbert_bytes'] / (pts[-1]['sketch_bytes'] - s['k'] * 96 * 2):.0f}"
    name = "quora" if "quora" in scal else next(iter(scal))
    pts, k = scal[name]["points"], scal[name]["k"]
    n = np.array([p["n_docs"] for p in pts]) / 1000
    mb = 1 / 2 ** 20
    fig, axes = plt.subplots(1, 2, figsize=(COL, FLAT))
    fig.subplots_adjust(left=0.115, right=0.99, top=TOP, bottom=BOTTOM, wspace=0.36)
    ax = axes[0]
    plot(ax, "colbert", n, [p["colbert_bytes"] * mb for p in pts], label="ColBERT")
    ax.plot(n, [p["raw_bytes"] * mb for p in pts], color=G2, marker="s", markersize=3.6, linewidth=1.1,
            markeredgecolor="white", markeredgewidth=0.6, label="Raw text", zorder=3)
    plot(ax, "reco", n, [(p["sketch_bytes"] - k * 96 * 2) * mb for p in pts], label="ReCo")
    ax.set_ylabel("Index size (MB)")
    ax = axes[1]
    for rule in ("fixed", "crc", "patience", "hazard"):
        plot(ax, rule, n, [p["rules"]["0.05"][rule]["R_mean"] for p in pts],
             label=STYLE[rule][0].replace(" budget", ""))
    ax.set_ylabel("Recomputations")
    for ax, cap in zip(axes, ("(a) Corpus size ($10^3$ documents)", "(b) Corpus size ($10^3$ documents)")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        log_ticks(ax, "x", [10, 30, 100, 300])
        frame(ax, "both")
        caption(ax, cap, below=9.5)
    log_ticks(axes[1], "y", [30, 50, 100, 200])
    hs, ls = [sum(p, []) for p in zip(*(list(map(list, a.get_legend_handles_labels())) for a in axes))]
    keep = {lab: h for h, lab in zip(hs, ls)}
    top_legend(fig, [keep[lab] for lab in ("ReCo", "ColBERT", "Raw text", "FixedBudget", "TA-Margin", "Patience")],
               ["ReCo", "ColBERT", "Raw text", "FixedBudget", "TA-Margin", "Patience"], top=TOP, ncol=6, columnspacing=0.55,
               handlelength=1.3, handletextpad=0.3, fontsize=6.6)
    fig.savefig(out / "scalability.pdf")
    plt.close(fig)
    return macros


# ------------------------------------------------------------------------------------------ Exp-5 recall and speed
def recall_throughput(plaid: list, one: list, wide: list, ceiling: float, out: Path) -> None:
    """Queries per second and median latency against recall of the exact top-10 on one GPU (MS MARCO). Each list holds
    (recall, QPS, p50 ms); the dotted line is the best recall of PLAID's CPU engine, which holds every setting."""
    fig, axes = plt.subplots(1, 2, figsize=(COL, FLAT))
    fig.subplots_adjust(left=0.095, right=0.995, top=TOP, bottom=BOTTOM, wspace=0.3)
    series = (("plaid", "PLAID", plaid, {}), ("reco", "ReCo, $C$=1", one, {}),
              ("hazard_doc", "ReCo, $C$=64", wide, {"linestyle": "--"}))
    for ax, col, ylab, ticks, cap in ((axes[0], 1, "QPS", [1, 3, 10, 30], "(a) Recall of the exact top-10"),
                                      (axes[1], 2, "p50 latency (ms)", [30, 100, 300, 1000, 3000, 10000],
                                       "(b) Recall of the exact top-10")):
        for key, lab, pts, kw in series:
            plot(ax, key, [p[0] for p in pts], [p[col] for p in pts], label=lab, **kw)
        ax.axvline(ceiling, color=INK2, linewidth=0.8, linestyle=":", zorder=2)
        ax.set_yscale("log")
        log_ticks(ax, "y", ticks)
        ax.set_xlim(0.795, 1.0)
        ax.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0])
        ax.set_xticklabels(["0.80", "0.85", "0.90", "0.95", "1.00"])
        ax.set_ylabel(ylab)
        frame(ax, "both")
        caption(ax, cap, below=9.5)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=INK2, linewidth=0.8, linestyle=":"))
    labels.append("PLAID limit (CPU)")
    top_legend(fig, handles, labels, top=TOP, x=0.55, ncol=4, columnspacing=0.8, handlelength=1.6)
    fig.savefig(out / "recall_qps.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------------------ Exp-11 context budget
def budget(panels: dict, out: Path, acc) -> None:
    """QA accuracy of the 4B reader versus the context it reads (top-3/10/30/100 units, and the whole memory)."""
    series = [("reco", "reco-K64k+R", "ReCo"), ("colbert", "colbert", "ColBERT (exact)"),
              ("dense", "bge-base", "bge-base"), ("bm25", "bm25", "BM25")]
    fig, axes = plt.subplots(1, len(panels), figsize=(COL, FLAT), sharey=True)
    fig.subplots_adjust(left=0.115, right=0.99, top=TOP, bottom=BOTTOM, wspace=0.08)
    for i, (ax, (title, rows_by_k)) in enumerate(zip(np.atleast_1d(axes), panels.items())):
        for key, name, lab in series[::-1]:
            pts = [(acc(rows_by_k[k], name)[1], acc(rows_by_k[k], name)[0]) for k in sorted(rows_by_k)
                   if acc(rows_by_k[k], name)[2]]
            plot(ax, key, [p[0] for p in pts], [p[1] for p in pts], label=lab)
        a, tok, n = acc(rows_by_k.get(10, {}), "full")
        if n:
            ax.plot([tok], [a], marker="*", markersize=7.5, color=INK, linestyle="none", zorder=6,
                    label="FullContext")
        for k in sorted(rows_by_k):
            a_c, tok_c, n_c = acc(rows_by_k[k], "reco-K64k+R")
            if n_c:
                ax.annotate(f"{k}", (tok_c, a_c), textcoords="offset points", xytext=(0, 4.5), fontsize=6,
                            color=INK2, ha="center")
        ax.margins(y=0.14)
        ax.set_xscale("log")
        frame(ax, "both")
        caption(ax, f"({chr(97 + i)}) {title}: context tokens", below=9.5)
    np.atleast_1d(axes)[0].set_ylabel("QA accuracy")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    order = [labels.index(l) for l in ("ReCo", "ColBERT (exact)", "bge-base", "BM25", "FullContext")
             if l in labels]
    top_legend(fig, [handles[j] for j in order], [labels[j] for j in order], top=TOP, x=0.56, ncol=5,
               columnspacing=0.7, handlelength=1.5)
    fig.savefig(out / "budget.pdf")
    plt.close(fig)
