"""End-to-end memory QA: an LLM answers from the units each retriever returns; one judge scores all answers.

Retrievers: gold evidence (oracle), exact ColBERT, ReCo sketch alone, ReCo + recomputation (hazard rule or
fixed budget, cross-calibrated over users at `delta`), BM25, bge; optionally the whole memory ("full", on a
subset; with max_context the oldest units are dropped to fit, as a device context budget would).
Readers: an OpenAI-compatible chat API (DeepSeek by default) or a local Hugging Face chat model
(reader="hf:<path>", e.g. an on-device-class 4B model). The judge is the API model with LongMemEval's yes/no
prompts; LoCoMo also gets token F1. Retrieved units are shown in chronological order with session dates.
Every LLM call is cached on disk keyed by the full request, so reruns cost nothing.

    python -m savemv qa --bench locomo --data data/memory/locomo10.json --max-questions 400
    python -m savemv qa --bench longmemeval --data data/memory/longmemeval_s_cleaned.json \
        --reader hf:models/Qwen3-4B-Instruct-2507 --full-subset 60 --max-context 32768
"""
from __future__ import annotations

import json
import re
import string
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ..data.memory import LOADERS
from ..llm import ChatClient, LocalReader
from ..reco.hazard import HazardSettings
from ..utils import get_logger, write_json
from .memory import TOP, MemoryConfig, calibrated_stops, rerank, score_all

log = get_logger(__name__)

ANSWER_SYSTEM = ("You answer questions about past conversations. You are given excerpts retrieved from the "
                 "conversation history, each with the date of its session. Use only these excerpts. If they do "
                 "not contain the answer, say that the information is not available.")
JUDGE = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
         "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct "
         "answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If "
         "the response only contains a subset of the information required by the answer, answer no.{extra}\n\n"
         "Question: {q}\n\nCorrect Answer: {a}\n\nModel Response: {r}\n\nIs the model response correct? "
         "Answer yes or no only.")
EXTRA = {
    "temporal-reasoning": " In addition, do not penalize off-by-one errors for the number of days. If the question "
                          "asks for the number of days/weeks/months, etc., and the model makes off-by-one errors "
                          "(e.g., predicting 19 days when the answer is 18), the model's response is still correct.",
    "knowledge-update": " If the response contains some previous information along with an updated answer, the "
                        "response should be considered as correct as long as the updated answer is the required "
                        "answer.",
    "locomo": " Dates and times may be written in a different format or relative to the session date; accept them "
              "if they denote the same date or period.",
}
PREFERENCE = ("I will give you a question, a rubric for desired personalized response, and a response from a model. "
              "Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model "
              "does not need to reflect all the points in the rubric. The response is correct as long as it recalls "
              "and utilizes the user's personal information correctly.\n\nQuestion: {q}\n\nRubric: {a}\n\n"
              "Model Response: {r}\n\nIs the model response correct? Answer yes or no only.")
LOCOMO_CATEGORIES = {"1": "multi-hop", "2": "temporal", "3": "open-domain", "4": "single-hop"}


@dataclass
class QAConfig:
    bench: str
    data: str
    out_root: str = "outputs/qa"
    model: str = "deepseek-flash"          # API model: answers (reader="api") and judges
    reader: str = "api"                    # "api" or "hf:<local model dir>"
    effort: str = "high"                   # reasoning effort of the API reader
    top: int = TOP
    retrievers: list[str] = field(default_factory=lambda: ["oracle", "colbert", "reco-K64k+R", "reco-K64k",
                                                           "bge-base", "bm25"])
    stop: str = "hazard"                   # recomputation rule of the "+R" retrievers: hazard or fixed
    delta: float = 0.05
    full_subset: int = 0                   # also answer this many questions from the whole memory
    max_context: int = 0                   # token budget of "full" for a local reader (oldest units dropped)
    max_questions: int = 0
    workers: int = 32
    judge2: str | None = None              # second judge model re-scoring a random sample (agreement check)
    judge2_sample: int = 0


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    po = float(np.mean(a == b))
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def token_f1(pred: str, gold: str) -> float:
    def norm(s):
        s = "".join(ch for ch in s.lower() if ch not in string.punctuation)
        return [w for w in s.split() if w not in ("a", "an", "the", "and")]
    p, g = norm(pred), norm(gold)
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def task_info(bench_name: str, data_path: str) -> dict[str, dict]:
    """qid -> gold answer, category, question date and session dates (LoCoMo adversarial questions skipped)."""
    raw = json.loads(Path(data_path).read_text(encoding="utf-8"))
    info = {}
    if bench_name == "longmemeval":
        for item in raw:
            ids = item["haystack_session_ids"]
            info[item["question_id"]] = {"answer": str(item["answer"]), "category": item["question_type"],
                                         "now": item.get("question_date"),
                                         "dates": dict(zip(ids, item.get("haystack_dates") or [None] * len(ids)))}
    else:
        for c, sample in enumerate(raw):
            conv = sample["conversation"]
            dates = {f"{c}:{k[:-len('_date_time')]}": v for k, v in conv.items() if k.endswith("_date_time")}
            for q, qa in enumerate(sample["qa"]):
                if "answer" in qa and str(qa["category"]) != "5":
                    info[f"{sample['sample_id']}-{q}"] = {"answer": str(qa["answer"]), "category": str(qa["category"]),
                                                         "now": None, "dates": dates,
                                                         "speakers": (conv["speaker_a"], conv["speaker_b"])}
    return info


def _chronological(bench_name: str, session: str, date: str, unit: int):
    if bench_name == "longmemeval":
        return (re.sub(r"\s*\([A-Za-z]+\)\s*", " ", date), unit)
    try:
        return (datetime.strptime(date, "%I:%M %p on %d %B, %Y").isoformat(), unit)
    except ValueError:
        return (session, unit)


def run_qa(cfg: QAConfig) -> dict:
    root = Path(cfg.out_root) / cfg.bench
    out = root if cfg.reader == "api" else root / Path(cfg.reader[3:]).name
    out.mkdir(parents=True, exist_ok=True)
    bench = LOADERS[cfg.bench](cfg.data)
    info = task_info(cfg.bench, cfg.data)
    keep = [i for i, t in enumerate(bench.tasks) if t.qid in info]
    if cfg.max_questions:
        keep = [int(i) for i in np.random.default_rng(0).permutation(keep)[:cfg.max_questions]]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scores, _, _, _, _, _ = score_all(MemoryConfig(cfg.bench, cfg.data), bench, Path(f"outputs/memory/{cfg.bench}"),
                                      device)
    tasks = bench.tasks
    orders = {n: [t.units[np.argsort(-s, kind="stable")] for t, s in zip(tasks, rows)] for n, rows in scores.items()}
    exact_maps = [dict(zip(t.units.tolist(), e.tolist())) for t, e in zip(tasks, scores["colbert"])]
    stops_used = {}
    for name in cfg.retrievers:
        if name.endswith("+R"):                     # sketch order, first R units recomputed (R by the stop rule)
            base = name[:-2]
            stops = calibrated_stops(scores[base], scores["colbert"], cfg.delta, HazardSettings(horizon=32, ahead=32))
            st = stops[cfg.stop]
            orders[name] = [rerank(o, m, int(s)) for o, m, s in zip(orders[base], exact_maps, st)]
            stops_used[name] = {"rule": cfg.stop, "R_mean": float(st.mean()), "R_p90": float(np.percentile(st, 90))}
            log.info("%s: %s rule recomputes %.1f units on average", name, cfg.stop, st.mean())
    orders["oracle"] = [t.relevant for t in tasks]

    answerer = (ChatClient(cfg.model, out / "cache.jsonl", reasoning_effort=cfg.effort) if cfg.reader == "api"
                else LocalReader(cfg.reader[3:], out / "cache.jsonl"))
    judge = ChatClient(cfg.model, root / "cache.jsonl", reasoning_effort="low", temperature=0)
    jobs = [(name, i) for name in cfg.retrievers for i in keep] + [("full", i) for i in keep[:cfg.full_subset]]
    judge2 = ChatClient(cfg.judge2, root / "cache.jsonl", reasoning_effort="low", temperature=0) if cfg.judge2 else None
    sampled = set(np.random.default_rng(1).choice(len(jobs), min(cfg.judge2_sample, len(jobs)), replace=False).tolist()
                  ) if judge2 else set()

    def run(indexed):
        j, (name, i) = indexed
        t, inf = tasks[i], info[tasks[i].qid]
        units = t.units if name == "full" else orders[name][i] if name == "oracle" else orders[name][i][:cfg.top]
        units = sorted((int(u) for u in units), key=lambda u: _chronological(
            cfg.bench, bench.sessions[u], inf["dates"].get(bench.sessions[u], ""), u))
        blocks = [f"(session date: {inf['dates'].get(bench.sessions[u], 'unknown')})\n{bench.texts[u]}" for u in units]
        if name == "full" and cfg.max_context and isinstance(answerer, LocalReader):
            sizes = [answerer.n_tokens(b) + 4 for b in blocks]       # keep the most recent units that fit
            start = len(blocks)
            while start > 0 and sum(sizes[start - 1:]) <= cfg.max_context:
                start -= 1
            blocks = blocks[start:]
        ctx = "\n\n".join(f"[{n + 1}] {b}" for n, b in enumerate(blocks))
        who = (f"The history is a conversation between {inf['speakers'][0]} and {inf['speakers'][1]}.\n\n"
               if "speakers" in inf else "")
        now = f"Current date: {inf['now']}\n" if inf["now"] else ""
        ans = answerer.chat([{"role": "system", "content": ANSWER_SYSTEM},
                             {"role": "user", "content": f"{who}Retrieved excerpts:\n\n{ctx}\n\n{now}Question: {t.query}"
                                                         f"\nAnswer concisely."}], max_tokens=8000)
        if inf["category"] == "single-session-preference":
            prompt = PREFERENCE.format(q=t.query, a=inf["answer"], r=ans["content"])
        else:
            extra = EXTRA.get(inf["category"], EXTRA["locomo"] if cfg.bench == "locomo" else "")
            prompt = JUDGE.format(extra=extra, q=t.query, a=inf["answer"], r=ans["content"])
        verdict = judge.chat([{"role": "user", "content": prompt}], max_tokens=2000)
        second = None
        if j in sampled:
            second = judge2.chat([{"role": "user", "content": prompt}], max_tokens=2000)["content"].strip().lower()
            second = second.startswith("yes")
        return (name, i, verdict["content"].strip().lower().startswith("yes"), token_f1(ans["content"], inf["answer"]),
                ans["usage"].get("prompt_tokens", 0)), second

    with ThreadPoolExecutor(cfg.workers) as pool:
        results = list(pool.map(run, enumerate(jobs)))
    rows = [r for r, _ in results]
    checked = [(r, s) for r, s in results if s is not None]

    cat = (lambda c: LOCOMO_CATEGORIES.get(c, c)) if cfg.bench == "locomo" else (lambda c: c)
    report = {"config": cfg.__dict__, "n_questions": len(keep), "recompute": stops_used, "retrievers": {}}
    base = {i: ok for n, i, ok, _, _ in rows if n == "colbert"}
    for name in cfg.retrievers + (["full"] if cfg.full_subset else []):
        mine = [r for r in rows if r[0] == name]
        by_cat = defaultdict(list)
        for _, i, ok, _, _ in mine:
            by_cat[cat(info[tasks[i].qid]["category"])].append(ok)
        entry = {"n": len(mine), "accuracy": float(np.mean([r[2] for r in mine])), "f1": float(np.mean([r[3] for r in mine])),
                 "prompt_tokens": float(np.mean([r[4] for r in mine])),
                 "by_category": {c: float(np.mean(v)) for c, v in sorted(by_cat.items())}}
        pairs = np.array([(ok, base[i]) for n, i, ok, _, _ in rows if n == name and i in base], dtype=float)
        if name != "colbert" and len(pairs) > 1:
            diff = pairs[:, 0] - pairs[:, 1]
            entry["vs_colbert"] = {"diff": float(diff.mean()), "ci95": float(1.96 * diff.std(ddof=1) / np.sqrt(len(diff)))}
        report["retrievers"][name] = entry
        log.info("%-14s n=%4d acc %.3f F1 %.3f ctx %6.0f tok%s", name, entry["n"], entry["accuracy"], entry["f1"],
                 entry["prompt_tokens"], "" if "vs_colbert" not in entry else
                 " | vs ColBERT %+.3f +- %.3f" % (entry["vs_colbert"]["diff"], entry["vs_colbert"]["ci95"]))
    if checked:
        a, b = [r[2] for r, _ in checked], [s for _, s in checked]
        by_ret = defaultdict(lambda: ([], []))
        for r, s in checked:
            by_ret[r[0]][0].append(r[2])
            by_ret[r[0]][1].append(s)
        report["judge_check"] = {"judge2": cfg.judge2, "n": len(checked), "agreement": float(np.mean(np.array(a) == np.array(b))),
                                 "kappa": cohen_kappa(a, b),
                                 "accuracy_by_retriever": {n: {"judge1": float(np.mean(x)), "judge2": float(np.mean(y)),
                                                               "n": len(x)} for n, (x, y) in sorted(by_ret.items())}}
        log.info("judge check (%s, n=%d): agreement %.3f, kappa %.3f", cfg.judge2, len(checked),
                 report["judge_check"]["agreement"], report["judge_check"]["kappa"])
    tag = f"top{cfg.top}" + (f"_full{cfg.max_context or ''}" if cfg.full_subset else "")
    write_json(out / f"qa_report_{tag}.json", report)
    write_json(out / f"qa_rows_{tag}.json", [{"retriever": n, "qid": tasks[i].qid, "category": cat(info[tasks[i].qid]["category"]),
                                              "correct": bool(ok), "f1": f, "prompt_tokens": p} for n, i, ok, f, p in rows])
    log.info("wrote %s", out / f"qa_report_{tag}.json")
    return report
