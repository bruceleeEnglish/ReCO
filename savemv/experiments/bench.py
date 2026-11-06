"""Systematic benchmark on one dataset: every configured method, one evaluation protocol.

    python -m savemv bench --config configs/beir_bench.yaml --set dataset=fiqa

Results are cached per method variant in <output_dir>/bench/<name>.json (keyed by the variant's
parameters and the evaluation protocol), so an interrupted or extended run only computes what is
missing. Built artifacts (pooled stores, encodings) are cached under <output_dir>/bench/artifacts/.
"""
from __future__ import annotations

import time

from ..config import ExperimentConfig
from ..reco.stopping import evaluate_stopping, exact_topk
from .pipeline import Pipeline, hazard_settings, mask_excluded
from ..utils import get_logger, read_json, write_json
from .bench_methods import Context, build_method, storage_ratio

log = get_logger(__name__)

PROTOCOL_VERSION = 3   # v2: stop paths, rank1/strict1 and conformal risk control; v3: patience and hazard rules


class Bench:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.pipeline = Pipeline(cfg)
        self.dir = cfg.output_dir / "bench"

    def protocol(self) -> dict:
        st, cfg, enc = self.cfg.stop, self.cfg, self.cfg.encoder
        return {"version": PROTOCOL_VERSION, "encoder": {"repo": enc.repo, "query_maxlen": enc.query_maxlen,
                                                       "doc_maxlen": enc.doc_maxlen}, "k": st.k, "deltas": st.deltas, "batch": st.batch,
                "folds": st.folds, "fit_fraction": st.fit_fraction, "split": st.split, "calib_top": st.calib_top,
                "seed": st.seed, "max_queries": cfg.max_queries, "exclude_identical_ids": cfg.exclude_identical_ids,
                "hazard": {"horizon": st.hazard_horizon, "ahead": st.hazard_ahead, "l2": st.hazard_l2,
                           "model": st.hazard_model}}

    def _cached(self, name: str, variant: dict) -> dict | None:
        path = self.dir / f"{name}.json"
        if not path.exists():
            return None
        record = read_json(path)
        if record.get("variant") != variant or record.get("protocol") != self.protocol():
            return None
        return record

    def run(self) -> dict:
        cfg, pipe = self.cfg, self.pipeline
        data = pipe.load_data()
        pipe._prepare_cache()
        store = pipe.token_store(data)
        queries = pipe.query_vectors(data)
        pairs = pipe.excluded_pairs(data)
        exact = mask_excluded(pipe.exact_scores(store, queries), pairs)
        topk = exact_topk(exact, cfg.stop.k)
        splits = pipe.query_splits(len(queries))
        ctx = Context(cfg, pipe, data, store, queries, pipe.device, self.dir / "artifacts")
        ctx.artifacts.mkdir(parents=True, exist_ok=True)
        raw = data.raw_bytes_per_doc()

        records: dict[str, dict] = {}
        for spec in cfg.bench.methods:
            method = build_method(spec)
            variants = method.variants()
            cached = {n: self._cached(n, v) for n, v in variants.items()}
            if all(cached.values()):
                records.update(cached)
                log.info("%s: cached", ", ".join(variants))
                continue
            t0 = time.perf_counter()
            for out in method.run(ctx):
                if out.name not in variants:
                    raise RuntimeError(f"{type(method).__name__} produced undeclared variant {out.name}")
                built = time.perf_counter() - t0
                k_sketch = None if out.info.get("universal_codebook") else out.info.get("codebook_size")
                ev = evaluate_stopping(mask_excluded(out.est, pairs), exact, topk, splits, cfg.stop.deltas,
                                       cfg.stop.k, cfg.stop.batch, cfg.stop.calib_top, data.query_ids, data.doc_ids,
                                       data.qrels, hazard=hazard_settings(cfg.stop, cfg.dataset, k_sketch))
                record = {"name": out.name, "family": out.family, "variant": variants[out.name],
                          "protocol": self.protocol(), "bytes_per_doc": out.bytes_per_doc,
                          "shared_bytes": out.shared_bytes,
                          "total_bytes_per_doc": out.bytes_per_doc + out.shared_bytes / data.n_docs,
                          "storage_ratio": storage_ratio(out.bytes_per_doc, out.shared_bytes, data.n_docs, raw),
                          "info": out.info, "build_and_score_s": built, "evaluation": ev}
                write_json(self.dir / f"{out.name}.json", record)
                records[out.name] = record
                d = ev["deltas"][str(self._main_delta())]["rank"]
                log.info("%-16s %8.1f B/doc (%.3fx raw) | estimate nDCG %.4f R@10 %.3f R95 %d | ReCo stop: R %.1f "
                         "recall %.3f nDCG %.4f", out.name, record["total_bytes_per_doc"], record["storage_ratio"],
                         ev["estimate_ndcg10"], ev["estimate_recall10"], ev["R_at_recall"]["0.95"], d["R_mean"],
                         d["recall"], d["ndcg10"])
                del out
                t0 = time.perf_counter()

        for path in sorted(self.dir.glob("*.json")):      # keep results of methods not in this config
            if path.name != "bench_results.json" and path.stem not in records:
                record = read_json(path)
                if record.get("protocol") == self.protocol():
                    records[path.stem] = record
        report = {"dataset": {"name": cfg.dataset, "n_docs": data.n_docs, "n_queries": data.n_queries,
                              "raw_bytes_per_doc": raw, "tokens_per_doc": store.n_tokens / store.n_docs,
                              "excluded_pairs": int(len(pairs[0]))},
                  "protocol": self.protocol(), "methods": records, "config": cfg.to_dict()}
        write_json(self.dir / "bench_results.json", report)
        (self.dir / "bench_summary.md").write_text(render_bench_summary(report, self._main_delta()), encoding="utf-8")
        log.info("wrote %s", self.dir / "bench_summary.md")
        return report

    def _main_delta(self) -> float:
        return 0.05 if 0.05 in self.cfg.stop.deltas else self.cfg.stop.deltas[0]


def render_bench_summary(report: dict, delta: float) -> str:
    ds = report["dataset"]
    methods = report["methods"]
    full = methods.get("full-fp16", {}).get("evaluation", {}).get("estimate_ndcg10")
    lines = [f"# {ds['name']}: {ds['n_docs']:,} docs, {ds['n_queries']} queries, raw {ds['raw_bytes_per_doc']:.0f} B/doc",
             "", f"Recomputation columns: rank-aware conformal stop at delta={delta} (k-fold calibrated).", "",
             "| method | family | B/doc | total B/doc | /raw | est. nDCG@10 | d vs full | est. R@10 | R95 "
             "| fixed R (recall) | stop R mean / p90 | recall | nDCG@10 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in sorted(methods.items(), key=lambda kv: kv[1]["total_bytes_per_doc"]):
        ev = r["evaluation"]
        d = ev["deltas"][str(delta)]
        gap = "" if full is None else f"{100 * (ev['estimate_ndcg10'] - full):+.1f}"
        lines.append(f"| {name} | {r['family']} | {r['bytes_per_doc']:.1f} | {r['total_bytes_per_doc']:.1f} "
                     f"| {r['storage_ratio']:.3f} | {ev['estimate_ndcg10']:.4f} | {gap} | {ev['estimate_recall10']:.3f} "
                     f"| {ev['R_at_recall']['0.95']} | {d['fixed']['R_mean']:.0f} ({d['fixed']['recall']:.3f}) "
                     f"| {d['rank']['R_mean']:.1f} / {d['rank']['R_p90']:.0f} | {d['rank']['recall']:.3f} "
                     f"| {d['rank']['ndcg10']:.4f} |")
    return "\n".join(lines) + "\n"
