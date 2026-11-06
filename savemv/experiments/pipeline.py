"""End-to-end experiment: encode -> exact ground truth -> sketches -> stopping evaluation -> latency.

Every expensive artifact is cached under `cfg.output_dir`, keyed by the encoder settings, so
re-running with different evaluation parameters reuses the encodings and ground truth.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..config import ExperimentConfig, StopConfig
from ..core.codebook import assign_to_codebook, train_spherical_kmeans, training_sample
from ..core.exact import maxsim_matrix
from ..core.metrics import ndcg_at_k, recall_vs_exact
from ..core.store import TokenStore
from ..data import RetrievalDataset, load_dataset
from ..encoders import ColBERTEncoder
from ..reco.engine import Calibration, ReCoSearcher
from ..reco.estimate import CentroidBagEstimator, TorchCentroidBagEstimator
from ..reco.hazard import HazardModel, HazardSettings
from ..reco.sketch import CentroidBagSketch
from ..reco.stopping import QuerySplit, evaluate_stopping, exact_topk, kfold_splits, split_queries
from ..utils import Timer, get_logger, read_json, resolve_device, write_json

log = get_logger(__name__)

EXCLUDED_SCORE = -1e9   # finite "minus infinity" for masked (query, document) pairs


def hazard_settings(st: StopConfig, dataset: str, k: int | None = None) -> HazardSettings:
    """Hazard settings of a run; the universal model (rule hazard_u) only for ReCo sketches with a model file."""
    universal = None
    if st.hazard_model and k is not None:
        path = Path(st.hazard_model.format(dataset=dataset, k=k))
        if path.exists():
            universal = HazardModel.load(path)
        else:
            log.warning("no universal hazard model at %s; rule hazard_u skipped", path)
    return HazardSettings(st.hazard_horizon, st.hazard_ahead, st.hazard_l2, seed=st.seed, universal=universal)


def mask_excluded(scores: np.ndarray, pairs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Copy of a (n_queries, n_docs) score matrix with the excluded pairs pushed to the bottom."""
    rows, cols = pairs
    if len(rows) == 0:
        return scores
    out = np.array(scores, dtype=np.float32)
    out[rows, cols] = EXCLUDED_SCORE
    return out


class Pipeline:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.out = cfg.output_dir
        self.device = resolve_device(cfg.encoder.device)
        self._encoder: ColBERTEncoder | None = None
        self.timings: dict[str, float] = {}

    # ------------------------------------------------------------ helpers
    @property
    def encoder(self) -> ColBERTEncoder:
        if self._encoder is None:
            self._encoder = ColBERTEncoder(self.cfg.encoder)
        return self._encoder

    def _cache_key(self) -> dict:
        e = self.cfg.encoder
        precision = "fp16-weights" if e.fp16 and self.device.type == "cuda" else "fp32"
        return {"dataset": self.cfg.dataset, "split": self.cfg.split, "max_docs": self.cfg.max_docs,
                "max_queries": self.cfg.max_queries, "repo": e.repo, "query_maxlen": e.query_maxlen, "doc_maxlen": e.doc_maxlen,
                "precision": precision, "encoder_impl": 2}

    def _prepare_cache(self) -> None:
        """Invalidate cached artifacts when the encoder or dataset settings changed."""
        meta = self.out / "cache_key.json"
        if meta.exists() and read_json(meta) != self._cache_key():
            log.warning("cache key changed; clearing cached artifacts in %s", self.out)
            for f in self.out.glob("*.np*"):
                f.unlink()
            for f in (self.out / "tokens").glob("*.npy"):
                f.unlink()
        write_json(meta, self._cache_key())

    @contextmanager
    def _timed(self, name: str) -> Iterator[Timer]:
        """Time a stage (CUDA-synchronized) and accumulate it into `self.timings[name]`."""
        with Timer(self.device) as timer:
            yield timer
        self.timings[name] = self.timings.get(name, 0.0) + timer.seconds

    # ------------------------------------------------------------- stages
    def load_data(self) -> RetrievalDataset:
        cfg = self.cfg
        return load_dataset(cfg.dataset, cfg.data_root, cfg.split, cfg.hf_endpoint, cfg.max_docs, cfg.max_queries,
                            cfg.stop.seed)

    def excluded_pairs(self, data: RetrievalDataset) -> tuple[np.ndarray, np.ndarray]:
        if not self.cfg.exclude_identical_ids:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        pairs = data.identical_pairs()
        if len(pairs[0]):
            log.info("excluding %d (query, document) pairs with identical ids", len(pairs[0]))
        return pairs

    def query_splits(self, n_queries: int) -> list[QuerySplit]:
        st = self.cfg.stop
        if st.folds > 1:
            return kfold_splits(n_queries, st.folds, st.fit_fraction, st.seed)
        return [split_queries(n_queries, st.split, st.seed)]

    def token_store(self, data: RetrievalDataset) -> TokenStore:
        directory = self.out / "tokens"
        if TokenStore.exists(directory):
            return TokenStore.load(directory)
        with self._timed("encode_corpus"):
            store = self.encoder.encode_corpus(data.doc_texts)
        store.save(directory)
        log.info("corpus: %d docs, %d tokens (%.1f / doc) in %.1fs", store.n_docs, store.n_tokens,
                 store.n_tokens / store.n_docs, self.timings["encode_corpus"])
        return TokenStore.load(directory)

    def query_vectors(self, data: RetrievalDataset) -> np.ndarray:
        path = self.out / "queries.npy"
        if path.exists():
            return np.load(path)
        with self._timed("encode_queries"):
            q = self.encoder.encode_queries(data.query_texts)
        np.save(path, q)
        return q

    def exact_scores(self, store: TokenStore, queries: np.ndarray) -> np.ndarray:
        path = self.out / "exact_scores.npy"
        if path.exists():
            return np.load(path)
        with self._timed("exact_scores"):
            scores = maxsim_matrix(store, queries, self.device)
        np.save(path, scores)
        log.info("exact MaxSim for %d queries x %d docs in %.1fs", *scores.shape, self.timings["exact_scores"])
        return scores

    def sketch(self, store: TokenStore, k: int) -> tuple[CentroidBagSketch, np.ndarray]:
        sk_path, cb_path = self.out / f"sketch_K{k}.npz", self.out / f"codebook_K{k}.npy"
        if sk_path.exists() and cb_path.exists():
            return CentroidBagSketch.load(sk_path), np.load(cb_path)
        sc = self.cfg.sketch
        with self._timed(f"codebook_K{k}"):
            rows = training_sample(store.n_tokens, k, sc.points_per_centroid, sc.min_train_points, sc.seed)
            centroids = train_spherical_kmeans(np.asarray(store.vectors[rows]), k, sc.kmeans_iters, sc.seed,
                                               self.device)
            ids, resid = assign_to_codebook(store.vectors, centroids, self.device)
            sketch = CentroidBagSketch.build(store.doc_of_token(), ids, resid, k, store.n_docs)
        sketch.save(sk_path)
        np.save(cb_path, centroids)
        log.info("sketch K=%d: %.1f centroids / doc, %.1f bytes / doc", k, sketch.sizes.mean(),
                 sketch.bytes_per_doc().mean())
        return sketch, centroids

    # --------------------------------------------------------------- main
    def run(self) -> dict:
        cfg = self.cfg
        data = self.load_data()
        self._prepare_cache()
        store = self.token_store(data)
        queries = self.query_vectors(data)
        pairs = self.excluded_pairs(data)
        exact = mask_excluded(self.exact_scores(store, queries), pairs)
        topk = exact_topk(exact, cfg.stop.k)
        splits = self.query_splits(len(queries))
        test_rows = np.concatenate([s.test for s in splits])

        raw = data.raw_bytes_per_doc()
        tokens_per_doc = store.n_tokens / store.n_docs
        dim = store.dim
        report: dict = {
            "dataset": {"name": cfg.dataset, "n_docs": data.n_docs, "n_queries": data.n_queries,
                        "n_test_queries": len(test_rows), "raw_bytes_per_doc": raw,
                        "tokens_per_doc": tokens_per_doc},
            "reference_storage_ratio": {
                "ColBERT fp16": tokens_per_doc * dim * 2 / raw,
                "ColBERTv2 2-bit": tokens_per_doc * (dim / 4 + 4) / raw,
                "ColBERTv2 1-bit": tokens_per_doc * (dim / 8 + 4) / raw},
            "full_colbert": {
                "ndcg10_all": ndcg_at_k(topk, data.query_ids, data.doc_ids, data.qrels, 10),
                "ndcg10_test": ndcg_at_k(topk[test_rows], [data.query_ids[i] for i in test_rows],
                                         data.doc_ids, data.qrels, 10)},
            "sketches": {},
            "config": cfg.to_dict(),
        }
        log.info("full ColBERT nDCG@10: all=%.4f test=%.4f", report["full_colbert"]["ndcg10_all"],
                 report["full_colbert"]["ndcg10_test"])

        for k in cfg.sketch.codebook_sizes:
            sketch, centroids = self.sketch(store, k)
            estimator = CentroidBagEstimator(sketch, centroids)
            estimator.score(queries[0])  # compile the numba kernel outside the timed region
            with Timer() as t:
                est = mask_excluded(estimator.score_many(queries), pairs)
            res = evaluate_stopping(est, exact, topk, splits, cfg.stop.deltas, cfg.stop.k, cfg.stop.batch,
                                    cfg.stop.calib_top, data.query_ids, data.doc_ids, data.qrels,
                                    hazard=hazard_settings(cfg.stop, cfg.dataset, k))
            res.update({"bytes_per_doc": float(sketch.bytes_per_doc().mean()),
                        "storage_ratio": float(sketch.bytes_per_doc().mean() / raw),
                        "centroids_per_doc": float(sketch.sizes.mean()),
                        "estimate_ms_per_query": t.ms / len(queries)})
            report["sketches"][str(k)] = res
            d = res["deltas"][str(0.05 if 0.05 in cfg.stop.deltas else cfg.stop.deltas[0])]
            log.info("K=%d storage=%.3fx | fixed R=%.0f | rank-aware R=%.1f recall=%.3f nDCG=%.4f | strict R=%.1f",
                     k, res["storage_ratio"], d["fixed"]["R_mean"], d["rank"]["R_mean"], d["rank"]["recall"],
                     d["rank"]["ndcg10"], d["strict"]["R_mean"])

        if cfg.latency.enabled:
            report["latency"] = self.latency(data, store, exact, topk, report)
        report["timings_s"] = self.timings
        write_json(self.out / "results.json", report)
        (self.out / "summary.md").write_text(render_summary(report), encoding="utf-8")
        log.info("wrote %s", self.out / "summary.md")
        return report

    def latency(self, data: RetrievalDataset, store: TokenStore, exact: np.ndarray, topk: np.ndarray,
                report: dict) -> dict:
        """Real recomputation from raw text: per-query timings, recall and score consistency.

        Each query uses the calibration (a, b, tau) of the fold that tests it. Excluded (identical-id)
        documents are not masked online, so run this on datasets without them.
        """
        lc, k = self.cfg.latency, self.cfg.stop.k
        sketch, centroids = self.sketch(store, lc.codebook_size)
        entry = report["sketches"].get(str(lc.codebook_size))
        if entry is None:
            raise ValueError("latency.codebook_size must be one of sketch.codebook_sizes")
        fixed = lc.rule in ("fixed", "fixed_strict")
        kind = "hazard" if lc.rule in ("hazard", "hazard_doc") else "patience" if lc.rule == "patience" else "margin"
        fold_cal = [(c["a"], c["b"], c["fixed_R" if fixed else "tau"][str(lc.delta)][lc.rule],
                     HazardModel.from_dict(c["hazard"]) if kind == "hazard" else None)
                    for c in entry["calibration"]]
        test_rows = np.asarray(entry["test_rows"])
        fold_of = np.repeat(np.arange(len(fold_cal)), [len(s.test) for s in self.query_splits(len(exact))])
        on_gpu = self.device.type == "cuda"
        estimator = (TorchCentroidBagEstimator(sketch, centroids, self.device) if on_gpu
                     else CentroidBagEstimator(sketch, centroids))
        picks = np.arange(min(lc.num_queries, len(test_rows)))
        out = {"codebook_size": lc.codebook_size, "delta": lc.delta, "rule": lc.rule,
               "param": [t for _, _, t, _ in fold_cal], "device": str(self.device),
               "estimator": type(estimator).__name__, "batches": {}}
        for batch in lc.batches:
            def calibration(i: int) -> Calibration:
                a, b, param, model = fold_cal[fold_of[i]]
                if fixed:
                    return Calibration(a, b, float("inf"), k, batch, budget=int(param))
                return Calibration(a, b, param, k, batch, rule=kind, hazard=model)

            searcher = ReCoSearcher(self.encoder, estimator, data.doc_texts, calibration(0))
            for i in picks[:3]:
                searcher.cal = calibration(i)
                searcher.search(data.query_texts[test_rows[i]])   # warm-up
            stats = {name: [] for name in ("total", "encode_query", "estimate", "recompute")}
            n_rec, recalls, diffs = [], [], []
            for i in picks:
                q = test_rows[i]
                searcher.cal = calibration(i)
                r = searcher.search(data.query_texts[q])
                for name in stats:
                    stats[name].append(r.timings_ms[name])
                n_rec.append(r.n_recomputed)
                recalls.append(recall_vs_exact(r.doc_indices, topk[q]))
                diffs.append(float(np.abs(r.recomputed_scores - exact[q, r.recomputed]).max()))
            total = np.array(stats["total"])
            # offline simulation used the same split order; it only matches exactly when batch == stop.batch
            offline = np.asarray(entry["deltas"][str(lc.delta)][lc.rule]["R_list"])[picks]
            same_r = float(np.mean(np.array(n_rec) == np.array(offline))) if batch == self.cfg.stop.batch else None
            out["batches"][str(batch)] = {
                "recomputed_matches_offline": same_r,
                "ms_p50": float(np.percentile(total, 50)), "ms_p90": float(np.percentile(total, 90)),
                "ms_p99": float(np.percentile(total, 99)),
                "ms_mean_by_stage": {name: float(np.mean(v)) for name, v in stats.items()},
                "recomputed_mean": float(np.mean(n_rec)), "recall_mean": float(np.mean(recalls)),
                "ms_per_recomputed_doc": float(np.sum(stats["recompute"]) / max(1, np.sum(n_rec))),
                "max_abs_score_diff_vs_stored": float(np.max(diffs)),
            }
            b = out["batches"][str(batch)]
            log.info("latency batch=%d: p50=%.1f p90=%.1f ms | recomputed %.1f docs (%.2f ms/doc) | recall %.3f",
                     batch, b["ms_p50"], b["ms_p90"], b["recomputed_mean"], b["ms_per_recomputed_doc"],
                     b["recall_mean"])
        return out


def render_summary(report: dict) -> str:
    """Markdown summary of a results report."""
    ds, full = report["dataset"], report["full_colbert"]
    lines = [f"# {ds['name']}: {ds['n_docs']:,} docs, {ds['n_queries']} queries ({ds['n_test_queries']} test)", "",
             f"- raw text: {ds['raw_bytes_per_doc']:.0f} B/doc, {ds['tokens_per_doc']:.1f} tokens/doc",
             f"- full ColBERT nDCG@10: all {full['ndcg10_all']:.4f}, test {full['ndcg10_test']:.4f}",
             "- reference index size / raw text: " + ", ".join(
                 f"{k} {v:.2f}x" for k, v in report["reference_storage_ratio"].items()), "",
             "| K | storage/raw | B/doc | delta | fixed R (recall) | rank-aware R mean / p90 (recall) "
             "| strict R (full-recall rate) | nDCG@10 | estimate-only nDCG |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k, s in report["sketches"].items():
        for delta, d in s["deltas"].items():
            lines.append(
                f"| {k} | {s['storage_ratio']:.3f} | {s['bytes_per_doc']:.0f} | {delta} "
                f"| {d['fixed']['R_mean']:.0f} ({d['fixed']['recall']:.3f}) "
                f"| {d['rank']['R_mean']:.1f} / {d['rank']['R_p90']:.0f} ({d['rank']['recall']:.3f}) "
                f"| {d['strict']['R_mean']:.1f} ({d['strict']['full_recall_rate']:.3f}) "
                f"| {d['rank']['ndcg10']:.4f} | {s['estimate_ndcg10']:.4f} |")
    lat = report.get("latency")
    if lat:
        lines += ["", f"## Online latency (K={lat['codebook_size']}, delta={lat['delta']}, device={lat['device']})", "",
                  "| batch | p50 ms | p90 ms | p99 ms | recomputed docs | ms / doc | recall | max score diff "
                  "| R = offline |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for batch, b in lat["batches"].items():
            match = "-" if b["recomputed_matches_offline"] is None else f"{b['recomputed_matches_offline']:.2f}"
            lines.append(f"| {batch} | {b['ms_p50']:.1f} | {b['ms_p90']:.1f} | {b['ms_p99']:.1f} "
                         f"| {b['recomputed_mean']:.1f} | {b['ms_per_recomputed_doc']:.2f} | {b['recall_mean']:.3f} "
                         f"| {b['max_abs_score_diff_vs_stored']:.4f} | {match} |")
    timings = report.get("timings_s", {})
    if timings:
        lines += ["", "Stage timings (s): " + ", ".join(f"{k} {v:.1f}" for k, v in timings.items())]
    return "\n".join(lines) + "\n"
