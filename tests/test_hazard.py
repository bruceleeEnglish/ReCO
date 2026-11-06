import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from savemv.core import assign_to_codebook, maxsim_matrix, train_spherical_kmeans
from savemv.reco import Calibration, CentroidBagEstimator, CentroidBagSketch, ReCoSearcher
from savemv.reco.hazard import FEATURES, HazardModel, HazardSettings, fit_hazard, hazard_gmax, logistic_fit
from savemv.reco.stopping import (RULES, StopPath, crc_margin, evaluate_stopping, exact_topk, kfold_splits,
                                    split_queries)
from tests.helpers import random_queries, random_store
from tests.test_engine import FakeEncoder

CPU = torch.device("cpu")


def reference_state(est, exact, k, batch, ahead, a, b):
    """Readable reference for StopPath(state=True).features: one row per batch boundary."""
    order = np.argsort(-est, kind="stable")
    e, s = est[order], exact[order]
    n, rows, best, last = len(e), [], [], -1
    for p in range(0, n, batch):
        for j in range(p, min(p + batch, n)):
            if len(best) < k:
                best.append(s[j])
                last = j
            elif s[j] > min(best):
                best[best.index(min(best))] = s[j]
                last = j
        end = min(p + batch, n)
        if end >= k and end < n:
            theta, nxt = min(best), a * e[end] + b
            offset = np.mean(s[:end] - a * e[:end])
            rows.append([theta - nxt, theta - (a * e[end] + offset), math.log(end), math.log1p(end - (last + 1)),
                         nxt - (a * e[min(end + ahead, n - 1)] + b), (a * e[0] + b) - theta, math.log(n)])
    return np.array(rows).reshape(-1, len(FEATURES))


def synthetic(nq, n_docs, k, seed, hetero=True):
    """Queries whose sketch error and score spread differ, so hazard has something to learn."""
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0.2, 1.0, size=(nq, 1)) if hetero else np.full((nq, 1), 0.5)
    exact = rng.standard_normal((nq, n_docs)) * rng.uniform(0.5, 2.0, size=(nq, 1))
    est = exact + noise * rng.standard_normal((nq, n_docs))
    return est, exact, exact_topk(exact, k)


class HazardTest(unittest.TestCase):
    def test_state_features_match_reference(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(60):
            n = int(rng.integers(15, 150))
            k, batch, ahead = int(rng.integers(1, 11)), int(rng.integers(1, 9)), int(rng.integers(1, 40))
            exact = rng.standard_normal(n)
            est = exact + 0.4 * rng.standard_normal(n)
            topk = exact_topk(exact[None], k)[0]
            path = StopPath(est, exact, topk, k, batch, state=True, ahead=ahead)
            plain = StopPath(est, exact, topk, k, batch)
            np.testing.assert_array_equal(path.ps, plain.ps)
            np.testing.assert_array_equal(path.theta, plain.theta)
            a, b = float(rng.uniform(0.5, 2)), float(rng.normal())
            np.testing.assert_allclose(path.features(a, b), reference_state(est, exact, k, batch, ahead, a, b),
                                       rtol=1e-4, atol=1e-4)

    def test_labels_and_patience(self) -> None:
        rng = np.random.default_rng(1)
        exact = rng.standard_normal(200)
        est = exact + 0.8 * rng.standard_normal(200)
        topk = exact_topk(exact[None], 10)[0]
        path = StopPath(est, exact, topk, 10, 8, state=True)
        for i, p in enumerate(path.ps):
            self.assertEqual(path.hazard_labels(32)[i], bool(np.any((path.ranks >= p) & (path.ranks < p + 32))))
        g = path.patience_gmax()
        self.assertTrue(np.all(np.diff(g) >= 0))
        self.assertTrue(np.all(path.since <= path.ps))

    def test_logistic_fit_matches_sklearn(self) -> None:
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            self.skipTest("scikit-learn not installed")
        rng = np.random.default_rng(2)
        x = rng.standard_normal((3000, 5))
        y = (x @ np.array([1.0, -2.0, 0.5, 0.0, 1.5]) + 0.3 * rng.standard_normal(3000) > 0.2).astype(float)
        w = rng.uniform(0.5, 3.0, 3000)
        coef, intercept = logistic_fit(x, y, w, l2=1.0)
        ref = LogisticRegression(C=1.0, tol=1e-10, max_iter=10000).fit(x, y, sample_weight=w)
        np.testing.assert_allclose(coef, ref.coef_[0], rtol=1e-3, atol=1e-3)
        self.assertAlmostEqual(intercept, float(ref.intercept_[0]), places=3)

    def test_model_roundtrip(self) -> None:
        est, exact, topk = synthetic(40, 120, 5, seed=3)
        paths = [StopPath(est[q], exact[q], topk[q], 5, 4, state=True) for q in range(40)]
        model = fit_hazard(paths, [(1.0, 0.0)] * 40, HazardSettings(horizon=16, ahead=16))
        with tempfile.TemporaryDirectory() as tmp:
            model.save(Path(tmp) / "m.json")
            again = HazardModel.load(Path(tmp) / "m.json")
        x = paths[0].features(1.0, 0.0)
        np.testing.assert_allclose(model.statistic(x), again.statistic(x))

    def test_crc_hazard_controls_recall_loss(self) -> None:
        """E[1 - recall] <= delta for the CRC-calibrated hazard rule on exchangeable synthetic queries."""
        nq, n_docs, k, delta = 400, 200, 5, 0.1
        est, exact, topk = synthetic(nq, n_docs, k, seed=4)
        paths = [StopPath(est[q], exact[q], topk[q], k, 4, state=True) for q in range(nq)]
        losses = []
        for seed in range(8):
            sp = split_queries(nq, [0.3, 0.35, 0.35], seed)
            model = fit_hazard([paths[q] for q in sp.fit], [(1.0, 0.0)] * len(sp.fit), HazardSettings(horizon=16))
            g = {q: hazard_gmax(model, paths[q], 1.0, 0.0) for q in np.r_[sp.cal, sp.test]}
            lam = crc_margin([paths[q] for q in sp.cal], [g[q] for q in sp.cal], delta, strict=False)
            losses += [paths[q].loss(paths[q].stop(g[q], lam), False) for q in sp.test]
        self.assertLessEqual(np.mean(losses), delta + 0.02)

    def test_evaluate_stopping_reports_new_rules(self) -> None:
        nq, n_docs, k = 60, 200, 10
        est, exact, topk = synthetic(nq, n_docs, k, seed=5)
        est, exact = est.astype(np.float32), exact.astype(np.float32)
        qids, doc_ids = [str(q) for q in range(nq)], [str(d) for d in range(n_docs)]
        qrels = {str(q): {str(int(topk[q, 0])): 1} for q in range(nq)}
        res = evaluate_stopping(est, exact, topk, kfold_splits(nq, 4, 0.4, 0), [0.1], k, 8, 50, qids, doc_ids, qrels)
        self.assertEqual(set(res["deltas"]["0.1"]), set(RULES))
        for rule in ("patience", "hazard", "hazard_doc"):
            self.assertGreaterEqual(res["deltas"]["0.1"][rule]["recall"], 0.8)
        self.assertIn("hazard", res["calibration"][0])
        universal = HazardModel.from_dict(res["calibration"][0]["hazard"])
        res_u = evaluate_stopping(est, exact, topk, kfold_splits(nq, 4, 0.4, 0), [0.1], k, 8, 50, qids, doc_ids,
                                  qrels, hazard=HazardSettings(universal=universal))
        self.assertIn("hazard_u", res_u["deltas"]["0.1"])

    def test_online_hazard_and_patience_match_offline(self) -> None:
        store = random_store(n_docs=150, dim=16, max_len=10, seed=5)
        queries = random_queries(n=8, m=4, dim=16, seed=6)
        vecs = np.asarray(store.vectors, dtype=np.float32)
        cb = train_spherical_kmeans(vecs, 32, iters=8, seed=0, device=CPU)
        ids, resid = assign_to_codebook(vecs, cb, CPU)
        estimator = CentroidBagEstimator(CentroidBagSketch.build(store.doc_of_token(), ids, resid, 32,
                                                                 store.n_docs), cb)
        exact = maxsim_matrix(store, queries, CPU)
        est = np.stack([estimator.score(q) for q in queries])
        topk = exact_topk(exact, 5)
        a, b, batch = 1.1, -0.2, 4
        paths = [StopPath(est[q], exact[q], topk[q], 5, batch, state=True, ahead=8) for q in range(len(queries))]
        model = fit_hazard(paths, [(a, b)] * len(paths), HazardSettings(horizon=8, ahead=8))
        texts = [f"d{j}" for j in range(store.n_docs)]
        for rule in ("hazard", "patience"):
            gs = [hazard_gmax(model, p, a, b) if rule == "hazard" else p.patience_gmax() for p in paths]
            values = np.unique(np.concatenate([g for g in gs if len(g)]))
            picks = (len(values) * np.array([0.2, 0.5, 0.8])).astype(int)
            for tau in (values[picks] + values[picks + 1]) / 2:     # midpoints: far from float32 rounding
                cal = Calibration(a, b, float(tau), k=5, batch=batch, rule=rule, hazard=model)
                searcher = ReCoSearcher(FakeEncoder(store, queries), estimator, texts, cal)
                for qi, (path, g) in enumerate(zip(paths, gs)):
                    self.assertEqual(searcher.search(f"q{qi}").n_recomputed, path.stop(g, float(tau)))


if __name__ == "__main__":
    unittest.main()
