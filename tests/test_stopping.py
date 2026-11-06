import math
import unittest

import numpy as np

from savemv.reco.stopping import (RULES, StopPath, adaptive_stop, conformal_quantile, crc_budget, crc_margin,
                                    evaluate_stopping, exact_topk, kfold_splits, nonconformity, split_queries,
                                    topk_ranks)


def reference_stop(shat: np.ndarray, exact: np.ndarray, tau: float, k: int, batch: int) -> int:
    """Readable reference for adaptive_stop."""
    n, p, seen = len(shat), 0, []
    while p < n:
        seen.extend(exact[p:p + batch].tolist())
        p = min(p + batch, n)
        if p >= k and p < n and sorted(seen)[-k] - shat[p] >= tau:
            break
    return p


class StoppingTest(unittest.TestCase):
    def test_adaptive_stop_matches_reference(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(200):
            n = int(rng.integers(5, 80))
            k = int(rng.integers(1, min(n, 12) + 1))
            batch = int(rng.integers(1, 9))
            exact = rng.standard_normal(n)
            shat = np.sort(exact + 0.3 * rng.standard_normal(n))[::-1].copy()
            tau = float(rng.choice([0.0, 0.2, 1.0, math.inf]))
            self.assertEqual(adaptive_stop(shat, exact, 1.0, 0.0, tau, k, batch),
                             reference_stop(shat, exact, tau, k, batch))
            a, b = float(rng.uniform(0.5, 2.0)), float(rng.normal())
            self.assertEqual(adaptive_stop(shat, exact, a, b, tau, k, batch),
                             reference_stop(a * shat + b, exact, tau, k, batch))

    def test_conformal_quantile_finite_sample(self) -> None:
        vals = np.arange(1.0, 10.0)                                   # n = 9
        self.assertEqual(conformal_quantile(vals, 0.1), 9.0)          # ceil(10 * 0.9) = 9
        self.assertEqual(conformal_quantile(vals, 0.5), 5.0)          # ceil(10 * 0.5) = 5
        self.assertTrue(math.isinf(conformal_quantile(vals, 0.05)))   # ceil(9.5) = 10 > 9

    def test_exact_topk_order(self) -> None:
        scores = np.random.default_rng(1).standard_normal((6, 50))
        np.testing.assert_array_equal(exact_topk(scores, 5), np.argsort(-scores, axis=1)[:, :5])

    def test_rank_aware_score_is_tighter(self) -> None:
        rng = np.random.default_rng(2)
        exact = rng.standard_normal((20, 100))
        est = exact + 0.5 * rng.standard_normal((20, 100))
        topk = exact_topk(exact, 10)
        s = nonconformity(est, exact, topk, np.arange(20), 1.0, 0.0)
        self.assertTrue(np.all(s["rank"] <= s["naive"] + 1e-12))

    def test_topk_ranks_match_stable_argsort(self) -> None:
        rng = np.random.default_rng(4)
        est = np.round(rng.standard_normal((7, 60)), 1)              # coarse values -> many ties
        topk = exact_topk(rng.standard_normal((7, 60)), 5)
        order = np.argsort(-est, axis=1, kind="stable")
        expected = np.array([[int(np.flatnonzero(order[q] == d)[0]) for d in topk[q]] for q in range(7)])
        np.testing.assert_array_equal(topk_ranks(est, topk), expected)

    def test_exact_topk_breaks_ties_by_doc_index(self) -> None:
        scores = np.array([[1.0, 3.0, 3.0, 2.0, 3.0, 0.0]])
        np.testing.assert_array_equal(exact_topk(scores, 4), [[1, 2, 4, 3]])

    def test_kfold_tests_every_query_once(self) -> None:
        splits = kfold_splits(53, 5, 0.4, seed=0)
        tests = np.concatenate([s.test for s in splits])
        np.testing.assert_array_equal(np.sort(tests), np.arange(53))
        for s in splits:
            self.assertEqual(len(set(s.fit) | set(s.cal) | set(s.test)), 53)
            self.assertFalse(set(s.fit) & set(s.cal))
            self.assertAlmostEqual(len(s.fit) / (len(s.fit) + len(s.cal)), 0.4, delta=0.03)

    def test_evaluate_stopping_kfold_is_consistent(self) -> None:
        rng = np.random.default_rng(5)
        nq, n_docs, k = 60, 200, 10
        exact = rng.standard_normal((nq, n_docs)).astype(np.float32)
        est = (exact + 0.3 * rng.standard_normal((nq, n_docs))).astype(np.float32)
        topk = exact_topk(exact, k)
        qids, doc_ids = [str(q) for q in range(nq)], [str(d) for d in range(n_docs)]
        qrels = {str(q): {str(int(topk[q, 0])): 1} for q in range(nq)}
        res = evaluate_stopping(est, exact, topk, kfold_splits(nq, 4, 0.4, 0), [0.1], k, 8, 50, qids, doc_ids, qrels)
        self.assertEqual(sorted(res["test_rows"]), list(range(nq)))
        self.assertEqual(set(res["deltas"]["0.1"]), set(RULES))
        rank = res["deltas"]["0.1"]["rank"]
        self.assertEqual(len(rank["R_list"]), nq)
        self.assertGreaterEqual(rank["recall"], 0.85)
        rank1 = res["deltas"]["0.1"]["rank1"]
        self.assertTrue(all(t1 <= t for t1, t in zip(rank1["param"], rank["param"])))   # k+1 margin is tighter
        self.assertLessEqual(rank1["R_mean"], rank["R_mean"])
        exact_as_est = evaluate_stopping(exact, exact, topk, kfold_splits(nq, 4, 0.4, 0), [0.1], k, 8, 50,
                                         qids, doc_ids, qrels)
        self.assertEqual(exact_as_est["estimate_recall10"], 1.0)
        self.assertEqual(exact_as_est["estimate_ndcg10"], 1.0)
        self.assertEqual(exact_as_est["R_at_recall"]["0.99"], 10)

    def test_stop_path_matches_simulator(self) -> None:
        rng = np.random.default_rng(6)
        for _ in range(100):
            n = int(rng.integers(12, 120))
            k, batch = int(rng.integers(1, 11)), int(rng.integers(1, 9))
            exact = rng.standard_normal(n)
            est = exact + 0.4 * rng.standard_normal(n)
            topk = exact_topk(exact[None], k)[0]
            path = StopPath(est, exact, topk, k, batch)
            order = np.argsort(-est, kind="stable")
            a, b = float(rng.uniform(0.5, 2)), float(rng.normal())
            gmax = path.gmax(a, b)
            for tau in (-1.0, 0.0, 0.3, 1.5, math.inf):
                self.assertEqual(path.stop(gmax, tau), adaptive_stop(est[order], exact[order], a, b, tau, k, batch))

    def test_crc_margin_and_budget_are_minimal(self) -> None:
        rng = np.random.default_rng(7)
        nq, n_docs, k, delta = 80, 150, 5, 0.1
        exact = rng.standard_normal((nq, n_docs))
        est = exact + 0.5 * rng.standard_normal((nq, n_docs))
        topk = exact_topk(exact, k)
        paths = [StopPath(est[q], exact[q], topk[q], k, 4) for q in range(nq)]
        gmaxes = [p.gmax(1.0, 0.0) for p in paths]
        for strict in (False, True):
            tau = crc_margin(paths, gmaxes, delta, strict)
            risk = lambda t: (sum(p.loss(p.stop(g, t), strict) for p, g in zip(paths, gmaxes)) + 1) / (nq + 1)
            self.assertLessEqual(risk(tau), delta)
            self.assertGreater(risk(tau - 1e-6), delta)
            ranks = np.stack([p.ranks for p in paths])
            r = crc_budget(ranks, n_docs, delta, strict)
            loss = lambda budget: ((ranks >= budget).any(1) if strict else (ranks >= budget).mean(1)).sum()
            self.assertLessEqual((loss(r) + 1) / (nq + 1), delta)
            self.assertGreater((loss(r - 1) + 1) / (nq + 1), delta)
        # pooled correction: the k top-k documents of every query are the units
        ranks = np.stack([p.ranks for p in paths])
        r = crc_budget(ranks, n_docs, delta, False, correction="doc")
        missed = lambda budget: (ranks >= budget).sum()                                   # noqa: E731
        self.assertLessEqual((missed(r) + 1) / (nq * k + 1), delta)
        self.assertGreater((missed(r - 1) + 1) / (nq * k + 1), delta)
        self.assertLessEqual(r, crc_budget(ranks, n_docs, delta, False))                 # pooled is never stricter here
        tau = crc_margin(paths, gmaxes, delta, False, correction="doc")
        self.assertLessEqual((k * sum(p.loss(p.stop(g, tau), False) for p, g in zip(paths, gmaxes)) + 1) / (nq * k + 1), delta)

    def test_crc_rule_controls_recall_loss(self) -> None:
        """E[1 - recall] <= delta for the CRC-calibrated adaptive rule on exchangeable synthetic queries."""
        rng = np.random.default_rng(8)
        nq, n_docs, k, delta = 300, 200, 5, 0.1
        exact = rng.standard_normal((nq, n_docs))
        est = exact + 0.5 * rng.standard_normal((nq, n_docs))
        topk = exact_topk(exact, k)
        paths = [StopPath(est[q], exact[q], topk[q], k, 4) for q in range(nq)]
        gmaxes = [p.gmax(1.0, 0.0) for p in paths]
        losses = []
        for seed in range(10):
            sp = split_queries(nq, [0.0, 0.5, 0.5], seed)
            tau = crc_margin([paths[q] for q in sp.cal], [gmaxes[q] for q in sp.cal], delta, strict=False)
            losses += [paths[q].loss(paths[q].stop(gmaxes[q], tau), False) for q in sp.test]
        self.assertLessEqual(np.mean(losses), delta + 0.02)

    def test_strict_rule_coverage_holds(self) -> None:
        """P[all true top-k recomputed] >= 1 - delta on exchangeable synthetic queries."""
        rng = np.random.default_rng(3)
        nq, n_docs, k, delta = 400, 300, 5, 0.1
        exact = rng.standard_normal((nq, n_docs))
        est = exact + 0.4 * rng.standard_normal((nq, n_docs))
        topk = exact_topk(exact, k)
        coverage = []
        for seed in range(10):
            sp = split_queries(nq, [0.0, 0.5, 0.5], seed)
            tau = conformal_quantile(nonconformity(est, exact, topk, sp.cal, 1.0, 0.0)["strict"], delta)
            for q in sp.test:
                order = np.argsort(-est[q])
                p = adaptive_stop(est[q, order], exact[q, order], 1.0, 0.0, tau, k, 4)
                coverage.append(set(topk[q].tolist()) <= set(order[:p].tolist()))
        self.assertGreaterEqual(np.mean(coverage), 1 - delta - 0.02)


if __name__ == "__main__":
    unittest.main()
