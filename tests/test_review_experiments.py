"""Checks of the optimality-gap and growth experiments (scripts/stopping/optimality_gap.py, scripts/stopping/growth.py)."""
import itertools
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

for folder in ("stopping", "official_baselines"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / folder))

from growth import calibrate_fold, summary, test_fold as run_fold  # noqa: E402
from optimality_gap import (build_paths, clairvoyant_bound, fit_hazard_fold, rank_correlation, reliability,  # noqa: E402
                            useful_stops)
from savemv.reco.stopping import StopPath, crc_margin, exact_topk, kfold_splits  # noqa: E402


def config(k: int = 3, batch: int = 2) -> SimpleNamespace:
    return SimpleNamespace(stop=SimpleNamespace(k=k, batch=batch, hazard_ahead=8, hazard_l2=1.0, calib_top=20))


def scores(n_q: int, n_d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    exact = rng.normal(size=(n_q, n_d)).astype(np.float32)
    return (exact + 0.5 * rng.normal(size=exact.shape)).astype(np.float32), exact


class SkippedQueries(unittest.TestCase):
    def test_a_failed_query_is_skipped_and_the_rest_keep_their_rows(self):
        from plaid_sweep_skip import run_point

        class Oom(Exception):
            pass

        released = []

        def search(q):
            if q == "too deep":
                raise Oom
            return [len(q)]
        ms, rankings, rows = run_point(search, ["a", "too deep", "ccc"], lambda: None, Oom, lambda: released.append(1))
        self.assertEqual(rows, [0, 2])
        self.assertEqual([r.tolist() for r in rankings], [[1], [3]])
        self.assertEqual(len(ms), 2)
        self.assertEqual(released, [1])


class UsefulStops(unittest.TestCase):
    def test_every_boundary_is_dominated(self):
        est, exact = scores(5, 60, 0)
        cfg = config()
        topk = exact_topk(exact, cfg.stop.k)
        for path in build_paths(est, exact, topk, cfg):
            stops, losses = useful_stops(path, cfg.stop.batch)
            self.assertTrue((np.diff(stops) >= 0).all())
            self.assertAlmostEqual(losses[-1], 0.0)
            for p in list(path.ps) + [path.n]:                       # any stop the rule could take
                loss = float((path.ranks >= p).mean())
                self.assertTrue(((stops <= p) & (losses <= loss + 1e-12)).any(), (p, stops, losses))

    def test_losses_follow_the_path(self):
        est, exact = scores(3, 40, 1)
        cfg = config()
        for path in build_paths(est, exact, exact_topk(exact, cfg.stop.k), cfg):
            stops, losses = useful_stops(path, cfg.stop.batch)
            for s, l in zip(stops, losses):
                self.assertAlmostEqual(l, path.loss(int(s), strict=False))


class Bound(unittest.TestCase):
    def brute(self, stops: np.ndarray, losses: np.ndarray, delta: float) -> float:
        n = len(stops)
        best = np.inf
        for pick in itertools.product(range(stops.shape[1]), repeat=n):
            idx = np.arange(n), np.array(pick)
            if losses[idx].mean() <= delta + 1e-12:
                best = min(best, stops[idx].mean())
        return best

    def test_against_integer_optimum_and_lp(self):
        from scipy.optimize import linprog
        for seed in range(8):
            rng = np.random.default_rng(seed)
            n, m = 5, 4
            stops = np.sort(rng.integers(1, 60, (n, m)), axis=1).astype(np.float64)
            losses = np.tile([1.0, 2 / 3, 1 / 3, 0.0], (n, 1))
            for delta in (0.0, 0.1, 0.3):
                lb = clairvoyant_bound(stops, losses, delta)
                self.assertLessEqual(lb, self.brute(stops, losses, delta) + 1e-6)
                c = stops.ravel() / n
                a_ub = [losses.ravel() / n]
                a_eq = np.zeros((n, n * m))
                for q in range(n):
                    a_eq[q, q * m:(q + 1) * m] = 1.0
                lp = linprog(c, A_ub=a_ub, b_ub=[delta], A_eq=a_eq, b_eq=np.ones(n), bounds=(0, 1))
                self.assertAlmostEqual(lb, lp.fun, places=5)

    def test_zero_budget_recomputes_every_top_document(self):
        stops = np.array([[2.0, 8.0], [2.0, 20.0]])
        losses = np.array([[0.5, 0.0], [0.5, 0.0]])
        self.assertAlmostEqual(clairvoyant_bound(stops, losses, 0.0), 14.0)
        self.assertAlmostEqual(clairvoyant_bound(stops, losses, 0.5), 2.0)

    def test_budget_goes_where_it_saves_most(self):
        stops = np.array([[2.0, 8.0], [2.0, 20.0]])
        losses = np.array([[0.5, 0.0], [0.5, 0.0]])
        self.assertAlmostEqual(clairvoyant_bound(stops, losses, 0.25), 5.0)   # query 2 stops early, query 1 does not


class Reliability(unittest.TestCase):
    def test_perfect_order(self):
        g = np.linspace(-3, 3, 2000)
        res = reliability(g, np.exp(-g))
        self.assertAlmostEqual(res["rank_correlation"], -1.0)
        self.assertEqual(res["inversions"], 0)
        self.assertEqual(res["bins"], 20)

    def test_ties_do_not_invent_an_order(self):
        g = np.arange(8.0)
        hazard = np.array([0.9, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])    # many bins without any hit, as deep in a path
        self.assertLess(rank_correlation(g, hazard), -0.8)

    def test_no_order(self):
        rng = np.random.default_rng(0)
        self.assertLess(abs(rank_correlation(rng.normal(size=5000), rng.normal(size=5000))), 0.1)


class Fold(unittest.TestCase):
    def setup_fold(self, seed=0):
        est, exact = scores(40, 300, seed)
        cfg = config(k=3, batch=2)
        paths = build_paths(est, exact, exact_topk(exact, 3), cfg)
        splits = kfold_splits(40, 5, 0.4, 0)
        return est, exact, cfg, paths, splits

    def test_statistic_is_the_hazard_rule(self):
        est, exact, cfg, paths, splits = self.setup_fold()
        sp = splits[0]
        ab, raw = fit_hazard_fold(sp, est, exact, paths, cfg, np.random.default_rng(0))
        self.assertEqual(len(ab), 2)
        g = raw(paths[int(sp.test[0])])
        self.assertEqual(len(g), len(paths[int(sp.test[0])].ps))
        self.assertTrue(np.isfinite(g).all())
        gmax = [np.maximum.accumulate(raw(paths[q])) for q in sp.cal]
        lam = crc_margin([paths[q] for q in sp.cal], gmax, 0.1, strict=False)
        self.assertTrue(np.isfinite(lam))

    def test_frozen_rules_equal_recalibrated_at_the_same_size(self):
        est, exact, cfg, paths, splits = self.setup_fold(1)
        deltas = [0.1]
        fold_a = calibrate_fold(splits[0], est, exact, paths, cfg, np.random.default_rng(0), deltas)
        fold_b = calibrate_fold(splits[0], est, exact, paths, cfg, np.random.default_rng(0), deltas)
        a = {(r, d): [] for r in ("hazard", "margin", "fixed") for d in deltas}
        b = {(r, d): [] for r in ("hazard", "margin", "fixed") for d in deltas}
        run_fold(fold_a, paths, deltas, a)
        run_fold(fold_b, paths, deltas, b)
        self.assertEqual(a, b)
        self.assertEqual(set(summary(a, deltas)["0.1"]), {"hazard", "margin", "fixed"})

    def test_grown_corpus_uses_its_own_size_feature(self):
        est, exact, cfg, paths, splits = self.setup_fold(2)
        sp = splits[0]
        _, raw = fit_hazard_fold(sp, est, exact, paths, cfg, np.random.default_rng(0))
        q = int(sp.test[0])
        keep = np.arange(150)
        small = StopPath(est[q, keep], exact[q, keep], exact_topk(exact[None, q, keep], 3)[0], 3, 2, state=True, ahead=8)
        self.assertNotEqual(small.n, paths[q].n)
        self.assertEqual(len(raw(small)), len(small.ps))


if __name__ == "__main__":
    unittest.main()
