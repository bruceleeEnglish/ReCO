import math
import unittest

import numpy as np
import torch

from savemv.core import assign_to_codebook, train_spherical_kmeans
from savemv.reco import CentroidBagEstimator, CentroidBagSketch, TorchCentroidBagEstimator
from tests.helpers import random_queries, random_store, unit

CPU = torch.device("cpu")


class SketchTest(unittest.TestCase):
    def test_build_matches_python_sets(self) -> None:
        rng = np.random.default_rng(0)
        n_docs, k = 40, 32
        doc_of_token = np.sort(rng.integers(0, n_docs, 600))
        ids = rng.integers(0, k, 600).astype(np.int32)
        resid = rng.random(600).astype(np.float32)
        sk = CentroidBagSketch.build(doc_of_token, ids, resid, k, n_docs)
        for d in range(n_docs):
            mask = doc_of_token == d
            got = sk.cent[sk.ptr[d]:sk.ptr[d + 1]]
            self.assertEqual(list(got), sorted(set(ids[mask].tolist())))
            for j, c in enumerate(got):
                self.assertAlmostEqual(float(sk.rmax[sk.ptr[d] + j]), float(resid[mask & (ids == c)].max()))

    def test_bytes_per_doc_uses_smaller_encoding(self) -> None:
        sk = CentroidBagSketch(256, np.array([0, 3, 203]), np.zeros(203, np.int32), None)
        # 3 ids * 8 bits = 24 bits < 256-bit bitmap; 200 ids * 8 bits > bitmap -> 256 bits
        np.testing.assert_allclose(sk.bytes_per_doc(), [3.0 + 4.0, 32.0 + 4.0])

    def test_kmeans_recovers_separated_clusters(self) -> None:
        rng = np.random.default_rng(1)
        centers = unit(rng.standard_normal((4, 16)))
        labels = rng.integers(0, 4, 2000)
        pts = unit(centers[labels] + 0.05 * rng.standard_normal((2000, 16))).astype(np.float32)
        cb = train_spherical_kmeans(pts, 4, iters=10, seed=0, device=CPU)
        ids, _ = assign_to_codebook(pts, cb, CPU)
        purity = sum(np.bincount(ids[labels == c]).max() for c in range(4)) / len(pts)
        self.assertGreater(purity, 0.98)

    def test_estimator_error_is_bounded_by_residual_radius(self) -> None:
        """Lemma: |max_x <q,x> - max_{c in C_d} <q,mu_c>| <= max_x ||x - mu_c(x)|| per query token."""
        store = random_store(n_docs=30, dim=8, max_len=12, seed=3)
        vecs = np.asarray(store.vectors, dtype=np.float32)
        cb = train_spherical_kmeans(vecs, 16, iters=8, seed=0, device=CPU)
        ids, resid = assign_to_codebook(vecs, cb, CPU)
        sk = CentroidBagSketch.build(store.doc_of_token(), ids, resid, 16, store.n_docs)
        est = CentroidBagEstimator(sk, cb)
        for q in random_queries(n=5, m=1, dim=8, seed=4):
            approx = est.score(q)
            for d in range(store.n_docs):
                exact = float((q @ store.doc(d).astype(np.float32).T).max())
                radius = float(resid[store.offsets[d]:store.offsets[d + 1]].max())
                self.assertLessEqual(abs(exact - approx[d]), radius + 1e-5)

    def test_storage_formula_bits(self) -> None:
        self.assertEqual(math.ceil(math.log2(16384)), 14)

    def test_torch_estimator_matches_numba(self) -> None:
        store = random_store(n_docs=60, dim=8, max_len=12, seed=7)
        vecs = np.asarray(store.vectors, dtype=np.float32)
        cb = train_spherical_kmeans(vecs, 16, iters=5, seed=0, device=CPU)
        ids, resid = assign_to_codebook(vecs, cb, CPU)
        sk = CentroidBagSketch.build(store.doc_of_token(), ids, resid, 16, store.n_docs)
        ref, fast = CentroidBagEstimator(sk, cb), TorchCentroidBagEstimator(sk, cb, CPU)
        for q in random_queries(n=4, m=6, dim=8, seed=8):
            np.testing.assert_allclose(fast.score(q), ref.score(q), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
