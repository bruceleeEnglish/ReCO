import unittest

import numpy as np
import torch

from savemv.core import assign_to_codebook, maxsim_matrix, train_spherical_kmeans
from savemv.reco import (Calibration, CentroidBagEstimator, CentroidBagSketch, ReCoSearcher,
                         TorchCentroidBagEstimator, adaptive_stop)
from tests.helpers import random_queries, random_store

CPU = torch.device("cpu")


class FakeEncoder:
    """Serves stored vectors for texts named 'q<i>' / 'd<j>', so recomputation is exact."""

    def __init__(self, store, queries) -> None:
        self.device = CPU
        self.store, self.queries = store, queries

    def encode_query_tensor(self, text: str) -> torch.Tensor:
        return torch.from_numpy(self.queries[int(text[1:])])

    def tokenize_docs(self, texts: list[str]):
        ids = np.array([[int(t[1:])] for t in texts], dtype=np.int64)   # "token ids" = document index
        return ids, np.ones_like(ids)

    def embed_docs(self, ids: np.ndarray, mask: np.ndarray):
        docs = [self.store.doc(int(i)).astype(np.float32) for i in ids[:, 0]]
        length = max(len(d) for d in docs)
        emb = np.zeros((len(docs), length, self.store.dim), np.float32)
        keep = np.zeros((len(docs), length), bool)
        for i, d in enumerate(docs):
            emb[i, :len(d)] = d
            keep[i, :len(d)] = True
        return torch.from_numpy(emb), torch.from_numpy(keep), torch.from_numpy(ids)


class EngineTest(unittest.TestCase):
    def test_online_search_matches_offline_simulation(self) -> None:
        store = random_store(n_docs=150, dim=16, max_len=10, seed=5)
        queries = random_queries(n=6, m=4, dim=16, seed=6)
        vecs = np.asarray(store.vectors, dtype=np.float32)
        cb = train_spherical_kmeans(vecs, 32, iters=8, seed=0, device=CPU)
        ids, resid = assign_to_codebook(vecs, cb, CPU)
        estimator = CentroidBagEstimator(CentroidBagSketch.build(store.doc_of_token(), ids, resid, 32,
                                                                 store.n_docs), cb)
        exact = maxsim_matrix(store, queries, CPU)
        texts = [f"d{j}" for j in range(store.n_docs)]
        torch_estimator = TorchCentroidBagEstimator(estimator.sketch, cb, CPU)
        for (tau, batch), est_impl in zip(((0.0, 4), (0.3, 8), (2.0, 3)), (estimator, torch_estimator, estimator)):
            cal = Calibration(a=1.0, b=0.0, tau=tau, k=5, batch=batch)
            searcher = ReCoSearcher(FakeEncoder(store, queries), est_impl, texts, cal)
            for qi in range(len(queries)):
                res = searcher.search(f"q{qi}")
                est = estimator.score(queries[qi])
                order = np.argsort(-est, kind="stable")
                p = adaptive_stop(est[order].astype(np.float64), exact[qi, order].astype(np.float64), 1.0, 0.0, tau,
                                  5, batch)
                self.assertEqual(res.n_recomputed, p)
                done = order[:p]
                expected = done[np.argsort(-exact[qi, done], kind="stable")[:5]]
                self.assertEqual(sorted(res.doc_indices.tolist()), sorted(expected.tolist()))
                np.testing.assert_allclose(res.recomputed_scores, exact[qi, res.recomputed], rtol=1e-5, atol=1e-5)

        for budget in (5, 13, 40):                      # fixed rule: exactly `budget` documents (>= k)
            cal = Calibration(a=1.0, b=0.0, tau=float("inf"), k=5, batch=8, budget=budget)
            searcher = ReCoSearcher(FakeEncoder(store, queries), estimator, texts, cal)
            for qi in range(len(queries)):
                res = searcher.search(f"q{qi}")
                order = np.argsort(-estimator.score(queries[qi]), kind="stable")
                self.assertEqual(res.n_recomputed, budget)
                done = order[:budget]
                expected = done[np.argsort(-exact[qi, done], kind="stable")[:5]]
                self.assertEqual(sorted(res.doc_indices.tolist()), sorted(expected.tolist()))

        # a candidate cap only truncates the processing order: same stops below it, never beyond it
        for est_impl in (estimator, torch_estimator):
            cal = Calibration(a=1.0, b=0.0, tau=0.3, k=5, batch=8)
            full = ReCoSearcher(FakeEncoder(store, queries), est_impl, texts, cal)
            capped = ReCoSearcher(FakeEncoder(store, queries), est_impl, texts, cal, max_candidates=24)
            for qi in range(len(queries)):
                r_full, r_cap = full.search(f"q{qi}"), capped.search(f"q{qi}")
                self.assertEqual(r_cap.n_recomputed, min(r_full.n_recomputed, 24))
                np.testing.assert_array_equal(r_cap.recomputed, r_full.recomputed[:r_cap.n_recomputed])


    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_triton_kernel_matches_numba(self) -> None:
        from savemv.reco.estimate import triton
        if triton is None:
            self.skipTest("triton not installed")
        store = random_store(n_docs=500, dim=16, max_len=30, seed=7)
        queries = random_queries(n=3, m=5, dim=16, seed=8)          # m = 5 exercises the power-of-two padding
        vecs = np.asarray(store.vectors, dtype=np.float32)
        cb = train_spherical_kmeans(vecs, 64, iters=5, seed=0, device=CPU)
        ids, resid = assign_to_codebook(vecs, cb, CPU)
        sketch = CentroidBagSketch.build(store.doc_of_token(), ids, resid, 64, store.n_docs)
        ref = CentroidBagEstimator(sketch, cb)
        gpu = torch.device("cuda")
        fast = TorchCentroidBagEstimator(sketch, cb, gpu, kernel="triton")
        half = TorchCentroidBagEstimator(sketch, cb, gpu, kernel="triton", half_table=True)
        for q in queries:
            np.testing.assert_allclose(fast.score(q), ref.score(q), atol=1e-5)
            np.testing.assert_allclose(half.score(q), ref.score(q), atol=5e-3 * q.shape[0])


if __name__ == "__main__":
    unittest.main()
