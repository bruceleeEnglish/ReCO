import math
import unittest

import numpy as np
import scipy.sparse as sp
import torch
from scipy.cluster import hierarchy

from savemv.baselines.dense import mean_pool_store, quantize
from savemv.baselines.fde import FDEConfig, ProductQuantizer, _rep_matrices, document_fdes, query_fdes
from savemv.baselines.pooling import pool_document, pool_store
from savemv.baselines.residual import ResidualCodec, plaid_num_centroids
from savemv.baselines.sparse import BM25, postings_bytes, sparse_index_bytes, sparse_scores
from tests.helpers import random_queries, random_store, unit

CPU = torch.device("cpu")


def pylate_pool(document: torch.Tensor, pool_factor: int, protected_tokens: int = 1) -> torch.Tensor:
    """PyLate ColBERT.pool_embeddings_hierarchical for one document (transcribed)."""
    protected = document[:protected_tokens]
    to_pool = document[protected_tokens:]
    num_embeddings = len(to_pool)
    num_clusters = max(num_embeddings // pool_factor, 1)
    if num_clusters >= num_embeddings:
        return document
    dist_full = 1 - torch.mm(to_pool, to_pool.t()).numpy()
    condensed = dist_full[np.triu_indices(num_embeddings, k=1)]
    labels = hierarchy.fcluster(hierarchy.linkage(condensed, method="ward"), t=num_clusters, criterion="maxclust")
    labels_tensor = torch.from_numpy(labels.astype(np.int64)) - 1
    sums = torch.zeros(num_clusters, to_pool.shape[1], dtype=to_pool.dtype)
    counts = torch.zeros(num_clusters, dtype=torch.long)
    sums.scatter_add_(0, labels_tensor.unsqueeze(1).expand_as(to_pool), to_pool)
    counts.scatter_add_(0, labels_tensor, torch.ones(num_embeddings, dtype=torch.long))
    mask = counts > 0
    return torch.cat([protected, sums[mask] / counts[mask].unsqueeze(1)], dim=0)


def cpp_fde(points: np.ndarray, cfg: FDEConfig, document: bool) -> np.ndarray:
    """Literal transcription of GenerateFixedDimensionalEncoding (fixed_dimensional_encoding.cc)."""
    n, dim = points.shape
    parts, proj = 2 ** cfg.simhash_bits, cfg.proj_dim or dim
    out = np.zeros(cfg.reps * parts * proj, dtype=np.float64)
    for rep in range(cfg.reps):
        simhash, ams = _rep_matrices(dim, cfg, rep)
        sketch = points @ simhash
        projected = points if ams is None else points @ ams
        sizes = np.zeros(parts)
        for i in range(n):
            idx = 0
            for v in sketch[i]:
                idx = (idx << 1) + (int(v > 0) ^ (idx & 1))          # AppendToGrayCode
            base = rep * parts * proj + idx * proj
            out[base:base + proj] += projected[i]
            sizes[idx] += 1
        if not document:
            continue
        for p in range(parts):
            base = rep * parts * proj + p * proj
            if sizes[p] == 0:
                if cfg.fill_empty:
                    binary = p ^ (p >> 1)                                # GrayCodeToBinary
                    dists = []
                    for i in range(n):
                        b, dist = binary, 0
                        for j in range(cfg.simhash_bits - 1, -1, -1):
                            dist += int(sketch[i, j] > 0) != (b & 1)
                            b >>= 1
                        dists.append(dist)
                    out[base:base + proj] = projected[int(np.argmin(dists))]
                continue
            out[base:base + proj] /= sizes[p]
    return out


class PoolingTest(unittest.TestCase):
    def test_matches_pylate(self) -> None:
        rng = np.random.default_rng(0)
        for n, factor in ((1, 2), (2, 2), (9, 2), (40, 3), (57, 4), (120, 8)):
            doc = unit(rng.standard_normal((n, 16))).astype(np.float32)
            ref = pylate_pool(torch.from_numpy(doc), factor).numpy()
            np.testing.assert_allclose(pool_document(doc, factor), ref, rtol=1e-5, atol=1e-6)

    def test_store_pooling_matches_documents(self) -> None:
        store = random_store(n_docs=25, dim=8, max_len=30, seed=2)
        pooled = pool_store(store, 4, workers=1)
        self.assertEqual(pooled.n_docs, store.n_docs)
        for d in range(store.n_docs):
            ref = pool_document(store.doc(d), 4)
            np.testing.assert_allclose(pooled.doc(d).astype(np.float32), ref, atol=2e-3)
            rest = len(store.doc(d)) - 1
            if max(rest // 4, 1) >= rest:
                self.assertEqual(len(ref), rest + 1)              # nothing to pool
            else:                                                 # older scipy may return one extra cluster on ties
                self.assertLessEqual(len(ref), 2 + max(rest // 4, 1))


class ResidualTest(unittest.TestCase):
    def test_plaid_codebook_size(self) -> None:
        self.assertEqual(plaid_num_centroids(6_460_000), 32768)
        self.assertEqual(plaid_num_centroids(1_000_000), 8192)

    def test_codes_reconstruct_from_bucket_weights(self) -> None:
        store = random_store(n_docs=200, dim=16, max_len=20, seed=3)
        vecs = np.asarray(store.vectors, dtype=np.float32)
        c2, c1 = ResidualCodec.fit(vecs, [2, 1], 32, iters=5, seed=0, device=CPU, min_train_points=500)
        np.testing.assert_array_equal(c2.centroids, c1.centroids)
        self.assertEqual((len(c1.weights), len(c2.weights)), (2, 4))
        rec1 = c1.reconstruct(vecs, CPU).astype(np.float32)
        rec2 = c2.reconstruct(vecs, CPU).astype(np.float32)
        np.testing.assert_allclose(np.linalg.norm(rec1, axis=1), 1.0, atol=2e-3)
        cos1, cos2 = (rec1 * vecs).sum(1).mean(), (rec2 * vecs).sum(1).mean()
        self.assertGreater(cos2, cos1)
        self.assertGreater(cos1, 0.6)
        self.assertEqual(c1.bytes_per_vector(16), 5 / 8 + 2)


class FDETest(unittest.TestCase):
    def test_query_and_document_fdes_match_cpp(self) -> None:
        rng = np.random.default_rng(4)
        docs = [unit(rng.standard_normal((n, 12))).astype(np.float32) for n in (1, 3, 7, 20)]
        offsets = np.r_[0, np.cumsum([len(d) for d in docs])]
        vectors = np.concatenate(docs)
        for cfg in (FDEConfig(reps=3, simhash_bits=3, proj_dim=4, fill_empty=True, seed=7),
                    FDEConfig(reps=2, simhash_bits=4, proj_dim=0, fill_empty=False, seed=1)):
            got = document_fdes(vectors, offsets, np.arange(len(docs)), cfg, CPU).numpy()
            for i, d in enumerate(docs):
                np.testing.assert_allclose(got[i], cpp_fde(d, cfg, True), rtol=1e-5, atol=1e-5)
            q = random_queries(n=3, m=5, dim=12, seed=5)
            gq = query_fdes(q, cfg, CPU)
            for i in range(3):
                np.testing.assert_allclose(gq[i], cpp_fde(q[i], cfg, False), rtol=1e-5, atol=1e-5)

    def test_fde_inner_product_tracks_chamfer(self) -> None:
        store = random_store(n_docs=60, dim=16, max_len=12, seed=6)
        q = random_queries(n=4, m=6, dim=16, seed=7)
        cfg = FDEConfig(reps=30, simhash_bits=3, proj_dim=0, fill_empty=True)
        est = query_fdes(q, cfg, CPU) @ document_fdes(store.vectors, store.offsets, np.arange(60), cfg, CPU).numpy().T
        exact = np.array([[(qq @ store.doc(d).astype(np.float32).T).max(1).sum() for d in range(60)] for qq in q])
        corr = np.mean([np.corrcoef(est[i], exact[i])[0, 1] for i in range(4)])
        self.assertGreater(corr, 0.7)

    def test_pq_roundtrip(self) -> None:
        rng = np.random.default_rng(8)
        centers = rng.standard_normal((4, 16)).astype(np.float32)       # 4 distinct rows -> exact with 256 codes
        x = torch.from_numpy(centers[rng.integers(0, 4, 300)])
        pq = ProductQuantizer.fit(x, sub_dim=8, n_centroids=256, iters=5)
        np.testing.assert_allclose(pq.decode(pq.encode(x), CPU).numpy(), x.numpy(), atol=1e-5)
        self.assertEqual((pq.bytes_per_vector(), pq.shared_bytes()), (2.0, 2 * 256 * 8 * 4))


class SingleVectorTest(unittest.TestCase):
    def test_quantizers(self) -> None:
        x = np.random.default_rng(9).standard_normal((50, 8)).astype(np.float32)
        rec, per_doc, shared = quantize(x, "binary")
        self.assertTrue(set(np.unique(rec)) <= {-1.0, 1.0})
        self.assertEqual((per_doc, shared), (1.0, 0.0))
        rec, per_doc, _ = quantize(x, "int8")
        step = (x.max(0) - x.min(0)) / 255
        self.assertTrue(np.all(np.abs(rec - x) <= step / 2 + 1e-6))
        self.assertEqual(per_doc, 8.0)

    def test_mean_pool(self) -> None:
        store = random_store(n_docs=10, dim=8, max_len=6, seed=10)
        got = mean_pool_store(store, CPU, chunk=7)
        for d in range(10):
            m = store.doc(d).astype(np.float32).mean(0)
            np.testing.assert_allclose(got[d], m / np.linalg.norm(m), atol=1e-5)


class SparseTest(unittest.TestCase):
    def test_postings_bytes(self) -> None:
        # term 0: docs [0, 5, 300] -> gaps 1, 5, 295 -> 1 + 1 + 2 bytes; term 1: doc [2] -> gap 3 -> 1 byte
        self.assertEqual(postings_bytes(np.array([0, 3, 4]), np.array([300, 0, 5, 2])), 5 + 4)
        docs = sp.csr_matrix(np.array([[1.0, 0], [0, 2.0], [3.0, 0]]))
        self.assertEqual(sparse_index_bytes(docs), 3 + 3)

    def test_sparse_scores(self) -> None:
        rng = np.random.default_rng(11)
        q, d = sp.random(5, 30, 0.3, format="csr", random_state=1), sp.random(40, 30, 0.2, format="csr", random_state=2)
        np.testing.assert_allclose(sparse_scores(q, d, chunk=2), q.toarray() @ d.toarray().T, rtol=1e-5)
        del rng

    def test_bm25_ranks_matching_documents(self) -> None:
        docs = ["the cat sat on the mat", "dogs chase cats in the garden", "stock markets fell sharply today",
                "a quiet afternoon"]
        bm25 = BM25()
        bm25.index(docs)
        s = bm25.scores(["cats", "market crash", "the of and"])
        self.assertEqual(set(np.flatnonzero(s[0] > 0)), {0, 1})
        self.assertEqual(int(np.argmax(s[1])), 2)
        self.assertTrue(np.all(s[2] == 0))
        self.assertGreater(bm25.index_bytes(), 0)
        self.assertTrue(math.isfinite(float(s.sum())))


if __name__ == "__main__":
    unittest.main()
