import unittest

import numpy as np

from savemv.reco.estimate import CentroidBagEstimator
from savemv.reco.inverted import InvertedSketchIndex
from savemv.reco.sketch import CentroidBagSketch


class InvertedTest(unittest.TestCase):
    def test_prefix_equals_full_scan(self) -> None:
        """The inverted index returns the head of the full-scan order with bit-identical estimates."""
        rng = np.random.default_rng(0)
        n, k, m, dim = 5000, 256, 8, 16
        sizes = rng.integers(1, 25, n)
        ptr = np.zeros(n + 1, np.int64)
        ptr[1:] = np.cumsum(sizes)
        cent = np.concatenate([np.sort(rng.choice(k, s, replace=False)) for s in sizes]).astype(np.int32)
        sketch = CentroidBagSketch(k, ptr, cent, None)
        cb = rng.standard_normal((k, dim)).astype(np.float32)
        cb /= np.linalg.norm(cb, axis=1, keepdims=True)
        full, index = CentroidBagEstimator(sketch, cb), InvertedSketchIndex(sketch, cb)
        for need, start in ((10, 1), (300, 2), (4999, 4), (5000, 64)):
            q = rng.standard_normal((m, dim)).astype(np.float32)
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            order, vals, _ = index.prefix(q, need, start=start, a=2.0, b=1.0)
            est = full.score(q)
            ref = np.argsort(-est, kind="stable")[:need]
            np.testing.assert_array_equal(order, ref)
            np.testing.assert_allclose(vals, 2.0 * est[ref].astype(np.float64) + 1.0, rtol=0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
