import unittest

import numpy as np
import torch

from savemv.core.exact import maxsim_matrix, maxsim_padded
from tests.helpers import naive_maxsim, random_queries, random_store


class ExactMaxSimTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = random_store(n_docs=23, dim=16, max_len=9)
        self.queries = random_queries(n=7, m=5, dim=16)

    def test_matrix_matches_naive_with_tiny_blocks(self) -> None:
        # tiny budgets force several query passes and many document blocks, with and without residency
        ref = naive_maxsim(self.store, self.queries)
        for resident_bytes in (1 << 30, 0):
            got = maxsim_matrix(self.store, self.queries, torch.device("cpu"), queries_per_pass=3,
                                max_block_elems=40, resident_bytes=resident_bytes)
            np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)

    def test_padded_matches_naive(self) -> None:
        docs = [self.store.doc(i).astype(np.float32) for i in range(6)]
        length = max(len(d) for d in docs)
        padded = np.zeros((len(docs), length, 16), np.float32)
        keep = np.zeros((len(docs), length), bool)
        for i, d in enumerate(docs):
            padded[i, :len(d)] = d
            keep[i, :len(d)] = True
        q = self.queries[0]
        got = maxsim_padded(torch.from_numpy(q), torch.from_numpy(padded), torch.from_numpy(keep)).numpy()
        ref = np.array([(q @ d.T).max(axis=1).sum() for d in docs])
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
