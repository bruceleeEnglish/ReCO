"""Encoder checks that need the real model; skipped when it cannot be loaded.

The model is taken from $SAVEMV_MODEL (a local directory or HF id), default the HF id.
"""
import os
import unittest

import numpy as np

from savemv.config import EncoderConfig

TEXTS = ["Short doc.", "A much longer document, with punctuation; numbers 12,345 and UNICODE café!" * 12,
         "", "Mixed CASE words and a [bracket] or two"]


def _load():
    try:
        from savemv.encoders import ColBERTEncoder
        return ColBERTEncoder(EncoderConfig(repo=os.environ.get("SAVEMV_MODEL",
                                                                "answerdotai/answerai-colbert-small-v1"),
                                            device="cpu"))
    except Exception as err:  # offline / model missing
        raise unittest.SkipTest(f"model unavailable: {err}")


class EncoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.enc = _load()

    def test_fast_doc_tokenization_matches_transformers(self) -> None:
        ids, mask = self.enc.tokenize_docs(TEXTS)
        ref = self.enc.tokenizer([". " + t for t in TEXTS], padding="longest", truncation=True,
                                 max_length=self.enc.cfg.doc_maxlen, return_tensors="np")
        ref_ids = ref["input_ids"].astype(np.int64)
        ref_ids[:, 1] = self.enc.d_marker
        np.testing.assert_array_equal(ids, ref_ids)
        np.testing.assert_array_equal(mask, ref["attention_mask"])

    def test_fast_query_tokenization_matches_transformers(self) -> None:
        ids, mask = self.enc._query_ids(TEXTS)
        ref = self.enc.tokenizer([". " + t for t in TEXTS], padding="max_length", truncation=True,
                                 max_length=self.enc.cfg.query_maxlen, return_tensors="np")
        ref_ids = ref["input_ids"].astype(np.int64)
        ref_ids[:, 1] = self.enc.q_marker
        ref_ids[ref_ids == self.enc.pad_id] = self.enc.mask_id
        np.testing.assert_array_equal(ids, ref_ids)
        np.testing.assert_array_equal(mask, ref["attention_mask"])

    def test_outputs_are_unit_vectors(self) -> None:
        emb, keep, _ = self.enc.encode_docs_padded(TEXTS[:2])
        norms = emb[keep].norm(dim=-1).numpy()
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)
        self.assertEqual(self.enc.encode_queries(TEXTS[:1]).shape, (1, self.enc.cfg.query_maxlen, self.enc.dim))


if __name__ == "__main__":
    unittest.main()
