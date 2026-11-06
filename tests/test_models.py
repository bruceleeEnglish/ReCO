"""Checks of the baseline encoders against their released models; skipped when a model is missing.

Model directories default to models/<name> and can be overridden with $SAVEMV_MODELS.
The ConstBERT check runs the official modeling code (trust_remote_code) on CPU in float32.
"""
import os
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("SAVEMV_MODELS", "models"))
CPU = torch.device("cpu")
QUERIES = ["What is the capital of France?", "latest advancements in AI"]
DOCS = ["Paris is the capital and most populous city of France.",
        "Artificial intelligence is rapidly evolving with new breakthroughs.",
        "The Eiffel Tower is a famous landmark in Paris."]


def _need(name: str) -> str:
    path = ROOT / name
    if not path.is_dir():
        raise unittest.SkipTest(f"{path} not found")
    return str(path)


class ConstBERTTest(unittest.TestCase):
    def test_matches_official_implementation(self) -> None:
        path = _need("ConstBERT")
        from transformers import AutoModel
        from savemv.baselines.constbert import ConstBERTEncoder
        if not hasattr(torch.amp, "GradScaler"):            # the official code targets torch >= 2.3
            torch.amp.GradScaler = torch.cuda.amp.GradScaler
        official = AutoModel.from_pretrained(path, trust_remote_code=True).eval()
        official.use_gpu = False                            # keep fp32 outputs on CPU
        ours = ConstBERTEncoder(path, CPU, fp16=False, query_marker="hf")   # same marker as the wrapper
        self.assertEqual(ours.q_marker, official.query_tokenizer.Q_marker_token_id)
        self.assertEqual(ConstBERTEncoder(path, CPU, fp16=False).q_marker,
                         official.query_tokenizer.tok.convert_tokens_to_ids("[unused0]"))
        with torch.no_grad():
            ref_q = official.encode_queries(QUERIES).float().cpu().numpy()
            ref_d = official.encode_documents(DOCS).float().cpu().numpy()
        np.testing.assert_allclose(ours.encode_queries(QUERIES), ref_q, atol=1e-4)
        np.testing.assert_allclose(ours.encode_documents(DOCS).astype(np.float32), ref_d, atol=2e-3)
        self.assertEqual(ref_d.shape, (3, 32, 128))


class SpladeTest(unittest.TestCase):
    def test_loads_and_ranks(self) -> None:
        path = _need("splade-cocondenser-ensembledistil")
        from savemv.baselines.sparse import SpladeEncoder, sparse_scores
        enc = SpladeEncoder(path, CPU, fp16=False, batch_size=2)
        q, d = enc.encode(QUERIES), enc.encode(DOCS)
        terms = {enc.tokenizer.convert_ids_to_tokens(int(i)) for i in q[0].indices}
        self.assertTrue({"paris", "france", "capital"} <= terms)
        s = sparse_scores(q, d)
        self.assertEqual(int(np.argmax(s[0])), 0)
        self.assertEqual(int(np.argmax(s[1])), 1)


class DenseTest(unittest.TestCase):
    def test_loads_and_ranks(self) -> None:
        path = _need("bge-small-en-v1.5")
        from savemv.baselines.dense import DenseEncoder
        enc = DenseEncoder(path, CPU, fp16=False, query_prefix="Represent this sentence for searching relevant passages: ")
        q, d = enc.encode(QUERIES, query=True), enc.encode(DOCS)
        np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-5)
        s = q @ d.T
        self.assertEqual(int(np.argmax(s[0])), 0)
        self.assertEqual(int(np.argmax(s[1])), 1)


if __name__ == "__main__":
    unittest.main()
