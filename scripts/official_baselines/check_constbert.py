"""Compare savemv's ConstBERT encoder with the official HF wrapper (pinecone/ConstBERT modeling.py).

    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/official_baselines/check_constbert.py models/ConstBERT

The official `ConstBERT.from_pretrained` builds `ColBERTConfig(name_or_path)`, whose first positional
dataclass field is `query_token_id`, so the wrapper uses the model path (-> [UNK]) as the query
marker instead of the trained [unused0]. This script shows that and checks both encoders agree
once the marker is the same.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from savemv.baselines.constbert import ConstBERTEncoder  # noqa: E402

QUERIES = ["What is the capital of France?", "latest advancements in AI"]
DOCS = ["Paris is the capital and most populous city of France.",
        "Artificial intelligence is rapidly evolving with new breakthroughs.",
        "The Eiffel Tower is a famous landmark in Paris."]


def main(path: str) -> None:
    from transformers import AutoModel
    if not hasattr(torch.amp, "GradScaler"):          # official code targets torch >= 2.3
        torch.amp.GradScaler = torch.cuda.amp.GradScaler
    official = AutoModel.from_pretrained(path, trust_remote_code=True).eval()
    official.use_gpu = False
    qt = official.query_tokenizer
    print("official query marker:", qt.Q_marker_token, repr(qt.config.query_token_id), "->", qt.Q_marker_token_id,
          "| doc marker id:", official.doc_tokenizer.D_marker_token_id)
    ours = ConstBERTEncoder(path, torch.device("cpu"), fp16=False)
    print("savemv query marker id:", ours.q_marker, "| doc marker id:", ours.d_marker)
    with torch.no_grad():
        ref_q = official.encode_queries(QUERIES).float().numpy()
        ref_d = official.encode_documents(DOCS).float().numpy()
    print("documents max |diff|:", float(np.abs(ours.encode_documents(DOCS).astype(np.float32) - ref_d).max()))
    print("queries (trained marker) max |diff|:", float(np.abs(ours.encode_queries(QUERIES) - ref_q).max()))
    ours.q_marker = qt.Q_marker_token_id
    print("queries (official marker) max |diff|:", float(np.abs(ours.encode_queries(QUERIES) - ref_q).max()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "models/ConstBERT")
