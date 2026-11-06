# Third-party code

No upstream source file is vendored in this repository. The modules below reimplement published methods, and each was
checked against the original.

| Module | Reimplements | Upstream |
|---|---|---|
| `savemv/baselines/residual.py` | ColBERTv2 and PLAID residual compression | [stanford-futuredata/ColBERT](https://github.com/stanford-futuredata/ColBERT) |
| `savemv/baselines/pooling.py` | hierarchical token pooling, following PyLate's `pool_embeddings_hierarchical` | [lightonai/pylate](https://github.com/lightonai/pylate) |
| `savemv/baselines/fde.py` | MUVERA fixed-dimensional encodings, matching the `muvera-py` port | `muvera-py` |
| `savemv/baselines/constbert.py` | the official ConstBERT model wrapper | `pinecone/ConstBERT` on Hugging Face |

Run as dependencies and not included here:

| Package | Use |
|---|---|
| `colbert-ai` 0.2.22 | official PLAID engine |
| [jlscheerer/xtr-warp](https://github.com/jlscheerer/xtr-warp) | official WARP engine, cloned and patched by `scripts/official_baselines/patch_warp.py` |
| `bm25s`, `PyStemmer` | BM25 |
| `lightgbm` | LAET and DARTH baselines |
