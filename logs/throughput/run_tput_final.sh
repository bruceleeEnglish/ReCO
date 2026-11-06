#!/bin/sh
. ./.env.sh
CORES=$(seq -s, 1 2 79)
nv=/usr/local/lib/python3.10/dist-packages/nvidia
( export LD_LIBRARY_PATH=$nv/cublas/lib:$nv/cuda_runtime/lib PATH=$PWD/.venv_plaid/bin:/usr/local/cuda-12.3/bin:$PATH CUDA_HOME=/usr/local/cuda-12.3
  CUDA_VISIBLE_DEVICES=1 taskset -c $CORES .venv_plaid/bin/python scripts/plaid_msmarco.py search --nbits 2 --device gpu )
.venv/bin/python - <<"PY"
import json
p = "outputs/msmarco/plaid_nbits2.json"; old = json.load(open("outputs/msmarco/plaid_nbits2.before_pinned.json"))
r = json.load(open(p)); r["gpu_pinned"] = r["gpu"]; r["gpu"] = old["gpu"]; json.dump(r, open(p, "w"), indent=1)
print("pinned PLAID search:", {k: v for k, v in r["gpu_pinned"].items() if k != "affinity"})
PY
CUDA_VISIBLE_DEVICES=1 taskset -c $CORES .venv/bin/python scripts/throughput.py --concurrency 1 4 16 64 --deltas 0.1 0.05 --queries 1000 --tokenizer-threads 8 --out outputs/msmarco
echo ALL DONE
