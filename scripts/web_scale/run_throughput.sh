#!/bin/sh
# Throughput comparison on MS MARCO (paper Exp-5): the official PLAID engine (per-query search and its batch API
# search_all) and ReCo at several concurrency levels, on the same GPU with the machine otherwise idle (no pinning).
#   scripts/web_scale/run_throughput.sh <gpu>
cd "$(dirname "$0")/../.." || exit 1
. ./.env.sh
gpu=${1:-1}
while pgrep -f "scripts/stopping/param_study.py" > /dev/null; do sleep 20; done      # wait for CPU-heavy jobs
nv=/usr/local/lib/python3.10/dist-packages/nvidia
cp outputs/msmarco/plaid_nbits2.json outputs/msmarco/plaid_nbits2.before_rerun.json
(
  export LD_LIBRARY_PATH=$nv/cublas/lib:$nv/cuda_runtime/lib PATH=$PWD/.venv_plaid/bin:/usr/local/cuda-12.3/bin:$PATH
  export CUDA_HOME=/usr/local/cuda-12.3 CUDA_VISIBLE_DEVICES=$gpu
  .venv_plaid/bin/python scripts/official_baselines/plaid_msmarco.py search --nbits 2 --device gpu
  .venv_plaid/bin/python scripts/official_baselines/plaid_msmarco.py throughput --nbits 2
)
# keep the original latency run as "gpu" and store the rerun next to it
.venv/bin/python -c "
import json
p = 'outputs/msmarco/plaid_nbits2.json'
old = json.load(open('outputs/msmarco/plaid_nbits2.before_rerun.json'))
r = json.load(open(p)); r['gpu_rerun'] = r['gpu']; r['gpu'] = old['gpu']; json.dump(r, open(p, 'w'), indent=1)
print('PLAID search rerun', r['gpu_rerun']['latency_ms'], 'search_all', r['throughput'])
"
CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/web_scale/throughput.py --concurrency 1 4 16 64 --deltas 0.1 0.05 \
  --queries 1000 --tokenizer-threads 8 --out outputs/msmarco
echo ALL DONE
