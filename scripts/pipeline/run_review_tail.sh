#!/usr/bin/env bash
# Last steps after the mock review, once run_review.sh and run_review_check.sh have finished and the machine is idle:
# PLAID's two middle operating points, on the GPU with the queries it cannot hold skipped, then on the CPU for recall.
#   scripts/pipeline/run_review_tail.sh <gpu>   -> outputs/logs/review_tail_<time>.log
set -u
cd "$(dirname "$0")/../.." || exit 1
. ./.env.sh
gpu=${1:-1}
exec > "outputs/logs/review_tail_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null
while pgrep -f "scripts/pipeline/run_review[.]sh|scripts/pipeline/run_review_check[.]sh" > /dev/null; do sleep 30; done
echo "start $(date -u +%FT%TZ)"
nv=/usr/local/lib/python3.10/dist-packages/nvidia
export LD_LIBRARY_PATH="$nv/cublas/lib:$nv/cuda_runtime/lib" PATH="$PWD/.venv_plaid/bin:/usr/local/cuda-12.3/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.3
CUDA_VISIBLE_DEVICES=$gpu .venv_plaid/bin/python scripts/official_baselines/plaid_sweep_skip.py
echo "[stage] plaid_gpu_skip rc=$? $(date -u +%FT%TZ)"
CUDA_VISIBLE_DEVICES= .venv_plaid/bin/python scripts/official_baselines/plaid_msmarco.py sweep --nbits 2 --device cpu --threads 16 \
  --start 2 --points 4 --result plaid_sweep_cpu_mid_nbits2.json
echo "[stage] plaid_cpu_mid rc=$? $(date -u +%FT%TZ)"
echo "ALL DONE $(date -u +%FT%TZ)"
