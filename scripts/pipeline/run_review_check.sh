#!/usr/bin/env bash
# After scripts/pipeline/run_review.sh: PLAID's default operating point on the CPU, same index, settings and 1,000 queries as the
# GPU run. The deepest points exist only on the CPU, so this shows whether the device changes the recall.
#   scripts/pipeline/run_review_check.sh   -> outputs/msmarco/plaid_sweep_cpu_check_nbits2.json
set -u
cd "$(dirname "$0")/../.." || exit 1
. ./.env.sh
exec > "outputs/logs/review_check_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null
while pgrep -f "scripts/pipeline/run_review[.]sh" > /dev/null; do sleep 30; done
nv=/usr/local/lib/python3.10/dist-packages/nvidia
export LD_LIBRARY_PATH="$nv/cublas/lib:$nv/cuda_runtime/lib" PATH="$PWD/.venv_plaid/bin:/usr/local/cuda-12.3/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.3 CUDA_VISIBLE_DEVICES=
.venv_plaid/bin/python scripts/official_baselines/plaid_msmarco.py sweep --nbits 2 --device cpu --threads 16 --points 1 \
  --result plaid_sweep_cpu_check_nbits2.json
echo "[stage] plaid_cpu_check rc=$? $(date -u +%FT%TZ)"
