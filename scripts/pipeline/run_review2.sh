#!/usr/bin/env bash
# Second round after the sub-agent reviews, one step after the other (no timing is reported from either step):
#   1. PLAID's CPU engine at lower centroid-score thresholds (recall only)
#   2. the learned stopping baselines again, now keeping FixedBudget, TA-Margin and Patience under the pooled correction
#   scripts/pipeline/run_review2.sh    -> outputs/logs/review2_<time>.log, one line "[stage] <name> rc=<code>" per step
set -u
cd "$(dirname "$0")/../.." || exit 1
. ./.env.sh
log="outputs/logs/review2_$(date +%Y%m%d_%H%M%S).log"
exec > "$log" 2>&1 < /dev/null
nv=/usr/local/lib/python3.10/dist-packages/nvidia

stage() {
  name=$1; shift
  echo "=== $(date -u +%FT%TZ) $name: $*"
  "$@"
  echo "[stage] $name rc=$? $(date -u +%FT%TZ)"
}

plaid_cpu() (
  export LD_LIBRARY_PATH="$nv/cublas/lib:$nv/cuda_runtime/lib" PATH="$PWD/.venv_plaid/bin:/usr/local/cuda-12.3/bin:$PATH"
  export CUDA_HOME=/usr/local/cuda-12.3 CUDA_VISIBLE_DEVICES=
  .venv_plaid/bin/python scripts/official_baselines/plaid_msmarco.py "$@"
)

echo "start $(date -u +%FT%TZ)"
stage plaid_low plaid_cpu sweep --nbits 2 --device cpu --threads 16 --grid low --result plaid_sweep_cpu_low_nbits2.json
stage stopping_pooled env CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/stopping/stopping_baselines.py configs/paper_bench.yaml \
  --datasets scifact nfcorpus "arguana:encoder.query_maxlen=300,output_root=outputs_q300" scidocs fiqa trec-covid \
  "quora:max_queries=1000" --out outputs/stopping_baselines_v2
echo "ALL DONE $(date -u +%FT%TZ)"
