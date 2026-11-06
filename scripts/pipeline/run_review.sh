#!/usr/bin/env bash
# Experiments added after the mock review, strictly one after the other on an otherwise idle machine. The two timing
# runs come first, so nothing else touches the GPU, the CPU or the page cache while they measure. They reuse the
# protocol of run_throughput.sh (same GPU, no pinning). plaid_recall_ceiling needs the CPU because the deepest settings
# exceed the GPU, and it reports recall only. Every result goes to a new file; no earlier output is changed.
#   scripts/pipeline/run_review.sh <gpu>   -> outputs/logs/review_<time>.log, one line "[stage] <name> rc=<code>" per step
# A failed step does not stop the next one: the steps share no state.
set -u
cd "$(dirname "$0")/../.." || exit 1
. ./.env.sh
gpu=${1:-1}
log="outputs/logs/review_$(date +%Y%m%d_%H%M%S).log"
exec > "$log" 2>&1 < /dev/null
nv=/usr/local/lib/python3.10/dist-packages/nvidia
config=configs/paper_bench.yaml

stage() {
  name=$1; shift
  echo "=== $(date -u +%FT%TZ) $name: $*"
  "$@"
  echo "[stage] $name rc=$? $(date -u +%FT%TZ)"
}

plaid() (
  export LD_LIBRARY_PATH="$nv/cublas/lib:$nv/cuda_runtime/lib" PATH="$PWD/.venv_plaid/bin:/usr/local/cuda-12.3/bin:$PATH"
  export CUDA_HOME=/usr/local/cuda-12.3 CUDA_VISIBLE_DEVICES=$gpu
  .venv_plaid/bin/python scripts/official_baselines/plaid_msmarco.py "$@"
)

plaid_cpu() ( gpu=; plaid "$@" )

reco() ( CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python "$@" )

echo "start $(date -u +%FT%TZ), gpu $gpu"
uptime
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

stage plaid_sweep plaid sweep --nbits 2 --points 4
stage reco_throughput_d01 reco scripts/web_scale/throughput.py --concurrency 1 16 64 --deltas 0.01 --queries 1000 \
  --tokenizer-threads 8 --out outputs/msmarco --result throughput_reco_d01.json
stage plaid_recall_ceiling plaid_cpu sweep --nbits 2 --device cpu --threads 16 --start 4 --result plaid_sweep_cpu_nbits2.json
stage optimality_gap reco scripts/stopping/optimality_gap.py $config --datasets scifact nfcorpus \
  "arguana:encoder.query_maxlen=300,output_root=outputs_q300" scidocs fiqa trec-covid "quora:max_queries=1000"
stage growth reco scripts/stopping/growth.py $config --datasets scidocs fiqa "quora:max_queries=1000"
echo "ALL DONE $(date -u +%FT%TZ)"
