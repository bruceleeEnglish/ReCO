#!/usr/bin/env bash
# Online latency (real recomputation from raw text) at the headline configurations of the G1 report.
#   scripts/pipeline/run_latency.sh [gpu]      -> outputs/logs/latency_<dataset>_K<k>_<rule>.json
set -uo pipefail
cd "$(dirname "$0")/../.."
source .env.sh
export CUDA_VISIBLE_DEVICES="${1:-0}"
run() {  # dataset K rule outdir [extra --set ...]
  local ds=$1 k=$2 rule=$3 out=$4; shift 4
  .venv/bin/python -m savemv run --config configs/beir_bench.yaml --set dataset=$ds --set output_root=$out \
    --set "sketch.codebook_sizes=[$k]" --set latency.enabled=true --set latency.codebook_size=$k \
    --set latency.rule=$rule --set latency.delta=0.05 --set "latency.batches=[8,32]" --set latency.num_queries=200 "$@"
  cp $out/$ds/results.json outputs/logs/latency_${ds}_K${k}_${rule}.json
}
run fiqa 16384 crc outputs
run fiqa 16384 fixed outputs
run scifact 4096 crc outputs
run trec-covid 65536 rank1 outputs
echo latency-done
