#!/usr/bin/env bash
# Run the benchmark on several datasets, one after another, on one GPU (in the background).
#   scripts/pipeline/run_bench.sh <gpu> <dataset>[:key=value,...] ...
#   e.g. scripts/pipeline/run_bench.sh 1 trec-covid quora:max_queries=1000
#        BENCH_CONFIG=configs/beir_universal.yaml scripts/pipeline/run_bench.sh 0 scifact
# Each dataset logs to outputs/logs/bench_<dataset>_<time>.log; extra --set pairs follow the colon.
set -euo pipefail
cd "$(dirname "$0")/../.."
gpu="${1:?usage: scripts/pipeline/run_bench.sh <gpu> <dataset>[:k=v,...] ...}"
shift
config="${BENCH_CONFIG:-configs/beir_bench.yaml}"
mkdir -p outputs/logs
queue_log="outputs/logs/queue_gpu${gpu}_$(date +%Y%m%d_%H%M%S).log"
(
  if [ -f .env.sh ]; then
    # shellcheck disable=SC1091
    source .env.sh
  fi
  export CUDA_VISIBLE_DEVICES="$gpu"
  for item in "$@"; do
    ds="${item%%:*}"
    sets=(--set "dataset=$ds")
    if [ "$item" != "$ds" ]; then
      IFS=',' read -r -a pairs <<< "${item#*:}"
      for p in "${pairs[@]}"; do sets+=(--set "$p"); done
    fi
    log="outputs/logs/bench_${ds}_$(date +%Y%m%d_%H%M%S).log"
    echo "$(date '+%F %T') start $ds -> $log"
    if .venv/bin/python -m savemv bench --config "$config" "${sets[@]}" > "$log" 2>&1; then
      echo "$(date '+%F %T') done  $ds"
    else
      echo "$(date '+%F %T') FAILED $ds (see $log)"
    fi
  done
) > "$queue_log" 2>&1 < /dev/null &
echo "queue on GPU $gpu started (pid $!) -> $queue_log"
