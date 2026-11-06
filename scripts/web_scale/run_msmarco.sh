#!/usr/bin/env bash
# MS MARCO passage (8.8M) scale experiment, all stages on one GPU (in the background).
#   scripts/web_scale/run_msmarco.sh <gpu> [out_dir]
# Needs data/msmarco/{corpus.jsonl,queries.jsonl,qrels/dev.tsv} (MTEB/BEIR layout), ~40 GB free disk for the
# evaluation-only exact score matrix, and ~1.5 h on an RTX 3090.
set -euo pipefail
cd "$(dirname "$0")/../.."
gpu="${1:?usage: scripts/web_scale/run_msmarco.sh <gpu> [out_dir]}"
out="${2:-outputs/msmarco}"
mkdir -p outputs/logs
log="outputs/logs/msmarco_$(date +%Y%m%d_%H%M%S).log"
(
  if [ -f .env.sh ]; then
    # shellcheck disable=SC1091
    source .env.sh
  fi
  export CUDA_VISIBLE_DEVICES="$gpu"
  py=".venv/bin/python scripts/web_scale/msmarco_scale.py"
  set -x
  $py encode --out "$out" --queries 1000
  $py eval --out "$out" --bm25
  $py eval --out "$out" --stop-batch 32
  $py latency --out "$out" --k 65536
  $py latency --out "$out" --k 65536 --stop-batch 32
  $py latency --out "$out" --k 16384
  echo "ALL DONE"
) > "$log" 2>&1 < /dev/null &
echo "MS MARCO run on GPU $gpu started (pid $!) -> $log"
