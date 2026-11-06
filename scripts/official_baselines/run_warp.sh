#!/bin/sh
# XTR/WARP baseline (paper Exp-1): build the official WARP index on the GPU and search on the CPU.
#   scripts/official_baselines/run_warp.sh <gpu> <nbits> <threads> <dataset> ...
cd "$(dirname "$0")/../.." || exit 1
. ./.env.sh
gpu=$1; nbits=$2; threads=$3; shift 3
nv=/usr/local/lib/python3.10/dist-packages/nvidia
export PATH=$PWD/.venv_warp/bin:/usr/local/cuda-12.3/bin:$PATH CUDA_HOME=/usr/local/cuda-12.3
export LD_LIBRARY_PATH=$nv/cublas/lib:$nv/cuda_runtime/lib
for ds in "$@"; do
  echo "== $ds index $(date)"
  CUDA_VISIBLE_DEVICES=$gpu .venv_warp/bin/python scripts/official_baselines/warp_beir.py index --datasets "$ds" --nbits "$nbits" || { echo "INDEX FAILED $ds"; continue; }
  echo "== $ds search $(date)"
  CUDA_VISIBLE_DEVICES= .venv_warp/bin/python scripts/official_baselines/warp_beir.py search --datasets "$ds" --nbits "$nbits" --threads "$threads" || echo "SEARCH FAILED $ds"
done
echo ALL DONE
