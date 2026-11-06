#!/bin/sh
. ./.env.sh
gpu=$1; shift
for ds in "$@"; do
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/stopping_baselines.py configs/paper_bench.yaml --datasets "$ds" --out outputs/stopping_baselines
done
echo ALL DONE
