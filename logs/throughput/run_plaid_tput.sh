#!/bin/sh
. ./.env.sh
nv=/usr/local/lib/python3.10/dist-packages/nvidia
export LD_LIBRARY_PATH=$nv/cublas/lib:$nv/cuda_runtime/lib PATH=$PWD/.venv_plaid/bin:/usr/local/cuda-12.3/bin:$PATH CUDA_HOME=/usr/local/cuda-12.3
CUDA_VISIBLE_DEVICES=1 taskset -c $(seq -s, 1 2 79) .venv_plaid/bin/python scripts/plaid_msmarco.py throughput --nbits 2
echo ALL DONE
