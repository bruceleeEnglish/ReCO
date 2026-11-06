#!/usr/bin/env bash
# Official PLAID (colbert-ai 0.2.22) on MS MARCO, as the baseline of scripts/web_scale/msmarco_scale.py (run that first).
#   scripts/official_baselines/run_plaid.sh [nbits]        -> outputs/msmarco/plaid_nbits<b>.json, log outputs/logs/plaid_<time>.log
# Setup (once): python3 -m venv --system-site-packages .venv_plaid
#   .venv_plaid/bin/pip install colbert-ai==0.2.22 transformers==4.57.6 pillow==12.3.0
#   .venv_plaid/bin/pip install --ignore-installed ninja                  (a user-site copy has no binary)
#   .venv_plaid/bin/pip install --no-deps faiss-gpu-cu12==1.14.1.post1      (keeps the system torch's CUDA libraries)
#   models/answerai-colbert-small-v1-colbertai = the model files + the hub repo's artifact.metadata
set -euo pipefail
cd "$(dirname "$0")/../.."
nbits="${1:-2}"
log="outputs/logs/plaid_$(date +%Y%m%d_%H%M%S).log"
(
  set -x
  if [ -f .env.sh ]; then
    # shellcheck disable=SC1091
    source .env.sh
  fi
  nv=/usr/local/lib/python3.10/dist-packages/nvidia
  export LD_LIBRARY_PATH="$nv/cublas/lib:$nv/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
  # torch builds colbert-ai's C++/CUDA extensions with the venv's ninja and nvcc
  export CUDA_HOME=/usr/local/cuda-12.3 PATH="$PWD/.venv_plaid/bin:/usr/local/cuda-12.3/bin:$PATH"
  py=.venv_plaid/bin/python
  [ -f outputs/plaid/collection.tsv ] || $py scripts/official_baselines/plaid_msmarco.py prepare
  CUDA_VISIBLE_DEVICES=0,1 $py scripts/official_baselines/plaid_msmarco.py index --nbits "$nbits" --nranks 2
  CUDA_VISIBLE_DEVICES=0 $py scripts/official_baselines/plaid_msmarco.py search --nbits "$nbits" --device gpu
  CUDA_VISIBLE_DEVICES="" taskset -c 0,2,4,6,8,10,12,14 $py scripts/official_baselines/plaid_msmarco.py search --nbits "$nbits" \
    --device cpu --threads 8 --num-queries 200
  echo "ALL DONE"
) > "$log" 2>&1 < /dev/null &
echo "PLAID run started (pid $!) -> $log"
