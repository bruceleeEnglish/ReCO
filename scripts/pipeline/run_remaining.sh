#!/usr/bin/env bash
# Overnight batch: every experiment still missing from the paper (launched 2026-10-25).
#   GPU0: cloud-reader QA with the hazard rule for "ReCo + recompute", LongMemEval at all 470 questions,
#         and a second-judge agreement check (deepseek-v4-pro on 300 sampled answers per benchmark)
#   GPU1: index build / insertion cost (FiQA, Quora), then the on-device-reader QA (Qwen3-4B) and its
#         context-budget sweep (top-3/10/30/100)
#   then: collect every result into results/
# Each step logs to outputs/logs/remaining_<step>.log; a failed step does not stop the others.
#   nohup scripts/pipeline/run_remaining.sh > outputs/logs/remaining.log 2>&1 < /dev/null &
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source .env.sh
PY=.venv/bin/python
LME=data/memory/longmemeval_s_cleaned.json
LOCOMO=data/memory/locomo10.json
READER=hf:models/Qwen3-4B-Instruct-2507
mkdir -p outputs/logs

step() {
  local name=$1; shift
  echo "$(date '+%F %T') start $name"
  if "$@" > "outputs/logs/remaining_$name.log" 2>&1; then
    echo "$(date '+%F %T') done  $name"
  else
    echo "$(date '+%F %T') FAILED $name (see outputs/logs/remaining_$name.log)"
  fi
}

gpu0() {
  export CUDA_VISIBLE_DEVICES=0
  step qa_locomo $PY -m savemv qa --bench locomo --data $LOCOMO --max-questions 400 --full-subset 200 \
    --judge2 deepseek-v4-pro --judge2-sample 300
  step qa_lme $PY -m savemv qa --bench longmemeval --data $LME --max-questions 470 --full-subset 60 \
    --judge2 deepseek-v4-pro --judge2-sample 300
}

gpu1() {
  export CUDA_VISIBLE_DEVICES=1
  step build_fiqa $PY scripts/efficiency/build_cost.py configs/paper_bench.yaml --dataset fiqa --k 16384
  step build_quora $PY scripts/efficiency/build_cost.py configs/paper_bench.yaml --dataset quora --set max_queries=1000 --k 16384
  step qwen_locomo $PY -m savemv qa --bench locomo --data $LOCOMO --reader $READER --max-questions 400 \
    --full-subset 200 --workers 8
  for top in 3 30 100; do
    step qwen_locomo_top$top $PY -m savemv qa --bench locomo --data $LOCOMO --reader $READER --max-questions 400 \
      --top $top --workers 8 --retrievers colbert reco-K64k+R bm25 bge-base
  done
  step qwen_lme $PY -m savemv qa --bench longmemeval --data $LME --reader $READER --max-questions 470 \
    --full-subset 60 --max-context 32768 --workers 8
  step qwen_lme_top3 $PY -m savemv qa --bench longmemeval --data $LME --reader $READER --max-questions 470 \
    --top 3 --workers 8 --retrievers colbert reco-K64k+R bm25 bge-base
}

gpu0 > outputs/logs/remaining_gpu0.log 2>&1 &
gpu1 > outputs/logs/remaining_gpu1.log 2>&1 &
wait
step collect scripts/pipeline/run_paper.sh collect
echo "$(date '+%F %T') all done"
