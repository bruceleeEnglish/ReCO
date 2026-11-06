#!/usr/bin/env bash
# Every experiment of the paper, stage by stage (see docs/EXPERIMENTS.md for what each one answers).
#   scripts/pipeline/run_paper.sh <stage> [...]
# Stages (in order; each can be rerun alone, results are cached):
#   test      unit tests
#   hazard    universal (leave-one-dataset-out) hazard models      -> models/hazard/          ~20 min
#   bench     E1-E4: all methods x all stopping rules, 7 BEIR sets  -> outputs*/<ds>/bench/   background, 2 GPUs
#   ablation  E8-E9: theory check (hazard vs residual-risk target) and ablations -> outputs/ablation/
#   scalability  nested corpus subsets (Quora, FiQA)                -> outputs/scalability/
#   build     index construction and insertion cost                 -> outputs/build_cost/
#   latency   E5: online latency with real recomputation            -> outputs/logs/latency_*.json
#   memory    E6: agent-memory retrieval + recompute rules          -> outputs/memory/<bench>/memory.json
#   qa        E7: end-to-end QA, cloud reader (DeepSeek API)        -> outputs/qa/<bench>/      costs API credit
#   qa_local  E7: end-to-end QA, on-device-class reader (Qwen3-4B)  -> outputs/qa/<bench>/Qwen3-4B-Instruct-2507/
#   collect   copy every result into results/ (then scripts/paper/paper_assets.py results paper_artifacts)
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -f .env.sh ]; then
  # shellcheck disable=SC1091
  source .env.sh
fi
PY=.venv/bin/python
[ -x "$PY" ] || PY=python
CONFIG=configs/paper_bench.yaml
GPU0_SETS=(scifact nfcorpus "arguana:encoder.query_maxlen=300,output_root=outputs_q300" scidocs)
GPU1_SETS=(fiqa trec-covid "quora:max_queries=1000")
LME=data/memory/longmemeval_s_cleaned.json
LOCOMO=data/memory/locomo10.json
READER=hf:models/Qwen3-4B-Instruct-2507
mkdir -p outputs/logs

latency_run() {   # dataset K rule [extra --set ...]
  local ds=$1 k=$2 rule=$3; shift 3
  $PY -m savemv run --config $CONFIG --set dataset="$ds" --set "sketch.codebook_sizes=[$k]" \
    --set latency.enabled=true --set latency.codebook_size="$k" --set latency.rule="$rule" --set latency.delta=0.05 \
    --set "latency.batches=[8,32]" --set latency.num_queries=200 "$@"
  cp "outputs/$ds/results.json" "outputs/logs/latency_${ds}_K${k}_${rule}.json"
}

for stage in "$@"; do
  echo "$(date '+%F %T') == $stage"
  case $stage in
    test)
      $PY -m unittest discover -s tests -t . ;;
    hazard)
      $PY scripts/stopping/train_universal_hazard.py $CONFIG --ks 1024 4096 16384 65536 --datasets "${GPU0_SETS[@]}" "${GPU1_SETS[@]}" ;;
    bench)
      BENCH_CONFIG=$CONFIG scripts/pipeline/run_bench.sh 0 "${GPU0_SETS[@]}"
      BENCH_CONFIG=$CONFIG scripts/pipeline/run_bench.sh 1 "${GPU1_SETS[@]}" ;;
    ablation)
      $PY scripts/stopping/hazard_ablation.py $CONFIG --datasets scifact nfcorpus scidocs fiqa --ks 4096 16384 ;;
    latency)
      for rule in hazard crc fixed; do
        latency_run fiqa 16384 $rule
        latency_run scifact 4096 $rule
        latency_run trec-covid 65536 $rule
      done ;;
    memory)
      $PY -m savemv memory --bench longmemeval --data $LME
      $PY -m savemv memory --bench locomo --data $LOCOMO ;;
    qa)
      $PY -m savemv qa --bench locomo --data $LOCOMO --max-questions 400 --full-subset 200 \
        --judge2 deepseek-v4-pro --judge2-sample 300
      $PY -m savemv qa --bench longmemeval --data $LME --max-questions 470 --full-subset 60 \
        --judge2 deepseek-v4-pro --judge2-sample 300 ;;
    build)
      $PY scripts/efficiency/build_cost.py $CONFIG --dataset fiqa --k 16384
      $PY scripts/efficiency/build_cost.py $CONFIG --dataset quora --set max_queries=1000 --k 16384 ;;
    scalability)
      $PY scripts/web_scale/scalability.py $CONFIG --dataset quora --set max_queries=1000 --k 16384
      $PY scripts/web_scale/scalability.py $CONFIG --dataset fiqa --k 16384 ;;
    qa_local)
      $PY -m savemv qa --bench locomo --data $LOCOMO --reader $READER --max-questions 400 --full-subset 200 --workers 8
      for top in 3 30 100; do
        $PY -m savemv qa --bench locomo --data $LOCOMO --reader $READER --max-questions 400 --top $top --workers 8 \
          --retrievers colbert reco-K64k+R bm25 bge-base
      done
      $PY -m savemv qa --bench longmemeval --data $LME --reader $READER --max-questions 470 --full-subset 60 \
        --max-context 32768 --workers 8
      $PY -m savemv qa --bench longmemeval --data $LME --reader $READER --max-questions 470 --top 3 --workers 8 \
        --retrievers colbert reco-K64k+R bm25 bge-base ;;
    collect)
      mkdir -p results/beir results/latency results/memory results/qa
      for item in "${GPU0_SETS[@]}" "${GPU1_SETS[@]}"; do
        ds=${item%%:*}
        root=outputs; [[ $item == *output_root=* ]] && root=${item##*output_root=} && root=${root%%,*}
        cp "$root/$ds/bench/bench_results.json" "results/beir/$ds.json"
      done
      cp outputs/logs/latency_*.json results/latency/ 2>/dev/null || true
      mkdir -p results/ablation && cp outputs/ablation/*.json results/ablation/ 2>/dev/null || true
      mkdir -p results/scalability results/build_cost
      cp outputs/scalability/*.json results/scalability/ 2>/dev/null || true
      cp outputs/build_cost/*.json results/build_cost/ 2>/dev/null || true
      for b in longmemeval locomo; do cp "outputs/memory/$b/memory.json" "results/memory/memory_$b.json"; done
      cp -r outputs/qa/. results/qa/ && find results/qa -name 'cache.jsonl' -delete
      cp -r models/hazard results/hazard_models ;;
    *)
      echo "unknown stage $stage"; exit 1 ;;
  esac
done
