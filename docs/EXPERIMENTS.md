# Experiments

Commands run from the repository root. `run_paper.sh <stage>` is `scripts/pipeline/run_paper.sh`. Log and result folders
share their names wherever one experiment produced them.

## Before the experiments

| Step | Command | Output |
|---|---|---|
| Hazard models for ReCo-U | `run_paper.sh hazard` | `models/hazard/` |
| Universal codebooks | `python scripts/stopping/train_universal_codebook.py configs/beir_bench.yaml --sizes 4096 16384 65536 --datasets scifact nfcorpus arguana scidocs fiqa trec-covid quora --points 3000000` | `models/codebooks/` |

## Comparison with existing methods

| Exp | In the paper | Command | Results | Logs |
|---|---|---|---|---|
| Exp-1 | Table IV, Fig. 3 | `run_paper.sh bench` and `scripts/official_baselines/run_warp.sh <gpu> <nbits> <threads> <datasets>` | `results/beir`, `results/warp` | `logs/beir`, `logs/warp` |
| Exp-2 | Table V | `run_paper.sh bench` and `python scripts/stopping/stopping_baselines.py configs/paper_bench.yaml --datasets ...` | `results/beir`, `results/stopping_baselines` | `logs/beir`, `logs/stopping_baselines` |
| Exp-2 | Table VI | `python scripts/stopping/train_calibration.py configs/paper_bench.yaml --datasets scifact nfcorpus fiqa` | `results/train_calibration` | `logs/train_calibration` |
| Exp-3 | Fig. 4 | part of `run_paper.sh bench` | `results/beir` | `logs/beir` |
| Exp-4 | Table VII | `scripts/web_scale/run_msmarco.sh <gpu>`, `scripts/official_baselines/run_plaid.sh 2` and `scripts/pipeline/run_review.sh <gpu>` | `results/msmarco` | `logs/msmarco`, `logs/plaid`, `logs/review` |
| Exp-5, latency | Table VIII | `run_paper.sh latency` | `results/latency` | `logs/latency` |
| Exp-5, throughput | Table IX, Fig. 5 | `scripts/web_scale/run_throughput.sh <gpu>`, then `scripts/pipeline/run_review.sh <gpu>`, `run_review_check.sh` and `run_review_tail.sh <gpu>` | `results/msmarco/throughput_reco*.json`, `plaid_sweep_*.json` | `logs/throughput`, `logs/review` |
| Exp-5, CPU scan | text | `python scripts/web_scale/inverted_eval.py --out outputs/msmarco --queries 200 --threads 8 --start 4` | `results/msmarco/inverted_cpu.json` | `logs/msmarco/inverted_rerun_20261030.*` |
| Exp-5, construction | Table X | `run_paper.sh build` | `results/build_cost` | `logs/build_cost` |
| Exp-5, storage cost | text | `python scripts/efficiency/cost_model.py --results results` | `results/cost` | none needed, deterministic |
| Exp-6, retrieval | Table XII | `run_paper.sh memory` | `results/memory` | `logs/ablation_and_memory` |
| Exp-6, answers | Table XI | `run_paper.sh qa qa_local` | `results/qa` | `logs/qa` |
| Exp-6, on device | Table XIII | `scripts/efficiency/cpu_latency.py`, `prepare` on the server, then `run` on each machine | `results/cpu_latency` | `logs/cpu_latency` |

## Parameter study

| Exp | In the paper | Command | Results | Logs |
|---|---|---|---|---|
| Exp-7, codebook size | Fig. 6 | part of `run_paper.sh bench` | `results/beir` | `logs/beir` |
| Exp-8, recall target | Fig. 7 | part of `run_paper.sh bench` | `results/beir`, `results/stopping_baselines` | `logs/beir`, `logs/stopping_baselines` |
| Exp-9, result size | Fig. 8 | `python scripts/stopping/param_study.py configs/paper_bench.yaml --datasets scidocs fiqa "quora:max_queries=1000"` | `results/param_study` | `logs/param_study` |
| Exp-10, calibration size | Fig. 9 | same command | `results/param_study` | `logs/param_study` |
| Exp-11, corpus size | Fig. 10 | `run_paper.sh scalability` | `results/scalability` | `logs/scalability` |
| Exp-11, growth after calibration | text | `python scripts/stopping/growth.py configs/paper_bench.yaml --datasets scidocs fiqa "quora:max_queries=1000"` | `results/growth` | `logs/review` |
| Exp-11, memories by size | text | `python scripts/stopping/memory_by_size.py --bench longmemeval --data data/memory/longmemeval_s_cleaned.json`, same for `locomo` | `results/memory/by_size_*.json` | `logs/review` |
| Exp-12, encoder | Table XIV | `BENCH_CONFIG=configs/paper_bench_colbertv2.yaml scripts/pipeline/run_bench.sh <gpu> scifact nfcorpus scidocs fiqa` | `results/beir_cv2` | `logs/beir_cv2` |
| Exp-13, context budget | Fig. 11 | part of `run_paper.sh qa_local` | `results/qa` | `logs/qa` |

## Ablation

| In the paper | Command | Results | Logs |
|---|---|---|---|
| Table XV | `run_paper.sh ablation` | `results/ablation` | `logs/ablation_and_memory` |
| Theorem 1, text | `python scripts/stopping/optimality_gap.py configs/paper_bench.yaml --datasets scifact nfcorpus "arguana:encoder.query_maxlen=300,output_root=outputs_q300" scidocs fiqa trec-covid "quora:max_queries=1000"` | `results/optimality_gap` | `logs/review` |

## From results to the paper

```bash
python scripts/paper/paper_assets.py results paper_artifacts
```

Almost every number of the paper is a macro in `paper_artifacts/numbers.tex`, and tables and figures come from the same
run. `python tools/verify_literals.py` recomputes the few numbers written directly in the prose.

## Reading the logs

Timestamps in the server logs are UTC. Paths are written as `<repo>`, `<workspace>` and `<home>`.

- `logs/throughput/tput_clean.log` is the final throughput run, on an idle machine and without CPU pinning. It also holds
  the PLAID rerun behind `gpu_rerun` and `throughput` in `plaid_nbits2.json`.
- `logs/throughput/plaid_tput_gpu1.log` and `tput_final.log` are earlier pinned attempts. Pinning slowed PLAID, so the
  paper does not use them.
- `logs/msmarco/msmarco_20261026_121437.log` covers the sketch, the BM25 cascade and latency at batches of 8 and 32.
- `logs/beir_cv2` has two runs per corpus: the main one and a later one for the pooling factor 32.
- `logs/msmarco/inverted_rerun_20261030.*` is a rerun, see [VERIFICATION.md](VERIFICATION.md).
- `logs/review/` holds the last experiments, run one after the other on an idle machine. `review_20261031_025724.log` is
  the main chain, `review_check_*` compares PLAID's GPU and CPU engines, `review_tail_*` finishes PLAID's middle settings,
  and `review_memory_by_size_*` is the check of agent memories by size. `review2_20261031_233333.log` is the second
  round (`scripts/pipeline/run_review2.sh`): PLAID's CPU engine at lower centroid thresholds
  (`results/msmarco/plaid_sweep_cpu_low_nbits2.json`), and the stopping baselines again with FixedBudget, TA-Margin and
  Patience also under the pooled correction. Every value of the earlier `results/stopping_baselines` is unchanged by the
  rerun, which only adds the keys `fixed_doc`, `margin_doc`, `patience` and `patience_doc`.
- All PLAID rows of Tables VII and IX come from one sweep. `gpu` in `plaid_nbits2.json` is the earlier single run, which
  gave the same recall and MRR@10 with slightly slower times, and the cost model still uses it.
