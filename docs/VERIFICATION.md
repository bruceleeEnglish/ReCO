# Verification

This folder was checked against the paper and the experiment server before release, last on 2026-11-06. The commands
in the table can be rerun by anyone from this folder, in a fresh environment made from `requirements/paper.txt`.

| Check | How | Outcome |
|---|---|---|
| Numbers, tables and figures of the paper come from `results/` | `python scripts/paper/paper_assets.py results paper_artifacts` | 600 macros and 13 tables identical to the paper's; the 9 figures pixel-identical with matplotlib 3.10.8 |
| Run logs agree with `results/` | `python tools/verify_logs.py` | 5,953 logged values, no mismatch |
| Numbers written in the prose | `python tools/verify_literals.py` | all 71 statements hold |
| Code still works after the reorganization | `python -m unittest discover -s tests -t .` | 64 tests run: 60 pass, 4 are skipped without local models |

Two more checks needed access to the server, so they are recorded here:

- Every code file of `savemv`, `scripts`, `configs` and `tests` was identical to the server's copy before the scripts
  were sorted into folders. Moving them changed only paths and added reading of `.json.gz` to `make_report.py`.
- Every result file that the paper reads was identical to the file the server wrote, byte by byte. Here, only paths in the
  stored configurations were replaced, and the largest files were compressed.

## Logs against results

`tools/verify_logs.py` parses what each log printed and compares it with `results/`, up to half a unit of the printed
precision. Changing a single stored value makes it fail.

## Limits of these checks

- **Inverted index.** The run reported in the paper left no log. `logs/msmarco/inverted_rerun_20261030.*` is a later rerun
  on the same machine with the same command. Its deterministic outcomes equal the reported run exactly: probe depth,
  candidates, memory, and 50 of 50 prefixes identical to the full scan. Mean times vary between the two runs, as wall-clock
  times do: 105, 124 and 178 ms reported, 116, 112 and 150 ms in the rerun, with a full scan of 553 and 516 ms.
- **PLAID's settings.** With (4, 0.4, 4096) and (6, 0.4, 4096) a few queries exceed our 24 GB GPU, so
  `plaid_sweep_nbits2.json` records those points as stopped. `scripts/official_baselines/plaid_sweep_skip.py` skips the
  failing queries instead: 1 and 5 of 1,000. It ran here for the first time on the GPU, after unit tests only, because a
  trial run would have disturbed the timing runs before it. Its results are consistent with the sweep's neighbouring points.
- **GPU and CPU engines of PLAID.** They differ at equal settings, 0.811 against 0.823 at the defaults for k <= 10, so the
  device matters. The GPU path scores in fp16 and the CPU path in fp32, and their pruning code differs too, so this
  repository does not attribute the gap to one cause. Times are compared on the GPU only.
- **Storage cost.** `results/cost/cost_model.json` is a deterministic function of the other results, so it has no log.
  Recomputing it with `scripts/efficiency/cost_model.py` gives the identical file.
- **Timing.** Logs confirm what was measured. A rerun will show different times on other hardware or load.
- **Stochastic steps.** Seeds are fixed in the configurations, but GPU training of codebooks is not bit-reproducible across
  machines, so a rerun gives close and not identical numbers.
- **Question answering.** Answers come from an external API and a local language model. `results/qa` keeps per-question
  outcomes without question text, so reader accuracy can be recomputed from them, but rerunning the models may differ.
