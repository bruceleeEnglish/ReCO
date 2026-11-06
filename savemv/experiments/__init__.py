"""Experiment drivers (imported lazily by the CLI).

    pipeline.py       ReCo alone on one BEIR dataset: encode, exact ground truth, sketches, stopping, latency
    bench.py          systematic comparison with all baselines on one BEIR dataset (bench_methods.py: the methods)
    memory.py         agent-memory retrieval (LongMemEval, LoCoMo) on per-user memories
"""
