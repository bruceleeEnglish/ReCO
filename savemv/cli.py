"""Command line.

    python -m savemv run    --config configs/reco_fiqa.yaml               # ReCo alone on one BEIR dataset
    python -m savemv bench  --config configs/beir_bench.yaml --set dataset=fiqa    # all methods, one dataset
    python -m savemv memory --bench locomo --data data/memory/locomo10.json        # agent-memory retrieval
    python -m savemv qa     --bench locomo --data data/memory/locomo10.json        # end-to-end memory QA (LLM)
    python -m savemv download fiqa                                                # fetch a BEIR dataset
"""
from __future__ import annotations

import argparse

import yaml

from .config import load_config
from .data import ensure_downloaded
from .utils import get_logger

log = get_logger(__name__)


def _overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got {pair!r}")
        out[key.strip()] = yaml.safe_load(value)   # typed: 8 -> int, [1, 2] -> list, null -> None
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="savemv", description="ReCo: late-interaction retrieval without token vectors")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("run", "ReCo alone on one BEIR dataset (YAML config)"),
                       ("bench", "all configured methods on one BEIR dataset (YAML config)")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--config", required=True)
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="override a config value, e.g. --set encoder.device=cpu (repeatable)")

    mem = sub.add_parser("memory", help="agent-memory retrieval on per-user memories (LongMemEval, LoCoMo)")
    mem.add_argument("--bench", choices=["longmemeval", "locomo"], required=True)
    mem.add_argument("--data", required=True)
    mem.add_argument("--out", default=None)
    mem.add_argument("--colbert", default="models/answerai-colbert-small-v1")
    mem.add_argument("--doc-maxlen", type=int, default=300)
    mem.add_argument("--codebooks", nargs="+", default=None, help="universal codebooks (.npy)")
    mem.add_argument("--dense", nargs="+", default=None, help="dense model directories")
    mem.add_argument("--delta", type=float, default=0.05)
    mem.add_argument("--cpu-threads", type=int, default=8)

    qa = sub.add_parser("qa", help="end-to-end memory QA: an LLM answers from each retriever's units")
    qa.add_argument("--bench", choices=["longmemeval", "locomo"], required=True)
    qa.add_argument("--data", required=True)
    qa.add_argument("--reader", default="api", help="'api' (the --model chat API) or hf:<local model dir>")
    qa.add_argument("--model", default="deepseek-flash", help="API model (answers with --reader api; always judges)")
    qa.add_argument("--retrievers", nargs="+", default=None)
    qa.add_argument("--top", type=int, default=10)
    qa.add_argument("--stop", choices=["hazard", "fixed"], default="hazard", help="rule of the '+R' retrievers")
    qa.add_argument("--delta", type=float, default=0.05)
    qa.add_argument("--full-subset", type=int, default=0)
    qa.add_argument("--max-context", type=int, default=0)
    qa.add_argument("--max-questions", type=int, default=0)
    qa.add_argument("--workers", type=int, default=32)
    qa.add_argument("--effort", default="high")
    qa.add_argument("--judge2", default=None, help="second judge model for an agreement check, e.g. deepseek-v4-pro")
    qa.add_argument("--judge2-sample", type=int, default=0)

    dl = sub.add_parser("download", help="download a BEIR dataset (MTEB layout)")
    dl.add_argument("dataset")
    dl.add_argument("--root", default="data")
    dl.add_argument("--split", default="test")
    dl.add_argument("--endpoint", default="https://huggingface.co")

    args = parser.parse_args(argv)
    if args.command == "download":
        log.info("dataset ready at %s", ensure_downloaded(args.dataset, args.root, args.split, args.endpoint))
        return 0
    if args.command == "memory":
        from .experiments.memory import MemoryConfig, run_memory   # heavy imports only when running
        cfg = MemoryConfig(args.bench, args.data, args.out, args.colbert, args.doc_maxlen, delta=args.delta,
                           cpu_threads=args.cpu_threads)
        if args.codebooks:
            cfg.codebooks = args.codebooks
        if args.dense:
            cfg.dense = args.dense
        run_memory(cfg)
        return 0
    if args.command == "qa":
        from .experiments.qa import QAConfig, run_qa
        cfg = QAConfig(args.bench, args.data, model=args.model, reader=args.reader, effort=args.effort, top=args.top,
                       stop=args.stop, delta=args.delta, full_subset=args.full_subset, max_context=args.max_context,
                       max_questions=args.max_questions, workers=args.workers, judge2=args.judge2,
                       judge2_sample=args.judge2_sample)
        if args.retrievers:
            cfg.retrievers = args.retrievers
        run_qa(cfg)
        return 0

    cfg = load_config(args.config, _overrides(args.set))
    if args.command == "bench":
        from .experiments.bench import Bench
        Bench(cfg).run()
    else:
        from .experiments.pipeline import Pipeline
        Pipeline(cfg).run()
    return 0
