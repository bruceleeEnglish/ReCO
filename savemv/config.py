"""Typed experiment configuration loaded from YAML."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EncoderConfig:
    repo: str = "answerdotai/answerai-colbert-small-v1"
    query_maxlen: int = 32
    doc_maxlen: int = 300
    batch_size: int = 128          # corpus encoding batch
    device: str = "cuda:0"
    fp16: bool = True              # autocast on CUDA only


@dataclass
class SketchConfig:
    codebook_sizes: list[int] = field(default_factory=lambda: [1024, 4096, 16384])
    kmeans_iters: int = 15
    points_per_centroid: int = 32  # k-means training sample = max(K * this, min_train_points)
    min_train_points: int = 200_000
    seed: int = 0


@dataclass
class StopConfig:
    k: int = 10
    deltas: list[float] = field(default_factory=lambda: [0.1, 0.05, 0.01])
    batch: int = 8                 # recomputations issued per step
    split: list[float] = field(default_factory=lambda: [0.2, 0.3, 0.5])  # fit / calibrate / test
    calib_top: int = 200           # estimator head used to fit the linear score map
    seed: int = 0
    folds: int = 0                 # > 1: k-fold cross-calibration (every query is tested once) instead of `split`
    fit_fraction: float = 0.4      # k-fold: share of the non-test queries used to fit the score map
    hazard_horizon: int = 64       # hazard label: an exact top-k document within the next H positions
    hazard_ahead: int = 64         # flatness feature look-ahead (documents)
    hazard_l2: float = 1.0         # ridge on the standardized logistic coefficients
    hazard_model: str | None = None   # universal hazard model JSON ({dataset}, {k} substituted) -> rule hazard_u


@dataclass
class LatencyConfig:
    enabled: bool = True
    codebook_size: int = 16384
    delta: float = 0.05
    num_queries: int = 100
    batches: list[int] = field(default_factory=lambda: [8, 32])
    rule: str = "rank"             # stopping rule: fixed, fixed_strict, or an adaptive one (rank, rank1, crc, ...)


@dataclass
class BenchConfig:
    """Systematic comparison: each entry of `methods` is {kind: ..., <params>} (see savemv.bench.methods)."""
    methods: list[dict[str, Any]] = field(default_factory=list)
    workers: int = 0               # CPU processes for token pooling (0 = all cores)


@dataclass
class ExperimentConfig:
    dataset: str
    data_root: str = "data"
    output_root: str = "outputs"
    split: str = "test"
    max_docs: int | None = None    # smoke tests: keep only the first max_docs documents
    max_queries: int | None = None # keep a seeded sample of at most this many judged queries
    exclude_identical_ids: bool = True   # BEIR convention: a query never retrieves the document with its own id
    hf_endpoint: str = "https://huggingface.co"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    sketch: SketchConfig = field(default_factory=SketchConfig)
    stop: StopConfig = field(default_factory=StopConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)

    @property
    def output_dir(self) -> Path:
        suffix = f"_n{self.max_docs}" if self.max_docs else ""
        return Path(self.output_root) / f"{self.dataset}{suffix}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS = {"encoder": EncoderConfig, "sketch": SketchConfig, "stop": StopConfig, "latency": LatencyConfig,
             "bench": BenchConfig}


def _build(cls: type, raw: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load a YAML config; `overrides` uses dotted keys, e.g. {"encoder.device": "cpu"}."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for key, value in (overrides or {}).items():
        node = raw
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    sections = {name: _build(cls, raw.pop(name, {}) or {}) for name, cls in _SECTIONS.items()}
    return _build(ExperimentConfig, {**raw, **sections})
