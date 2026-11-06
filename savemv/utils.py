"""Small shared helpers: logging, devices, timing and JSON output."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_LOG_FORMAT = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the `savemv` namespace, configuring the root handler once."""
    root = logging.getLogger("savemv")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, "%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    return logging.getLogger(name if name.startswith("savemv") else f"savemv.{name}")


def resolve_device(name: str) -> torch.device:
    """`cuda[:i]` falls back to CPU when CUDA is unavailable."""
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Timer:
    """Wall-clock timer that synchronizes CUDA so GPU work is fully counted."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device
        self.seconds = 0.0

    def __enter__(self) -> "Timer":
        if self.device is not None:
            synchronize(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        if self.device is not None:
            synchronize(self.device)
        self.seconds = time.perf_counter() - self._t0

    @property
    def ms(self) -> float:
        return 1000.0 * self.seconds


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_to_builtin), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
