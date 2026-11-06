"""Hazard stopping: when to stop verifying sketch candidates.

Recomputing candidates in decreasing estimate order is a sequential decision. Under a guarantee on average
recall, min E[R] s.t. E[missed fraction of the exact top-k] <= delta decomposes per query through its
Lagrangian: each recomputation costs 1 and each missed top-k document lambda / k. With the hazard
h(p) = E[number of exact top-k documents among the next recomputations | history at p] non-increasing
along the path, the one-step look-ahead rule "stop once h(p) <= eta, with eta common to all queries" is
optimal (monotone case of optimal stopping). Two consequences: rules that equalize each query's residual
risk (probabilistic top-k, tail-sum rules) over-spend on hard queries, and the fixed budget, the margin
rule and patience are this rule with h seen through a single feature.

ReCo estimates h with a logistic model on state features that are all available online:

  margin          theta_k - s_hat(next)                                    (sketch evidence)
  margin_online   theta_k - (a E(next) + mean_{recomputed}(s - a E))        (per-query offset)
  log_p           log(documents recomputed so far)
  log_patience    log(1 + recomputations since the running top-k last changed)   (outcome evidence)
  flatness        s_hat(next) - s_hat(next + ahead)                        (candidates still close)
  top_gap         s_hat(first) - theta_k
  log_n           log(corpus size)                                         (constant per corpus)

The stopping statistic is -logit(h_hat); a query stops at the first batch boundary where it reaches
lambda, so the stop is monotone in lambda and conformal risk control calibrates lambda with a
finite-sample guarantee whatever the model's accuracy: the model only has to rank states. Labels come
from exact scores alone (a final top-k document within the next `horizon` positions), no judgments.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FEATURES = ("margin", "margin_online", "log_p", "log_patience", "flatness", "top_gap", "log_n")


@dataclass
class HazardSettings:
    """How evaluate_stopping fits / uses the hazard model."""
    horizon: int = 64                 # label: an exact top-k document within the next `horizon` positions
    ahead: int = 64                   # flatness look-ahead (in documents)
    l2: float = 1.0                   # ridge on standardized coefficients
    dense: int = 256                  # training rows: every boundary up to `dense`, then a uniform sample
    budget: int = 2000
    seed: int = 0
    universal: "HazardModel | None" = field(default=None, repr=False)   # fixed model -> extra rule hazard_u


@dataclass
class HazardModel:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    horizon: int = 64
    ahead: int = 64
    features: tuple[str, ...] = FEATURES

    def statistic(self, x: np.ndarray) -> np.ndarray:
        """-logit(h_hat) for state rows x (n, F); larger means safer to stop."""
        return -(((np.asarray(x, dtype=np.float64) - self.mean) / self.scale) @ self.coef + self.intercept)

    def to_dict(self) -> dict:
        return {"features": list(self.features), "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coef": self.coef.tolist(), "intercept": float(self.intercept), "horizon": self.horizon,
                "ahead": self.ahead}

    @classmethod
    def from_dict(cls, d: dict) -> "HazardModel":
        if tuple(d["features"]) != FEATURES:
            raise ValueError(f"hazard model features {d['features']} != {list(FEATURES)}")
        return cls(np.asarray(d["mean"], dtype=np.float64), np.asarray(d["scale"], dtype=np.float64),
                   np.asarray(d["coef"], dtype=np.float64), float(d["intercept"]), int(d["horizon"]),
                   int(d["ahead"]))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HazardModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def logistic_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 1.0, iters: int = 100,
                 tol: float = 1e-10) -> tuple[np.ndarray, float]:
    """Weighted ridge logistic regression by Newton's method: min sum_i w_i loss_i + l2 / 2 |coef|^2
    (intercept not penalized; same objective as scikit-learn's LogisticRegression with C = 1 / l2)."""
    n, f = x.shape
    design = np.c_[x, np.ones(n)]
    reg = np.r_[np.full(f, l2), 0.0]
    beta = np.zeros(f + 1)
    y = np.asarray(y, dtype=np.float64)
    for _ in range(iters):
        p = 0.5 * (1.0 + np.tanh(0.5 * (design @ beta)))
        grad = design.T @ (w * (p - y)) + reg * beta
        hess = (design * (w * p * (1.0 - p))[:, None]).T @ design + np.diag(reg + 1e-9)
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.max(np.abs(step)) < tol:
            break
    return beta[:f], float(beta[f])


def training_rows(n: int, dense: int, budget: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Boundary indices to learn from (all of the first `dense`, a uniform sample of the rest) and weights
    that make every path count with its full length."""
    head = np.arange(min(n, dense))
    if n <= dense:
        return head, np.ones(len(head))
    take = min(n - dense, budget)
    tail = np.sort(rng.choice(np.arange(dense, n), take, replace=False))
    return np.r_[head, tail], np.r_[np.ones(len(head)), np.full(take, (n - dense) / take)]


def training_set(paths, maps, settings: HazardSettings, rng: np.random.Generator, weights=None):
    """Stacked (states, labels, weights) of StopPaths built with state=True; `maps` holds each path's (a, b)."""
    xs, ys, ws = [], [], []
    for i, (path, (a, b)) in enumerate(zip(paths, maps)):
        rows, w = training_rows(len(path.ps), settings.dense, settings.budget, rng)
        if not len(rows):
            continue
        xs.append(path.features(a, b)[rows])
        ys.append(path.hazard_labels(settings.horizon)[rows])
        ws.append(w * (1.0 if weights is None else weights[i]))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)


def fit_hazard(paths, maps, settings: HazardSettings, weights=None) -> HazardModel:
    """Logistic hazard model from StopPaths (state=True) and their score maps (a, b)."""
    x, y, w = training_set(paths, maps, settings, np.random.default_rng(settings.seed), weights)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0                       # constant features (log_n within one corpus)
    coef, intercept = logistic_fit((x - mean) / scale, y, w, settings.l2)
    return HazardModel(mean, scale, coef, intercept, settings.horizon, settings.ahead)


def hazard_gmax(model: HazardModel, path, a: float, b: float) -> np.ndarray:
    """Running maximum of the stopping statistic along a path (StopPath.stop binary-searches it)."""
    g = model.statistic(path.features(a, b))
    return np.maximum.accumulate(g) if len(g) else g
