"""ReCo, the proposed method: centroid-bag sketch -> MaxSim estimate -> recompute from raw text + calibrated stop.

    sketch.py     index: per document, the set of codebook centroids its tokens fall into
    estimate.py   s_hat(q, d) = sum_i max_{c in C_d} <q_i, mu_c>
    stopping.py   stopping rules (fixed / conformal / k+1 / CRC / patience / hazard) and their offline evaluation
    hazard.py     the learned hazard stopping statistic (state features, logistic model)
    engine.py     online search: estimate, recompute candidates from raw text, stop
"""
from .engine import Calibration, ReCoSearcher, SearchResult
from .estimate import CentroidBagEstimator, TorchCentroidBagEstimator
from .hazard import FEATURES, HazardModel, HazardSettings, fit_hazard
from .sketch import CentroidBagSketch
from .stopping import (RULES, QuerySplit, StopPath, adaptive_stop, best_docs, calibrate, conformal_quantile,
                       crc_budget, crc_margin, evaluate_stopping, exact_topk, fit_linear_map, kfold_splits,
                       nonconformity, split_queries, topk_ranks)

__all__ = ["CentroidBagSketch", "CentroidBagEstimator", "TorchCentroidBagEstimator", "Calibration", "ReCoSearcher",
           "SearchResult", "FEATURES", "HazardModel", "HazardSettings", "fit_hazard", "RULES", "QuerySplit", "StopPath", "adaptive_stop", "best_docs", "calibrate",
           "conformal_quantile", "crc_budget", "crc_margin", "evaluate_stopping", "exact_topk", "fit_linear_map",
           "kfold_splits", "nonconformity", "split_queries", "topk_ranks"]
