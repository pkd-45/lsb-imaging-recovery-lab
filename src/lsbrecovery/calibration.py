from __future__ import annotations

from itertools import pairwise

import numpy as np


def binary_calibration_metrics(
    probability: np.ndarray,
    truth: np.ndarray,
    *,
    nuisance: np.ndarray | None = None,
    n_bins: int = 10,
    candidate_floor: float = 0.05,
) -> dict[str, float]:
    """Brier score and ECE on scientifically relevant candidate pixels.

    Easy empty sky can dominate pixel calibration statistics. Candidate pixels therefore include
    any true feature, any known nuisance, and any pixel assigned non-trivial feature probability.
    """
    p = np.asarray(probability, dtype=float)
    t = np.asarray(truth, dtype=bool)
    if p.shape != t.shape:
        raise ValueError("probability and truth must have the same shape")
    candidate = t | (p >= candidate_floor)
    if nuisance is not None:
        n = np.asarray(nuisance, dtype=bool)
        if n.shape != p.shape:
            raise ValueError("nuisance must match probability shape")
        candidate |= n
    if not np.any(candidate):
        return {"brier": 0.0, "ece": 0.0, "candidate_fraction": 0.0}
    pp = np.clip(p[candidate], 0.0, 1.0)
    yy = t[candidate].astype(float)
    brier = float(np.mean((pp - yy) ** 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(pp)
    for lo, hi in pairwise(edges):
        if hi == 1.0:
            take = (pp >= lo) & (pp <= hi)
        else:
            take = (pp >= lo) & (pp < hi)
        if not np.any(take):
            continue
        conf = float(pp[take].mean())
        freq = float(yy[take].mean())
        ece += float(take.sum()) / n_total * abs(conf - freq)
    return {
        "brier": brier,
        "ece": float(ece),
        "candidate_fraction": float(candidate.mean()),
    }
