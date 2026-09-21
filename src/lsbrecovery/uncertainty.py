from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _binary_auc(score: np.ndarray, label: np.ndarray) -> float:
    score = np.asarray(score, dtype=float).ravel()
    label = np.asarray(label, dtype=bool).ravel()
    n_pos = int(label.sum())
    n_neg = int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(score, method="average")
    pos_rank_sum = float(ranks[label].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def uncertainty_summary(
    mean_prob: np.ndarray,
    std_prob: np.ndarray,
    truth: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    """Diagnostic association between MC-dropout dispersion and segmentation error.

    This is deliberately not labelled a calibrated Bayesian uncertainty estimate.
    """
    p = np.asarray(mean_prob)
    s = np.asarray(std_prob)
    t = np.asarray(truth, dtype=bool)
    pred = p >= threshold
    error = pred != t
    candidate = np.logical_or(t, p >= 0.10)
    if not np.any(candidate):
        return {
            "error_auc": 0.5,
            "mean_std_error": 0.0,
            "mean_std_correct": 0.0,
            "error_to_correct_std_ratio": 1.0,
        }
    c_error = error[candidate]
    c_std = s[candidate]
    mean_error = float(c_std[c_error].mean()) if np.any(c_error) else 0.0
    mean_correct = float(c_std[~c_error].mean()) if np.any(~c_error) else 0.0
    return {
        "error_auc": float(_binary_auc(c_std, c_error)),
        "mean_std_error": mean_error,
        "mean_std_correct": mean_correct,
        "error_to_correct_std_ratio": mean_error / max(mean_correct, 1e-9),
    }
