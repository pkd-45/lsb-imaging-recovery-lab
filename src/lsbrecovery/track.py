from __future__ import annotations

from itertools import pairwise

import numpy as np


def _polar_points(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.nonzero(np.asarray(mask, dtype=bool))
    if len(xx) == 0:
        return np.empty(0), np.empty(0)
    cy = (mask.shape[0] - 1) / 2.0
    cx = (mask.shape[1] - 1) / 2.0
    dx = xx - cx
    dy = yy - cy
    theta = np.arctan2(dy, dx)
    radius = np.sqrt(dx**2 + dy**2)
    return theta, radius


def stream_track_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    n_bins: int = 24,
    min_truth_pixels: int = 2,
) -> dict[str, float]:
    """Approximate a STRRINGS-like stream ridgeline from angular radial bins.

    This intentionally evaluates whether a segmentation is useful for downstream geometry, rather
    than claiming to reproduce the STRRINGS Gaussian-profile track fitter.
    """
    p_theta, p_radius = _polar_points(pred)
    t_theta, t_radius = _polar_points(truth)
    if len(t_theta) == 0:
        return {"track_completeness": 0.0, "track_radial_mae_fraction": 1.0, "usable": 0.0}

    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    truth_bins = 0
    matched = 0
    errors: list[float] = []
    norm = max(min(truth.shape) / 2.0, 1.0)
    for lo, hi in pairwise(edges):
        t_sel = (t_theta >= lo) & (t_theta < hi)
        if int(t_sel.sum()) < min_truth_pixels:
            continue
        truth_bins += 1
        truth_r = float(np.median(t_radius[t_sel]))
        p_sel = (p_theta >= lo) & (p_theta < hi)
        if not np.any(p_sel):
            continue
        matched += 1
        pred_r = float(np.median(p_radius[p_sel]))
        errors.append(abs(pred_r - truth_r) / norm)
    completeness = matched / max(truth_bins, 1)
    mae = float(np.mean(errors)) if errors else 1.0
    # Synthetic quality gate: chosen only to make downstream utility explicit, not a scientific cut.
    usable = float(completeness >= 0.60 and mae <= 0.12)
    return {
        "track_completeness": float(completeness),
        "track_radial_mae_fraction": mae,
        "usable": usable,
    }


def batch_stream_track_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    kinds: list[str],
) -> dict[str, float]:
    values = [
        stream_track_metrics(p, t)
        for p, t, kind in zip(pred, truth, kinds, strict=True)
        if kind == "stream"
    ]
    if not values:
        return {
            "n_streams": 0.0,
            "track_completeness": 0.0,
            "track_radial_mae_fraction": 1.0,
            "usable_fraction": 0.0,
        }
    return {
        "n_streams": float(len(values)),
        "track_completeness": float(np.mean([v["track_completeness"] for v in values])),
        "track_radial_mae_fraction": float(
            np.mean([v["track_radial_mae_fraction"] for v in values])
        ),
        "usable_fraction": float(np.mean([v["usable"] for v in values])),
    }
