from __future__ import annotations

import numpy as np


def segmentation_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    p = np.asarray(pred, dtype=bool)
    t = np.asarray(truth, dtype=bool)
    if p.shape != t.shape:
        raise ValueError("pred and truth must have the same shape")
    tp = float(np.logical_and(p, t).sum())
    fp = float(np.logical_and(p, ~t).sum())
    fn = float(np.logical_and(~p, t).sum())
    union = tp + fp + fn
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "iou": tp / max(union, 1.0),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def batch_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    values = [segmentation_metrics(p, t) for p, t in zip(pred, truth, strict=True)]
    if not values:
        raise ValueError("empty batch")
    return {key: float(np.mean([v[key] for v in values])) for key in values[0]}


def nuisance_false_positive_rate(
    pred: np.ndarray, nuisance: np.ndarray, truth: np.ndarray
) -> float:
    """Fraction of known nuisance pixels incorrectly labelled as tidal structure."""
    p = np.asarray(pred, dtype=bool)
    n = np.asarray(nuisance, dtype=bool)
    t = np.asarray(truth, dtype=bool)
    if not (p.shape == n.shape == t.shape):
        raise ValueError("pred, nuisance and truth must have the same shape")
    region = np.logical_and(n, ~t)
    return float(np.logical_and(p, region).sum() / max(region.sum(), 1))


def companion_leakage_rate(pred: np.ndarray, companion: np.ndarray, truth: np.ndarray) -> float:
    """Measure deblending failure specifically on companion-galaxy pixels."""
    p = np.asarray(pred, dtype=bool)
    c = np.asarray(companion, dtype=bool)
    t = np.asarray(truth, dtype=bool)
    if not (p.shape == c.shape == t.shape):
        raise ValueError("pred, companion and truth must have the same shape")
    region = np.logical_and(c, ~t)
    return float(np.logical_and(p, region).sum() / max(region.sum(), 1))


def cross_talk_rate(
    tidal_pred: np.ndarray,
    nuisance_pred: np.ndarray,
    tidal_truth: np.ndarray,
    nuisance_truth: np.ndarray,
) -> float:
    """Average wrong-head activation on pixels belonging uniquely to the other class."""
    tp = np.asarray(tidal_pred, dtype=bool)
    npred = np.asarray(nuisance_pred, dtype=bool)
    tt = np.asarray(tidal_truth, dtype=bool)
    nt = np.asarray(nuisance_truth, dtype=bool)
    nuisance_only = nt & ~tt
    tidal_only = tt & ~nt
    tidal_on_nuisance = float((tp & nuisance_only).sum() / max(nuisance_only.sum(), 1))
    nuisance_on_tidal = float((npred & tidal_only).sum() / max(tidal_only.sum(), 1))
    return 0.5 * (tidal_on_nuisance + nuisance_on_tidal)


def _radial_extent(mask: np.ndarray) -> float:
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return 0.0
    cy = (mask.shape[0] - 1) / 2.0
    cx = (mask.shape[1] - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return float(np.quantile(rr, 0.95))


def characterisation_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    p = np.asarray(pred, dtype=bool)
    t = np.asarray(truth, dtype=bool)
    area_t = max(float(t.sum()), 1.0)
    area_ratio = float(p.sum()) / area_t
    ext_t = max(_radial_extent(t), 1e-6)
    ext_p = _radial_extent(p)
    return {
        "area_ratio": area_ratio,
        "area_fractional_error": abs(area_ratio - 1.0),
        "radial_extent_ratio": ext_p / ext_t,
        "radial_extent_fractional_error": abs(ext_p / ext_t - 1.0),
    }


def batch_characterisation_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    values = [characterisation_metrics(p, t) for p, t in zip(pred, truth, strict=True)]
    if not values:
        raise ValueError("empty batch")
    return {key: float(np.mean([v[key] for v in values])) for key in values[0]}
