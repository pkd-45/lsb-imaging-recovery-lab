from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_opening, gaussian_filter, label


def classical_segment(residual: np.ndarray, threshold_sigma: float = 0.85) -> np.ndarray:
    """Smooth residual + robust threshold baseline."""
    if residual.ndim != 2:
        raise ValueError("residual must be 2-D")
    smoothed = gaussian_filter(residual, sigma=1.0, mode="reflect")
    med = np.median(smoothed)
    mad = np.median(np.abs(smoothed - med))
    sigma = max(1.4826 * mad, 1e-4)
    mask = smoothed > med + threshold_sigma * sigma
    mask = binary_opening(mask, iterations=1)
    labels, n = label(mask)
    if n == 0:
        return mask.astype(np.float32)
    counts = np.bincount(labels.ravel())
    keep = counts >= 8
    keep[0] = False
    return keep[labels].astype(np.float32)
