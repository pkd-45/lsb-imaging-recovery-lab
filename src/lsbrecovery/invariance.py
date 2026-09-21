from __future__ import annotations

import numpy as np

from .train import predict_proba


def _transform(batch: np.ndarray, code: int) -> np.ndarray:
    k = code % 4
    flipped = code >= 4
    out = batch
    if flipped:
        out = np.flip(out, axis=-1)
    out = np.rot90(out, k=k, axes=(-2, -1))
    return np.ascontiguousarray(out)


def _invert(batch: np.ndarray, code: int) -> np.ndarray:
    k = code % 4
    flipped = code >= 4
    out = np.rot90(batch, k=-k, axes=(-2, -1))
    if flipped:
        out = np.flip(out, axis=-1)
    return np.ascontiguousarray(out)


def d4_probability_disagreement(
    model,
    images: np.ndarray,
    device: str,
    *,
    head: int = 0,
) -> float:
    """Mean absolute disagreement after undoing 8 rotations/reflections."""
    aligned: list[np.ndarray] = []
    for code in range(8):
        transformed = _transform(images, code)
        prob = predict_proba(model, transformed, device=device)[:, head : head + 1]
        aligned.append(_invert(prob, code))
    stack = np.stack(aligned, axis=0)
    mean = stack.mean(axis=0)
    return float(np.mean(np.abs(stack - mean)))
