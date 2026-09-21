from __future__ import annotations

import numpy as np

_INPUTS = {
    "original": (0,),
    "model": (1,),
    "residual": (2,),
    "triplet": (0, 1, 2),
    "original_residual": (0, 2),
}


def preprocess_batch(channels: np.ndarray, mode: str = "triplet") -> np.ndarray:
    """Select a reproducible view of original/model/residual channels."""
    if channels.ndim != 4 or channels.shape[1] != 3:
        raise ValueError("channels must have shape (N, 3, H, W)")
    if mode not in _INPUTS:
        raise ValueError(f"unknown mode={mode!r}; choose from {sorted(_INPUTS)}")
    return np.ascontiguousarray(channels[:, _INPUTS[mode]], dtype=np.float32)


def in_channels_for_mode(mode: str) -> int:
    if mode not in _INPUTS:
        raise ValueError(f"unknown mode={mode!r}")
    return len(_INPUTS[mode])
