import numpy as np
import pytest

from lsbrecovery.preprocess import in_channels_for_mode, preprocess_batch


def test_preprocess_modes():
    x = np.zeros((2, 3, 16, 16), dtype=np.float32)
    assert preprocess_batch(x, "original").shape == (2, 1, 16, 16)
    assert preprocess_batch(x, "residual").shape == (2, 1, 16, 16)
    assert preprocess_batch(x, "triplet").shape == (2, 3, 16, 16)
    assert in_channels_for_mode("original_residual") == 2


def test_preprocess_rejects_bad_mode():
    x = np.zeros((2, 3, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError):
        preprocess_batch(x, "bad")
