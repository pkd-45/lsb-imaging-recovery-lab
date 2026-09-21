import numpy as np

from lsbrecovery.baseline import classical_segment


def test_classical_segment_contract():
    x = np.zeros((32, 32), dtype=float)
    x[8:10, 5:25] = 4
    y = classical_segment(x)
    assert y.shape == x.shape
    assert set(np.unique(y)).issubset({0.0, 1.0})
