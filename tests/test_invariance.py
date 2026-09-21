import numpy as np

from lsbrecovery.invariance import d4_probability_disagreement
from lsbrecovery.model import TinyUNet


def test_d4_disagreement_is_finite_for_each_head():
    model = TinyUNet(in_channels=3, dropout=0.0)
    x = np.zeros((2, 3, 32, 32), dtype=np.float32)
    for head in (0, 1):
        value = d4_probability_disagreement(model, x, "cpu", head=head)
        assert np.isfinite(value)
        assert value >= 0.0
