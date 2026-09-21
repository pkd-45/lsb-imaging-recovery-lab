import numpy as np

from lsbrecovery.calibration import binary_calibration_metrics


def test_perfect_binary_probabilities_have_zero_brier_and_ece():
    truth = np.array([[0, 1, 1, 0]], dtype=bool)
    prob = truth.astype(float)
    out = binary_calibration_metrics(prob, truth, n_bins=5)
    assert out["brier"] == 0.0
    assert out["ece"] == 0.0
