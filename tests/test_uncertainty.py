import numpy as np

from lsbrecovery.uncertainty import uncertainty_summary


def test_uncertainty_auc_prefers_errors():
    truth = np.array([[1, 1, 0, 0]], dtype=bool)
    prob = np.array([[0.9, 0.2, 0.8, 0.1]])
    std = np.array([[0.1, 0.9, 0.8, 0.1]])
    out = uncertainty_summary(prob, std, truth, threshold=0.5)
    assert out["error_auc"] > 0.5
    assert out["error_to_correct_std_ratio"] > 1.0
