import numpy as np

from lsbrecovery.metrics import (
    batch_characterisation_metrics,
    batch_metrics,
    companion_leakage_rate,
    cross_talk_rate,
    nuisance_false_positive_rate,
)


def test_perfect_metrics():
    truth = np.zeros((2, 16, 16), dtype=bool)
    truth[:, 4:8, 5:10] = True
    values = batch_metrics(truth, truth)
    assert values["f1"] == 1.0
    char = batch_characterisation_metrics(truth, truth)
    assert char["area_fractional_error"] == 0.0
    assert char["radial_extent_fractional_error"] == 0.0


def test_nuisance_and_companion_leakage():
    truth = np.zeros((8, 8), dtype=bool)
    nuisance = np.zeros((8, 8), dtype=bool)
    companion = np.zeros((8, 8), dtype=bool)
    nuisance[0:2] = True
    companion[0:1] = True
    pred = nuisance.copy()
    assert nuisance_false_positive_rate(pred, nuisance, truth) == 1.0
    assert companion_leakage_rate(pred, companion, truth) == 1.0


def test_head_cross_talk_zero_when_heads_are_clean():
    tidal = np.zeros((8, 8), dtype=bool)
    nuisance = np.zeros((8, 8), dtype=bool)
    tidal[1:3, 1:3] = True
    nuisance[5:7, 5:7] = True
    assert cross_talk_rate(tidal, nuisance, tidal, nuisance) == 0.0
