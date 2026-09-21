import numpy as np

from lsbrecovery.simulate import ObservationConfig, simulate_batch, simulate_one


def test_simulate_one_contract():
    channels, truth, nuisance, companion, meta = simulate_one(11)
    assert channels.shape == (3, 64, 64)
    assert truth.shape == nuisance.shape == companion.shape == (64, 64)
    assert channels.dtype == np.float32
    assert np.isfinite(channels).all()
    assert meta["kind"] in {"stream", "shell", "tail"}
    assert 0 < truth.mean() < 0.5


def test_simulate_batch_is_deterministic():
    a = simulate_batch(4, 99)
    b = simulate_batch(4, 99)
    for idx in range(4):
        assert np.allclose(a[idx], b[idx])


def test_resampling_preserves_shape():
    cfg = ObservationConfig(resample_factor=0.5)
    channels, truth, nuisance, companion, _ = simulate_one(5, cfg)
    assert channels.shape == (3, 64, 64)
    assert truth.shape == nuisance.shape == companion.shape == (64, 64)


def test_unseen_trail_marks_nuisance():
    base = simulate_one(7, ObservationConfig(trail_strength=0.0))[2]
    trail = simulate_one(7, ObservationConfig(trail_strength=0.2))[2]
    assert trail.sum() >= base.sum()
