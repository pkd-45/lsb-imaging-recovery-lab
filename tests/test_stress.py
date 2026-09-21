from lsbrecovery.model import TinyUNet
from lsbrecovery.simulate import ObservationConfig
from lsbrecovery.stress import domain_shift_suite


def test_stress_suite_contract():
    model = TinyUNet(in_channels=3, dropout=0.0)
    out = domain_shift_suite(
        model,
        input_mode="triplet",
        base_config=ObservationConfig(size=32),
        n=2,
        seed=3,
        device="cpu",
        tidal_threshold=0.5,
        nuisance_threshold=0.5,
    )
    assert "nominal" in out
    assert "subtraction_mismatch" in out
    assert "coarse_resampling" in out
    assert "crowded_deblend" in out
    assert "unseen_satellite_trail" in out
    assert "companion_leakage_rate" in out["nominal"]
