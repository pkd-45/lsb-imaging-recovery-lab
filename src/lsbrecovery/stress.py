from __future__ import annotations

from dataclasses import replace

from .metrics import (
    batch_metrics,
    companion_leakage_rate,
    cross_talk_rate,
    nuisance_false_positive_rate,
)
from .preprocess import preprocess_batch
from .simulate import ObservationConfig, simulate_batch
from .train import predict_proba


def domain_shift_suite(
    model,
    *,
    input_mode: str,
    base_config: ObservationConfig,
    n: int,
    seed: int,
    device: str,
    tidal_threshold: float,
    nuisance_threshold: float,
) -> dict[str, dict[str, float]]:
    variants = {
        "nominal": base_config,
        "broader_psf": replace(base_config, psf_sigma=1.9),
        "higher_noise": replace(base_config, sky_sigma=0.085),
        "sky_gradient": replace(base_config, sky_gradient=0.065),
        "strong_cirrus": replace(base_config, cirrus_strength=0.080),
        "strong_ghosts": replace(base_config, ghost_strength=0.090),
        "dimmer_features": replace(base_config, feature_scale=0.60),
        "coarse_resampling": replace(base_config, resample_factor=0.5),
        "subtraction_mismatch": replace(
            base_config,
            model_flux_bias=0.07,
            model_psf_scale=1.18,
            model_shift_pixels=0.55,
        ),
        "crowded_deblend": replace(
            base_config,
            companion_strength=0.65,
            companion_distance=(0.20, 0.42),
            model_flux_bias=0.05,
            model_shift_pixels=0.35,
        ),
        # Satellite trails are absent from nominal training: this is explicit unseen-nuisance shift.
        "unseen_satellite_trail": replace(base_config, trail_strength=0.13),
    }
    out: dict[str, dict[str, float]] = {}
    for name, cfg in variants.items():
        channels, masks, nuisance, companion, _ = simulate_batch(n=n, seed=seed, config=cfg)
        prob = predict_proba(model, preprocess_batch(channels, input_mode), device=device)
        tidal_pred = prob[:, 0] >= tidal_threshold
        nuisance_pred = prob[:, 1] >= nuisance_threshold
        truth = masks[:, 0] >= 0.5
        nuisance_truth = nuisance[:, 0] >= 0.5
        companion_truth = companion[:, 0] >= 0.5
        values = batch_metrics(tidal_pred, truth)
        nuisance_values = batch_metrics(nuisance_pred, nuisance_truth)
        values.update({f"nuisance_{k}": v for k, v in nuisance_values.items()})
        values["nuisance_false_positive_rate"] = nuisance_false_positive_rate(
            tidal_pred, nuisance_truth, truth
        )
        values["companion_leakage_rate"] = companion_leakage_rate(
            tidal_pred, companion_truth, truth
        )
        values["head_cross_talk_rate"] = cross_talk_rate(
            tidal_pred, nuisance_pred, truth, nuisance_truth
        )
        out[name] = values
    return out
