from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _bounded(value: float, lo: float = 0.0, hi: float = 1.0) -> bool:
    return math.isfinite(value) and lo <= value <= hi


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("output_dir", type=Path)
    args = p.parse_args()
    required = [
        "metrics.json",
        "overview.png",
        "representation_ablation.png",
        "domain_shift.png",
        "quality_and_calibration.png",
        "uncertainty_diagnostic.png",
        "original_model.pt",
        "residual_model.pt",
        "triplet_model.pt",
        "primary_model.pt",
    ]
    for name in required:
        path = args.output_dir / name
        if not path.exists() or path.stat().st_size == 0:
            raise SystemExit(f"missing/empty output: {path}")

    metrics = json.loads((args.output_dir / "metrics.json").read_text())
    if metrics.get("version") != "0.3.0":
        raise SystemExit("unexpected metrics version")
    if metrics.get("primary_mode") != "triplet":
        raise SystemExit("primary mode must be the predeclared triplet")

    for family in metrics["input_ablation"].values():
        for key in [
            "iou",
            "precision",
            "recall",
            "f1",
            "nuisance_f1",
            "nuisance_false_positive_rate",
            "companion_leakage_rate",
            "head_cross_talk_rate",
        ]:
            if not _bounded(float(family[key])):
                raise SystemExit(f"invalid metric {key}={family[key]}")

    expected_shifts = {
        "nominal",
        "broader_psf",
        "higher_noise",
        "sky_gradient",
        "strong_cirrus",
        "strong_ghosts",
        "dimmer_features",
        "coarse_resampling",
        "subtraction_mismatch",
        "crowded_deblend",
        "unseen_satellite_trail",
    }
    if set(metrics["domain_shift"]) != expected_shifts:
        raise SystemExit("domain-shift suite incomplete")
    for values in metrics["domain_shift"].values():
        for key in ["f1", "nuisance_f1", "nuisance_false_positive_rate", "companion_leakage_rate"]:
            if not _bounded(float(values[key])):
                raise SystemExit(f"invalid shift metric {key}")
        if not math.isfinite(float(values["relative_f1"])) or values["relative_f1"] < 0:
            raise SystemExit("invalid relative F1")

    track = metrics["primary_stream_track"]
    for key in ["track_completeness", "usable_fraction"]:
        if not _bounded(float(track[key])):
            raise SystemExit(f"invalid track metric {key}")
    if not math.isfinite(float(track["track_radial_mae_fraction"])):
        raise SystemExit("invalid track radial error")

    calibration = metrics["primary_tidal_calibration"]
    for key in ["brier", "ece", "candidate_fraction"]:
        if not _bounded(float(calibration[key])):
            raise SystemExit(f"invalid calibration metric {key}")

    uncertainty = metrics["primary_mc_dropout_diagnostic"]
    if not _bounded(float(uncertainty["error_auc"])):
        raise SystemExit("invalid uncertainty AUC")
    print("OUTPUT_VALIDATION=PASS")


if __name__ == "__main__":
    main()
