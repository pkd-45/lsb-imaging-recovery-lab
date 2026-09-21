from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lsbrecovery.preprocess import preprocess_batch
from lsbrecovery.simulate import ObservationConfig, simulate_batch
from lsbrecovery.train import load_checkpoint, predict_proba


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("products/reference"))
    parser.add_argument("--output", type=Path, default=Path("products/reference/hero_figure.png"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def display_limits(image: np.ndarray, lower: float = 2.0, upper: float = 99.3) -> tuple[float, float]:
    lo, hi = np.percentile(image, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(image)), float(np.max(image))
    return float(lo), float(hi)


def main() -> None:
    args = parse_args()
    metrics = json.loads((args.run_dir / "metrics.json").read_text())
    seed = 20260921
    n_test = int(metrics["n_test"])
    channels, tidal, _, _, meta = simulate_batch(
        n_test,
        seed + 10_000,
        ObservationConfig(),
    )
    truth = tidal[:, 0] >= 0.5

    model, mode = load_checkpoint(args.run_dir / "primary_model.pt", args.device)
    prob = predict_proba(model, preprocess_batch(channels, mode), device=args.device)[:, 0]
    threshold = float(metrics["threshold_by_mode"]["triplet"]["tidal"])

    stream_indices = [i for i, item in enumerate(meta) if item["kind"] == "stream"]
    example = max(stream_indices, key=lambda idx: float(meta[idx]["truth_fraction"]))

    observed = channels[example, 0]
    residual = channels[example, 2]
    truth_example = truth[example]
    probability = prob[example]

    shifts = metrics["domain_shift"]
    order = sorted(shifts, key=lambda key: shifts[key]["relative_f1"], reverse=True)
    label_map = {
        "nominal": "Nominal",
        "subtraction_mismatch": "Subtraction mismatch",
        "crowded_deblend": "Crowded / deblend",
        "higher_noise": "Higher noise",
        "unseen_satellite_trail": "Unseen satellite trail",
        "broader_psf": "Broader PSF",
        "strong_ghosts": "Strong ghosts",
        "sky_gradient": "Sky gradient",
        "coarse_resampling": "Coarse resampling",
        "dimmer_features": "Dimmer features",
        "strong_cirrus": "Strong cirrus",
    }
    labels = [label_map.get(key, key.replace("_", " ")) for key in order]
    values = [float(shifts[key]["relative_f1"]) for key in order]

    fig = plt.figure(figsize=(18.0, 8.8))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=(0.95, 0.95, 0.32, 2.75),
        left=0.045,
        right=0.985,
        top=0.88,
        bottom=0.16,
        wspace=0.18,
        hspace=0.14,
    )
    ax_obs = fig.add_subplot(grid[0, 0])
    ax_res = fig.add_subplot(grid[0, 1])
    ax_truth = fig.add_subplot(grid[1, 0])
    ax_prob = fig.add_subplot(grid[1, 1])
    ax_shift = fig.add_subplot(grid[:, 3])

    lo, hi = display_limits(observed, 4.0, 98.8)
    ax_obs.imshow(observed, origin="lower", cmap="gray", vmin=lo, vmax=hi)
    ax_obs.set_title("Observed image")

    lo, hi = display_limits(residual, 2.0, 99.0)
    ax_res.imshow(residual, origin="lower", cmap="gray", vmin=lo, vmax=hi)
    ax_res.set_title("Residual after source model")

    ax_truth.imshow(truth_example, origin="lower", cmap="gray", vmin=0, vmax=1)
    ax_truth.set_title("Known tidal truth")

    ax_prob.imshow(probability, origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax_prob.contour(truth_example, levels=[0.5], linewidths=1.2)
    ax_prob.set_title("Recovered tidal probability")

    for ax in (ax_obs, ax_res, ax_truth, ax_prob):
        ax.set_xticks([])
        ax.set_yticks([])

    ypos = np.arange(len(order))
    bars = ax_shift.barh(ypos, values)
    ax_shift.set_yticks(ypos, labels)
    ax_shift.invert_yaxis()
    ax_shift.set_xlim(0, 1.08)
    ax_shift.set_xlabel("Tidal F1 relative to nominal")
    ax_shift.set_title("Recovery under observational and nuisance shift")
    ax_shift.axvline(1.0, linestyle="--", linewidth=1.2)
    ax_shift.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, values, strict=True):
        ax_shift.text(
            min(value + 0.016, 1.035),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=10.5,
        )

    cirrus_idx = order.index("strong_cirrus")
    ax_shift.annotate(
        f"Strong cirrus: {(1.0 - values[cirrus_idx]) * 100:.0f}% loss in tidal F1",
        xy=(values[cirrus_idx], cirrus_idx),
        xytext=(0.50, cirrus_idx - 1.15),
        arrowprops={"arrowstyle": "->", "lw": 1.0},
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.9},
        fontsize=10.5,
    )

    ablation = metrics["input_ablation"]
    baseline = metrics["classical_residual_baseline"]
    track = metrics["primary_stream_track"]
    summary = (
        "Committed CPU quick reference\n"
        f"Triplet F1: {ablation['triplet']['f1']:.3f}   "
        f"Classical baseline: {baseline['f1']:.3f}\n"
        f"Stream-track completeness: {track['track_completeness']:.3f}"
    )
    ax_shift.text(
        0.98,
        0.025,
        summary,
        transform=ax_shift.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "alpha": 0.9},
    )

    fig.suptitle(
        "Faint tidal structure can be recovered — but reliability is condition-dependent",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.055,
        "Controlled synthetic benchmark: known truth → realistic imaging nuisances → segmentation → downstream recoverability",
        ha="center",
        fontsize=11.5,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"HERO_FIGURE={args.output}")
    print(f"EXAMPLE_INDEX={example}")
    print(f"TRIPLET_THRESHOLD={threshold:.2f}")


if __name__ == "__main__":
    main()
