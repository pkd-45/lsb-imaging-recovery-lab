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
    parser.add_argument(
        "--seed-summary",
        type=Path,
        default=Path("products/seed_sweep/summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("products/reference/hero_figure.png"),
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def display_limits(
    image: np.ndarray, lower: float = 2.0, upper: float = 99.3
) -> tuple[float, float]:
    lo, hi = np.percentile(image, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(image)), float(np.max(image))
    return float(lo), float(hi)


def core_false_positive(probability: np.ndarray, truth: np.ndarray) -> tuple[int, int]:
    h, w = probability.shape
    yy, xx = np.indices(probability.shape)
    rr = np.sqrt((yy - (h - 1) / 2.0) ** 2 + (xx - (w - 1) / 2.0) ** 2)
    core = rr < 0.24 * min(h, w)
    candidate = np.where(core & ~truth, probability, -np.inf)
    if not np.isfinite(candidate).any():
        candidate = np.where(~truth, probability, -np.inf)
    y, x = np.unravel_index(int(np.nanargmax(candidate)), candidate.shape)
    return int(y), int(x)


def main() -> None:
    args = parse_args()
    metrics = json.loads((args.run_dir / "metrics.json").read_text())
    sweep = json.loads(args.seed_summary.read_text())

    seed = 20260921
    n_test = int(metrics["n_test"])
    channels, tidal, _, _, meta = simulate_batch(
        n_test,
        seed + 10_000,
        ObservationConfig(),
    )
    truth = tidal[:, 0] >= 0.5

    model, mode = load_checkpoint(args.run_dir / "primary_model.pt", args.device)
    prob = predict_proba(
        model, preprocess_batch(channels, mode), device=args.device
    )[:, 0]

    stream_indices = [i for i, item in enumerate(meta) if item["kind"] == "stream"]
    example = max(stream_indices, key=lambda idx: float(meta[idx]["truth_fraction"]))

    observed = channels[example, 0]
    residual = channels[example, 2]
    truth_example = truth[example]
    probability = prob[example]

    shift_summary = sweep["domain_shift_relative_f1"]
    order = sorted(
        shift_summary,
        key=lambda key: float(shift_summary[key]["mean"]),
        reverse=True,
    )
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
    values = [float(shift_summary[key]["mean"]) for key in order]
    errors = [float(shift_summary[key]["sd"]) for key in order]

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
    ax_prob.contour(truth_example, levels=[0.5], linewidths=1.4)
    ax_prob.set_title("Recovered tidal probability")

    fp_y, fp_x = core_false_positive(probability, truth_example)
    ax_prob.annotate(
        "host-core subtraction residual\n(false positive)",
        xy=(fp_x, fp_y),
        xytext=(3, 8),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "lw": 1.2},
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "alpha": 0.88},
        fontsize=8.7,
        ha="left",
        va="bottom",
    )

    for ax in (ax_obs, ax_res, ax_truth, ax_prob):
        ax.set_xticks([])
        ax.set_yticks([])

    ypos = np.arange(len(order))
    bars = ax_shift.barh(
        ypos,
        values,
        xerr=errors,
        capsize=3,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )
    ax_shift.set_yticks(ypos, labels)
    ax_shift.invert_yaxis()

    xmax = max(v + e for v, e in zip(values, errors, strict=True))
    ax_shift.set_xlim(0, max(1.12, xmax + 0.08))
    ax_shift.set_xlabel("Tidal F1 relative to nominal")
    ax_shift.set_title("Recovery under observational and nuisance shift — 5-seed mean ± SD")
    ax_shift.axvline(1.0, linestyle="--", linewidth=1.2)
    ax_shift.grid(axis="x", alpha=0.2)

    for bar, value, error in zip(bars, values, errors, strict=True):
        ax_shift.text(
            min(value + error + 0.018, ax_shift.get_xlim()[1] - 0.045),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=10.2,
        )

    cirrus_idx = order.index("strong_cirrus")
    cirrus_mean = values[cirrus_idx]
    cirrus_sd = errors[cirrus_idx]
    loss_pct = 100.0 * (1.0 - cirrus_mean)
    loss_sd_pct = 100.0 * cirrus_sd
    ax_shift.annotate(
        f"Strong cirrus: {loss_pct:.0f} ± {loss_sd_pct:.0f}% loss in tidal F1",
        xy=(cirrus_mean, cirrus_idx),
        xycoords="data",
        xytext=(0.53, 0.14),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1.0},
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.92},
        fontsize=10.5,
    )

    summary = (
        f"{sweep['n_seeds']}-seed CPU quick sweep\n"
        f"Triplet F1: {sweep['triplet_f1']['mean']:.3f} ± "
        f"{sweep['triplet_f1']['sd']:.3f}   "
        f"Residual: {sweep['residual_f1']['mean']:.3f} ± "
        f"{sweep['residual_f1']['sd']:.3f}\n"
        f"Classical: {sweep['classical_f1']['mean']:.3f} ± "
        f"{sweep['classical_f1']['sd']:.3f}   "
        f"Track completeness: "
        f"{sweep['stream_track_completeness']['mean']:.3f} ± "
        f"{sweep['stream_track_completeness']['sd']:.3f}"
    )
    ax_shift.text(
        0.98,
        0.025,
        summary,
        transform=ax_shift.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.0,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "alpha": 0.92},
    )

    fig.suptitle(
        "Faint tidal structure can be recovered — but reliability is condition-dependent",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.055,
        "Controlled synthetic benchmark: known truth → imaging nuisances → "
        "segmentation → downstream recoverability",
        ha="center",
        fontsize=11.5,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"HERO_FIGURE={args.output}")
    print(f"EXAMPLE_INDEX={example}")
    print(f"FALSE_POSITIVE_PIXEL=({fp_y},{fp_x})")
    print(f"STRONG_CIRRUS_LOSS={loss_pct:.2f} +/- {loss_sd_pct:.2f} percent")


if __name__ == "__main__":
    main()
