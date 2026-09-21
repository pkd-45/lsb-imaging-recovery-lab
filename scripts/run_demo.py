from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lsbrecovery.baseline import classical_segment
from lsbrecovery.calibration import binary_calibration_metrics
from lsbrecovery.invariance import d4_probability_disagreement
from lsbrecovery.metrics import (
    batch_characterisation_metrics,
    batch_metrics,
    companion_leakage_rate,
    cross_talk_rate,
    nuisance_false_positive_rate,
)
from lsbrecovery.preprocess import preprocess_batch
from lsbrecovery.simulate import ObservationConfig, simulate_batch
from lsbrecovery.stress import domain_shift_suite
from lsbrecovery.track import batch_stream_track_metrics
from lsbrecovery.train import predict_mc, predict_proba, save_checkpoint, train_model
from lsbrecovery.uncertainty import uncertainty_summary

MODES = ("original", "residual", "triplet")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("products"))
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--device", default="auto")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def tune_threshold(probability: np.ndarray, truth: np.ndarray) -> float:
    grid = np.linspace(0.25, 0.85, 13)
    scores = [batch_metrics(probability >= t, truth)["f1"] for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate_heads(
    probability: np.ndarray,
    tidal_truth: np.ndarray,
    nuisance_truth: np.ndarray,
    companion_truth: np.ndarray,
    tidal_threshold: float,
    nuisance_threshold: float,
) -> dict[str, float]:
    tidal_pred = probability[:, 0] >= tidal_threshold
    nuisance_pred = probability[:, 1] >= nuisance_threshold
    values = batch_metrics(tidal_pred, tidal_truth)
    nuisance_values = batch_metrics(nuisance_pred, nuisance_truth)
    values.update({f"nuisance_{k}": v for k, v in nuisance_values.items()})
    values["nuisance_false_positive_rate"] = nuisance_false_positive_rate(
        tidal_pred, nuisance_truth, tidal_truth
    )
    values["companion_leakage_rate"] = companion_leakage_rate(
        tidal_pred, companion_truth, tidal_truth
    )
    values["head_cross_talk_rate"] = cross_talk_rate(
        tidal_pred, nuisance_pred, tidal_truth, nuisance_truth
    )
    return values


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_train, n_val, n_test, epochs = (192, 48, 60, 4) if args.quick else (520, 112, 144, 7)
    cfg = ObservationConfig()

    train_channels, train_tidal, train_nuisance, _, _ = simulate_batch(
        n_train, args.seed, cfg
    )
    val_channels, val_tidal, val_nuisance, _, _ = simulate_batch(
        n_val, args.seed + 5_000, cfg
    )
    test_channels, test_tidal, test_nuisance, test_companion, test_meta = simulate_batch(
        n_test, args.seed + 10_000, cfg
    )
    train_targets = np.concatenate([train_tidal, train_nuisance], axis=1)
    tidal_truth = test_tidal[:, 0] >= 0.5
    nuisance_truth = test_nuisance[:, 0] >= 0.5
    companion_truth = test_companion[:, 0] >= 0.5
    val_tidal_truth = val_tidal[:, 0] >= 0.5
    val_nuisance_truth = val_nuisance[:, 0] >= 0.5

    fits = {}
    ablation = {}
    thresholds: dict[str, dict[str, float]] = {}
    probabilities = {}
    for j, mode in enumerate(MODES):
        fit = train_model(
            preprocess_batch(train_channels, mode),
            train_targets,
            seed=args.seed + j,
            epochs=epochs,
            batch_size=16,
            device=args.device,
        )
        fits[mode] = fit
        val_prob = predict_proba(
            fit.model, preprocess_batch(val_channels, mode), device=fit.device
        )
        tidal_threshold = tune_threshold(val_prob[:, 0], val_tidal_truth)
        nuisance_threshold = tune_threshold(val_prob[:, 1], val_nuisance_truth)
        thresholds[mode] = {
            "tidal": tidal_threshold,
            "nuisance": nuisance_threshold,
        }
        prob = predict_proba(
            fit.model, preprocess_batch(test_channels, mode), device=fit.device
        )
        probabilities[mode] = prob
        ablation[mode] = evaluate_heads(
            prob,
            tidal_truth,
            nuisance_truth,
            companion_truth,
            tidal_threshold,
            nuisance_threshold,
        )
        save_checkpoint(fit.model, args.output_dir / f"{mode}_model.pt", mode=mode)

    # Fixed a priori: the primary method receives the same original/model/residual triplet used in
    # the public STRRINGS workflow. We do not select the primary method by test-set performance.
    primary_mode = "triplet"
    main_fit = fits[primary_mode]
    primary_thresholds = thresholds[primary_mode]
    primary_x = preprocess_batch(test_channels, primary_mode)
    primary_prob = probabilities[primary_mode]
    primary_tidal_pred = primary_prob[:, 0] >= primary_thresholds["tidal"]
    save_checkpoint(main_fit.model, args.output_dir / "primary_model.pt", mode=primary_mode)

    baseline_pred = np.stack(
        [classical_segment(item[2]) >= 0.5 for item in test_channels], axis=0
    )
    baseline_metrics = {
        **batch_metrics(baseline_pred, tidal_truth),
        "nuisance_false_positive_rate": nuisance_false_positive_rate(
            baseline_pred, nuisance_truth, tidal_truth
        ),
        "companion_leakage_rate": companion_leakage_rate(
            baseline_pred, companion_truth, tidal_truth
        ),
    }

    char_metrics = batch_characterisation_metrics(primary_tidal_pred, tidal_truth)
    track_metrics = batch_stream_track_metrics(
        primary_tidal_pred,
        tidal_truth,
        [str(item["kind"]) for item in test_meta],
    )
    calibration = binary_calibration_metrics(
        primary_prob[:, 0], tidal_truth, nuisance=nuisance_truth, candidate_floor=0.20
    )
    tidal_invariance = d4_probability_disagreement(
        main_fit.model,
        primary_x[: min(12, len(primary_x))],
        main_fit.device,
        head=0,
    )
    nuisance_invariance = d4_probability_disagreement(
        main_fit.model,
        primary_x[: min(12, len(primary_x))],
        main_fit.device,
        head=1,
    )

    mc_mean, mc_std = predict_mc(
        main_fit.model,
        primary_x[: min(36, len(primary_x))],
        device=main_fit.device,
        passes=10 if args.quick else 18,
    )
    uncertainty = uncertainty_summary(
        mc_mean[:, 0],
        mc_std[:, 0],
        tidal_truth[: len(mc_mean)],
        threshold=primary_thresholds["tidal"],
    )

    shifts = domain_shift_suite(
        main_fit.model,
        input_mode=primary_mode,
        base_config=cfg,
        n=24 if args.quick else 72,
        seed=args.seed + 20_000,
        device=main_fit.device,
        tidal_threshold=primary_thresholds["tidal"],
        nuisance_threshold=primary_thresholds["nuisance"],
    )
    nominal_f1 = shifts["nominal"]["f1"]
    for values in shifts.values():
        values["relative_f1"] = values["f1"] / max(nominal_f1, 1e-9)

    metrics = {
        "scope": (
            "controlled synthetic methods benchmark; not a real-survey performance claim and "
            "not a reproduction of the Cambridge pipeline"
        ),
        "version": "0.3.0",
        "scientific_question": (
            "when does a tidal-feature segmentation remain useful for downstream stream "
            "characterisation under nuisance, deblending and observational shift?"
        ),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "epochs_per_ablation": epochs,
        "device": main_fit.device,
        "primary_mode": primary_mode,
        "primary_choice_policy": "fixed before test evaluation; original+model+residual triplet",
        "threshold_by_mode": thresholds,
        "input_ablation": ablation,
        "training_loss": {mode: fits[mode].losses for mode in MODES},
        "classical_residual_baseline": baseline_metrics,
        "primary_characterisation": char_metrics,
        "primary_stream_track": track_metrics,
        "primary_tidal_calibration": calibration,
        "primary_d4_mean_probability_disagreement": {
            "tidal": tidal_invariance,
            "nuisance": nuisance_invariance,
        },
        "primary_mc_dropout_diagnostic": uncertainty,
        "domain_shift": shifts,
        "feature_kind_counts": {
            kind: sum(item["kind"] == kind for item in test_meta)
            for kind in ["stream", "shell", "tail"]
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # Figure 1: STRRINGS-inspired original/model/residual information pattern plus both heads.
    stream_indices = [i for i, item in enumerate(test_meta) if item["kind"] == "stream"]
    i = max(stream_indices, key=lambda k: float(test_meta[k]["truth_fraction"]))
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.0), constrained_layout=True)
    panels = [
        (test_channels[i, 0], "Original"),
        (test_channels[i, 1], "Source model"),
        (test_channels[i, 2], "Residual"),
        (tidal_truth[i], "Known tidal truth"),
        (nuisance_truth[i], "Known nuisance truth"),
        (baseline_pred[i], "Classical residual baseline"),
        (primary_prob[i, 0], "Tidal probability"),
        (primary_prob[i, 1], "Nuisance probability"),
    ]
    for ax, (image, title) in zip(axes.ravel(), panels, strict=True):
        ax.imshow(image, origin="lower", cmap="gray", vmin=None, vmax=None)
        if title == "Tidal probability":
            ax.contour(tidal_truth[i], levels=[0.5], linewidths=0.7)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(args.output_dir / "overview.png", dpi=170)
    plt.close(fig)

    # Figure 2: input representation is an explicit ablation, not an implicit choice.
    labels = list(MODES) + ["classical"]
    f1s = [ablation[m]["f1"] for m in MODES] + [baseline_metrics["f1"]]
    leakage = [ablation[m]["companion_leakage_rate"] for m in MODES] + [
        baseline_metrics["companion_leakage_rate"]
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    xx = np.arange(len(labels))
    ax.bar(xx - 0.18, f1s, width=0.36, label="tidal F1")
    ax.bar(xx + 0.18, leakage, width=0.36, label="companion leakage")
    ax.set_xticks(xx, labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Metric")
    ax.legend()
    fig.savefig(args.output_dir / "representation_ablation.png", dpi=170)
    plt.close(fig)

    # Figure 3: observational and nuisance shift, including deliberately unseen satellite trails.
    labels = list(shifts)
    rel = [shifts[k]["relative_f1"] for k in labels]
    nuisance_fpr = [shifts[k]["nuisance_false_positive_rate"] for k in labels]
    fig, ax = plt.subplots(figsize=(10.0, 4.8), constrained_layout=True)
    xx = np.arange(len(labels))
    ax.plot(xx, rel, marker="o", label="tidal F1 / nominal")
    ax.plot(xx, nuisance_fpr, marker="s", label="tidal FPR on nuisances")
    ax.axhline(1.0, linewidth=1.0)
    ax.set_xticks(xx, labels, rotation=30, ha="right")
    ax.set_ylabel("Metric")
    ax.set_title("Recovery under observation, subtraction, crowding and unseen-nuisance shift")
    ax.legend()
    fig.savefig(args.output_dir / "domain_shift.png", dpi=170)
    plt.close(fig)

    # Figure 4: downstream utility and reliability diagnostics.
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), constrained_layout=True)
    axes[0].bar(
        ["track\ncompleteness", "usable\nstreams"],
        [track_metrics["track_completeness"], track_metrics["usable_fraction"]],
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Stream-track utility")
    axes[1].bar(["Brier", "ECE"], [calibration["brier"], calibration["ece"]])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Candidate-pixel calibration")
    axes[2].bar(
        ["tidal", "nuisance"],
        [tidal_invariance, nuisance_invariance],
    )
    axes[2].set_title("D4 probability disagreement")
    fig.savefig(args.output_dir / "quality_and_calibration.png", dpi=170)
    plt.close(fig)

    # Figure 5: MC-dropout dispersion is only an error-association diagnostic.
    j = i
    mc_example_x = preprocess_batch(test_channels[j : j + 1], primary_mode)
    ex_mean, ex_std = predict_mc(
        main_fit.model,
        mc_example_x,
        device=main_fit.device,
        passes=10 if args.quick else 18,
    )
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.0), constrained_layout=True)
    axes[0].imshow(test_channels[j, 2], origin="lower", cmap="gray")
    axes[0].set_title("Residual")
    axes[1].imshow(tidal_truth[j], origin="lower", cmap="gray")
    axes[1].set_title("Truth")
    axes[2].imshow(ex_mean[0, 0], origin="lower", cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("MC mean probability")
    axes[3].imshow(ex_std[0, 0], origin="lower", cmap="magma")
    axes[3].set_title("MC-dropout dispersion")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(args.output_dir / "uncertainty_diagnostic.png", dpi=170)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
