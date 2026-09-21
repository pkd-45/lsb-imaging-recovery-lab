from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("validation_runs/seed_sweep"))
    p.add_argument("--summary", type=Path, default=Path("products/seed_sweep/summary.json"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=20260921)
    return p.parse_args()


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, float | int]] = []
    for i in range(args.n_seeds):
        seed = args.base_seed + i
        run_dir = args.output_dir / f"seed_{seed}"
        cmd = [
            sys.executable,
            "scripts/run_demo.py",
            "--quick",
            "--device",
            args.device,
            "--seed",
            str(seed),
            "--output-dir",
            str(run_dir),
        ]
        print(f"SEED_SWEEP_RUN seed={seed}", flush=True)
        subprocess.run(cmd, check=True)
        subprocess.run(
            [sys.executable, "scripts/validate_outputs.py", str(run_dir)],
            check=True,
        )

        metrics = json.loads((run_dir / "metrics.json").read_text())
        triplet = float(metrics["input_ablation"]["triplet"]["f1"])
        residual = float(metrics["input_ablation"]["residual"]["f1"])
        classical = float(metrics["classical_residual_baseline"]["f1"])
        track = float(metrics["primary_stream_track"]["track_completeness"])
        cirrus = float(metrics["domain_shift"]["strong_cirrus"]["relative_f1"])
        records.append(
            {
                "seed": seed,
                "triplet_f1": triplet,
                "residual_f1": residual,
                "classical_f1": classical,
                "triplet_minus_residual": triplet - residual,
                "stream_track_completeness": track,
                "strong_cirrus_relative_f1": cirrus,
            }
        )

    summary = {
        "scope": "five-seed CPU quick stability check; synthetic benchmark, not a real-survey performance claim",
        "device": args.device,
        "n_seeds": args.n_seeds,
        "base_seed": args.base_seed,
        "records": records,
        "triplet_f1": mean_sd([float(r["triplet_f1"]) for r in records]),
        "residual_f1": mean_sd([float(r["residual_f1"]) for r in records]),
        "classical_f1": mean_sd([float(r["classical_f1"]) for r in records]),
        "triplet_minus_residual": mean_sd(
            [float(r["triplet_minus_residual"]) for r in records]
        ),
        "stream_track_completeness": mean_sd(
            [float(r["stream_track_completeness"]) for r in records]
        ),
        "strong_cirrus_relative_f1": mean_sd(
            [float(r["strong_cirrus_relative_f1"]) for r in records]
        ),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")

    print("SEED_SWEEP=PASS")
    print(f"N_SEEDS={args.n_seeds}")
    for key in (
        "triplet_f1",
        "residual_f1",
        "classical_f1",
        "triplet_minus_residual",
        "stream_track_completeness",
        "strong_cirrus_relative_f1",
    ):
        value = summary[key]
        print(f"{key.upper()}={value['mean']:.6f} +/- {value['sd']:.6f}")
    print(f"SUMMARY={args.summary}")


if __name__ == "__main__":
    main()
