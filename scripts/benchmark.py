from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from lsbrecovery.preprocess import preprocess_batch
from lsbrecovery.simulate import simulate_batch
from lsbrecovery.train import load_checkpoint, resolve_device


def timed(fn, repeats: int = 3) -> float:
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    return min(values)


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--n", type=int, default=128)
    args = p.parse_args()

    device = resolve_device(args.device)
    sim_cache = {}

    def make_data() -> None:
        sim_cache["batch"] = simulate_batch(args.n, 9917)[0]

    sim_seconds = timed(make_data)
    channels = sim_cache["batch"]
    model, mode = load_checkpoint(args.model, device)

    prep_cache = {}

    def make_inputs() -> None:
        prep_cache["batch"] = preprocess_batch(channels, mode)

    prep_seconds = timed(make_inputs)
    images = prep_cache["batch"]
    x = torch.as_tensor(images, dtype=torch.float32, device=device)
    with torch.no_grad():
        _ = model(x[: min(8, len(x))])
        _sync(device)

        def infer() -> None:
            _ = model(x)
            _sync(device)

        inference_seconds = timed(infer, repeats=5)

    report = {
        "device": device,
        "input_mode": mode,
        "n_images": args.n,
        "image_shape": list(images.shape[1:]),
        "model_parameters": int(sum(p.numel() for p in model.parameters())),
        "simulation_images_per_second": args.n / sim_seconds,
        "preprocess_images_per_second": args.n / prep_seconds,
        "inference_images_per_second": args.n / inference_seconds,
        "simulation_seconds": sim_seconds,
        "preprocess_seconds": prep_seconds,
        "inference_seconds": inference_seconds,
        "note": "Local microbenchmark only; not a cross-hardware performance claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
