from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .model import TinyUNet


@dataclass(frozen=True)
class TrainResult:
    model: TinyUNet
    losses: list[float]
    device: str


def _dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (-2, -1)
    num = 2.0 * torch.sum(prob * target, dim=dims) + 1.0
    den = torch.sum(prob + target, dim=dims) + 1.0
    return 1.0 - (num / den).mean()


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_model(
    images: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int = 7,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "auto",
    dropout: float = 0.08,
) -> TrainResult:
    if targets.ndim != 4 or targets.shape[1] != 2:
        raise ValueError("targets must have shape (N, 2, H, W): tidal, nuisance")
    torch.manual_seed(seed)
    np.random.seed(seed)
    selected = resolve_device(device)
    model = TinyUNet(in_channels=images.shape[1], out_channels=2, dropout=dropout).to(selected)
    x = torch.as_tensor(images, dtype=torch.float32)
    y = torch.as_tensor(targets, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    positives = y.sum(dim=(0, 2, 3)).numpy()
    total_per_head = float(y.shape[0] * y.shape[2] * y.shape[3])
    weights = []
    for pos in positives:
        neg = total_per_head - float(pos)
        weights.append(min(max(neg / max(float(pos), 1.0), 1.0), 20.0))
    pos_weight = torch.tensor(weights, dtype=torch.float32, device=selected).view(1, 2, 1, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        running = 0.0
        count = 0
        for xb, yb in loader:
            xb, yb = xb.to(selected), yb.to(selected)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            bce = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            loss = 0.65 * bce + 0.35 * _dice_loss(logits, yb)
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu()) * len(xb)
            count += len(xb)
        losses.append(running / max(count, 1))
    return TrainResult(model=model, losses=losses, device=selected)


def predict_proba(model: TinyUNet, images: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(images, dtype=torch.float32, device=device)
        return torch.sigmoid(model(x)).cpu().numpy()


def predict_mc(
    model: TinyUNet,
    images: np.ndarray,
    *,
    device: str,
    passes: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    if passes < 2:
        raise ValueError("passes must be >= 2")
    model.train()  # activates dropout; model contains no batch norm
    x = torch.as_tensor(images, dtype=torch.float32, device=device)
    draws = []
    with torch.no_grad():
        for _ in range(passes):
            draws.append(torch.sigmoid(model(x)).cpu().numpy())
    stack = np.stack(draws, axis=0)
    model.eval()
    return stack.mean(axis=0), stack.std(axis=0)


def save_checkpoint(model: TinyUNet, path: Path, *, mode: str) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_channels": model.in_channels,
            "out_channels": model.out_channels,
            "base": model.base,
            "dropout": model.dropout,
            "input_mode": mode,
        },
        path,
    )


def load_checkpoint(path: Path, device: str) -> tuple[TinyUNet, str]:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = TinyUNet(
        in_channels=int(payload["in_channels"]),
        out_channels=int(payload.get("out_channels", 2)),
        base=int(payload.get("base", 8)),
        dropout=float(payload.get("dropout", 0.08)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, str(payload.get("input_mode", "triplet"))
