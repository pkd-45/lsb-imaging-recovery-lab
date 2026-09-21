from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    """Small inspectable U-Net with tidal and nuisance segmentation heads."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        base: int = 8,
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base = base
        self.dropout = dropout
        self.enc1 = ConvBlock(in_channels, base, dropout)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, 2 * base, dropout)
        self.bottleneck = ConvBlock(2 * base, 4 * base, dropout)
        self.up2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, stride=2)
        self.dec2 = ConvBlock(4 * base, 2 * base, dropout)
        self.up1 = nn.ConvTranspose2d(2 * base, base, 2, stride=2)
        self.dec1 = ConvBlock(2 * base, base, dropout)
        self.head = nn.Conv2d(base, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)
