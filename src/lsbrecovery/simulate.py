from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, shift, zoom


@dataclass(frozen=True)
class ObservationConfig:
    size: int = 64
    psf_sigma: float = 1.25
    sky_sigma: float = 0.050
    sky_gradient: float = 0.025
    cirrus_strength: float = 0.030
    ghost_strength: float = 0.035
    feature_contrast: tuple[float, float] = (0.10, 0.26)
    feature_scale: float = 1.0
    n_point_sources: tuple[int, int] = (1, 5)
    companion_strength: float = 0.35
    companion_distance: tuple[float, float] = (0.40, 0.78)
    trail_strength: float = 0.0  # zero in training; non-zero is an unseen-nuisance stress test
    model_flux_bias: float = 0.0
    model_psf_scale: float = 1.0
    model_shift_pixels: float = 0.0
    resample_factor: float = 1.0


def _grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    return np.meshgrid(axis, axis, indexing="xy")


def _robust_normalize(image: np.ndarray, lo: float = -6.0, hi: float = 12.0) -> np.ndarray:
    med = float(np.median(image))
    mad = float(np.median(np.abs(image - med)))
    scale = max(1.4826 * mad, 1e-4)
    return np.clip((image - med) / scale, lo, hi).astype(np.float32)


def _elliptical_component(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    cx: float = 0.0,
    cy: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    theta = rng.uniform(0, np.pi)
    ct, st = np.cos(theta), np.sin(theta)
    xr = ct * (x - cx) + st * (y - cy)
    yr = -st * (x - cx) + ct * (y - cy)
    q = rng.uniform(0.45, 0.9)
    re = rng.uniform(0.20, 0.36)
    r = np.sqrt(xr**2 + (yr / q) ** 2)
    n = rng.uniform(1.0, 3.0)
    body = np.exp(-np.power(np.maximum(r / re, 1e-5), 1.0 / n))
    bulge = rng.uniform(0.05, 0.25) * np.exp(-0.5 * (r / rng.uniform(0.05, 0.11)) ** 2)
    return (scale * (body + bulge)).astype(np.float32)


def _stream(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a curved, host-centred stream suitable for a track-recovery benchmark."""
    rr = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    centre = rng.uniform(-np.pi, np.pi)
    span = rng.uniform(0.75, 1.55)
    dphi = np.arctan2(np.sin(phi - centre), np.cos(phi - centre))
    base_radius = rng.uniform(0.42, 0.72)
    slope = rng.uniform(-0.12, 0.12)
    curvature = rng.uniform(-0.06, 0.06)
    ridge_radius = base_radius + slope * dphi + curvature * dphi**2
    width = rng.uniform(0.018, 0.040)
    angular = np.exp(-0.5 * (dphi / span) ** 4)
    ridge = np.exp(-0.5 * ((rr - ridge_radius) / width) ** 2)
    signal = ridge * angular
    return signal.astype(np.float32), signal > 0.22


def _shell(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    cx, cy = rng.uniform(-0.12, 0.12, size=2)
    rr = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    radius = rng.uniform(0.46, 0.74)
    width = rng.uniform(0.024, 0.050)
    phi = np.arctan2(y - cy, x - cx)
    centre = rng.uniform(-np.pi, np.pi)
    span = rng.uniform(0.65, 1.35)
    dphi = np.arctan2(np.sin(phi - centre), np.cos(phi - centre))
    angular = np.exp(-0.5 * (dphi / span) ** 4)
    signal = np.exp(-0.5 * ((rr - radius) / width) ** 2) * angular
    return signal.astype(np.float32), signal > 0.24


def _tail(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    phi = rng.uniform(0, 2 * np.pi)
    ct, st = np.cos(phi), np.sin(phi)
    xr = ct * x + st * y
    yr = -st * x + ct * y
    slope = rng.uniform(-0.45, 0.45)
    centre = slope * (xr + 0.15) + rng.uniform(-0.12, 0.12)
    width = rng.uniform(0.035, 0.074)
    ridge = np.exp(-0.5 * ((yr - centre) / width) ** 2)
    gate = 1.0 / (1.0 + np.exp(-18.0 * (xr - rng.uniform(-0.18, 0.08))))
    fade = np.exp(-np.maximum(xr, 0) / rng.uniform(0.45, 0.8))
    signal = ridge * gate * fade
    return signal.astype(np.float32), signal > 0.24


def _point_sources(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_range: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros_like(x, dtype=np.float32)
    mask = np.zeros_like(x, dtype=bool)
    n = int(rng.integers(n_range[0], n_range[1] + 1))
    for _ in range(n):
        cx, cy = rng.uniform(-0.95, 0.95, size=2)
        sigma = rng.uniform(0.010, 0.030)
        amp = rng.uniform(0.20, 0.85)
        source = amp * np.exp(-0.5 * (((x - cx) / sigma) ** 2 + ((y - cy) / sigma) ** 2))
        image += source.astype(np.float32)
        mask |= source > 0.04
    return image, mask


def _companion(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    strength: float,
    distance: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if strength <= 0:
        return np.zeros_like(x, dtype=np.float32), np.zeros_like(x, dtype=bool)
    r = rng.uniform(*distance)
    phi = rng.uniform(0, 2 * np.pi)
    cx, cy = r * np.cos(phi), r * np.sin(phi)
    image = _elliptical_component(x, y, rng, cx=cx, cy=cy, scale=strength)
    # The leakage mask marks the compact companion footprint rather than the full Sersic tail.
    footprint = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) < rng.uniform(0.10, 0.16)
    return image, footprint


def _cirrus(
    shape: tuple[int, int], rng: np.random.Generator, strength: float
) -> tuple[np.ndarray, np.ndarray]:
    field = gaussian_filter(rng.normal(size=shape), sigma=rng.uniform(3.5, 7.0))
    field -= field.mean()
    field /= max(field.std(), 1e-6)
    nuisance = field > 1.0
    return (strength * field).astype(np.float32), nuisance


def _ghost_ring(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    if strength <= 0:
        return np.zeros_like(x, dtype=np.float32), np.zeros_like(x, dtype=bool)
    cx, cy = rng.uniform(-0.65, 0.65, size=2)
    rr = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    radius = rng.uniform(0.24, 0.55)
    width = rng.uniform(0.035, 0.075)
    ring = np.exp(-0.5 * ((rr - radius) / width) ** 2)
    return (strength * ring).astype(np.float32), ring > 0.35


def _satellite_trail(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    if strength <= 0:
        return np.zeros_like(x, dtype=np.float32), np.zeros_like(x, dtype=bool)
    phi = rng.uniform(0, np.pi)
    ct, st = np.cos(phi), np.sin(phi)
    cross = -st * x + ct * y - rng.uniform(-0.5, 0.5)
    along = ct * x + st * y
    width = rng.uniform(0.012, 0.028)
    line = np.exp(-0.5 * (cross / width) ** 2) * (np.abs(along) < 1.2)
    return (strength * line).astype(np.float32), line > 0.2


def _resample_square(image: np.ndarray, factor: float, order: int) -> np.ndarray:
    if factor == 1.0:
        return image
    if not (0.25 <= factor <= 1.0):
        raise ValueError("resample_factor must be in [0.25, 1.0]")
    small = zoom(image, factor, order=order, mode="reflect", prefilter=False)
    restored = zoom(small, 1.0 / factor, order=order, mode="reflect", prefilter=False)
    target = image.shape[0]
    out = np.zeros_like(image)
    h = min(target, restored.shape[0])
    w = min(target, restored.shape[1])
    y0_src = max((restored.shape[0] - h) // 2, 0)
    x0_src = max((restored.shape[1] - w) // 2, 0)
    y0_dst = max((target - h) // 2, 0)
    x0_dst = max((target - w) // 2, 0)
    out[y0_dst : y0_dst + h, x0_dst : x0_dst + w] = restored[
        y0_src : y0_src + h, x0_src : x0_src + w
    ]
    return out


def simulate_one(
    seed: int,
    config: ObservationConfig | None = None,
    feature_kind: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | str]]:
    """Return original/model/residual, tidal truth, nuisance truth, companion mask, metadata.

    This is a transparent toy forward model. It deliberately contains source-model mismatch and
    contaminant classes so recovery and deblending failure can be measured under known truth.
    """
    cfg = config or ObservationConfig()
    rng = np.random.default_rng(seed)
    x, y = _grid(cfg.size)
    host = _elliptical_component(x, y, rng)

    kind = feature_kind or str(rng.choice(["stream", "shell", "tail"]))
    generators = {"stream": _stream, "shell": _shell, "tail": _tail}
    if kind not in generators:
        raise ValueError(f"unknown feature_kind={kind!r}")
    feature_shape, truth = generators[kind](x, y, rng)
    contrast = rng.uniform(*cfg.feature_contrast) * cfg.feature_scale
    tidal = (contrast * feature_shape).astype(np.float32)

    stars, star_mask = _point_sources(x, y, rng, cfg.n_point_sources)
    companion, companion_mask = _companion(
        x, y, rng, cfg.companion_strength, cfg.companion_distance
    )
    cirrus, cirrus_mask = _cirrus(host.shape, rng, cfg.cirrus_strength)
    ghost, ghost_mask = _ghost_ring(x, y, rng, cfg.ghost_strength)
    trail, trail_mask = _satellite_trail(x, y, rng, cfg.trail_strength)

    astrophysical = host + tidal + stars + companion
    observed = gaussian_filter(astrophysical, cfg.psf_sigma, mode="reflect")

    gx, gy = rng.normal(size=2)
    gradient = cfg.sky_gradient * (gx * x + gy * y) / max(np.hypot(gx, gy), 1e-6)
    observed = observed + gradient + cirrus + ghost + trail
    observed = observed + rng.normal(0.0, cfg.sky_sigma, size=observed.shape)

    # A deliberately imperfect high-surface-brightness source model. Companion/point sources are
    # modelled, but mismatch leaves residuals: a compact stand-in for deblending/subtraction errors.
    model_sigma = max(cfg.psf_sigma * cfg.model_psf_scale, 0.25)
    model = gaussian_filter(host + stars + companion, model_sigma, mode="reflect")
    model *= 1.0 + cfg.model_flux_bias
    if cfg.model_shift_pixels != 0.0:
        model = shift(
            model,
            shift=(cfg.model_shift_pixels, -0.65 * cfg.model_shift_pixels),
            order=1,
            mode="nearest",
            prefilter=False,
        )

    residual = observed - model
    nuisance = cirrus_mask | ghost_mask | star_mask | companion_mask | trail_mask

    if cfg.resample_factor != 1.0:
        observed = _resample_square(observed, cfg.resample_factor, order=1)
        model = _resample_square(model, cfg.resample_factor, order=1)
        residual = observed - model
        truth = _resample_square(truth.astype(np.float32), cfg.resample_factor, order=0) > 0.5
        nuisance = _resample_square(nuisance.astype(np.float32), cfg.resample_factor, order=0) > 0.5
        companion_mask = (
            _resample_square(companion_mask.astype(np.float32), cfg.resample_factor, order=0) > 0.5
        )

    channels = np.stack(
        [
            _robust_normalize(observed),
            _robust_normalize(model),
            _robust_normalize(residual, lo=-10.0, hi=10.0),
        ],
        axis=0,
    ).astype(np.float32)

    meta: dict[str, float | str] = {
        "kind": kind,
        "contrast": float(contrast),
        "psf_sigma": float(cfg.psf_sigma),
        "sky_sigma": float(cfg.sky_sigma),
        "cirrus_strength": float(cfg.cirrus_strength),
        "ghost_strength": float(cfg.ghost_strength),
        "companion_strength": float(cfg.companion_strength),
        "trail_strength": float(cfg.trail_strength),
        "feature_scale": float(cfg.feature_scale),
        "model_flux_bias": float(cfg.model_flux_bias),
        "model_psf_scale": float(cfg.model_psf_scale),
        "model_shift_pixels": float(cfg.model_shift_pixels),
        "resample_factor": float(cfg.resample_factor),
        "truth_fraction": float(truth.mean()),
        "nuisance_fraction": float(nuisance.mean()),
        "companion_fraction": float(companion_mask.mean()),
    }
    return (
        channels,
        truth.astype(np.float32),
        nuisance.astype(np.float32),
        companion_mask.astype(np.float32),
        meta,
    )


def simulate_batch(
    n: int,
    seed: int,
    config: ObservationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, float | str]]]:
    channels, masks, nuisances, companions, meta = [], [], [], [], []
    for i in range(n):
        c, m, nu, co, info = simulate_one(seed + 104729 * i, config=config)
        channels.append(c)
        masks.append(m[None])
        nuisances.append(nu[None])
        companions.append(co[None])
        meta.append(info)
    return (
        np.stack(channels),
        np.stack(masks),
        np.stack(nuisances),
        np.stack(companions),
        meta,
    )
