# Scientific scope and assumptions

## What this synthetic benchmark is for

The simulator exists to make failure measurable. Every generated example has known tidal truth and known nuisance/deblending masks. That allows controlled tests of whether a learned segmentation degrades when the observation or subtraction differs from nominal training conditions.

## What the forward model contains

- compact analytic host-light profile;
- one stream, shell or tail;
- point sources and a companion galaxy;
- Gaussian PSF;
- sky noise and a low-order gradient;
- smooth cirrus-like background structure;
- optical ghost ring;
- optional satellite trail used as an unseen nuisance;
- imperfect bright-source model with flux/PSF/centroid mismatch;
- optional coarse downsample/upsample transformation.

## Important simplifications

The simulator has no calibrated photometric zeropoint, correlated detector noise, real sky background distribution, multi-band SED, survey masks, detector cosmetics, scattered-light physics, WCS, real Tractor source model, spatially varying PSF or cosmological forward model. Therefore a quantity called “feature contrast” is a model parameter, not a surface brightness in mag/arcsec².

The companion mask marks a compact footprint used only to quantify leakage. It is not a deblender truth catalogue.

## Reliability diagnostics

- Pixel metrics quantify segmentation overlap.
- Nuisance FPR quantifies contamination of the tidal head.
- Companion leakage isolates one deblending failure mode.
- Brier/ECE tests candidate-pixel probability calibration.
- MC-dropout standard deviation is reported only as an empirical error-ranking diagnostic.
- D4 disagreement tests one limited invariance family.
- Stress tests are deterministic controlled shifts, not estimates of real survey uncertainty.
- Track recovery asks whether a stream mask preserves enough geometry for a downstream ridgeline measurement.

## Synthetic quality gate

The `usable` flag in `track.py` is deliberately labelled **synthetic**. Its completeness/error thresholds are not taken from STRRINGS, Chemaly et al., DESI-LS, Euclid or Rubin. It exists to demonstrate how a segmentation pipeline can pass a downstream scientific-utility contract rather than merely maximize image-space F1.
