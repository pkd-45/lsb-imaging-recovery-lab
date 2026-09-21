# Design rationale and traceability

This document maps public scientific and software-motivated design requirements to the corresponding benchmark components. It does not claim that any external group uses this exact implementation.

| Public role / research signal | Repository response | What it demonstrates |
|---|---|---|
| LSB detection and characterisation | known stream/shell/tail truth + segmentation + geometry metrics | moves past generic image classification |
| Source detection / deblending | point sources + companion galaxy + explicit companion leakage | measurable crowded-field failure |
| Residual-image workflow in STRRINGS | original/model/residual channels and ablation | source-model subtraction is part of the experiment |
| Cirrus/artefact confusion in Richards et al. | nuisance segmentation head + nuisance FPR/cross-talk | contaminants are first-class labels |
| Simulation/generative approaches | transparent synthetic forward model | exact truth and controlled interventions |
| Robust simulation-based inference literature | domain-shift suite + no transfer claim | training-domain performance is not assumed to generalise |
| GaMPEN uncertainty/calibration | Brier/ECE + MC-dropout error diagnostic | separates calibration from point performance |
| Invariance under observational transformations | D4, dimming, PSF and resampling tests | tests observation-induced transformations |
| STRRINGS track extraction | angular/radial track-recovery diagnostic | asks whether a segmentation is useful downstream |
| Stream-track physical inference | synthetic usable-track gate | makes the segmentation→physics interface explicit |
| GPU/HPC / benchmarking | device-aware PyTorch + throughput script | performance is measured, not asserted |
| Git/GitHub/testing/open software | package structure, pytest, CI, output validator | inspectable research software |

## Deliberate non-features

The benchmark does not attempt to look impressive by adding every modern method. In particular, v0.3 does not include a diffusion model, a foundation model, real Rubin/Euclid cutouts, or a claim of production-scale throughput. Those are appropriate later steps only after the baseline validation contract is clear.

## Natural next version after a local/GitHub gate

A scientifically stronger v0.4 would use real DESI Legacy Survey/SGA cutouts or realistic injections, preserve survey PSF/inverse-variance metadata, compare a pretrained astronomical representation against the compact U-Net, and test whether the same quality gate predicts downstream track failure on real annotated examples.
