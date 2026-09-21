# Literature and group scan behind the benchmark

This document records the **public** material used to decide what belongs in this repository. It distinguishes published/public work from inference about the advertised post. It is not intended to imply access to unpublished Cambridge plans.

## 1. Scientific context

The benchmark is motivated by public work on low-surface-brightness tidal features, contaminant-aware segmentation, robust simulation-trained inference, astronomical image invariance, and the use of recovered stream geometry for downstream physical inference.

## 2. Low-surface-brightness discovery with machine learning

The most important point missed in the first scan is that Belokurov was PI of the 2021 Leverhulme Research Project Grant **“The Faint Universe Made Visible with Machine Learning”** (RPG-2021-205). A Cambridge postdoc funded under that grant explicitly sought expertise spanning galaxy dynamics, numerical galaxy-formation simulations, machine learning/computational statistics, spectroscopy, wide-field surveys and large datasets.

Sources:
- https://people.ast.cam.ac.uk/~vasily/cv/
- https://aas.org/jobregister/ad/b7a0e6bc

Belokurov’s broader programme uses large surveys to recover assembly history from streams, dwarfs and other faint structures. He participates in survey ecosystems including LSST, DES, DESI, 4MOST and Euclid.

## 3. Cambridge Project 28: discovery → characterisation → galaxy physics

The 2024–25 IoA project **“Discovering the Low Surface Brightness Universe. Faint dwarfs and streams in the DECaLS imaging survey”**, supervised by Elisabeth Sola, Vasily Belokurov and Wyn Evans, lays out three explicit steps:

1. discover tidal features and faint dwarfs in **residual DECaLS images**, including with ML;
2. extract useful information from the recovered structures, again potentially with ML;
3. correlate features with host mass/environment to learn about galaxy formation and evolution.

It also motivates narrow streams as probes of the host mass distribution/dark matter.

Source: Cambridge IoA 2024–25 research-project booklet, Project 28.

This is why the repository contains a downstream geometry test rather than ending at pixel segmentation.

## 4. Sola et al.: labels, contaminants, host properties and rotational support

Sola and collaborators built deep-image annotation sets containing tidal features **and nuisance/artefact classes**. Their 2025 CFHT study uses 475 nearby massive galaxies across multiple environments and relates low-surface-brightness features to host properties. The published analysis reports a tentative relation between tidal-feature incidence and rotational support, while noting the importance of stellar mass as a possible driver.

This is directly relevant to an applicant coming from resolved galaxy kinematics: the external tidal record and internal rotational support are two complementary views of assembly.

Representative sources:
- Sola et al. 2022, *Characterization of Low Surface Brightness structures in annotated deep images*.
- Sola et al. 2025, MNRAS 541, 3015, https://arxiv.org/abs/2503.18480

## 5. Richards et al.: segmentation must include contaminants

Richards, Paiement, Xie, Sola & Duc (2024), **“Panoptic Segmentation of Galactic Structures in LSB Images”**, explicitly frame Galactic dust/cirrus as a major confusion source. Their approach combines Mask R-CNN, a contaminant-specialised network, adaptive preprocessing and human-in-the-loop label improvement.

Source: https://arxiv.org/abs/2407.07494

This is the reason v0.3 uses a second nuisance head and measures cross-talk and false positives on known nuisance pixels. A single binary stream mask is too weak a proxy for the public problem.

## 6. STRRINGS: original/model/residual is a real analysis pattern

Sola et al. (2025), **“STRRINGS: STReams in Residual Images of Nearby GalaxieS”**, inspect 19,387 SGA-2020 galaxies using **original, model and residual** DESI Legacy Survey images. Residual images enhance faint asymmetries. The work releases a 35-galaxy stream sample and measures stream geometry, surface brightness, colours and stellar masses.

Crucially, segmentation is followed by track reconstruction. The public method segments non-overlapping stream portions in azimuth, extracts radial profiles in angular bins and fits one or more Gaussians to reconstruct the unwrapped stream track. The paper also discusses track failures caused by bright residual pixels, imperfect Tractor modelling and self-overlap.

Sources:
- https://arxiv.org/abs/2508.02154
- MNRAS 544, 735 (2025), DOI 10.1093/mnras/staf1647

This motivates:
- the fixed primary **original/model/residual triplet**;
- explicit source-model mismatch;
- companion/deblending leakage;
- a downstream stream-track metric.

## 7. Chemaly et al.: image recovery ultimately feeds physical inference

Chemaly et al. (2026) develop a hierarchical Bayesian framework for inferring the population distribution of dark-matter halo flattening from projected extragalactic stream tracks. The simulation paper validates an end-to-end forward-modelling pipeline and stresses model mismatch. The follow-up application to 32 STRRINGS streams introduces an additional variance term for model mismatch and track systematics and identifies a higher-quality 17-stream subset with useful constraining power.

Sources:
- MNRAS 550, stag1161 (2026), DOI 10.1093/mnras/stag1161
- https://arxiv.org/abs/2607.05510

This is the strongest reason not to optimize only segmentation F1. For the scientific programme, **a visually plausible mask can still be useless or biased for downstream inference**.

## 8. Robust scientific ML, uncertainty, scale and software

Several strands of public scientific-ML work motivate the validation design:

### Robust simulation-based inference
Lemos et al. (2023), with Cranmer, show that simulation-trained inference can become biased and overconfident under training-to-real distribution shift and study Bayesian-neural-network mitigation.

https://arxiv.org/abs/2207.08435

### GaMPEN
Ghosh et al. (2022), with Cranmer, estimate joint Bayesian posteriors for galaxy morphology and explicitly test whether predicted uncertainty tracks regions of poor performance.

https://arxiv.org/abs/2207.05107

### AION-1
The AION-1 collaboration, including Cranmer, trains an omnimodal astronomical foundation model on more than 200 million observations from Legacy Survey, HSC, SDSS, DESI and Gaia, with downstream tasks including morphology, retrieval and image segmentation.

https://arxiv.org/abs/2510.17960

### Bifrost
Cranmer led Bifrost, an open-source Python/C++ framework for high-throughput CPU/GPU astronomy stream processing. This predates the current imaging programme but is directly relevant to the job’s software/HPC emphasis.

https://arxiv.org/abs/1708.00720

Together, these argue for **distribution-shift tests, reliability diagnostics, machine-readable evaluation and throughput measurement**, not simply a notebook that trains a neural network.

## 9. Invariance and continuous astronomical representations

The 2025–26 Cambridge IoA Project 37, supervised by Belokurov, Sola and Koposov, asks for **invariant similarity metrics for galaxy images** and explicitly lists rotation, parity flips, cosmological dimming, pixel resampling and PSF changes as nuisance transformations unrelated to intrinsic morphology. It proposes rigorous benchmarking at Rubin/Euclid scale.

Cambridge PhD student George Vassilakis publicly describes his Cranmer/Belokurov project as ML + astronomical imaging for Stage-IV surveys and his current work as **TheCube**, a continuous spatio-spectral representation of the multi-survey sky.

Sources:
- Cambridge IoA 2025–26 research-project booklet, Project 37.
- https://github.com/GeorgeVassilakis

These public projects are adjacent evidence, not proof of the exact unpublished LE50980 work plan. They motivate the D4 consistency test and explicit PSF/resampling stress suite in this repository.

## 10. Design conclusion

The public evidence supports a more specific benchmark than the original v0.1 idea:

**known tidal truth → realistic nuisance + crowded bright sources → original/model/residual representation → joint tidal/nuisance segmentation → deblending leakage → invariance/domain-shift checks → stream-track recovery → quality gate for downstream science**.

That is the architecture implemented in v0.3.
