# lsb-imaging-recovery-lab

**Release 0.3.2:** formatting-only local-gate repair for the exact Ruff 0.16.7 I001 findings reported on macOS; scientific logic and stored reference products are unchanged from 0.3.0.


A compact, auditable benchmark for a specific astronomical-imaging question:

> **When does a segmentation of faint tidal structure remain useful for downstream stream characterisation once PSF, sky, subtraction, crowding and contaminants change?**

This repository is a **methods demonstrator**, not a reproduction of the Cambridge LE50980 research programme and not a real-survey performance claim. It was designed after reading the public research trail around Miles Cranmer, Vasily Belokurov, Elisabeth Sola and the current Cambridge astronomical-imaging work rather than from the job title alone.

## Why this benchmark is shaped this way

The public Cambridge work points to an end-to-end problem, not merely a binary segmentation task:

1. **detect faint structure in wide-field imaging**;
2. **separate astrophysical tidal features from contaminants and deblending/subtraction artefacts**;
3. **use original, model and residual images together**;
4. **test invariance and robustness to observational nuisance transformations**;
5. **extract geometry from the recovered structures**;
6. **decide whether that geometry is good enough for physical inference**;
7. **build the workflow as scalable, testable scientific software**.

The strongest public clues are summarized in [`docs/LITERATURE_NOTES.md`](docs/LITERATURE_NOTES.md) and mapped to implementation choices in [`docs/CAMBRIDGE_ALIGNMENT.md`](docs/CAMBRIDGE_ALIGNMENT.md).

## What is implemented

- synthetic host galaxies with known stream, shell or tail truth;
- compact point-source and companion-galaxy crowding/deblending nuisance;
- cirrus-like diffuse contamination and optical ghost rings;
- a deliberately **unseen satellite-trail** stress case;
- PSF convolution, sky noise, sky gradients and coarse resampling;
- an imperfect bright-source model and explicit original/model/residual triplets;
- a transparent classical residual-threshold baseline;
- a small two-head U-Net for **tidal + nuisance segmentation**;
- original-only, residual-only and triplet input ablations;
- separate validation-set threshold selection for both heads;
- tidal F1/IoU/precision/recall and nuisance segmentation metrics;
- nuisance false-positive and companion-leakage diagnostics;
- D4 rotation/reflection consistency tests;
- candidate-pixel Brier score and expected calibration error;
- MC-dropout dispersion as an **error-association diagnostic only**;
- stream-track recovery from angular/radial bins;
- an explicitly synthetic downstream “usable stream” gate;
- stress tests for PSF, noise, sky, cirrus, ghosting, dimming, resampling, model mismatch, crowding and unseen nuisance;
- local throughput microbenchmarking;
- deterministic tests and machine-readable outputs.

## Why track recovery is included

The public STRRINGS workflow does not end at segmentation. Sola et al. use residual images to identify streams and then reconstruct stream tracks from angular bins and radial profiles. Recent Cambridge work then uses projected tracks for population-level dark-matter-halo inference, explicitly introducing extra variance for model mismatch/track systematics and defining a higher-quality subset.

Accordingly, this repository asks a stricter question than “did the network paint approximately the right pixels?” It measures whether the recovered stream geometry is still usable by a simple downstream track extractor. The track metric here is intentionally lightweight and **is not the STRRINGS fitter**.

## Quick start — Conda

```bash
conda env create -f environment.yml
conda activate lsb-imaging-recovery
export PYTHONPATH="$PWD/src"
pytest -q
python scripts/run_demo.py --quick --output-dir validation_runs/quick
python scripts/validate_outputs.py validation_runs/quick
python scripts/benchmark.py \
  --model validation_runs/quick/primary_model.pt \
  --output validation_runs/quick/benchmark.json
```

The reference workflow is Conda-only; the source tree is exposed through `PYTHONPATH` for local validation.

## Main outputs

`run_demo.py` produces:

- `metrics.json` — full machine-readable audit;
- `overview.png` — original/model/residual + truth + tidal/nuisance predictions;
- `representation_ablation.png` — original vs residual vs triplet vs classical;
- `domain_shift.png` — recovery under observational/nuisance shift;
- `quality_and_calibration.png` — downstream track utility, calibration and invariance;
- `uncertainty_diagnostic.png` — MC-dropout dispersion without a calibration claim;
- three ablation checkpoints and one fixed-primary checkpoint.

The **primary model is fixed to the original/model/residual triplet before test evaluation**. The code does not pick whichever input wins on the test set.


## Recorded quick reference run

The committed `products/reference/` directory is a deterministic CPU **quick** run used only as a software/scientific smoke test. In this run:

- residual-only tidal F1: `0.442`;
- fixed-primary triplet tidal F1: `0.423`;
- classical residual baseline tidal F1: `0.289`;
- mean stream-track completeness: `0.813`;
- synthetic usable-stream fraction: `0.615`;
- strong-cirrus tidal F1 relative to nominal: `0.376`;
- candidate-pixel ECE: `0.436`;
- MC-dropout error-ranking AUC: `0.437`.

The last two values are intentionally kept rather than hidden: in this small quick run the model probabilities are poorly calibrated and MC-dropout dispersion does **not** rank errors usefully. That is a failure diagnosis, not a successful uncertainty claim. Likewise, residual-only happens to outperform the triplet in this smoke run; the triplet remains the fixed primary because the input policy was declared before test evaluation rather than selected after looking at the test set.

## Scientific boundaries

This repository deliberately does **not** claim that:

- the simulator is a faithful DESI-LS, Rubin or Euclid forward model;
- the U-Net is state of the art;
- the synthetic thresholds define an astrophysical quality cut;
- MC dropout is a calibrated posterior;
- the contaminant model reproduces Galactic cirrus statistics;
- the track extractor reproduces STRRINGS/Jafar/Gaussian-profile fitting;
- synthetic performance transfers to real low-surface-brightness imaging;
- a better pixel F1 necessarily implies better physical inference.

The point is the **validation architecture**: known truth → observation/model/residual → multi-class recovery → nuisance/deblending stress → geometry recovery → reliability gate.

## Repository map

```text
src/lsbrecovery/simulate.py      known-truth image + nuisance forward model
src/lsbrecovery/preprocess.py    original/model/residual input policies
src/lsbrecovery/baseline.py      classical residual baseline
src/lsbrecovery/model.py         small two-head U-Net
src/lsbrecovery/train.py         deterministic training/checkpointing
src/lsbrecovery/metrics.py       segmentation, leakage and characterisation metrics
src/lsbrecovery/calibration.py   Brier/ECE candidate-pixel diagnostics
src/lsbrecovery/invariance.py    D4 consistency benchmark
src/lsbrecovery/uncertainty.py   MC-dropout error-association diagnostic
src/lsbrecovery/track.py         downstream stream-track utility test
src/lsbrecovery/stress.py        observational/deblending/unseen-nuisance shifts
scripts/run_demo.py              end-to-end benchmark
scripts/validate_outputs.py      generated-product contract
scripts/benchmark.py             local throughput microbenchmark
```

## Licence

MIT. Dependencies retain their own licences.
