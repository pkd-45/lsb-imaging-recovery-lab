# Project decision after the second literature scan

## Why v0.1 was not enough

The first concept was a reasonable generic low-surface-brightness injection/recovery benchmark, but it under-read the specific Cambridge research ecosystem. The deeper scan revealed four concrete facts that materially change the design:

1. Belokurov has led a Leverhulme programme literally titled **“The Faint Universe Made Visible with Machine Learning.”**
2. The Cambridge LSB workflow publicly uses **residual survey images** and then **extracts physical/geometric information**, rather than stopping at detection.
3. The ML work in this ecosystem treats **contaminants/cirrus as explicit classes**, and the STRRINGS analysis documents errors from imperfect source modelling and overlapping structures.
4. Recent stream work turns **projected track quality into dark-matter inference**, including explicit model-mismatch variance and a high-quality subset.

## Consequence

The repo was therefore rebuilt around the interface between **ML recovery and downstream science**. v0.3 adds joint nuisance segmentation, companion/deblending leakage, an unseen nuisance, fixed triplet inputs, calibration, invariance and stream-track utility.

This is more relevant to the people involved than a larger network or a prettier segmentation demo because it asks the same methodological question visible across their public work: **what information survives the measurement/model pipeline, and when is it safe to use?**
