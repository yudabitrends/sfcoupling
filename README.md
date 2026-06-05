# sfcoupling

Code for the paper **"Gray matter and functional connectivity share a low-dimensional
coupling subspace"** (Bi & Calhoun, TReNDS Center; submitted to *Imaging Neuroscience*).

The study benchmarks multivariate methods for mapping gray-matter (GM) morphometry to
resting-state functional network connectivity (FNC), and characterizes the *geometry* of the
learned mapping (subspace overlap, principal angles, spectral rank) rather than predictive
accuracy alone, across three cohorts (a schizophrenia case–control discovery set, an external
validation set, and the UK Biobank).

## What is here

This repository contains the **analysis and figure code** only. It does **not** contain any
subject-level data (see *Data availability* below).

```
models/        core fitting + metrics (Ridge, RRR, PLS, Nuclear Norm/ISTA, OptShrink, NN-Init MLP)
train/         training / evaluation drivers + configs (subspace analysis, bootstrap SV, CV)
preprocess/    feature-construction pipeline + synthetic unit-test fixtures
scripts/       analyses and figures for the paper, including:
  overlap_permutation_null.py        structured break-pairing null on the overlap statistic O
  hierarchy_break_pairing_null.py    per-tier break-pairing null
  hierarchy_resolved_dissociation.py per-tier overlap / R^2
  analyze_svd_modes.py               SVD coupling-mode decomposition + domain fingerprints
  compute_theory_hierarchy_summary.py  spiked-matrix (BBP) plug-in (negative sensitivity check)
  reviewer_revision_M4_scrambled_null.py / stratified_null.py   null analyses
  tiv_icv_validation.py              total-GM-proxy vs measured UK Biobank ICV
  reviewer_revision_figures/         figure generators (figA–figG)
```

## Reproducibility without data

The full analyses require the cohort feature matrices (not distributable; see below). The
pipeline can be exercised end-to-end on **synthetic data** with no external data:

```bash
pip install -r requirements.txt
python scripts/smoketest_synthetic.py     # runs the pipeline on generated synthetic features
pytest preprocess/tests                    # unit tests use the synthetic fixtures in preprocess/tests/fixtures
```

Configuration files under `train/` and `preprocess/` reference local data paths from the
compute environment used for the submission; adjust them to point at your own feature files
to reproduce the reported numbers.

## Data availability

- **Discovery (DS1) and external (DS2) cohorts** — derived neuroimaging features are available
  from the corresponding author on reasonable request, subject to the data-use agreements of
  the constituent studies (COBRE, FBIRN, MPRC, and a multi-site Chinese cohort).
- **UK Biobank** — data are available to approved researchers through the UK Biobank Access
  Management System (https://www.ukbiobank.ac.uk/); they cannot be redistributed here.

No subject-level data are included in this repository.

## Citation

> Bi Y., Calhoun V.D. *Gray matter and functional connectivity share a low-dimensional
> coupling subspace.* (under review).

## License

Released under the MIT License (see `LICENSE`).
