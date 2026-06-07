# Population-scale follow-up

Analysis and figure code for the follow-up paper **"A surrogate-floor control for cross-cohort
comparison of structure–function coupling subspaces"** (NeuroImage). This directory scales the
Riemannian Spiked Coupling Model (RSCM) — introduced in the companion paper (see the root
`README.md`) — to a UK Biobank *N*-ladder, and adds the surrogate-floor / matrix-normal control
for cross-cohort subspace comparison.

It reuses the RSCM runner shipped at the repository root: `train/run_rscm.py`.

As with the rest of this repository, **no subject-level data are included**. UK Biobank data are
available to approved researchers through the UK Biobank Access Management System
(<https://www.ukbiobank.ac.uk/>); the discovery / external clinical cohorts are available from the
corresponding author on reasonable request, subject to the relevant data-use agreements.

## Layout

```
population_scale/
  prepare/      build run_rscm-ready aligned features from UK Biobank feature arrays/chunks
  configs/      RSCM run configs for each N tier (paths are repo-relative)
  slurm/        SLURM submission templates (edit account/partition for your cluster)
  figures/      self-contained figure generators (matplotlib)
```

## The N-ladder

| Tier | N | prepare script | config |
|------|---|----------------|--------|
| smoke  | 1,079  | `prepare/prepare_ukb1079.py`             | `configs/config_rscm_smoke_ukb1079_le_harmon.yaml` |
| tier 2 | 11,820 | `prepare/prepare_ukb11820_from_chunks.py` | `configs/config_rscm_ukb11820_le_harmon_lam03.yaml` |
| tier 3 | 37,775 | `prepare/prepare_ukb37775_from_chunks.py` | `configs/config_rscm_ukb37775_le_harmon_lam03.yaml` |

## Running

All commands are run from the repository root.

```bash
pip install -r requirements.txt

# 1. Build aligned features (paths configurable via environment variables; see each script's
#    docstring). Source UK Biobank feature arrays/chunks are not distributed with this repo.
export UKB_CHUNK_DIR=/path/to/ukb_features/chunks
export ALIGNED_DIR=data/aligned_features_ukb37775
export SPLITS_DIR=data/splits_ukb37775
python population_scale/prepare/prepare_ukb37775_from_chunks.py

# 2. Fit RSCM across the N tier (config points at data/aligned_features_* and writes to results/).
python train/run_rscm.py \
    --config population_scale/configs/config_rscm_ukb37775_le_harmon_lam03.yaml \
    --seeds 42 43 44 --d 53 --rank_cap 50 --pca_ks 3 5 10 20 50 --lambda_grid 0.3 \
    --save_decomposition --out_dir results/rscm_ukb37775_le_harmon_lam03
# On a SLURM cluster, edit and submit the templates in population_scale/slurm/ instead.

# 3. Regenerate figures (each reads decompositions from results/ and writes a PDF/PNG beside it).
cd population_scale/figures
python fig_spectral.py
python fig_surrogate.py
# ... etc.
```

### Environment variables used by `prepare/`

| Variable | Meaning | Default |
|----------|---------|---------|
| `UKB_PILOT_DIR` | source pilot aligned-feature arrays (smoke tier) | `data/ukb_pilot` |
| `UKB_CHUNK_DIR` | source `chunk_0NN.npz` directory (tiers 2–3)     | `data/ukb_features/chunks` |
| `ALIGNED_DIR`   | output `aligned_features_*` directory             | `data/aligned_features_ukbNNNN` |
| `SPLITS_DIR`    | output `splits_*` directory                       | `data/splits_ukbNNNN` |

Generated feature arrays, splits, results, and figure images stay local (they are git-ignored).
