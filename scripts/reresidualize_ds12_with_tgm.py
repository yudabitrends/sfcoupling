#!/usr/bin/env python3
"""
Re-residualize DS1/DS2 aligned features with total_gm (TIV proxy) added
to the design matrix.

Reads raw arrays + covariates from existing aligned_features, computes
total_gm = sum(GM_ROI_values) per subject, and re-runs residualization
with design matrix [1, Age, Gender, total_gm].

Overwrites the scaled arrays (dataset{1,2}_{X,Y}.npy) in place.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALIGNED = Path("/data/users1/ybi3/cVAE/aligned_features")
SPLITS = ALIGNED.parent / "splits"


def main():
    # Load raw arrays
    X1_raw = np.load(ALIGNED / "dataset1_X_raw.npy").astype(np.float64)
    Y1_raw = np.load(ALIGNED / "dataset1_Y_raw.npy").astype(np.float64)
    X2_raw = np.load(ALIGNED / "dataset2_X_raw.npy").astype(np.float64)
    Y2_raw = np.load(ALIGNED / "dataset2_Y_raw.npy").astype(np.float64)
    logger.info("Loaded raw: X1=%s Y1=%s X2=%s Y2=%s", X1_raw.shape, Y1_raw.shape, X2_raw.shape, Y2_raw.shape)

    # Load covariates
    cov1 = pd.read_csv(ALIGNED / "meta" / "dataset1_subjects.tsv", sep="\t")
    cov2 = pd.read_csv(ALIGNED / "meta" / "dataset2_subjects.tsv", sep="\t")

    # Load splits to identify train subjects
    split1 = json.loads((SPLITS / "dataset1_split.json").read_text())
    train_ids = set(str(s) for s in split1["train"])
    sub1_ids = cov1["SubjectID"].astype(str).values
    train_mask = np.array([s in train_ids for s in sub1_ids])
    logger.info("Train mask: %d / %d", train_mask.sum(), len(train_mask))

    # Compute total_gm (TIV proxy)
    tgm1 = X1_raw.sum(axis=1)
    tgm2 = X2_raw.sum(axis=1)
    logger.info("total_gm DS1: mean=%.2f std=%.2f | DS2: mean=%.2f std=%.2f",
                tgm1.mean(), tgm1.std(), tgm2.mean(), tgm2.std())

    # Build design matrices: [1, Age, Gender, total_gm]
    def _build_design(cov_df, tgm):
        age = pd.to_numeric(cov_df["Age"], errors="coerce").values.astype(np.float64)
        gender = pd.to_numeric(cov_df["Gender"], errors="coerce").values.astype(np.float64)
        age_med = float(np.nanmedian(age))
        gender_med = float(np.nanmedian(gender))
        age = np.nan_to_num(age, nan=age_med)
        gender = np.nan_to_num(gender, nan=gender_med)
        return np.column_stack([
            np.ones(len(age), dtype=np.float64),
            age,
            gender,
            tgm.astype(np.float64),
        ])

    design1 = _build_design(cov1, tgm1)
    design2 = _build_design(cov2, tgm2)
    design_train = design1[train_mask]
    logger.info("Design matrix shape: DS1=%s DS2=%s (cols: intercept, Age, Gender, total_gm)", design1.shape, design2.shape)

    # Residualize X (GM) — fit on train only
    lr_x = LinearRegression().fit(design_train, X1_raw[train_mask])
    X1_res = X1_raw - lr_x.predict(design1)
    X2_res = X2_raw - lr_x.predict(design2)

    # Residualize Y (FNC) — fit on train only
    lr_y = LinearRegression().fit(design_train, Y1_raw[train_mask])
    Y1_res = Y1_raw - lr_y.predict(design1)
    Y2_res = Y2_raw - lr_y.predict(design2)
    logger.info("Residualization complete")

    # Z-score — fit on train only
    scaler_x = StandardScaler().fit(X1_res[train_mask])
    scaler_y = StandardScaler().fit(Y1_res[train_mask])
    X1_z = scaler_x.transform(X1_res).astype(np.float32)
    Y1_z = scaler_y.transform(Y1_res).astype(np.float32)
    X2_z = scaler_x.transform(X2_res).astype(np.float32)
    Y2_z = scaler_y.transform(Y2_res).astype(np.float32)
    logger.info("Z-scoring complete")

    # Save (overwrite scaled arrays)
    np.save(ALIGNED / "dataset1_X.npy", X1_z)
    np.save(ALIGNED / "dataset1_Y.npy", Y1_z)
    np.save(ALIGNED / "dataset2_X.npy", X2_z)
    np.save(ALIGNED / "dataset2_Y.npy", Y2_z)

    # Also save residualized (pre-zscore) arrays
    np.save(ALIGNED / "dataset1_X_resid.npy", X1_res.astype(np.float32))
    np.save(ALIGNED / "dataset1_Y_resid.npy", Y1_res.astype(np.float32))
    np.save(ALIGNED / "dataset2_X_resid.npy", X2_res.astype(np.float32))
    np.save(ALIGNED / "dataset2_Y_resid.npy", Y2_res.astype(np.float32))

    # Update covariates metadata
    covariates_info = {
        "covariate_cols": ["Age", "Gender", "total_gm"],
        "design_cols": ["intercept", "Age", "Gender", "total_gm"],
    }
    (ALIGNED / "meta" / "covariates_used.json").write_text(json.dumps(covariates_info, indent=2))
    logger.info("Saved updated arrays and metadata to %s", ALIGNED)


if __name__ == "__main__":
    main()
