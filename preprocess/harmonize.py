"""
Tangent-space harmonization for RSCM.

The NeuroImage main paper residualizes Age/Gender directly in edge space
(off-diagonal Fisher-z vectors). This breaks PD-ness of the resulting
"correlation" matrices (empirical: 12% of eigenvalues go negative on DS1).

The MIA follow-up's thread-2 contribution is to defer confound removal to
the SPD tangent space: reconstruct raw Y as an SPD matrix, log-map to
tangent at the Frechet mean, then fit and subtract a linear confound model
in the flat tangent space. The SPD structure is preserved throughout;
downstream RSCM regression sees confound-free tangent vectors.

Fit on training covariates only (no leakage), apply to train + held-out
splits with the same design matrix convention.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np


def _add_intercept(C: np.ndarray) -> np.ndarray:
    """Prepend a column of 1s to a (n, k) covariate matrix."""
    if C.ndim != 2:
        raise ValueError(f"Expected (n, k), got {C.shape}")
    n = C.shape[0]
    return np.hstack([np.ones((n, 1), dtype=C.dtype), C])


def fit_tangent_residualizer(
    T_train: np.ndarray,
    covariates_train: np.ndarray,
) -> np.ndarray:
    """OLS of T_train on covariates_train (with intercept). Returns beta.

    T_train       : (n, q) tangent vectors (q = d(d+1)/2 for SPD tangent)
    covariates_train : (n, k) covariate matrix (e.g., [Age, Gender])

    The returned `beta` has shape (k+1, q) and is meant to be used with
    `apply_tangent_residualizer` on both train and test splits.
    """
    if T_train.shape[0] != covariates_train.shape[0]:
        raise ValueError("T_train and covariates_train row count mismatch.")
    D_tr = _add_intercept(covariates_train.astype(np.float64))
    T_tr64 = T_train.astype(np.float64)
    # Closed-form OLS: beta = pinv(D) @ T
    beta, *_ = np.linalg.lstsq(D_tr, T_tr64, rcond=None)
    return beta.astype(np.float64)


def apply_tangent_residualizer(
    T: np.ndarray,
    covariates: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Subtract covariates @ beta from tangent vectors T.

    T          : (n, q)  tangent vectors
    covariates : (n, k)  covariate matrix, SAME columns/order as fit
    beta       : (k+1, q) from fit_tangent_residualizer
    """
    if T.shape[0] != covariates.shape[0]:
        raise ValueError("T and covariates row count mismatch.")
    D = _add_intercept(covariates.astype(np.float64))
    projection = D @ beta  # (n, q)
    return (T.astype(np.float64) - projection).astype(T.dtype)


def residualize_splits(
    T_train: np.ndarray,
    covariates_train: np.ndarray,
    T_others: Iterable[np.ndarray],
    covariates_others: Iterable[np.ndarray],
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Convenience wrapper: fit beta on (T_train, covariates_train), then
    residualize train and an iterable of other splits with the same beta.

    Returns (T_train_resid, [T_other_resid, ...], beta).
    """
    beta = fit_tangent_residualizer(T_train, covariates_train)
    T_train_resid = apply_tangent_residualizer(T_train, covariates_train, beta)
    others_resid = [
        apply_tangent_residualizer(T_o, cov_o, beta)
        for T_o, cov_o in zip(T_others, covariates_others)
    ]
    return T_train_resid, others_resid, beta


def load_covariates(
    subjects_tsv_path,
    subject_ids: Iterable[str],
    columns: Iterable[str] = ("Age", "Gender"),
) -> np.ndarray:
    """Load covariate matrix aligned to `subject_ids` from a subjects TSV.

    The TSV must have SubjectID in the first column and the requested
    covariate columns. Missing values are median-imputed (matches the
    behaviour of preprocess/covariates.py).
    """
    import pandas as pd

    df = pd.read_csv(subjects_tsv_path, sep="\t")
    df = df.set_index("SubjectID")
    df.index = df.index.astype(str)
    cols = list(columns)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing covariate columns: {missing}")
    C = df.loc[[str(sid) for sid in subject_ids], cols].apply(pd.to_numeric,
                                                              errors="coerce")
    # Median impute column-wise (train-only means / medians are the norm,
    # but for covariates like Age/Gender this is usually already clean)
    medians = C.median(numeric_only=True)
    C = C.fillna(medians)
    return C.to_numpy(dtype=np.float64)
