"""
Diagnostics: alignment proofs, leakage checks, distribution checks.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def compute_alignment_hash(subject_order: List[str]) -> str:
    """Compute deterministic hash of subject ID order."""
    s = "|".join(str(x) for x in subject_order)
    return hashlib.sha256(s.encode()).hexdigest()


def compute_row_integrity_checksums(
    X: np.ndarray,
    Y: np.ndarray,
    subject_ids: List[str],
    sample_size: int = 20,
    seed: int = 1337,
) -> List[Dict[str, str]]:
    """
    Deterministic sampled row checksums for alignment integrity proof.
    Hash payload: subject_id|repr(round(X_row,8))|repr(round(Y_row,8))
    """
    n = len(subject_ids)
    if n == 0:
        return []
    if X.shape[0] != n or Y.shape[0] != n:
        raise ValueError("X/Y row count must match subject_ids length.")
    rng = np.random.default_rng(seed)
    if n <= sample_size:
        idx = np.arange(n)
    else:
        idx = np.sort(rng.choice(n, size=sample_size, replace=False))
    out = []
    for i in idx.tolist():
        x_row = np.round(np.asarray(X[i], dtype=np.float64), 8).tolist()
        y_row = np.round(np.asarray(Y[i], dtype=np.float64), 8).tolist()
        payload = f"{subject_ids[i]}|{repr(x_row)}|{repr(y_row)}"
        out.append(
            {
                "index": int(i),
                "subject_id": str(subject_ids[i]),
                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            }
        )
    return out


def run_diagnostics(
    X: np.ndarray,
    Y: np.ndarray,
    subject_ids: List[str],
    scalers_fit_on_train: bool = True,
    train_mask: Optional[np.ndarray] = None,
) -> Dict:
    """
    Run full diagnostics: alignment, value checks, leakage hints.
    Returns report dict.
    """
    report: Dict = {
        "n_subjects": len(subject_ids),
        "dx": int(X.shape[1]),
        "dy": int(Y.shape[1]),
        "subject_order_hash": compute_alignment_hash(subject_ids),
        "subject_order_sha256": compute_alignment_hash(subject_ids),
        "alignment": {
            "x_rows_match_subjects": X.shape[0] == len(subject_ids),
            "y_rows_match_subjects": Y.shape[0] == len(subject_ids),
            "x_y_same_n": X.shape[0] == Y.shape[0],
        },
        "value_checks": {
            "x_has_nan": bool(np.any(np.isnan(X))),
            "x_has_inf": bool(np.any(np.isinf(X))),
            "y_has_nan": bool(np.any(np.isnan(Y))),
            "y_has_inf": bool(np.any(np.isinf(Y))),
        },
        "distribution_pre": {},
        "distribution_post": {},
        "leakage_checks": {
            "scalers_fit_on_train_only": scalers_fit_on_train,
            "residualization_fit_on_train_only": scalers_fit_on_train,
        },
        "row_integrity_checksums": compute_row_integrity_checksums(X, Y, subject_ids),
    }

    report["distribution_pre"]["x_mean"] = float(np.nanmean(X))
    report["distribution_pre"]["x_std"] = float(np.nanstd(X))
    report["distribution_pre"]["y_mean"] = float(np.nanmean(Y))
    report["distribution_pre"]["y_std"] = float(np.nanstd(Y))

    if train_mask is not None:
        report["distribution_post"]["train_x_mean"] = float(np.nanmean(X[train_mask]))
        report["distribution_post"]["train_x_std"] = float(np.nanstd(X[train_mask]))
        if np.any(~train_mask):
            report["distribution_post"]["val_x_mean"] = float(np.nanmean(X[~train_mask]))
            report["distribution_post"]["val_x_std"] = float(np.nanstd(X[~train_mask]))
        else:
            report["distribution_post"]["val_x_mean"] = None
            report["distribution_post"]["val_x_std"] = None

    report["sanity_passed"] = (
        report["alignment"]["x_rows_match_subjects"]
        and report["alignment"]["y_rows_match_subjects"]
        and report["alignment"]["x_y_same_n"]
        and not report["value_checks"]["x_has_nan"]
        and not report["value_checks"]["x_has_inf"]
        and not report["value_checks"]["y_has_nan"]
        and not report["value_checks"]["y_has_inf"]
    )
    return report


def augment_report(
    base_report: Dict,
    dataset_name: str,
    alignment_proof: Dict,
    split_proof: Dict,
    fisherz_decision: Optional[Dict] = None,
    feature_map_checks: Optional[Dict] = None,
    design_matrix_proof: Optional[Dict] = None,
) -> Dict:
    """Attach extended audit fields to diagnostics report."""
    report = dict(base_report)
    report["dataset"] = dataset_name
    report["alignment_proof"] = alignment_proof
    report["split_proof"] = split_proof
    report["feature_map_checks"] = feature_map_checks or {}
    report["design_matrix_proof"] = design_matrix_proof or {}
    report["fisher_z_decision"] = fisherz_decision or {}
    return report


def write_alignment_report(report: Dict, out_path: str) -> None:
    """Write report to JSON file."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
