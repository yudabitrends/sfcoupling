"""
Subject ID normalization and modality alignment.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


def normalize_id(sid: str, rules: Optional[Dict] = None) -> str:
    """
    Normalize subject ID according to rules.
    rules: { "trim": true, "lower": true, "upper": false, "prefix": "", "suffix": "" }
    """
    rules = rules or {}
    s = str(sid)
    if rules.get("trim", False):
        s = s.strip()
    if rules.get("lower", False):
        s = s.lower()
    if rules.get("upper", False):
        s = s.upper()
    prefix = rules.get("prefix", "")
    suffix = rules.get("suffix", "")
    if prefix and s.startswith(prefix):
        s = s[len(prefix) :]
    if suffix and s.endswith(suffix):
        s = s[:-len(suffix)]
    return s


def apply_normalization(
    ids: Set[str], rules: Optional[Dict] = None
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Apply normalization to a set of IDs.
    Returns: (normalized -> original), (original -> normalized)
    """
    norm_to_orig: Dict[str, str] = {}
    orig_to_norm: Dict[str, str] = {}
    for oid in ids:
        nid = normalize_id(oid, rules)
        if nid in norm_to_orig and norm_to_orig[nid] != oid:
            raise ValueError(
                f"ID collision: '{oid}' and '{norm_to_orig[nid]}' both normalize to '{nid}'"
            )
        norm_to_orig[nid] = oid
        orig_to_norm[oid] = nid
    return norm_to_orig, orig_to_norm


def intersect_subjects(
    gm_ids: Set[str],
    fnc_ids: Set[str],
    cov_ids: Set[str],
) -> Set[str]:
    """Return intersection of subjects present in all three modalities."""
    return gm_ids & fnc_ids & cov_ids


def align_arrays(
    gm_dict: Dict[str, np.ndarray],
    fnc_dict: Dict[str, np.ndarray],
    subject_order: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stack GM and FNC dicts into (N, dx) and (N, dy) arrays in subject_order.
    Raises KeyError if any subject is missing.
    """
    X_list = []
    Y_list = []
    for sid in subject_order:
        X_list.append(gm_dict[sid])
        Y_list.append(fnc_dict[sid])
    X = np.stack(X_list, axis=0).astype(np.float32)
    Y = np.stack(Y_list, axis=0).astype(np.float32)
    return X, Y


def build_missingness_report(
    gm_ids: Set[str],
    fnc_ids: Set[str],
    cov_ids: Set[str],
    all_subjects: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Build a report of which subjects are missing from which modality.
    all_subjects: union of all IDs across modalities.
    """
    all_s = all_subjects or (gm_ids | fnc_ids | cov_ids)
    rows = []
    for sid in sorted(all_s):
        has_gm = sid in gm_ids
        has_fnc = sid in fnc_ids
        has_cov = sid in cov_ids
        rows.append(
            {
                "subject_id": sid,
                "has_gm": has_gm,
                "has_fnc": has_fnc,
                "has_cov": has_cov,
                "complete": has_gm and has_fnc and has_cov,
            }
        )
    return pd.DataFrame(rows)


def compute_drop_reason_counts(
    gm_ids: Set[str],
    fnc_ids: Set[str],
    cov_ids: Set[str],
) -> Dict[str, int]:
    """
    Count dropped subjects by missingness reason before strict intersection.
    A single subject can contribute to multiple reason counts.
    """
    all_ids = gm_ids | fnc_ids | cov_ids
    missing_gm = 0
    missing_fnc = 0
    missing_cov = 0
    for sid in all_ids:
        if sid not in gm_ids:
            missing_gm += 1
        if sid not in fnc_ids:
            missing_fnc += 1
        if sid not in cov_ids:
            missing_cov += 1
    return {
        "missing_gm": int(missing_gm),
        "missing_fnc": int(missing_fnc),
        "missing_covariates": int(missing_cov),
    }


def build_alignment_proof_payload(
    gm_ids: Set[str],
    fnc_ids: Set[str],
    cov_ids: Set[str],
    aligned_subjects: Set[str],
) -> Dict:
    """Build dataset-level alignment proof counts for audit reports."""
    union_ids = gm_ids | fnc_ids | cov_ids
    return {
        "counts_pre_alignment": {
            "gm": int(len(gm_ids)),
            "fnc": int(len(fnc_ids)),
            "covariates": int(len(cov_ids)),
            "union": int(len(union_ids)),
        },
        "counts_post_alignment": {
            "intersection": int(len(aligned_subjects)),
        },
        "dropped_reasons": compute_drop_reason_counts(gm_ids, fnc_ids, cov_ids),
    }
