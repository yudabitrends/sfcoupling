"""
FNC (functional network connectivity) feature extraction.
Supports: H5 with sFNC column, CSV. Applies Fisher-z and upper-triangle vectorization.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def build_edge_names(n: int) -> List[str]:
    """Build stable edge names for upper triangle: IC_i--IC_j for i < j."""
    names = []
    for i in range(n):
        for j in range(i + 1, n):
            names.append(f"IC_{i}--IC_{j}")
    return names


def vectorize_upper_triangle(matrix: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
    """
    Extract upper triangle (i < j) from square matrix in stable order.
    If matrix is flattened (row-major), reshape to (n, n) first.
    n_components: if matrix is 1D, infer n from n*(n-1)/2 = len; else use this.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 1:
        n_sq = arr.size
        n = int(np.sqrt(n_sq))
        if n * n != n_sq:
            raise ValueError(
                f"Flattened FNC size {n_sq} is not a perfect square. "
                f"Expected n^2 for n components."
            )
        mat = arr.reshape(n, n)
    else:
        mat = arr
        n = mat.shape[0]
        if mat.shape[1] != n:
            raise ValueError(f"Expected square matrix, got shape {mat.shape}")

    triu_idx = np.triu_indices(n, k=1)
    return mat[triu_idx].astype(np.float32)


def detect_correlation_range(arr: np.ndarray) -> bool:
    """True if values appear to be correlations in [-1, 1]."""
    a = np.asarray(arr).flatten()
    if a.size == 0:
        return False
    in_range = np.all((a >= -1.01) & (a <= 1.01))
    finite = np.all(np.isfinite(a))
    return bool(in_range and finite)


def apply_fisher_z(r: np.ndarray, clip_eps: float = 1e-6) -> Tuple[np.ndarray, bool]:
    """Fisher z-transform for correlation values. Clips to avoid arctanh(1) = inf."""
    r = np.asarray(r, dtype=np.float64)
    lower = -1.0 + clip_eps
    upper = 1.0 - clip_eps
    clip_applied = bool(np.any(r < lower) or np.any(r > upper))
    r = np.clip(r, lower, upper)
    return np.arctanh(r).astype(np.float32), clip_applied


def _stats(arr: np.ndarray) -> Dict[str, float]:
    """Basic summary stats for reporting."""
    a = np.asarray(arr, dtype=np.float64).flatten()
    return {
        "min": float(np.nanmin(a)) if a.size else 0.0,
        "max": float(np.nanmax(a)) if a.size else 0.0,
        "mean": float(np.nanmean(a)) if a.size else 0.0,
        "std": float(np.nanstd(a)) if a.size else 0.0,
    }


def build_fisherz_decision(
    values: np.ndarray,
    apply_fisher_z_config: bool,
    input_was_matrix: bool,
    diag_values: Optional[np.ndarray] = None,
    force_correlation: bool = False,
    clip_eps: float = 1e-6,
) -> Dict:
    """
    Build explicit Fisher-z decision object.
    Rules:
      - correlation_detected if values mostly in [-1,1] and (diag near 1 for matrix OR vectorized data)
      - apply fisher only when correlation_detected and config enabled
    """
    arr = np.asarray(values, dtype=np.float64).flatten()
    tol = 0.01
    in_range_ratio = float(np.mean((arr >= -1.0 - tol) & (arr <= 1.0 + tol))) if arr.size else 0.0
    mostly_in_range = in_range_ratio >= 0.98
    diag_near_one = True
    if input_was_matrix and diag_values is not None and len(diag_values) > 0:
        d = np.asarray(diag_values, dtype=np.float64).flatten()
        diag_near_one = bool(np.mean(np.abs(d - 1.0) <= 0.05) >= 0.95)
    correlation_detected = bool(force_correlation or (mostly_in_range and diag_near_one))
    apply_flag = bool(correlation_detected and apply_fisher_z_config)
    decision = {
        "correlation_detected": correlation_detected,
        "apply_fisher_z": apply_flag,
        "clip_applied": False,
        "clip_eps": float(clip_eps),
        "stats_before": _stats(arr),
        "stats_after": None,
        "in_range_ratio": in_range_ratio,
        "diag_near_one": bool(diag_near_one),
    }
    return decision


def extract_fnc_from_h5(
    df: pd.DataFrame,
    fnc_col: str,
    id_col: str,
    n_components: int = 53,
    apply_fisher_z_flag: bool = True,
    extract_upper_triangle: bool = True,
    clip_eps: float = 1e-6,
    force_correlation: bool = False,
    force_fisher_z: bool = False,
    audit_out: Optional[Dict] = None,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Extract FNC from H5 DataFrame. Each row has fnc_col as array (53x53 or flattened).
    Returns: (subject_id -> FNC vector dict, edge_names)
    """
    if fnc_col not in df.columns:
        raise ValueError(f"Column {fnc_col} not found. Available: {list(df.columns)}")
    if id_col not in df.columns:
        raise ValueError(f"Column {id_col} not found. Available: {list(df.columns)}")

    fnc_dict: Dict[str, np.ndarray] = {}
    edge_names = build_edge_names(n_components)
    decision_agg: Optional[Dict] = None
    raw_concat = []
    post_concat = []

    for _, row in df.iterrows():
        sid = str(row[id_col]).strip()
        val = row[fnc_col]
        if pd.isna(val):
            continue
        arr = np.asarray(val, dtype=np.float64)
        input_was_matrix = arr.ndim != 1
        diag_vals = None
        if arr.ndim == 1:
            vec = vectorize_upper_triangle(arr, n_components) if extract_upper_triangle else arr
        else:
            arr = arr.reshape(n_components, n_components)
            diag_vals = np.diag(arr)
            vec = vectorize_upper_triangle(arr) if extract_upper_triangle else arr[np.triu_indices(n_components, k=1)]
        decision = build_fisherz_decision(
            vec,
            apply_fisher_z_config=apply_fisher_z_flag,
            input_was_matrix=input_was_matrix,
            diag_values=diag_vals,
            force_correlation=force_correlation,
            clip_eps=clip_eps,
        )
        if force_fisher_z:
            decision["apply_fisher_z"] = True
            decision["correlation_detected"] = True
        raw_concat.append(vec.astype(np.float64))
        if decision["apply_fisher_z"]:
            vec, clip_applied = apply_fisher_z(vec, clip_eps=clip_eps)
            decision["clip_applied"] = clip_applied
            decision["stats_after"] = _stats(vec)
        post_concat.append(vec.astype(np.float64))
        # Keep first decision structure and refresh aggregate stats later.
        if decision_agg is None:
            decision_agg = decision
        else:
            decision_agg["correlation_detected"] = bool(
                decision_agg["correlation_detected"] or decision["correlation_detected"]
            )
            decision_agg["apply_fisher_z"] = bool(
                decision_agg["apply_fisher_z"] or decision["apply_fisher_z"]
            )
            decision_agg["clip_applied"] = bool(
                decision_agg["clip_applied"] or decision["clip_applied"]
            )
        fnc_dict[sid] = vec.astype(np.float32)

    if audit_out is not None:
        if raw_concat:
            raw_vals = np.concatenate(raw_concat)
            post_vals = np.concatenate(post_concat)
            if decision_agg is None:
                decision_agg = build_fisherz_decision(
                    raw_vals, apply_fisher_z_flag, input_was_matrix=False, clip_eps=clip_eps
                )
            decision_agg["stats_before"] = _stats(raw_vals)
            decision_agg["stats_after"] = _stats(post_vals) if decision_agg["apply_fisher_z"] else None
        else:
            decision_agg = build_fisherz_decision(
                np.array([]), apply_fisher_z_flag, input_was_matrix=False, clip_eps=clip_eps
            )
        audit_out.update(
            {
                "extract_upper_triangle": bool(extract_upper_triangle),
                "apply_fisher_z_config": bool(apply_fisher_z_flag),
                "force_fisher_z": bool(force_fisher_z),
                "decision": decision_agg,
            }
        )

    return fnc_dict, edge_names


def extract_fnc_from_csv(
    path: str,
    id_col: Optional[str] = None,
    n_components: Optional[int] = 53,
    apply_fisher_z_flag: bool = True,
    subject_ids: Optional[List[str]] = None,
    clip_eps: float = 1e-6,
    force_correlation: bool = False,
    force_fisher_z: bool = False,
    audit_out: Optional[Dict] = None,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Extract FNC from CSV. Rows are subjects; columns are FNC values (full matrix or upper-tri).
    If id_col is None, use subject_ids for row order (must match row count).
    """
    df = pd.read_csv(path, low_memory=False)
    n_rows = len(df)
    if id_col and id_col in df.columns:
        ids = df[id_col].astype(str).str.strip().tolist()
        fnc_cols = [c for c in df.columns if c != id_col]
    else:
        if subject_ids is None:
            ids = [f"sub_{i}" for i in range(n_rows)]
        else:
            if len(subject_ids) != n_rows:
                raise ValueError(
                    f"subject_ids length {len(subject_ids)} != rows {n_rows}"
                )
            ids = [str(s).strip() for s in subject_ids]
        fnc_cols = list(df.columns)

    data = df[fnc_cols].values.astype(np.float64)
    n_expected = n_components * (n_components - 1) // 2 if n_components else None
    full_size = n_components * n_components if n_components else None

    fnc_dict = {}
    edge_names = build_edge_names(n_components) if n_components else [
        f"edge_{i}" for i in range(data.shape[1])
    ]
    raw_concat = []
    post_concat = []
    decision_agg: Optional[Dict] = None

    for i, sid in enumerate(ids):
        row = data[i]
        if data.shape[1] == full_size:
            vec = vectorize_upper_triangle(row, n_components)
        elif n_expected and data.shape[1] == n_expected:
            vec = row
        else:
            vec = row
        decision = build_fisherz_decision(
            vec,
            apply_fisher_z_config=apply_fisher_z_flag,
            input_was_matrix=False,
            force_correlation=force_correlation,
            clip_eps=clip_eps,
        )
        if force_fisher_z:
            decision["apply_fisher_z"] = True
            decision["correlation_detected"] = True
        raw_concat.append(vec.astype(np.float64))
        if decision["apply_fisher_z"]:
            vec, clip_applied = apply_fisher_z(vec, clip_eps=clip_eps)
            decision["clip_applied"] = clip_applied
            decision["stats_after"] = _stats(vec)
        post_concat.append(vec.astype(np.float64))
        if decision_agg is None:
            decision_agg = decision
        else:
            decision_agg["correlation_detected"] = bool(
                decision_agg["correlation_detected"] or decision["correlation_detected"]
            )
            decision_agg["apply_fisher_z"] = bool(
                decision_agg["apply_fisher_z"] or decision["apply_fisher_z"]
            )
            decision_agg["clip_applied"] = bool(
                decision_agg["clip_applied"] or decision["clip_applied"]
            )
        fnc_dict[sid] = vec.astype(np.float32)

    if audit_out is not None:
        if raw_concat:
            raw_vals = np.concatenate(raw_concat)
            post_vals = np.concatenate(post_concat)
            if decision_agg is None:
                decision_agg = build_fisherz_decision(
                    raw_vals, apply_fisher_z_flag, input_was_matrix=False, clip_eps=clip_eps
                )
            decision_agg["stats_before"] = _stats(raw_vals)
            decision_agg["stats_after"] = _stats(post_vals) if decision_agg["apply_fisher_z"] else None
        else:
            decision_agg = build_fisherz_decision(
                np.array([]), apply_fisher_z_flag, input_was_matrix=False, clip_eps=clip_eps
            )
        audit_out.update(
            {
                "extract_upper_triangle": bool(n_components is not None and data.shape[1] == full_size),
                "apply_fisher_z_config": bool(apply_fisher_z_flag),
                "force_fisher_z": bool(force_fisher_z),
                "decision": decision_agg,
            }
        )

    return fnc_dict, edge_names
