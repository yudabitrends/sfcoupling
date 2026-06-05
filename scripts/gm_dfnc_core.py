#!/usr/bin/env python3
"""Core utilities for GM-dFNC analyses.

This module is intentionally config-agnostic and focuses on:
  - loading static GM->FNC maps and dynamic FNC arrays
  - geometric state/subspace statistics
  - subject-level dynamic phenotype extraction
  - hierarchy, prediction, and clinical helper routines

The main orchestration lives in ``scripts/run_gm_dfnc_analysis.py``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import linalg, stats
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score

from models.baselines import fit_ridge_grid
from models.utils import load_config, load_training_contracts, save_json, set_seed
from train.run_kernel_spectral_regression import fit_linear_optshrink
from train.run_multivariate_methods import fit_nuclear_norm, fit_pls, fit_rrr


FNC_DOMAIN_RANGES = {
    "SC": (0, 5),
    "AUD": (5, 7),
    "SM": (7, 16),
    "VS": (16, 25),
    "CC": (25, 42),
    "DM": (42, 49),
    "CB": (49, 53),
}

PRIMARY_TIER_MAP = {
    "SM": "sensorimotor",
    "VS": "sensorimotor",
    "AUD": "sensorimotor",
    "CC": "heteromodal",
    "CB": "heteromodal",
    "DM": "transmodal",
    "SC": "transmodal",
}


def infer_table_sep(path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    return ","


def load_table(path: str | Path, sep: Optional[str] = None) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p, sep=sep or infer_table_sep(p))


def load_array(path: str | Path, key: Optional[str] = None) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    if p.suffix == ".npz":
        payload = np.load(p, allow_pickle=False)
        if key is not None:
            return payload[key]
        keys = list(payload.keys())
        if len(keys) != 1:
            raise ValueError(f"{p} contains keys {keys}; provide an explicit key")
        return payload[keys[0]]
    raise ValueError(f"Unsupported array format: {p}")


def vectorize_symmetric_block(arr: np.ndarray, k: int = 1) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array; got shape {arr.shape}")
    n_items, nc, nc2 = arr.shape
    if nc != nc2:
        raise ValueError(f"Expected symmetric matrices; got {arr.shape}")
    idx = np.triu_indices(nc, k=k)
    return np.asarray([arr[i][idx] for i in range(n_items)], dtype=np.float64)


def vectorize_single_symmetric(mat: np.ndarray, k: int = 1) -> np.ndarray:
    if mat.ndim == 1:
        return mat
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected 1D vector or square matrix; got {mat.shape}")
    idx = np.triu_indices(mat.shape[0], k=k)
    return mat[idx]


def ensure_2d_centroids(centroids: np.ndarray, k: int = 1) -> np.ndarray:
    if centroids.ndim == 2:
        return centroids
    if centroids.ndim == 3:
        return vectorize_symmetric_block(centroids, k=k)
    raise ValueError(f"Unexpected centroid shape: {centroids.shape}")


def save_tsv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)


def as_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_jsonable(v) for v in obj]
    return str(obj)


def orthonormalize(V: np.ndarray) -> np.ndarray:
    if V.size == 0:
        return np.zeros_like(V)
    q, _ = linalg.qr(V, mode="economic")
    return q


def principal_angles(V1: np.ndarray, V2: np.ndarray) -> np.ndarray:
    if V1.size == 0 or V2.size == 0:
        return np.zeros(0, dtype=np.float64)
    M = orthonormalize(V1).T @ orthonormalize(V2)
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    return np.clip(s, 0.0, 1.0)


def subspace_overlap(V1: np.ndarray, V2: np.ndarray) -> float:
    cos_angles = principal_angles(V1, V2)
    if cos_angles.size == 0:
        return 0.0
    return float(np.mean(cos_angles ** 2))


def top_k_subspace(Y: np.ndarray, k: int) -> np.ndarray:
    if Y.ndim != 2:
        raise ValueError(f"Expected 2D matrix; got {Y.shape}")
    if Y.shape[0] < 2 or Y.shape[1] < 1:
        return np.zeros((Y.shape[1], 0), dtype=np.float64)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    _, s, Vt = np.linalg.svd(Yc, full_matrices=False)
    rank = int(np.sum(s > 1e-12))
    k_eff = min(k, rank)
    return Vt[:k_eff].T


def select_rank(Sigma: np.ndarray, mode: Any) -> int:
    if Sigma is None:
        raise ValueError("Rank selection requires singular values")
    if isinstance(mode, str):
        mode_lower = mode.lower()
        if mode_lower == "eff":
            tol = max(1e-10, 1e-10 * float(Sigma[0]))
            return max(int(np.sum(Sigma > tol)), 1)
        if mode_lower.startswith("energy:"):
            frac = float(mode_lower.split(":", 1)[1])
            cum = np.cumsum(Sigma ** 2) / np.sum(Sigma ** 2)
            return int(np.searchsorted(cum, frac) + 1)
        if mode_lower.startswith("fixed:"):
            return max(int(mode_lower.split(":", 1)[1]), 1)
    if isinstance(mode, (int, np.integer)):
        return max(int(mode), 1)
    if isinstance(mode, float) and 0 < mode < 1:
        cum = np.cumsum(Sigma ** 2) / np.sum(Sigma ** 2)
        return int(np.searchsorted(cum, mode) + 1)
    raise ValueError(f"Unsupported rank selector: {mode}")


def compute_subspace_retention(
    V: np.ndarray,
    vectors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    V_orth = orthonormalize(V)
    coeffs = vectors @ V_orth
    proj = coeffs @ V_orth.T
    resid = vectors - proj
    norms_sq = np.sum(vectors ** 2, axis=1)
    proj_sq = np.sum(proj ** 2, axis=1)
    rho = np.divide(proj_sq, np.maximum(norms_sq, 1e-30))
    return rho, proj, resid


def rotation_null_distribution(
    V: np.ndarray,
    centroids: np.ndarray,
    n_perm: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    q, r = V.shape
    K = centroids.shape[0]
    rho_obs, _, _ = compute_subspace_retention(V, centroids)
    rho_null = np.zeros((n_perm, K), dtype=np.float64)
    for i in range(n_perm):
        z = rng.standard_normal((q, r))
        V_perm = orthonormalize(z)
        rho_null[i], _, _ = compute_subspace_retention(V_perm, centroids)
    p_vals = np.mean(rho_null >= rho_obs[None, :], axis=0)
    mean_obs = float(np.mean(rho_obs))
    mean_null = np.mean(rho_null, axis=1)
    p_mean = float(np.mean(mean_null >= mean_obs))
    return rho_obs, rho_null, p_vals, p_mean


def compute_centroids_from_labels(
    windows: np.ndarray,
    labels: np.ndarray,
    K: int,
) -> np.ndarray:
    centroids = []
    for state in range(K):
        mask = labels == state
        if not np.any(mask):
            centroids.append(np.zeros(windows.shape[1], dtype=np.float64))
        else:
            centroids.append(np.mean(windows[mask], axis=0))
    return np.asarray(centroids, dtype=np.float64)


def bootstrap_state_retention(
    V: np.ndarray,
    windows: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    K: int,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(subject_ids)
    boot_rho = []
    boot_mean = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        chunks = []
        chunk_labels = []
        for sid in sampled:
            mask = subject_ids == sid
            chunks.append(windows[mask])
            chunk_labels.append(labels[mask])
        win_b = np.vstack(chunks)
        lab_b = np.concatenate(chunk_labels)
        cent_b = compute_centroids_from_labels(win_b, lab_b, K)
        rho_b, _, _ = compute_subspace_retention(V, cent_b)
        boot_rho.append(rho_b)
        boot_mean.append(np.mean(rho_b))
    arr = np.asarray(boot_rho, dtype=np.float64)
    mean_arr = np.asarray(boot_mean, dtype=np.float64)
    return {
        "rho_ci_lo": np.percentile(arr, 2.5, axis=0),
        "rho_ci_hi": np.percentile(arr, 97.5, axis=0),
        "rho_mean_boot": np.mean(arr, axis=0),
        "mean_rho_ci_lo": float(np.percentile(mean_arr, 2.5)),
        "mean_rho_ci_hi": float(np.percentile(mean_arr, 97.5)),
        "mean_rho_boot": float(np.mean(mean_arr)),
    }


def compute_delta_retention(
    V: np.ndarray,
    centroids: np.ndarray,
) -> Dict[str, Any]:
    deltas = []
    labels = []
    for i in range(centroids.shape[0]):
        for j in range(i + 1, centroids.shape[0]):
            deltas.append(centroids[j] - centroids[i])
            labels.append((i, j))
    if not deltas:
        return {"pairs": [], "rho": np.zeros(0), "mean_rho": float("nan")}
    delta_mat = np.asarray(deltas, dtype=np.float64)
    rho, proj, resid = compute_subspace_retention(V, delta_mat)
    return {
        "pairs": labels,
        "rho": rho,
        "mean_rho": float(np.mean(rho)),
        "proj": proj,
        "resid": resid,
    }


def pairwise_distance_alignment(
    full_vectors: np.ndarray,
    projected_vectors: np.ndarray,
    n_perm: int,
    seed: int,
) -> Dict[str, float]:
    d_full = pdist(full_vectors, metric="euclidean")
    d_proj = pdist(projected_vectors, metric="euclidean")
    pearson = float(np.corrcoef(d_full, d_proj)[0, 1]) if d_full.size else 1.0
    spearman = float(stats.spearmanr(d_full, d_proj).correlation) if d_full.size else 1.0

    rng = np.random.default_rng(seed)
    if d_full.size == 0:
        return {"pearson": 1.0, "spearman": 1.0, "mantel_p": 1.0}

    full_square = squareform(d_full)
    proj_square = squareform(d_proj)
    null_stats = np.zeros(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm = rng.permutation(full_square.shape[0])
        permuted = proj_square[perm][:, perm]
        null_stats[i] = np.corrcoef(d_full, squareform(permuted))[0, 1]
    mantel_p = float(np.mean(null_stats >= pearson))
    return {"pearson": pearson, "spearman": spearman, "mantel_p": mantel_p}


def per_mode_contribution(V: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    V_orth = orthonormalize(V)
    norms_sq = np.maximum(np.sum(vectors ** 2, axis=1, keepdims=True), 1e-30)
    coeffs = vectors @ V_orth
    return (coeffs ** 2) / norms_sq


def analyze_residuals(V: np.ndarray, centroids: np.ndarray) -> Dict[str, Any]:
    _, _, resid = compute_subspace_retention(V, centroids)
    norms = np.maximum(linalg.norm(resid, axis=1), 1e-30)
    cosine = (resid @ resid.T) / (norms[:, None] * norms[None, :])
    if resid.shape[0] >= 2:
        _, s, Vt = np.linalg.svd(resid, full_matrices=False)
        explained = (s ** 2) / np.maximum(np.sum(s ** 2), 1e-30)
    else:
        s = np.asarray([norms[0]], dtype=np.float64)
        Vt = resid / norms[0]
        explained = np.asarray([1.0], dtype=np.float64)
    return {
        "resid_norms": norms,
        "resid_cosines": cosine,
        "resid_singular_values": s,
        "resid_explained_variance_ratio": explained,
        "resid_components": Vt,
    }


def retention_vs_rank_curve(
    V_full: np.ndarray,
    centroids: np.ndarray,
    ranks: Sequence[int],
) -> Dict[str, Any]:
    q = V_full.shape[0]
    rho_rows = []
    chance = []
    for r in ranks:
        V_r = V_full[:, :r]
        rho_r, _, _ = compute_subspace_retention(V_r, centroids)
        rho_rows.append(rho_r)
        chance.append(float(r / q))
    return {
        "ranks": np.asarray(ranks, dtype=np.int64),
        "rho_per_rank": np.asarray(rho_rows, dtype=np.float64),
        "chance_curve": np.asarray(chance, dtype=np.float64),
    }


def between_within_variance(
    windows: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    V: np.ndarray,
) -> Dict[str, float]:
    assigned = centroids[labels]
    within = windows - assigned

    rho_total, proj_total, _ = compute_subspace_retention(V, windows)
    rho_between, proj_between, _ = compute_subspace_retention(V, assigned)
    rho_within, proj_within, _ = compute_subspace_retention(V, within)

    total_ss = float(np.sum(windows ** 2))
    between_ss = float(np.sum(assigned ** 2))
    within_ss = float(np.sum(within ** 2))

    return {
        "total_var": total_ss,
        "between_var": between_ss,
        "within_var": within_ss,
        "between_var_frac": between_ss / max(total_ss, 1e-30),
        "within_var_frac": within_ss / max(total_ss, 1e-30),
        "coupled_total_frac": float(np.sum(proj_total ** 2) / max(total_ss, 1e-30)),
        "coupled_between_frac": float(np.sum(proj_between ** 2) / max(between_ss, 1e-30)),
        "coupled_within_frac": float(np.sum(proj_within ** 2) / max(within_ss, 1e-30)),
        "mean_window_rho": float(np.mean(rho_total)),
        "mean_between_rho": float(np.mean(rho_between)),
        "mean_within_rho": float(np.mean(rho_within)),
    }


def within_state_local_overlaps(
    windows: np.ndarray,
    labels: np.ndarray,
    V: np.ndarray,
    max_k: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    q = windows.shape[1]
    for state in sorted(np.unique(labels).tolist()):
        mask = labels == state
        n_win = int(np.sum(mask))
        if n_win < 3:
            rows.append({
                "state": int(state),
                "n_windows": n_win,
                "k_local": 0,
                "overlap": float("nan"),
                "chance": float("nan"),
            })
            continue
        local = windows[mask] - np.mean(windows[mask], axis=0, keepdims=True)
        k_local = min(max_k, n_win - 1, q, V.shape[1])
        V_local = top_k_subspace(local, k_local)
        overlap = subspace_overlap(V_local, V[:, :k_local])
        rows.append({
            "state": int(state),
            "n_windows": n_win,
            "k_local": int(k_local),
            "overlap": float(overlap),
            "chance": float(k_local / q),
        })
    return rows


def _run_lengths(labels: np.ndarray) -> List[Tuple[int, int]]:
    if labels.size == 0:
        return []
    runs = []
    cur = int(labels[0])
    length = 1
    for lab in labels[1:]:
        if int(lab) == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur = int(lab)
            length = 1
    runs.append((cur, length))
    return runs


def build_subject_dynamic_summary(
    subject_ids: np.ndarray,
    window_idx: np.ndarray,
    labels: np.ndarray,
    windows: np.ndarray,
    V: np.ndarray,
    K: int,
) -> pd.DataFrame:
    rho_w, proj_w, resid_w = compute_subspace_retention(V, windows)
    total_energy = np.sum(windows ** 2, axis=1)
    coupled_energy = np.sum(proj_w ** 2, axis=1)
    uncoupled_energy = np.sum(resid_w ** 2, axis=1)

    frame = pd.DataFrame({
        "subject_id": subject_ids.astype(str),
        "window_idx": window_idx.astype(int),
        "state": labels.astype(int),
        "rho_window": rho_w,
        "total_energy": total_energy,
        "coupled_energy": coupled_energy,
        "uncoupled_energy": uncoupled_energy,
    })
    frame = frame.sort_values(["subject_id", "window_idx"]).reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    for sid, grp in frame.groupby("subject_id", sort=False):
        labs = grp["state"].to_numpy(dtype=np.int64)
        n = len(grp)
        counts = np.bincount(labs, minlength=K).astype(np.float64)
        occupancy = counts / max(float(n), 1.0)

        dwell = np.zeros(K, dtype=np.float64)
        dwell_counts = np.zeros(K, dtype=np.float64)
        for state, length in _run_lengths(labs):
            dwell[state] += length
            dwell_counts[state] += 1.0
        dwell = np.divide(dwell, np.maximum(dwell_counts, 1.0))

        trans_counts = np.zeros((K, K), dtype=np.float64)
        if n >= 2:
            for a, b in zip(labs[:-1], labs[1:]):
                trans_counts[int(a), int(b)] += 1.0
        trans_total = np.sum(trans_counts)
        trans_prob_global = trans_counts / max(trans_total, 1.0)
        nonzero = trans_prob_global[trans_prob_global > 0]
        trans_entropy = float(-np.sum(nonzero * np.log(nonzero))) if nonzero.size else 0.0
        switching = float(np.sum(labs[1:] != labs[:-1]) / max(n - 1, 1))

        row = {
            "subject_id": str(sid),
            "n_windows": int(n),
            "switching_rate": switching,
            "transition_entropy": trans_entropy,
        }
        for state in range(K):
            state_mask = labs == state
            row[f"occupancy_s{state}"] = float(occupancy[state])
            row[f"dwell_s{state}"] = float(dwell[state])
            if np.any(state_mask):
                sub = grp.loc[state_mask]
                row[f"full_energy_s{state}"] = float(sub["total_energy"].mean())
                row[f"coupled_energy_s{state}"] = float(sub["coupled_energy"].mean())
                row[f"uncoupled_energy_s{state}"] = float(sub["uncoupled_energy"].mean())
                row[f"mean_rho_s{state}"] = float(sub["rho_window"].mean())
            else:
                row[f"full_energy_s{state}"] = 0.0
                row[f"coupled_energy_s{state}"] = 0.0
                row[f"uncoupled_energy_s{state}"] = 0.0
                row[f"mean_rho_s{state}"] = 0.0
            row_sum = np.sum(trans_counts[state])
            for nxt in range(K):
                row[f"trans_count_s{state}_to_s{nxt}"] = float(trans_counts[state, nxt])
                row[f"trans_prob_s{state}_to_s{nxt}"] = float(
                    trans_counts[state, nxt] / max(row_sum, 1.0)
                )
        rows.append(row)

    return pd.DataFrame(rows)


def clr_transform(X: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
    Xp = np.asarray(X, dtype=np.float64) + pseudocount
    Xp = Xp / np.maximum(np.sum(Xp, axis=1, keepdims=True), 1e-30)
    log_x = np.log(Xp)
    return log_x - np.mean(log_x, axis=1, keepdims=True)


def rowwise_clr_transition(flat_probs: np.ndarray, K: int, pseudocount: float = 1e-6) -> np.ndarray:
    arr = np.asarray(flat_probs, dtype=np.float64).reshape(-1, K, K)
    out = np.zeros_like(arr)
    for row in range(K):
        probs = arr[:, row, :] + pseudocount
        probs = probs / np.maximum(np.sum(probs, axis=1, keepdims=True), 1e-30)
        log_p = np.log(probs)
        out[:, row, :] = log_p - np.mean(log_p, axis=1, keepdims=True)
    return out.reshape(arr.shape[0], K * K)


def build_prediction_targets(
    subject_df: pd.DataFrame,
    K: int,
) -> Dict[str, Dict[str, Any]]:
    occupancy_cols = [f"occupancy_s{s}" for s in range(K)]
    dwell_cols = [f"dwell_s{s}" for s in range(K)]
    coupled_cols = [f"coupled_energy_s{s}" for s in range(K)]
    transition_cols = [f"trans_prob_s{i}_to_s{j}" for i in range(K) for j in range(K)]

    occupancy = clr_transform(subject_df[occupancy_cols].to_numpy())
    dwell = np.log1p(subject_df[dwell_cols].to_numpy(dtype=np.float64))
    coupled = np.log1p(subject_df[coupled_cols].to_numpy(dtype=np.float64))
    switching = np.log1p(subject_df[["switching_rate"]].to_numpy(dtype=np.float64))
    entropy = np.log1p(subject_df[["transition_entropy"]].to_numpy(dtype=np.float64))
    transitions = rowwise_clr_transition(subject_df[transition_cols].to_numpy(dtype=np.float64), K)

    targets = {
        "occupancy": {"Y": occupancy, "feature_names": occupancy_cols},
        "dwell": {"Y": dwell, "feature_names": dwell_cols},
        "state_coupled_energy": {"Y": coupled, "feature_names": coupled_cols},
        "switching_rate": {"Y": switching, "feature_names": ["switching_rate"]},
        "transition_entropy": {"Y": entropy, "feature_names": ["transition_entropy"]},
        "transition_matrix": {"Y": transitions, "feature_names": transition_cols},
    }
    targets["slow_bundle"] = {
        "Y": np.concatenate([occupancy, dwell, coupled], axis=1),
        "feature_names": occupancy_cols + dwell_cols + coupled_cols,
    }
    targets["fast_bundle"] = {
        "Y": np.concatenate([switching, entropy, transitions], axis=1),
        "feature_names": ["switching_rate", "transition_entropy"] + transition_cols,
    }
    return targets


def regression_summary(Y_true: np.ndarray, Y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(Y_true, dtype=np.float64)
    y_pred = np.asarray(Y_pred, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    comp_r2 = []
    comp_corr = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        if np.std(yt) < 1e-12:
            continue
        comp_r2.append(float(r2_score(yt, yp)))
        if np.std(yp) < 1e-12:
            comp_corr.append(0.0)
        else:
            comp_corr.append(float(np.corrcoef(yt, yp)[0, 1]))
    flat_true = y_true.ravel()
    flat_pred = y_pred.ravel()
    slope = float(np.polyfit(flat_pred, flat_true, deg=1)[0]) if np.std(flat_pred) > 1e-12 else 0.0
    return {
        "r2_mean": float(np.mean(comp_r2)) if comp_r2 else float("nan"),
        "r2_median": float(np.median(comp_r2)) if comp_r2 else float("nan"),
        "corr_mean": float(np.mean(comp_corr)) if comp_corr else float("nan"),
        "corr_median": float(np.median(comp_corr)) if comp_corr else float("nan"),
        "calibration_slope": slope,
        "n_targets": int(y_true.shape[1]),
    }


def fit_logistic_grid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    Cs: Sequence[float],
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    best_model = None
    best_auc = -np.inf
    trials = []
    for C in Cs:
        clf = LogisticRegression(
            C=float(C),
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
        )
        clf.fit(X_train, y_train)
        score = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        trials.append({"C": float(C), "val_auc": float(score)})
        if score > best_auc:
            best_auc = score
            best_model = clf
    assert best_model is not None
    return best_model, {"best_C": float(best_model.C), "val_best_auc": float(best_auc), "trials": trials}


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    auc_arr = np.asarray(aucs, dtype=np.float64)
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "auc_ci_lo": float(np.percentile(auc_arr, 2.5)) if auc_arr.size else float("nan"),
        "auc_ci_hi": float(np.percentile(auc_arr, 97.5)) if auc_arr.size else float("nan"),
    }


def cohen_d(x0: np.ndarray, x1: np.ndarray) -> float:
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    n0 = len(x0)
    n1 = len(x1)
    if n0 < 2 or n1 < 2:
        return float("nan")
    s0 = np.var(x0, ddof=1)
    s1 = np.var(x1, ddof=1)
    pooled = math.sqrt(((n0 - 1) * s0 + (n1 - 1) * s1) / max(n0 + n1 - 2, 1))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(x1) - np.mean(x0)) / pooled)


def case_control_state_effects(
    subject_df: pd.DataFrame,
    diagnosis_col: str,
    K: int,
    rho_state: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    rows = []
    y = subject_df[diagnosis_col].to_numpy(dtype=np.int64)
    for metric_prefix in ["occupancy", "dwell", "coupled_energy"]:
        for state in range(K):
            col = f"{metric_prefix}_s{state}"
            ctrl = subject_df.loc[y == 0, col].to_numpy(dtype=np.float64)
            case = subject_df.loc[y == 1, col].to_numpy(dtype=np.float64)
            t_res = stats.ttest_ind(case, ctrl, equal_var=False, nan_policy="omit")
            rows.append({
                "metric": metric_prefix,
                "state": int(state),
                "mean_control": float(np.mean(ctrl)) if ctrl.size else float("nan"),
                "mean_case": float(np.mean(case)) if case.size else float("nan"),
                "cohen_d": cohen_d(ctrl, case),
                "t_stat": float(t_res.statistic) if np.isfinite(t_res.statistic) else float("nan"),
                "p_value": float(t_res.pvalue) if np.isfinite(t_res.pvalue) else float("nan"),
                "rho_state": float(rho_state[state]) if rho_state is not None and state < len(rho_state) else float("nan"),
            })
    out = {"rows": rows}
    if rho_state is not None:
        df = pd.DataFrame(rows)
        assoc = []
        for metric in sorted(df["metric"].unique()):
            sub = df[df["metric"] == metric]
            corr = stats.spearmanr(np.abs(sub["cohen_d"]), sub["rho_state"], nan_policy="omit")
            assoc.append({
                "metric": metric,
                "spearman_abs_effect_vs_rho": float(corr.correlation),
                "p_value": float(corr.pvalue) if np.isfinite(corr.pvalue) else float("nan"),
            })
        out["rho_association"] = assoc
    return out


def get_fnc_domain(ic_idx: int) -> str:
    for domain, (lo, hi) in FNC_DOMAIN_RANGES.items():
        if lo <= ic_idx < hi:
            return domain
    return "Other"


def parse_fnc_edges(fnc_names: Iterable[str]) -> List[Tuple[int, int]]:
    edges = []
    for name in fnc_names:
        left, right = name.split("--")
        edges.append((int(left.replace("IC_", "")), int(right.replace("IC_", ""))))
    return edges


def build_tier_masks(
    fnc_names: Sequence[str],
    tier_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, np.ndarray]:
    tier_map = tier_map or PRIMARY_TIER_MAP
    edges = parse_fnc_edges(fnc_names)
    masks = {"sensorimotor": [], "heteromodal": [], "transmodal": []}
    for edge_idx, (ic_i, ic_j) in enumerate(edges):
        tier_i = tier_map.get(get_fnc_domain(ic_i), "other")
        tier_j = tier_map.get(get_fnc_domain(ic_j), "other")
        if tier_i == tier_j and tier_i in masks:
            masks[tier_i].append(edge_idx)
    return {k: np.asarray(v, dtype=np.int64) for k, v in masks.items()}


def tier_retention(
    V: np.ndarray,
    centroids: np.ndarray,
    fnc_names: Sequence[str],
) -> List[Dict[str, Any]]:
    rows = []
    masks = build_tier_masks(fnc_names)
    for tier, idx in masks.items():
        if idx.size == 0:
            continue
        V_sub = V[idx, :]
        cent_sub = centroids[:, idx]
        rho, _, _ = compute_subspace_retention(V_sub, cent_sub)
        q_sub = cent_sub.shape[1]
        r_sub = orthonormalize(V_sub).shape[1]
        for state, value in enumerate(rho):
            rows.append({
                "tier": tier,
                "state": int(state),
                "rho": float(value),
                "chance": float(r_sub / q_sub),
                "lift": float(value / max(r_sub / q_sub, 1e-30)),
                "n_edges": int(q_sub),
            })
    return rows


def fit_static_method(
    cfg_path: str | Path,
    method: str,
    seed: int,
) -> Dict[str, Any]:
    cfg = load_config(str(cfg_path))
    set_seed(seed)
    data = load_training_contracts(cfg)

    idx_tr = data["idx1_train"]
    idx_val = data["idx1_val"]
    idx_te = data["idx1_test"]
    idx_ext = data["idx2_external"]

    Xtr = data["X1"][idx_tr].astype(np.float64)
    Ytr = data["Y1"][idx_tr].astype(np.float64)
    Xva = data["X1"][idx_val].astype(np.float64)
    Yva = data["Y1"][idx_val].astype(np.float64)
    Xte = data["X1"][idx_te].astype(np.float64)
    Yte = data["Y1"][idx_te].astype(np.float64)
    Xext = data["X2"][idx_ext].astype(np.float64)
    Yext = data["Y2"][idx_ext].astype(np.float64)

    ridge_alphas = cfg.get("ridge", {}).get("alphas", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])

    method_l = method.lower()
    if method_l == "nuclear_norm":
        fit = fit_nuclear_norm(Xtr, Ytr, Xva, Yva)
        B = fit["B"]
        meta = {
            "optimal_lambda": fit["optimal_lambda"],
            "best_val_r2": fit["best_val_r2"],
        }
    elif method_l == "rrr":
        fit = fit_rrr(Xtr, Ytr, Xva, Yva, ridge_alphas=ridge_alphas)
        B = fit["B"]
        meta = {
            "optimal_rank": fit["optimal_rank"],
            "best_val_r2": fit["best_val_r2"],
        }
    elif method_l == "pls":
        fit = fit_pls(Xtr, Ytr, Xva, Yva)
        B = fit["B"]
        meta = {
            "optimal_n": fit["optimal_n"],
            "best_val_r2": fit["best_val_r2"],
        }
    elif method_l == "linear_optshrink":
        fit = fit_linear_optshrink(Xtr, Ytr, Xva, Yva, ridge_alphas=ridge_alphas)
        B = fit["B"]
        meta = {
            "ridge_alpha": fit["ridge_alpha"],
            "sigma_noise_mult": fit["sigma_noise_mult"],
            "best_val_r2": fit["best_val_r2"],
        }
    elif method_l == "ridge":
        model, info = fit_ridge_grid(Xtr, Ytr, Xva, Yva, ridge_alphas)
        B = model.coef_.T
        meta = {
            "ridge_alpha": info["best_alpha"],
            "best_val_r2": info["val_best_r2_global"],
        }
    else:
        raise ValueError(f"Unsupported static method: {method}")

    _, Sigma, Vt = np.linalg.svd(B, full_matrices=False)
    return {
        "cfg_path": str(cfg_path),
        "method": method,
        "seed": int(seed),
        "B": B,
        "Sigma": Sigma,
        "V_full": Vt.T,
        "meta": meta,
        "contracts": data,
        "splits": {
            "dataset1_train": idx_tr,
            "dataset1_val": idx_val,
            "dataset1_test": idx_te,
            "dataset1_all": np.arange(len(data["ids1"]), dtype=np.int64),
            "dataset2_external": idx_ext,
            "dataset2_all": np.arange(len(data["ids2"]), dtype=np.int64),
        },
        "X_eval": {
            "dataset1_train": Xtr,
            "dataset1_val": Xva,
            "dataset1_test": Xte,
            "dataset2_external": Xext,
        },
        "Y_eval": {
            "dataset1_train": Ytr,
            "dataset1_val": Yva,
            "dataset1_test": Yte,
            "dataset2_external": Yext,
        },
    }


def split_subject_frame(static_fit: Dict[str, Any], split_name: str) -> pd.DataFrame:
    contracts = static_fit["contracts"]
    if split_name.startswith("dataset1"):
        subjects = contracts["subjects1"].copy()
        ids = np.asarray(contracts["ids1"])
        X = contracts["X1"]
    elif split_name.startswith("dataset2"):
        subjects = contracts["subjects2"].copy()
        ids = np.asarray(contracts["ids2"])
        X = contracts["X2"]
    else:
        raise ValueError(f"Unknown split name: {split_name}")

    if split_name.endswith("_all"):
        idx = np.arange(len(ids), dtype=np.int64)
    else:
        idx = static_fit["splits"][split_name]
    out = subjects.iloc[idx].copy().reset_index(drop=True)
    out["subject_id"] = ids[idx].astype(str)
    out["_row_idx"] = idx
    out["_X_index"] = idx
    out["_X_matrix"] = "X1" if split_name.startswith("dataset1") else "X2"
    return out


def match_subject_rows(
    subject_df: pd.DataFrame,
    subject_summary: pd.DataFrame,
    id_col_summary: str = "subject_id",
) -> Tuple[pd.DataFrame, np.ndarray]:
    merged = subject_df.merge(
        subject_summary,
        left_on="subject_id",
        right_on=id_col_summary,
        how="inner",
    )
    return merged, merged["_X_index"].to_numpy(dtype=np.int64)


def write_summary_json(path: str | Path, payload: Dict[str, Any]) -> None:
    save_json(Path(path), as_jsonable(payload))
