from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA


def r2_global(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true.reshape(-1), y_pred.reshape(-1)))


def r2_edgewise(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    vals = r2_score(y_true, y_pred, multioutput="raw_values")
    vals = np.asarray(vals, dtype=np.float64)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    return vals


def r2_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    edge = r2_edgewise(y_true, y_pred)
    return {
        "r2_global": r2_global(y_true, y_pred),
        "r2_edge_mean": float(np.mean(edge)),
        "r2_edge_median": float(np.median(edge)),
    }


def residual_burden(y_true: np.ndarray, y_struct: np.ndarray) -> np.ndarray:
    r = y_true - y_struct
    return np.linalg.norm(r, axis=1)


def residual_burden_summary(
    y_true: np.ndarray,
    y_struct: np.ndarray,
    subjects_df: Optional[pd.DataFrame] = None,
    diagnosis_col: str = "Diagnosis",
) -> Dict:
    b = residual_burden(y_true, y_struct)
    out = {
        "burden_mean": float(np.mean(b)),
        "burden_std": float(np.std(b)),
        "burden_median": float(np.median(b)),
    }
    if subjects_df is not None and diagnosis_col in subjects_df.columns:
        groups = {}
        for g, gdf in subjects_df.groupby(diagnosis_col):
            idx = gdf.index.values
            groups[str(g)] = float(np.mean(b[idx]))
        out["group_burden_mean"] = groups
    return out


def network_aggregate_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    edge_names: List[str],
) -> Dict[str, float]:
    """
    Aggregate edge-wise R2 by node involvement if names like IC_i--IC_j.
    """
    edge_r2 = r2_edgewise(y_true, y_pred)
    node_to_vals: Dict[str, List[float]] = {}
    for i, nm in enumerate(edge_names):
        if "--" not in nm:
            continue
        a, b = nm.split("--", 1)
        node_to_vals.setdefault(a, []).append(float(edge_r2[i]))
        node_to_vals.setdefault(b, []).append(float(edge_r2[i]))
    return {k: float(np.mean(v)) for k, v in node_to_vals.items() if v}


def leakage_from_ridge_probes(
    train_feat_s: np.ndarray,
    train_feat_full: np.ndarray,
    train_target: np.ndarray,
    eval_feat_s: np.ndarray,
    eval_feat_full: np.ndarray,
    eval_target: np.ndarray,
    alpha: float = 1.0,
) -> Dict[str, float]:
    ridge_s = Ridge(alpha=alpha, random_state=0)
    ridge_f = Ridge(alpha=alpha, random_state=0)
    ridge_s.fit(train_feat_s, train_target)
    ridge_f.fit(train_feat_full, train_target)
    pred_s = ridge_s.predict(eval_feat_s)
    pred_f = ridge_f.predict(eval_feat_full)
    r2_s = r2_summary(eval_target, pred_s)
    r2_f = r2_summary(eval_target, pred_f)
    return {
        "r2_shared_only_global": r2_s["r2_global"],
        "r2_full_global": r2_f["r2_global"],
        "leak_global": float(r2_f["r2_global"] - r2_s["r2_global"]),
        "r2_shared_only_edge_mean": r2_s["r2_edge_mean"],
        "r2_full_edge_mean": r2_f["r2_edge_mean"],
        "leak_edge_mean": float(r2_f["r2_edge_mean"] - r2_s["r2_edge_mean"]),
    }


def fit_pca_on_train(y_train: np.ndarray, k: int, seed: int = 42, solver: str = "randomized") -> PCA:
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    k_eff = min(k, y_train.shape[1], y_train.shape[0])
    if k_eff < 1:
        raise ValueError("Invalid PCA dimensionality for given training data.")
    solver = solver or "auto"
    if solver not in {"auto", "full", "randomized"}:
        raise ValueError(f"Unsupported PCA solver: {solver}")
    pca = PCA(n_components=k_eff, svd_solver=solver, random_state=seed)
    pca.fit(y_train)
    return pca


def pc_space_r2_from_pca(y_true: np.ndarray, y_pred: np.ndarray, pca: PCA) -> Dict[str, float]:
    yt = pca.transform(y_true)
    yp = pca.transform(y_pred)
    per_pc = r2_score(yt, yp, multioutput="raw_values")
    per_pc = np.asarray(per_pc, dtype=np.float64)
    per_pc = np.where(np.isfinite(per_pc), per_pc, 0.0)
    return {
        "pc_k": int(per_pc.shape[0]),
        "pc_r2_mean": float(np.mean(per_pc)),
        "pc_r2_median": float(np.median(per_pc)),
        "pc_r2_first5": [float(x) for x in per_pc[:5]],
        "pc_r2_all": [float(x) for x in per_pc],
    }


# --- SPD manifold metrics (sfcoupling v2) --------------------------------


def _sym_eigh_clipped(A: np.ndarray, min_eig: float = 1e-10):
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    return np.maximum(w, min_eig), V


def frechet_error_airm(y_true_spd: np.ndarray, y_pred_spd: np.ndarray) -> np.ndarray:
    """
    Squared AIRM distance per subject: d²(F̂_i, F_i) for i = 1..N.
    Inputs: (N, d, d) stacks of SPD matrices. Returns (N,) float64 vector.
    """
    if y_true_spd.shape != y_pred_spd.shape:
        raise ValueError(
            f"shape mismatch {y_true_spd.shape} vs {y_pred_spd.shape}"
        )
    n = y_true_spd.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        w_a, V_a = _sym_eigh_clipped(y_true_spd[i])
        A_inv_sqrt = V_a @ np.diag(1.0 / np.sqrt(w_a)) @ V_a.T
        M = A_inv_sqrt @ y_pred_spd[i] @ A_inv_sqrt
        w_m, _ = _sym_eigh_clipped(M)
        out[i] = float(np.sum(np.log(w_m) ** 2))
    return out


def log_euclidean_error(y_true_spd: np.ndarray, y_pred_spd: np.ndarray) -> np.ndarray:
    """Per-subject ||log(F) - log(F̂)||_F squared."""
    n = y_true_spd.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        w_t, V_t = _sym_eigh_clipped(y_true_spd[i])
        w_p, V_p = _sym_eigh_clipped(y_pred_spd[i])
        log_t = V_t @ np.diag(np.log(w_t)) @ V_t.T
        log_p = V_p @ np.diag(np.log(w_p)) @ V_p.T
        out[i] = float(np.linalg.norm(log_t - log_p, ord="fro") ** 2)
    return out


def bures_wasserstein_error(
    y_true_spd: np.ndarray, y_pred_spd: np.ndarray
) -> np.ndarray:
    """Per-subject Bures-Wasserstein distance squared."""
    n = y_true_spd.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        w_a, V_a = _sym_eigh_clipped(y_true_spd[i])
        A_sqrt = V_a @ np.diag(np.sqrt(w_a)) @ V_a.T
        inner = A_sqrt @ y_pred_spd[i] @ A_sqrt
        w_i, V_i = _sym_eigh_clipped(inner)
        inner_sqrt_trace = float(np.sum(np.sqrt(w_i)))
        val = (
            float(np.trace(y_true_spd[i]))
            + float(np.trace(y_pred_spd[i]))
            - 2.0 * inner_sqrt_trace
        )
        out[i] = max(val, 0.0)
    return out


def spd_manifold_summary(
    y_true_spd: np.ndarray, y_pred_spd: np.ndarray
) -> Dict[str, float]:
    """Mean / std / median per-subject errors for all three SPD metrics."""
    airm = frechet_error_airm(y_true_spd, y_pred_spd)
    loge = log_euclidean_error(y_true_spd, y_pred_spd)
    bw = bures_wasserstein_error(y_true_spd, y_pred_spd)
    return {
        "airm_mean": float(np.mean(airm)),
        "airm_std": float(np.std(airm)),
        "airm_median": float(np.median(airm)),
        "logE_mean": float(np.mean(loge)),
        "logE_std": float(np.std(loge)),
        "logE_median": float(np.median(loge)),
        "bw_mean": float(np.mean(bw)),
        "bw_std": float(np.std(bw)),
        "bw_median": float(np.median(bw)),
    }

