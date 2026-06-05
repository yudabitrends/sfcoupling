#!/usr/bin/env python3
"""
Kernel Spectral Regression (KSR): GM -> FNC via RFF + Optimal Shrinkage.

Two innovations over Nuclear Norm regression:
  1. Random Fourier Features (RFF) capture nonlinear structure-function coupling
  2. Gavish-Donoho Optimal Spectral Shrinkage replaces uniform nuclear norm
     penalty with per-singular-value data-adaptive shrinkage from RMT

Ablation variants included:
  - Linear-OptShrink: OptShrink on Ridge B (no kernel) — isolates shrinkage
  - KSR-NuclearNorm: RFF + uniform soft-threshold — isolates kernel

Usage:
    python train/run_kernel_spectral_regression.py \
        --config train/config_baselines.yaml \
        --seeds 42 43 44 45 46 47 48 \
        --D 500 --sigma_mults 0.5 1.0 2.0 \
        --pca_ks 5 10 20 50 \
        --n_perm 1000 --n_boot 10000 \
        --save_decomposition \
        --out_dir results/kernel_spectral_regression
"""
import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.spatial.distance import pdist
from sklearn.linear_model import Ridge

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.baselines import fit_ridge_grid
from models.metrics import fit_pca_on_train, pc_space_r2_from_pca, r2_summary
from models.utils import load_config, load_training_contracts, save_json, set_seed
from train.statistical_analysis import bootstrap_bca_ci, run_paired_comparisons

T_TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def _stats(values):
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    mean = float(np.mean(arr))
    if n <= 1:
        return {"mean": mean, "std": float("nan"), "ci95": float("nan")}
    std = float(np.std(arr, ddof=1))
    t = T_TABLE.get(n - 1, 2.0 if n - 1 < 30 else 1.96)
    ci95 = float(t * std / math.sqrt(n))
    return {"mean": mean, "std": std, "ci95": ci95}


# ---------------------------------------------------------------------------
# Random Fourier Features
# ---------------------------------------------------------------------------

def median_bandwidth_heuristic(X: np.ndarray, max_pairs: int = 2000) -> float:
    """Estimate RBF kernel bandwidth from median pairwise distance.

    sigma = sqrt(median_squared_distance / 2) so that the median kernel
    value is exp(-1) ~ 0.368.
    """
    n = X.shape[0]
    if n > max_pairs:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, max_pairs, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    dists = pdist(X_sub, metric="sqeuclidean")
    sigma = float(np.sqrt(np.median(dists) / 2.0))
    return max(sigma, 1e-6)


def make_rff_params(
    dx: int, D: int, sigma: float, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate RFF projection parameters W, b."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((dx, D)) / sigma  # (dx, D)
    b = rng.uniform(0, 2 * np.pi, size=D)      # (D,)
    return W, b


def apply_rff(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Apply RFF transform: phi(x) = sqrt(2/D) * cos(X @ W + b)."""
    D = W.shape[1]
    return np.sqrt(2.0 / D) * np.cos(X @ W + b[np.newaxis, :])


# ---------------------------------------------------------------------------
# Marchenko-Pastur & Optimal Shrinkage (Gavish-Donoho 2014, 2017)
# ---------------------------------------------------------------------------

# Cache MP medians to avoid repeated numerical integration
_MP_MEDIAN_CACHE: Dict[float, float] = {}


def marchenko_pastur_median(beta: float) -> float:
    """Median of the Marchenko-Pastur distribution (continuous part).

    beta = min(m, n) / max(m, n) for an m x n random matrix.
    """
    beta = float(round(beta, 6))  # round for cache hits
    if beta in _MP_MEDIAN_CACHE:
        return _MP_MEDIAN_CACHE[beta]

    beta_plus = (1.0 + np.sqrt(beta)) ** 2
    beta_minus = (1.0 - np.sqrt(beta)) ** 2

    def mp_density(x):
        if x <= beta_minus or x >= beta_plus:
            return 0.0
        return (np.sqrt((beta_plus - x) * (x - beta_minus))
                / (2.0 * np.pi * beta * x))

    def cdf_minus_half(m):
        val, _ = quad(mp_density, beta_minus, m, limit=100)
        return val - 0.5

    med = brentq(cdf_minus_half, beta_minus + 1e-10, beta_plus - 1e-10,
                 xtol=1e-12)
    _MP_MEDIAN_CACHE[beta] = med
    return med


def estimate_noise_sigma(singular_values: np.ndarray, beta: float) -> float:
    """Estimate noise level from singular values via MP median.

    sigma_est = median(sv) / sqrt(mu_beta)
    where mu_beta is the MP median of squared singular values.
    """
    mu_beta = marchenko_pastur_median(beta)
    med_sv = float(np.median(singular_values))
    # The MP distribution describes squared singular values of the noise
    # matrix. The singular values themselves have median ~ sqrt(mu_beta) * sigma.
    sigma_est = med_sv / np.sqrt(mu_beta)
    return max(sigma_est, 1e-10)


def optimal_singular_value_shrinkage(
    singular_values: np.ndarray,
    beta: float,
    sigma: float,
) -> np.ndarray:
    """Gavish-Donoho (2017) optimal nonlinear shrinkage for Frobenius loss.

    eta*(y) = (1/y) * sqrt(max((y^2 - beta_plus*sigma^2)(y^2 - beta_minus*sigma^2), 0))

    Singular values below sqrt(beta_plus) * sigma (MP bulk edge) are zeroed.
    """
    beta_plus = (1.0 + np.sqrt(beta)) ** 2
    beta_minus = (1.0 - np.sqrt(beta)) ** 2

    S = np.asarray(singular_values, dtype=np.float64)
    S_shrunk = np.zeros_like(S)

    bulk_edge = np.sqrt(beta_plus) * sigma

    for i, y in enumerate(S):
        if y <= bulk_edge:
            S_shrunk[i] = 0.0
        else:
            y2 = y * y
            num = (y2 - beta_plus * sigma * sigma) * (y2 - beta_minus * sigma * sigma)
            S_shrunk[i] = np.sqrt(max(num, 0.0)) / y

    return S_shrunk


# ---------------------------------------------------------------------------
# Method: KSR (full)
# ---------------------------------------------------------------------------

def fit_ksr(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    sigma_kernel_multipliers: Optional[List[float]] = None,
    D: int = 500,
    ridge_alphas: Optional[List[float]] = None,
    sigma_noise_multipliers: Optional[List[float]] = None,
    rff_seed: int = 0,
) -> Dict:
    """Kernel Spectral Regression: RFF + Ridge + Optimal Shrinkage.

    Grid search over (sigma_kernel, ridge_alpha, sigma_noise_mult).
    sigma_noise_mult scales the MP-estimated noise level to compensate
    for violation of i.i.d. noise assumptions in Ridge coefficients.
    """
    if sigma_kernel_multipliers is None:
        sigma_kernel_multipliers = [0.5, 1.0, 2.0]
    if ridge_alphas is None:
        ridge_alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    if sigma_noise_multipliers is None:
        sigma_noise_multipliers = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

    sigma_median = median_bandwidth_heuristic(X_train)
    dx = X_train.shape[1]
    dy = Y_train.shape[1]

    best_val_r2 = -np.inf
    best_result = None
    hp_search = []

    for mult in sigma_kernel_multipliers:
        sigma_k = mult * sigma_median
        W, b = make_rff_params(dx, D, sigma_k, seed=rff_seed)
        Phi_train = apply_rff(X_train, W, b)
        Phi_val = apply_rff(X_val, W, b)

        for alpha in ridge_alphas:
            ridge = Ridge(alpha=float(alpha), random_state=0)
            ridge.fit(Phi_train, Y_train)
            B_ridge = ridge.coef_.T  # (D, dy)

            U, S, Vt = np.linalg.svd(B_ridge, full_matrices=False)
            beta = min(D, dy) / max(D, dy)
            sigma_noise_base = estimate_noise_sigma(S, beta)

            for snm in sigma_noise_multipliers:
                sigma_noise = sigma_noise_base * snm
                S_shrunk = optimal_singular_value_shrinkage(S, beta, sigma_noise)
                eff_rank = int(np.sum(S_shrunk > 0))

                B_ksr = (U * S_shrunk) @ Vt
                Y_val_pred = Phi_val @ B_ksr
                val_r2 = r2_summary(Y_val, Y_val_pred)["r2_global"]

                hp_search.append({
                    "sigma_mult": float(mult),
                    "sigma_kernel": float(sigma_k),
                    "ridge_alpha": float(alpha),
                    "sigma_noise_mult": float(snm),
                    "val_r2_global": float(val_r2),
                    "effective_rank": eff_rank,
                    "sigma_noise_base": float(sigma_noise_base),
                    "sigma_noise_used": float(sigma_noise),
                })

                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    best_result = {
                        "rff_params": (W, b),
                        "B_ksr": B_ksr,
                        "sigma_kernel": float(sigma_k),
                        "sigma_mult": float(mult),
                        "ridge_alpha": float(alpha),
                        "sigma_noise_mult": float(snm),
                        "singular_values_original": [float(s) for s in S],
                        "singular_values_shrunk": [float(s) for s in S_shrunk],
                        "sigma_noise_base": float(sigma_noise_base),
                        "sigma_noise_used": float(sigma_noise),
                        "effective_rank": eff_rank,
                        "n_rff_dims": D,
                    }

    best_result["hp_search"] = hp_search
    best_result["best_val_r2"] = float(best_val_r2)
    best_result["sigma_median_heuristic"] = float(sigma_median)
    return best_result


# ---------------------------------------------------------------------------
# Ablation 1: Linear-OptShrink (no kernel)
# ---------------------------------------------------------------------------

def fit_linear_optshrink(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    ridge_alphas: Optional[List[float]] = None,
    sigma_noise_multipliers: Optional[List[float]] = None,
) -> Dict:
    """Ridge + OptShrink in original (dx, dy) space — no kernel features.

    Sweeps (ridge_alpha, sigma_noise_mult) on validation. This avoids
    the issue where fit_ridge_grid picks the best alpha for raw Ridge,
    which may not be optimal when combined with OptShrink.
    """
    if ridge_alphas is None:
        ridge_alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    if sigma_noise_multipliers is None:
        sigma_noise_multipliers = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

    best_val_r2 = -np.inf
    best_result = None
    hp_search = []

    for alpha in ridge_alphas:
        ridge = Ridge(alpha=float(alpha), random_state=0)
        ridge.fit(X_train, Y_train)
        B_ridge = ridge.coef_.T  # (dx, dy)
        dx, dy = B_ridge.shape

        U, S, Vt = np.linalg.svd(B_ridge, full_matrices=False)
        beta = min(dx, dy) / max(dx, dy)
        sigma_noise_base = estimate_noise_sigma(S, beta)

        for snm in sigma_noise_multipliers:
            sigma_noise = sigma_noise_base * snm
            S_shrunk = optimal_singular_value_shrinkage(S, beta, sigma_noise)
            eff_rank = int(np.sum(S_shrunk > 0))

            B_opt = (U * S_shrunk) @ Vt
            Y_val_pred = X_val @ B_opt
            val_r2 = r2_summary(Y_val, Y_val_pred)["r2_global"]

            hp_search.append({
                "ridge_alpha": float(alpha),
                "sigma_noise_mult": float(snm),
                "val_r2_global": float(val_r2),
                "effective_rank": eff_rank,
                "sigma_noise_base": float(sigma_noise_base),
                "sigma_noise_used": float(sigma_noise),
            })

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_result = {
                    "B": B_opt,
                    "ridge_alpha": float(alpha),
                    "sigma_noise_mult": float(snm),
                    "singular_values_original": [float(s) for s in S],
                    "singular_values_shrunk": [float(s) for s in S_shrunk],
                    "sigma_noise_base": float(sigma_noise_base),
                    "sigma_noise_used": float(sigma_noise),
                    "effective_rank": eff_rank,
                    "best_val_r2": float(val_r2),
                }

    best_result["hp_search"] = hp_search
    return best_result


# ---------------------------------------------------------------------------
# Ablation 2: KSR-NuclearNorm (kernel + uniform shrinkage)
# ---------------------------------------------------------------------------

def fit_ksr_nuclear_norm(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    sigma_kernel_multipliers: Optional[List[float]] = None,
    D: int = 500,
    ridge_alphas: Optional[List[float]] = None,
    rff_seed: int = 0,
) -> Dict:
    """RFF + Ridge + uniform SVD soft-thresholding (nuclear norm ablation)."""
    if sigma_kernel_multipliers is None:
        sigma_kernel_multipliers = [0.5, 1.0, 2.0]
    if ridge_alphas is None:
        ridge_alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    sigma_median = median_bandwidth_heuristic(X_train)
    dx = X_train.shape[1]

    # First select best (sigma_k, alpha) by Ridge val R²
    best_ridge_r2 = -np.inf
    best_ridge_cfg = None

    for mult in sigma_kernel_multipliers:
        sigma_k = mult * sigma_median
        W, b = make_rff_params(dx, D, sigma_k, seed=rff_seed)
        Phi_train = apply_rff(X_train, W, b)
        Phi_val = apply_rff(X_val, W, b)

        for alpha in ridge_alphas:
            ridge = Ridge(alpha=float(alpha), random_state=0)
            ridge.fit(Phi_train, Y_train)
            Y_val_pred = ridge.predict(Phi_val)
            val_r2 = r2_summary(Y_val, Y_val_pred)["r2_global"]
            if val_r2 > best_ridge_r2:
                best_ridge_r2 = val_r2
                best_ridge_cfg = {
                    "W": W, "b": b, "sigma_kernel": sigma_k,
                    "sigma_mult": mult, "ridge_alpha": alpha,
                    "B_ridge": ridge.coef_.T,
                    "Phi_val": Phi_val,
                }

    # Now sweep uniform soft-threshold on best Ridge B
    B_ridge = best_ridge_cfg["B_ridge"]
    Phi_val = best_ridge_cfg["Phi_val"]
    U, S, Vt = np.linalg.svd(B_ridge, full_matrices=False)

    # Threshold grid: fractions of max singular value
    thresholds = [0.0] + [S[0] * r for r in
                          [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]]
    best_val_r2 = -np.inf
    best_tau = 0.0
    best_B_nn = B_ridge.copy()

    for tau in thresholds:
        S_thresh = np.maximum(S - tau, 0.0)
        B_nn = (U * S_thresh) @ Vt
        Y_val_pred = Phi_val @ B_nn
        val_r2 = r2_summary(Y_val, Y_val_pred)["r2_global"]
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_tau = tau
            best_B_nn = B_nn.copy()

    eff_rank = int(np.sum(np.maximum(S - best_tau, 0.0) > 0))

    return {
        "rff_params": (best_ridge_cfg["W"], best_ridge_cfg["b"]),
        "B": best_B_nn,
        "sigma_kernel": best_ridge_cfg["sigma_kernel"],
        "sigma_mult": best_ridge_cfg["sigma_mult"],
        "ridge_alpha": best_ridge_cfg["ridge_alpha"],
        "optimal_threshold": float(best_tau),
        "effective_rank": eff_rank,
        "best_val_r2": float(best_val_r2),
        "n_rff_dims": D,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(
    Y_pred: np.ndarray,
    Y_test: np.ndarray,
    Y_train: np.ndarray,
    pca_ks: List[int],
    seed: int,
) -> Dict:
    """Evaluate pre-computed predictions (for kernel methods)."""
    edge_r2 = r2_summary(Y_test, Y_pred)

    pc_r2_by_k = {}
    for k in pca_ks:
        pca = fit_pca_on_train(Y_train, k=k, seed=seed)
        pc_r2 = pc_space_r2_from_pca(Y_test, Y_pred, pca)
        pc_r2_by_k[f"k{k}"] = pc_r2

    var_struct = float(np.sum(Y_pred ** 2))
    var_total = float(np.sum(Y_test ** 2))

    return {
        "edge_r2": edge_r2,
        "pc_r2_by_k": pc_r2_by_k,
        "var_explained_ratio": float(var_struct / (var_total + 1e-15)),
    }


def evaluate_B(
    B: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    Y_train: np.ndarray,
    pca_ks: List[int],
    seed: int,
) -> Dict:
    """Evaluate a coefficient matrix B in original space."""
    Y_pred = X_test @ B
    return evaluate_predictions(Y_pred, Y_test, Y_train, pca_ks, seed)


# ---------------------------------------------------------------------------
# Permutation test for kernel rank dimensions
# ---------------------------------------------------------------------------

def permutation_test_kernel_rank_dims(
    B_ksr: np.ndarray,
    Phi_test: np.ndarray,
    Y_test: np.ndarray,
    n_perm: int = 1000,
    seed: int = 42,
    early_stop_consecutive: int = 3,
) -> List[Dict]:
    """Permutation test for incremental R² of each rank dim in kernel space."""
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(seed)
    U, S, Vt = np.linalg.svd(B_ksr, full_matrices=False)
    n_samples = Phi_test.shape[0]
    max_dim = min(len(S), int(np.sum(S > 1e-12)) + 3)  # beyond last nonzero
    max_dim = min(max_dim, len(S))

    def _r2_rank(r, Y_target):
        if r == 0:
            Y_pred = np.zeros_like(Y_target)
        else:
            B_r = (U[:, :r] * S[:r]) @ Vt[:r, :]
            Y_pred = Phi_test @ B_r
        return float(r2_score(Y_target.ravel(), Y_pred.ravel()))

    results = []
    consecutive_ns = 0

    for r in range(1, max_dim + 1):
        obs_r2_r = _r2_rank(r, Y_test)
        obs_r2_prev = _r2_rank(r - 1, Y_test)
        obs_incr = obs_r2_r - obs_r2_prev

        count_ge = 0
        perm_incrs = np.empty(n_perm, dtype=np.float64)
        for i in range(n_perm):
            perm_idx = rng.permutation(n_samples)
            Y_perm = Y_test[perm_idx]
            perm_r2_r = _r2_rank(r, Y_perm)
            perm_r2_prev = _r2_rank(r - 1, Y_perm)
            perm_incrs[i] = perm_r2_r - perm_r2_prev
            if perm_incrs[i] >= obs_incr:
                count_ge += 1

        p_value = (count_ge + 1) / (n_perm + 1)

        results.append({
            "rank_dim": r,
            "observed_incr_r2": float(obs_incr),
            "cumulative_r2": float(obs_r2_r),
            "p_value": float(p_value),
            "perm_incr_mean": float(np.mean(perm_incrs)),
            "perm_incr_std": float(np.std(perm_incrs)),
            "significant": p_value < 0.05,
        })

        if p_value >= 0.05:
            consecutive_ns += 1
        else:
            consecutive_ns = 0
        if consecutive_ns >= early_stop_consecutive:
            break

    return results


# ---------------------------------------------------------------------------
# Single seed runner
# ---------------------------------------------------------------------------

def run_single_seed(
    cfg: Dict,
    seed: int,
    D: int,
    sigma_mults: List[float],
    sigma_noise_mults: List[float],
    pca_ks: List[int],
    n_perm: int,
    save_decomposition: bool,
    out_dir: Path,
) -> Dict:
    """Run KSR + 2 ablations for one seed."""
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

    ridge_alphas = cfg.get("ridge", {}).get(
        "alphas", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    )

    results = {"seed": seed, "data_shapes": {
        "train": list(Xtr.shape), "val": list(Xva.shape),
        "test": list(Xte.shape), "external": list(Xext.shape),
    }}

    # Use fixed rff_seed=0 across data seeds to isolate data variability
    rff_seed = 0

    # --- KSR (full) ---
    print(f"  [seed {seed}] Fitting KSR (RFF + OptShrink) ...", flush=True)
    t0 = time.time()
    ksr = fit_ksr(Xtr, Ytr, Xva, Yva, sigma_mults, D, ridge_alphas,
                   sigma_noise_mults, rff_seed)
    W, b = ksr["rff_params"]
    Phi_te = apply_rff(Xte, W, b)
    Phi_ext = apply_rff(Xext, W, b)
    Y_pred_ds1 = Phi_te @ ksr["B_ksr"]
    Y_pred_ds2 = Phi_ext @ ksr["B_ksr"]
    ds1_ksr = evaluate_predictions(Y_pred_ds1, Yte, Ytr, pca_ks, seed)
    ds2_ksr = evaluate_predictions(Y_pred_ds2, Yext, Ytr, pca_ks, seed)
    results["ksr"] = {
        "sigma_kernel": ksr["sigma_kernel"],
        "sigma_mult": ksr["sigma_mult"],
        "sigma_median_heuristic": ksr["sigma_median_heuristic"],
        "ridge_alpha": ksr["ridge_alpha"],
        "sigma_noise_mult": ksr["sigma_noise_mult"],
        "n_rff_dims": ksr["n_rff_dims"],
        "effective_rank": ksr["effective_rank"],
        "sigma_noise_base": ksr["sigma_noise_base"],
        "sigma_noise_used": ksr["sigma_noise_used"],
        "singular_values_original": ksr["singular_values_original"][:30],
        "singular_values_shrunk": ksr["singular_values_shrunk"][:30],
        "hp_search": ksr["hp_search"],
        "dataset1_test": ds1_ksr,
        "dataset2_external": ds2_ksr,
        "fit_time_s": round(time.time() - t0, 2),
    }

    # --- Linear-OptShrink (ablation) ---
    print(f"  [seed {seed}] Fitting Linear-OptShrink ...", flush=True)
    t0 = time.time()
    los = fit_linear_optshrink(Xtr, Ytr, Xva, Yva, ridge_alphas,
                               sigma_noise_mults)
    B_los = los["B"]
    ds1_los = evaluate_B(B_los, Xte, Yte, Ytr, pca_ks, seed)
    ds2_los = evaluate_B(B_los, Xext, Yext, Ytr, pca_ks, seed)
    results["linear_optshrink"] = {
        "ridge_alpha": los["ridge_alpha"],
        "sigma_noise_mult": los["sigma_noise_mult"],
        "effective_rank": los["effective_rank"],
        "sigma_noise_base": los["sigma_noise_base"],
        "sigma_noise_used": los["sigma_noise_used"],
        "singular_values_original": los["singular_values_original"][:30],
        "singular_values_shrunk": los["singular_values_shrunk"][:30],
        "dataset1_test": ds1_los,
        "dataset2_external": ds2_los,
        "fit_time_s": round(time.time() - t0, 2),
    }

    # --- KSR-NuclearNorm (ablation) ---
    print(f"  [seed {seed}] Fitting KSR-NuclearNorm ...", flush=True)
    t0 = time.time()
    knn = fit_ksr_nuclear_norm(Xtr, Ytr, Xva, Yva, sigma_mults, D,
                                ridge_alphas, rff_seed)
    W2, b2 = knn["rff_params"]
    Phi_te2 = apply_rff(Xte, W2, b2)
    Phi_ext2 = apply_rff(Xext, W2, b2)
    Y_pred_ds1_knn = Phi_te2 @ knn["B"]
    Y_pred_ds2_knn = Phi_ext2 @ knn["B"]
    ds1_knn = evaluate_predictions(Y_pred_ds1_knn, Yte, Ytr, pca_ks, seed)
    ds2_knn = evaluate_predictions(Y_pred_ds2_knn, Yext, Ytr, pca_ks, seed)
    results["ksr_nuclear_norm"] = {
        "sigma_kernel": knn["sigma_kernel"],
        "sigma_mult": knn["sigma_mult"],
        "ridge_alpha": knn["ridge_alpha"],
        "optimal_threshold": knn["optimal_threshold"],
        "effective_rank": knn["effective_rank"],
        "n_rff_dims": knn["n_rff_dims"],
        "dataset1_test": ds1_knn,
        "dataset2_external": ds2_knn,
        "fit_time_s": round(time.time() - t0, 2),
    }

    # --- Permutation test on KSR rank dims ---
    if n_perm > 0:
        print(f"  [seed {seed}] Permutation test ({n_perm} perms) ...",
              flush=True)
        t0 = time.time()
        perm_results = permutation_test_kernel_rank_dims(
            ksr["B_ksr"], Phi_te, Yte, n_perm=n_perm, seed=seed,
        )
        results["ksr"]["perm_test_rank_dims"] = perm_results
        results["ksr"]["perm_test_time_s"] = round(time.time() - t0, 2)

    # --- Save decompositions ---
    if save_decomposition:
        dec_dir = out_dir / "decompositions"
        dec_dir.mkdir(parents=True, exist_ok=True)
        # KSR
        np.save(dec_dir / f"ksr_seed{seed}_B_kernel.npy", ksr["B_ksr"])
        np.save(dec_dir / f"ksr_seed{seed}_rff_W.npy", W)
        np.save(dec_dir / f"ksr_seed{seed}_rff_b.npy", b)
        np.save(dec_dir / f"ksr_seed{seed}_Y_struct_ds1.npy",
                Y_pred_ds1.astype(np.float32))
        np.save(dec_dir / f"ksr_seed{seed}_Y_resid_ds1.npy",
                (Yte - Y_pred_ds1).astype(np.float32))
        # Linear-OptShrink
        np.save(dec_dir / f"linear_optshrink_seed{seed}_B.npy", B_los)

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(
    all_results: List[Dict], n_boot: int, boot_seed: int,
) -> Dict:
    seeds = [r["seed"] for r in all_results]
    summary = {"n_seeds": len(seeds), "seeds": seeds}
    eval_pca_k = "k20"

    methods = ["ksr", "linear_optshrink", "ksr_nuclear_norm"]

    for method in methods:
        d1_pc, d2_pc, d1_edge, d2_edge, effs = [], [], [], [], []
        for r in all_results:
            m = r[method]
            d1 = m["dataset1_test"]
            d2 = m["dataset2_external"]
            d1_pc.append(d1["pc_r2_by_k"][eval_pca_k]["pc_r2_mean"])
            d2_pc.append(d2["pc_r2_by_k"][eval_pca_k]["pc_r2_mean"])
            d1_edge.append(d1["edge_r2"]["r2_edge_mean"])
            d2_edge.append(d2["edge_r2"]["r2_edge_mean"])
            effs.append(m["effective_rank"])

        summary[method] = {
            "pc_r2_mean_d1": _stats(d1_pc),
            "pc_r2_mean_d2": _stats(d2_pc),
            "edge_r2_mean_d1": _stats(d1_edge),
            "edge_r2_mean_d2": _stats(d2_edge),
            "effective_dim": _stats(effs),
        }

        d1_arr = np.array(d1_pc)
        d2_arr = np.array(d2_pc)
        if len(d1_arr) > 1 and n_boot > 0:
            summary[method]["d1_bootstrap_ci"] = bootstrap_bca_ci(
                d1_arr, n_boot=n_boot, seed=boot_seed)
            summary[method]["d2_bootstrap_ci"] = bootstrap_bca_ci(
                d2_arr, n_boot=n_boot, seed=boot_seed)

        # Per-PCA-k breakdown
        for r in all_results:
            for pk in r[method]["dataset1_test"]["pc_r2_by_k"]:
                vals = [ar[method]["dataset1_test"]["pc_r2_by_k"][pk]["pc_r2_mean"]
                        for ar in all_results]
                summary[method].setdefault("per_pca_k", {})[pk] = _stats(vals)
            break

    return summary


def run_statistical_comparisons(all_results: List[Dict]) -> Dict:
    eval_pca_k = "k20"
    cond_d1, cond_d2 = {}, {}

    for method in ["ksr", "linear_optshrink", "ksr_nuclear_norm"]:
        d1 = np.array([r[method]["dataset1_test"]["pc_r2_by_k"][eval_pca_k]["pc_r2_mean"]
                        for r in all_results])
        d2 = np.array([r[method]["dataset2_external"]["pc_r2_by_k"][eval_pca_k]["pc_r2_mean"]
                        for r in all_results])
        cond_d1[method] = d1
        cond_d2[method] = d2

    pairs = [
        ("ksr", "linear_optshrink"),
        ("ksr", "ksr_nuclear_norm"),
        ("linear_optshrink", "ksr_nuclear_norm"),
    ]

    return {
        "dataset1_test": run_paired_comparisons(cond_d1, pairs),
        "dataset2_external": run_paired_comparisons(cond_d2, pairs),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Kernel Spectral Regression: RFF + Optimal Shrinkage",
    )
    parser.add_argument("--config", type=str, default="train/config_baselines.yaml")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 43, 44, 45, 46, 47, 48])
    parser.add_argument("--D", type=int, default=500,
                        help="Number of random Fourier features")
    parser.add_argument("--sigma_mults", type=float, nargs="+",
                        default=[0.5, 1.0, 2.0])
    parser.add_argument("--sigma_noise_mults", type=float, nargs="+",
                        default=[0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
                        help="Multipliers for MP-estimated sigma_noise")
    parser.add_argument("--pca_ks", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--n_perm", type=int, default=1000)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--save_decomposition", action="store_true")
    parser.add_argument("--out_dir", type=str,
                        default="results/kernel_spectral_regression")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Kernel Spectral Regression (KSR)")
    print(f"  Variants: KSR, Linear-OptShrink, KSR-NuclearNorm")
    print(f"  Seeds:    {args.seeds}")
    print(f"  D (RFF):  {args.D}")
    print(f"  sigma_mults: {args.sigma_mults}")
    print(f"  sigma_noise_mults: {args.sigma_noise_mults}")
    print(f"  PCA ks:   {args.pca_ks}")
    print(f"  Perms:    {args.n_perm}")
    print(f"  Output:   {out_dir}")
    print("=" * 70)

    all_results = []
    for i, seed in enumerate(args.seeds):
        t0 = time.time()
        print(f"\n[{i+1}/{len(args.seeds)}] Seed={seed}", flush=True)
        result = run_single_seed(
            cfg, seed=seed, D=args.D, sigma_mults=args.sigma_mults,
            sigma_noise_mults=args.sigma_noise_mults,
            pca_ks=args.pca_ks, n_perm=args.n_perm,
            save_decomposition=args.save_decomposition, out_dir=out_dir,
        )
        elapsed = time.time() - t0

        ksr_pc = result["ksr"]["dataset1_test"]["pc_r2_by_k"]["k20"]["pc_r2_mean"]
        los_pc = result["linear_optshrink"]["dataset1_test"]["pc_r2_by_k"]["k20"]["pc_r2_mean"]
        knn_pc = result["ksr_nuclear_norm"]["dataset1_test"]["pc_r2_by_k"]["k20"]["pc_r2_mean"]
        print(f"  KSR: {ksr_pc:.4f}  Lin-OS: {los_pc:.4f}  "
              f"KSR-NN: {knn_pc:.4f}  [{elapsed:.1f}s]", flush=True)

        all_results.append(result)
        save_json(out_dir / f"seed_{seed}.json", result)

    summary = aggregate_results(all_results, n_boot=args.n_boot, boot_seed=42)
    save_json(out_dir / "summary.json", summary)

    if len(all_results) > 1:
        comparisons = run_statistical_comparisons(all_results)
        save_json(out_dir / "statistical_comparisons.json", comparisons)

    # Print summary
    print("\n" + "=" * 76)
    print("KSR SUMMARY")
    print("=" * 76)
    print(f"{'Method':<22s} {'DS1 pc_r2(k20)':>16s} {'DS2 pc_r2(k20)':>16s} "
          f"{'DS1 edge_r2':>14s} {'Eff.dim':>9s}")
    print("-" * 76)
    for method in ["ksr", "linear_optshrink", "ksr_nuclear_norm"]:
        s = summary[method]
        d1 = s["pc_r2_mean_d1"]
        d2 = s["pc_r2_mean_d2"]
        e1 = s["edge_r2_mean_d1"]
        ed = s["effective_dim"]
        print(f"{method:<22s} {d1['mean']:>7.4f}+/-{d1['ci95']:.4f} "
              f"{d2['mean']:>7.4f}+/-{d2['ci95']:.4f} "
              f"{e1['mean']:>7.4f}+/-{e1['ci95']:.4f} "
              f"{ed['mean']:>5.1f}+/-{ed['ci95']:.1f}")
    print("=" * 76)

    # Permutation test summary
    if args.n_perm > 0 and "perm_test_rank_dims" in all_results[-1]["ksr"]:
        print("\nKSR Rank Permutation Test (last seed):")
        for e in all_results[-1]["ksr"]["perm_test_rank_dims"]:
            sig = "*" if e["significant"] else " "
            print(f"  dim {e['rank_dim']:2d}: "
                  f"incr_R2={e['observed_incr_r2']:.6f}  "
                  f"p={e['p_value']:.4f} {sig}")

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
