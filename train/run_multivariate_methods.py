#!/usr/bin/env python3
"""
Multivariate decomposition: GM -> FNC via classical methods.

Implements three complementary approaches:
  1. Reduced Rank Regression (RRR) — rank-constrained Ridge
  2. PLS Regression — community standard for structure-function
  3. Nuclear Norm Regularization — smooth rank penalty via ISTA

Each method produces B such that Y_struct = X @ B, Y_resid = Y - Y_struct.
Evaluation uses the same metrics as run_baselines_multiseed.py for direct comparison.

Usage:
    python train/run_multivariate_methods.py \
        --config train/config_baselines.yaml \
        --seeds 42 43 44 45 46 47 48 \
        --max_rank 30 \
        --pca_ks 5 10 20 50 \
        --n_perm 1000 \
        --n_boot 10000 \
        --save_decomposition \
        --out_dir results/multivariate_methods
"""
import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.baselines import fit_ridge_grid
from models.metrics import fit_pca_on_train, pc_space_r2_from_pca, r2_summary
from models.utils import load_config, load_training_contracts, save_json, set_seed
from train.statistical_analysis import (
    bootstrap_bca_ci,
    paired_t_test,
    run_paired_comparisons,
    t_interval_ci,
)

T_TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def _stats(values):
    """Mean, std, 95% CI from an array of per-seed values."""
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
# Method 1: Reduced Rank Regression
# ---------------------------------------------------------------------------

def fit_rrr(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    max_rank: int = 30,
    ridge_alphas: Optional[List[float]] = None,
) -> Dict:
    """Reduced Rank Regression via Ridge + SVD truncation.

    1. Fit Ridge on (X_train, Y_train) selecting alpha by val R².
    2. SVD of coefficient matrix B_ridge.
    3. Truncate to rank r = 1..max_rank, pick r maximising val R².

    Returns dict with B_rrr, rank_spectrum, optimal_rank, singular_values.
    """
    if ridge_alphas is None:
        ridge_alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    # Step 1: select best alpha via validation
    ridge_model, ridge_info = fit_ridge_grid(
        X_train, Y_train, X_val, Y_val, ridge_alphas,
    )
    # B_ridge shape: (dx, dy) — sklearn stores coef_ as (dy, dx)
    B_ridge = ridge_model.coef_.T  # (dx, dy)

    # Step 2: SVD of coefficient matrix
    U, S, Vt = np.linalg.svd(B_ridge, full_matrices=False)
    # U: (dx, min(dx,dy)), S: (min(dx,dy),), Vt: (min(dx,dy), dy)

    max_rank = min(max_rank, len(S))

    # Step 3: rank sweep on validation
    rank_spectrum = []
    best_rank = 1
    best_val_r2 = -np.inf

    for r in range(1, max_rank + 1):
        B_r = (U[:, :r] * S[:r]) @ Vt[:r, :]
        Y_val_pred = X_val @ B_r
        val_r2 = r2_summary(Y_val, Y_val_pred)["r2_global"]
        rank_spectrum.append({"rank": r, "val_r2_global": float(val_r2)})
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_rank = r

    # Optimal B
    B_rrr = (U[:, :best_rank] * S[:best_rank]) @ Vt[:best_rank, :]

    return {
        "B": B_rrr,
        "rank_spectrum": rank_spectrum,
        "optimal_rank": best_rank,
        "singular_values": [float(s) for s in S[:max_rank]],
        "ridge_alpha": ridge_info["best_alpha"],
        "best_val_r2": float(best_val_r2),
    }


# ---------------------------------------------------------------------------
# Method 2: PLS Regression
# ---------------------------------------------------------------------------

def fit_pls(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    max_components: int = 30,
) -> Dict:
    """PLS Regression with component sweep on validation.

    scale=False because data is already z-scored.

    Returns dict with best model's B matrix, component_spectrum, optimal_n.
    """
    max_components = min(max_components, X_train.shape[1], X_train.shape[0])

    component_spectrum = []
    best_n = 1
    best_val_r2 = -np.inf
    best_model = None

    for n in range(1, max_components + 1):
        pls = PLSRegression(n_components=n, scale=False, max_iter=1000)
        pls.fit(X_train, Y_train)
        Y_val_pred = pls.predict(X_val)
        val_r2 = r2_summary(Y_val, Y_val_pred)["r2_global"]
        component_spectrum.append({"n_components": n, "val_r2_global": float(val_r2)})
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_n = n
            best_model = pls

    # Extract coefficient matrix: Y ≈ X @ B
    # PLS coef_ shape: (dy, dx), so transpose to (dx, dy)
    B_pls = best_model.coef_.T

    return {
        "B": B_pls,
        "component_spectrum": component_spectrum,
        "optimal_n": best_n,
        "best_val_r2": float(best_val_r2),
        "model": best_model,
    }


# ---------------------------------------------------------------------------
# Method 3: Nuclear Norm Regularization via ISTA
# ---------------------------------------------------------------------------

def _svd_soft_threshold(M: np.ndarray, threshold: float) -> Tuple[np.ndarray, float]:
    """Proximal operator for nuclear norm: SVD soft-thresholding.

    Returns (thresholded matrix, effective rank).
    """
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    S_thresh = np.maximum(S - threshold, 0.0)
    eff_rank = float(np.sum(S_thresh > 0))
    B_new = (U * S_thresh) @ Vt
    return B_new, eff_rank


def fit_nuclear_norm(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    lambdas: Optional[List[float]] = None,
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> Dict:
    """Nuclear norm regularized regression via ISTA.

    min_B  (1/2n)||Y - XB||_F^2 + lambda * ||B||_*

    Warm-starts from large to small lambda.

    Returns dict with B_nn, regularization_path, optimal_lambda.
    """
    n = X_train.shape[0]
    XtX = X_train.T @ X_train  # (dx, dx)
    XtY = X_train.T @ Y_train  # (dx, dy)

    # Step size: 1 / max_eigenvalue(XtX / n)
    eig_max = np.linalg.eigvalsh(XtX / n)[-1]
    step = 1.0 / (eig_max + 1e-10)

    # Auto lambda path
    if lambdas is None:
        # lambda_max: smallest lambda that gives B=0
        # At B=0, gradient = -XtY/n, so lambda_max = ||XtY/n||_op
        lambda_max = np.linalg.svd(XtY / n, compute_uv=False)[0]
        ratios = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        lambdas = [float(lambda_max * r) for r in ratios]

    # Sort large to small for warm start
    lambdas = sorted(lambdas, reverse=True)

    B = np.zeros((X_train.shape[1], Y_train.shape[1]), dtype=np.float64)
    reg_path = []
    best_lam = lambdas[-1]
    best_val_r2 = -np.inf
    best_B = B.copy()

    for lam in lambdas:
        # ISTA iterations
        for it in range(max_iter):
            grad = (XtX @ B - XtY) / n
            B_candidate = B - step * grad
            B_new, eff_rank = _svd_soft_threshold(B_candidate, step * lam)

            # Convergence check
            diff_norm = np.linalg.norm(B_new - B)
            b_norm = np.linalg.norm(B) + 1e-15
            B = B_new
            if diff_norm / b_norm < tol:
                break

        Y_val_pred = X_val @ B
        val_r2 = r2_summary(Y_val, Y_val_pred)["r2_global"]
        reg_path.append({
            "lambda": float(lam),
            "val_r2_global": float(val_r2),
            "effective_rank": float(eff_rank),
            "n_iter": it + 1,
        })

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_lam = lam
            best_B = B.copy()

    return {
        "B": best_B,
        "regularization_path": reg_path,
        "optimal_lambda": float(best_lam),
        "best_val_r2": float(best_val_r2),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_B(
    B: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    Y_train: np.ndarray,
    pca_ks: List[int],
    seed: int,
) -> Dict:
    """Evaluate a coefficient matrix B on test data.

    Returns edge R², PC-space R² at multiple k, and variance explained ratio.
    """
    Y_struct = X_test @ B
    Y_resid = Y_test - Y_struct

    edge_r2 = r2_summary(Y_test, Y_struct)

    pc_r2_by_k = {}
    for k in pca_ks:
        pca = fit_pca_on_train(Y_train, k=k, seed=seed)
        pc_r2 = pc_space_r2_from_pca(Y_test, Y_struct, pca)
        pc_r2_by_k[f"k{k}"] = pc_r2

    # Variance explained ratio: ||Y_struct||_F^2 / ||Y||_F^2
    var_struct = float(np.sum(Y_struct ** 2))
    var_total = float(np.sum(Y_test ** 2))
    var_explained_ratio = var_struct / (var_total + 1e-15)

    return {
        "edge_r2": edge_r2,
        "pc_r2_by_k": pc_r2_by_k,
        "var_explained_ratio": float(var_explained_ratio),
    }


# ---------------------------------------------------------------------------
# Permutation test for rank dimensions
# ---------------------------------------------------------------------------

def permutation_test_rank_dims(
    B: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    n_perm: int = 1000,
    seed: int = 42,
    early_stop_consecutive: int = 3,
) -> List[Dict]:
    """Permutation test for incremental R² of each rank dimension.

    For each rank r, tests whether the incremental R² (rank r vs rank r-1)
    is significantly greater than expected under permutation.
    Early stops after `early_stop_consecutive` non-significant dimensions.
    """
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(seed)
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    n_samples = X_test.shape[0]
    max_dim = min(len(S), X_test.shape[1])

    def _global_r2_rank(r, Y_target):
        """R² for rank-r approximation against Y_target."""
        if r == 0:
            # Predict zeros (mean model R² would be different, but here
            # we use the explicit prediction Y_pred = 0 for consistency)
            Y_pred = np.zeros_like(Y_target)
        else:
            B_r = (U[:, :r] * S[:r]) @ Vt[:r, :]
            Y_pred = X_test @ B_r
        return float(r2_score(Y_target.ravel(), Y_pred.ravel()))

    results = []
    consecutive_ns = 0

    for r in range(1, max_dim + 1):
        # Observed incremental R²
        obs_r2_r = _global_r2_rank(r, Y_test)
        obs_r2_prev = _global_r2_rank(r - 1, Y_test)
        obs_incr = obs_r2_r - obs_r2_prev

        # Permutation null
        count_ge = 0
        perm_incrs = np.empty(n_perm, dtype=np.float64)
        for i in range(n_perm):
            perm_idx = rng.permutation(n_samples)
            Y_perm = Y_test[perm_idx]
            perm_r2_r = _global_r2_rank(r, Y_perm)
            perm_r2_prev = _global_r2_rank(r - 1, Y_perm)
            perm_incr = perm_r2_r - perm_r2_prev
            perm_incrs[i] = perm_incr
            if perm_incr >= obs_incr:
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
    max_rank: int,
    pca_ks: List[int],
    n_perm: int,
    save_decomposition: bool,
    out_dir: Path,
) -> Dict:
    """Run all three methods for a single seed."""
    set_seed(seed)
    data = load_training_contracts(cfg)

    idx_tr = data["idx1_train"]
    idx_val = data["idx1_val"]
    idx_te = data["idx1_test"]
    idx_ext = data["idx2_external"]

    Xtr, Ytr = data["X1"][idx_tr], data["Y1"][idx_tr]
    Xva, Yva = data["X1"][idx_val], data["Y1"][idx_val]
    Xte, Yte = data["X1"][idx_te], data["Y1"][idx_te]
    Xext, Yext = data["X2"][idx_ext], data["Y2"][idx_ext]

    # Cast to float64 for numerical stability in SVD / ISTA
    Xtr64 = Xtr.astype(np.float64)
    Ytr64 = Ytr.astype(np.float64)
    Xva64 = Xva.astype(np.float64)
    Yva64 = Yva.astype(np.float64)
    Xte64 = Xte.astype(np.float64)
    Yte64 = Yte.astype(np.float64)
    Xext64 = Xext.astype(np.float64)
    Yext64 = Yext.astype(np.float64)

    ridge_alphas = cfg.get("ridge", {}).get(
        "alphas", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    )

    results = {"seed": seed, "data_shapes": {
        "train": list(Xtr.shape), "val": list(Xva.shape),
        "test": list(Xte.shape), "external": list(Xext.shape),
    }}

    # --- RRR ---
    print(f"  [seed {seed}] Fitting RRR ...", flush=True)
    t0 = time.time()
    rrr = fit_rrr(Xtr64, Ytr64, Xva64, Yva64,
                  max_rank=max_rank, ridge_alphas=ridge_alphas)
    B_rrr = rrr["B"]
    ds1_rrr = evaluate_B(B_rrr, Xte64, Yte64, Ytr64, pca_ks, seed)
    ds2_rrr = evaluate_B(B_rrr, Xext64, Yext64, Ytr64, pca_ks, seed)
    results["rrr"] = {
        "optimal_rank": rrr["optimal_rank"],
        "ridge_alpha": rrr["ridge_alpha"],
        "singular_values": rrr["singular_values"],
        "rank_spectrum": rrr["rank_spectrum"],
        "dataset1_test": ds1_rrr,
        "dataset2_external": ds2_rrr,
        "fit_time_s": round(time.time() - t0, 2),
    }

    # --- PLS ---
    print(f"  [seed {seed}] Fitting PLS ...", flush=True)
    t0 = time.time()
    pls = fit_pls(Xtr64, Ytr64, Xva64, Yva64, max_components=max_rank)
    B_pls = pls["B"]
    ds1_pls = evaluate_B(B_pls, Xte64, Yte64, Ytr64, pca_ks, seed)
    ds2_pls = evaluate_B(B_pls, Xext64, Yext64, Ytr64, pca_ks, seed)
    results["pls"] = {
        "optimal_n": pls["optimal_n"],
        "component_spectrum": pls["component_spectrum"],
        "dataset1_test": ds1_pls,
        "dataset2_external": ds2_pls,
        "fit_time_s": round(time.time() - t0, 2),
    }

    # --- Nuclear Norm ---
    print(f"  [seed {seed}] Fitting Nuclear Norm ...", flush=True)
    t0 = time.time()
    nn = fit_nuclear_norm(Xtr64, Ytr64, Xva64, Yva64)
    B_nn = nn["B"]
    ds1_nn = evaluate_B(B_nn, Xte64, Yte64, Ytr64, pca_ks, seed)
    ds2_nn = evaluate_B(B_nn, Xext64, Yext64, Ytr64, pca_ks, seed)
    results["nuclear_norm"] = {
        "optimal_lambda": nn["optimal_lambda"],
        "regularization_path": nn["regularization_path"],
        "dataset1_test": ds1_nn,
        "dataset2_external": ds2_nn,
        "fit_time_s": round(time.time() - t0, 2),
    }

    # --- Permutation test on RRR rank dimensions ---
    if n_perm > 0:
        print(f"  [seed {seed}] Permutation test ({n_perm} perms) ...", flush=True)
        t0 = time.time()
        perm_results = permutation_test_rank_dims(
            B_rrr, Xte64, Yte64, n_perm=n_perm, seed=seed,
        )
        results["rrr"]["perm_test_rank_dims"] = perm_results
        results["rrr"]["perm_test_time_s"] = round(time.time() - t0, 2)

    # --- Save decompositions ---
    if save_decomposition:
        dec_dir = out_dir / "decompositions"
        dec_dir.mkdir(parents=True, exist_ok=True)
        for method_name, B_mat in [("rrr", B_rrr), ("pls", B_pls),
                                    ("nuclear_norm", B_nn)]:
            np.save(dec_dir / f"{method_name}_seed{seed}_B.npy", B_mat)
            Y_struct_ds1 = Xte64 @ B_mat
            Y_resid_ds1 = Yte64 - Y_struct_ds1
            np.save(dec_dir / f"{method_name}_seed{seed}_Y_struct_ds1.npy",
                    Y_struct_ds1.astype(np.float32))
            np.save(dec_dir / f"{method_name}_seed{seed}_Y_resid_ds1.npy",
                    Y_resid_ds1.astype(np.float32))

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(all_results: List[Dict], n_boot: int, boot_seed: int) -> Dict:
    """Aggregate per-seed results into summary statistics."""
    seeds = [r["seed"] for r in all_results]
    summary = {"n_seeds": len(seeds), "seeds": seeds}

    methods = ["rrr", "pls", "nuclear_norm"]
    eval_pca_k = "k20"  # primary metric

    for method in methods:
        # Collect per-seed metrics
        d1_pc_r2 = []
        d2_pc_r2 = []
        d1_edge_r2 = []
        d2_edge_r2 = []
        effective_dims = []

        for r in all_results:
            m = r[method]
            d1 = m["dataset1_test"]
            d2 = m["dataset2_external"]
            d1_pc_r2.append(d1["pc_r2_by_k"][eval_pca_k]["pc_r2_mean"])
            d2_pc_r2.append(d2["pc_r2_by_k"][eval_pca_k]["pc_r2_mean"])
            d1_edge_r2.append(d1["edge_r2"]["r2_edge_mean"])
            d2_edge_r2.append(d2["edge_r2"]["r2_edge_mean"])

            if method == "rrr":
                effective_dims.append(m["optimal_rank"])
            elif method == "pls":
                effective_dims.append(m["optimal_n"])
            elif method == "nuclear_norm":
                # Get effective rank at optimal lambda
                opt_lam = m["optimal_lambda"]
                for entry in m["regularization_path"]:
                    if abs(entry["lambda"] - opt_lam) < 1e-12:
                        effective_dims.append(entry["effective_rank"])
                        break

        d1_pc_arr = np.array(d1_pc_r2)
        d2_pc_arr = np.array(d2_pc_r2)

        summary[method] = {
            "pc_r2_mean_d1": _stats(d1_pc_r2),
            "pc_r2_mean_d2": _stats(d2_pc_r2),
            "edge_r2_mean_d1": _stats(d1_edge_r2),
            "edge_r2_mean_d2": _stats(d2_edge_r2),
            "effective_dim": _stats(effective_dims),
        }

        # Bootstrap CIs
        if len(d1_pc_arr) > 1 and n_boot > 0:
            summary[method]["d1_bootstrap_ci"] = bootstrap_bca_ci(
                d1_pc_arr, n_boot=n_boot, seed=boot_seed,
            )
            summary[method]["d2_bootstrap_ci"] = bootstrap_bca_ci(
                d2_pc_arr, n_boot=n_boot, seed=boot_seed,
            )

        # Per-PCA-k breakdown
        for r in all_results:
            for pk in r[method]["dataset1_test"]["pc_r2_by_k"]:
                if pk not in summary[method].get("per_pca_k", {}):
                    vals = [
                        ar[method]["dataset1_test"]["pc_r2_by_k"][pk]["pc_r2_mean"]
                        for ar in all_results
                    ]
                    summary[method].setdefault("per_pca_k", {})[pk] = _stats(vals)
            break  # only need one pass to discover keys

    return summary


def run_statistical_comparisons(
    all_results: List[Dict],
) -> Dict:
    """Paired comparisons between methods across seeds."""
    eval_pca_k = "k20"
    conditions_d1 = {}
    conditions_d2 = {}

    for method in ["rrr", "pls", "nuclear_norm"]:
        d1_vals = np.array([
            r[method]["dataset1_test"]["pc_r2_by_k"][eval_pca_k]["pc_r2_mean"]
            for r in all_results
        ])
        d2_vals = np.array([
            r[method]["dataset2_external"]["pc_r2_by_k"][eval_pca_k]["pc_r2_mean"]
            for r in all_results
        ])
        conditions_d1[method] = d1_vals
        conditions_d2[method] = d2_vals

    pairs = [
        ("rrr", "pls"),
        ("rrr", "nuclear_norm"),
        ("pls", "nuclear_norm"),
    ]

    return {
        "dataset1_test": run_paired_comparisons(conditions_d1, pairs),
        "dataset2_external": run_paired_comparisons(conditions_d2, pairs),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multivariate GM->FNC decomposition (RRR + PLS + Nuclear Norm)",
    )
    parser.add_argument("--config", type=str, default="train/config_baselines.yaml")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 43, 44, 45, 46, 47, 48])
    parser.add_argument("--max_rank", type=int, default=30)
    parser.add_argument("--pca_ks", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--n_perm", type=int, default=1000)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--save_decomposition", action="store_true")
    parser.add_argument("--out_dir", type=str,
                        default="results/multivariate_methods")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Multivariate GM -> FNC Decomposition")
    print(f"  Methods: RRR, PLS, Nuclear Norm")
    print(f"  Seeds:   {args.seeds}")
    print(f"  Max rank / components: {args.max_rank}")
    print(f"  PCA ks for evaluation: {args.pca_ks}")
    print(f"  Permutations: {args.n_perm}")
    print(f"  Output: {out_dir}")
    print("=" * 70)

    all_results = []
    for i, seed in enumerate(args.seeds):
        t0 = time.time()
        print(f"\n[{i+1}/{len(args.seeds)}] Seed={seed}", flush=True)
        result = run_single_seed(
            cfg, seed=seed, max_rank=args.max_rank,
            pca_ks=args.pca_ks, n_perm=args.n_perm,
            save_decomposition=args.save_decomposition,
            out_dir=out_dir,
        )
        elapsed = time.time() - t0

        # Quick printout
        rrr_pc = result["rrr"]["dataset1_test"]["pc_r2_by_k"]["k20"]["pc_r2_mean"]
        pls_pc = result["pls"]["dataset1_test"]["pc_r2_by_k"]["k20"]["pc_r2_mean"]
        nn_pc = result["nuclear_norm"]["dataset1_test"]["pc_r2_by_k"]["k20"]["pc_r2_mean"]
        print(f"  RRR (rank={result['rrr']['optimal_rank']}): "
              f"pc_r2={rrr_pc:.4f}  "
              f"PLS (n={result['pls']['optimal_n']}): "
              f"pc_r2={pls_pc:.4f}  "
              f"NN: pc_r2={nn_pc:.4f}  [{elapsed:.1f}s]", flush=True)

        all_results.append(result)
        save_json(out_dir / f"seed_{seed}.json", result)

    # Aggregate
    summary = aggregate_results(all_results, n_boot=args.n_boot, boot_seed=42)
    save_json(out_dir / "summary.json", summary)

    # Statistical comparisons between methods
    if len(all_results) > 1:
        comparisons = run_statistical_comparisons(all_results)
        save_json(out_dir / "statistical_comparisons.json", comparisons)

    # Print summary table
    print("\n" + "=" * 70)
    print("MULTIVARIATE METHODS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20s} {'DS1 pc_r2(k20)':>16s} {'DS2 pc_r2(k20)':>16s} "
          f"{'DS1 edge_r2':>14s} {'Eff. dim':>10s}")
    print("-" * 70)
    for method in ["rrr", "pls", "nuclear_norm"]:
        s = summary[method]
        d1 = s["pc_r2_mean_d1"]
        d2 = s["pc_r2_mean_d2"]
        e1 = s["edge_r2_mean_d1"]
        ed = s["effective_dim"]
        print(f"{method:<20s} {d1['mean']:>7.4f}+/-{d1['ci95']:.4f} "
              f"{d2['mean']:>7.4f}+/-{d2['ci95']:.4f} "
              f"{e1['mean']:>7.4f}+/-{e1['ci95']:.4f} "
              f"{ed['mean']:>5.1f}+/-{ed['ci95']:.1f}")
    print("=" * 70)

    # Rank permutation test summary (from last seed for display)
    if args.n_perm > 0 and "perm_test_rank_dims" in all_results[-1]["rrr"]:
        print("\nRRR Rank Dimension Permutation Test (last seed):")
        for entry in all_results[-1]["rrr"]["perm_test_rank_dims"]:
            sig = "*" if entry["significant"] else " "
            print(f"  dim {entry['rank_dim']:2d}: "
                  f"incr_R2={entry['observed_incr_r2']:.6f}  "
                  f"p={entry['p_value']:.4f} {sig}")

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
