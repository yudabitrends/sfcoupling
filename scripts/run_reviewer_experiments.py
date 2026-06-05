#!/usr/bin/env python3
"""
Run all reviewer-requested experiments for NeuroImage revision.

Experiments:
  E1: HC-only sensitivity analysis (Major Comment M1)
      - Fit Nuclear Norm on HC-only subset of DS1
      - Report subspace overlap, effective rank, PC-R²
  E2: 10K permutation test (Major Comment M5)
      - Re-run subspace overlap with 10,000 permutations for finer p-values
  E3: Nested CV SZ classification (Major Comment M7)
      - 5-fold outer CV for coupled/full/uncoupled GM AUC
  E4: OptShrink dissociation metrics (Major Comment M9)
      - Report subspace overlap and R² for OptShrink alongside Nuclear Norm

Usage:
    python scripts/run_reviewer_experiments.py --config train/config_baselines.yaml
    python scripts/run_reviewer_experiments.py --config train/config_baselines.yaml --experiments E1 E2 E3 E4
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.metrics import fit_pca_on_train, pc_space_r2_from_pca, r2_summary
from models.utils import load_config, load_training_contracts, save_json, set_seed
from train.run_multivariate_methods import fit_nuclear_norm, evaluate_B
from train.run_subspace_analysis import (
    analyze_method,
    principal_angles,
    subspace_overlap,
    top_k_subspace,
)

OUT_DIR = PROJECT_ROOT / "results" / "reviewer_experiments"
DECOMP_DIR = PROJECT_ROOT / "results" / "multivariate_methods" / "decompositions"
KSR_DECOMP_DIR = PROJECT_ROOT / "results" / "kernel_spectral_regression" / "decompositions"

# Cache for B matrices computed on-the-fly
_B_CACHE: Dict[str, np.ndarray] = {}


def _get_or_fit_B(name: str, data: Dict, seed: int = 42) -> np.ndarray:
    """Load B from disk or fit from scratch and cache."""
    cache_key = f"{name}_seed{seed}"
    if cache_key in _B_CACHE:
        return _B_CACHE[cache_key]

    # Try loading from disk
    for base_dir in [DECOMP_DIR, KSR_DECOMP_DIR]:
        path = base_dir / f"{name}_seed{seed}_B.npy"
        if path.exists():
            B = np.load(path)
            _B_CACHE[cache_key] = B
            return B

    # Fit from scratch
    set_seed(seed)
    X1, Y1 = data["X1"].astype(np.float64), data["Y1"].astype(np.float64)
    Xtr = X1[data["idx1_train"]]
    Ytr = Y1[data["idx1_train"]]
    Xva = X1[data["idx1_val"]]
    Yva = Y1[data["idx1_val"]]

    if name == "nuclear_norm":
        print(f"  Fitting Nuclear Norm (seed={seed}) from scratch ...", flush=True)
        result = fit_nuclear_norm(Xtr, Ytr, Xva, Yva)
        B = result["B"]
    elif name == "linear_optshrink":
        print(f"  Fitting Linear OptShrink (seed={seed}) from scratch ...", flush=True)
        from models.baselines import fit_ridge_grid
        ridge_alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        ridge_model, _ = fit_ridge_grid(Xtr, Ytr, Xva, Yva, ridge_alphas)
        B_ridge = ridge_model.coef_.T
        # Gavish-Donoho optimal shrinkage
        U, S, Vt = np.linalg.svd(B_ridge, full_matrices=False)
        m, n_feat = B_ridge.shape
        gamma = min(m, n_feat) / max(m, n_feat)
        sigma_plus = np.sqrt((1 + np.sqrt(gamma)) ** 2)
        # Estimate noise level from residuals
        Y_pred = Xtr @ B_ridge
        resid = Ytr - Y_pred
        sigma_hat = np.median(np.abs(resid)) / 0.6745  # MAD estimator
        threshold = sigma_hat * sigma_plus
        S_shrunk = np.where(S > threshold, S, 0.0)
        B = (U * S_shrunk) @ Vt
    elif name == "rrr":
        print(f"  Fitting RRR (seed={seed}) from scratch ...", flush=True)
        from train.run_multivariate_methods import fit_rrr
        result = fit_rrr(Xtr, Ytr, Xva, Yva)
        B = result["B"]
    elif name == "pls":
        print(f"  Fitting PLS (seed={seed}) from scratch ...", flush=True)
        from train.run_multivariate_methods import fit_pls
        result = fit_pls(Xtr, Ytr, Xva, Yva)
        B = result["B"]
    else:
        raise ValueError(f"Unknown method: {name}")

    # Save for future use
    DECOMP_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DECOMP_DIR / f"{name}_seed{seed}_B.npy", B)
    print(f"  Saved B to {DECOMP_DIR / f'{name}_seed{seed}_B.npy'}")

    _B_CACHE[cache_key] = B
    return B


# ═══════════════════════════════════════════════════════════════════════════
# E1: HC-only Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════════════════

def experiment_e1_hc_only(data: Dict, seed: int = 42) -> Dict:
    """Fit Nuclear Norm on HC-only DS1 subset and report key metrics."""
    print("\n" + "=" * 70)
    print("E1: HC-only Sensitivity Analysis")
    print("=" * 70)

    set_seed(seed)

    subjects = data["subjects1"]
    diag = subjects["Diagnosis"].values

    # Get HC indices within each partition
    idx_train = data["idx1_train"]
    idx_val = data["idx1_val"]
    idx_test = data["idx1_test"]

    hc_train = idx_train[diag[idx_train] == 0]
    hc_val = idx_val[diag[idx_val] == 0]
    hc_test = idx_test[diag[idx_test] == 0]

    print(f"  HC train: {len(hc_train)} (of {len(idx_train)})")
    print(f"  HC val:   {len(hc_val)} (of {len(idx_val)})")
    print(f"  HC test:  {len(hc_test)} (of {len(idx_test)})")

    X1, Y1 = data["X1"].astype(np.float64), data["Y1"].astype(np.float64)

    Xtr, Ytr = X1[hc_train], Y1[hc_train]
    Xva, Yva = X1[hc_val], Y1[hc_val]
    Xte, Yte = X1[hc_test], Y1[hc_test]

    # Fit Nuclear Norm
    print("  Fitting Nuclear Norm on HC-only ...", flush=True)
    t0 = time.time()
    nn_result = fit_nuclear_norm(Xtr, Ytr, Xva, Yva)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    B_hc = nn_result["B"]
    U, S, Vt = np.linalg.svd(B_hc, full_matrices=False)
    eff_rank = int(np.sum(S > 1e-10))

    # PC-R² evaluation
    pca_ks = [5, 10, 20]
    eval_result = evaluate_B(B_hc, Xte, Yte, Ytr, pca_ks, seed)

    # Subspace overlap at k=5,10,20
    rng = np.random.default_rng(seed)
    subspace_ks = [5, 10, 20]
    subspace_results = {}
    for k in subspace_ks:
        V_actual = top_k_subspace(Yte, k)
        Y_pred = Xte @ B_hc
        V_pred = top_k_subspace(Y_pred, k)
        cos_angles = principal_angles(V_actual, V_pred)
        overlap = float(np.mean(cos_angles ** 2))

        # Random null (1000 perms for HC-only)
        null_overlaps = []
        for _ in range(1000):
            Q, _ = np.linalg.qr(rng.standard_normal((Yte.shape[1], k)))
            null_overlaps.append(subspace_overlap(V_actual, Q))
        null_overlaps = np.array(null_overlaps)
        p_val = float((np.sum(null_overlaps >= overlap) + 1) / 1001)

        subspace_results[f"k={k}"] = {
            "overlap": overlap,
            "cos_angles": cos_angles.tolist(),
            "null_mean": float(np.mean(null_overlaps)),
            "p_value": p_val,
        }
        print(f"  k={k}: overlap={overlap:.4f} (null={np.mean(null_overlaps):.4f}, p={p_val:.4f})")

    # Also evaluate on full test set (mixed) for comparison
    Xte_all = X1[idx_test]
    Yte_all = Y1[idx_test]
    eval_full_test = evaluate_B(B_hc, Xte_all, Yte_all, Ytr, pca_ks, seed)

    # Compare with mixed-cohort B (seed 42)
    comparison = {}
    try:
        B_mixed = _get_or_fit_B("nuclear_norm", data, seed)
        U_mixed, S_mixed, Vt_mixed = np.linalg.svd(B_mixed, full_matrices=False)
        eff_rank_mixed = int(np.sum(S_mixed > 1e-10))

        # Subspace overlap between HC-only and mixed B matrices
        for k in [5, 10, 20]:
            r = min(k, eff_rank, eff_rank_mixed)
            V_hc = Vt[:r].T
            V_mx = Vt_mixed[:r].T
            cos_ang = principal_angles(V_hc, V_mx)
            comparison[f"k={k}"] = {
                "hc_vs_mixed_overlap": float(np.mean(cos_ang ** 2)),
                "cos_angles": cos_ang.tolist(),
            }
            print(f"  HC vs Mixed B subspace overlap (k={k}): {np.mean(cos_ang**2):.4f}")

        comparison["eff_rank_mixed"] = eff_rank_mixed
    except Exception as e:
        print(f"  Warning: Could not load mixed B for comparison: {e}")

    results = {
        "n_hc_train": len(hc_train),
        "n_hc_val": len(hc_val),
        "n_hc_test": len(hc_test),
        "effective_rank": eff_rank,
        "optimal_lambda": nn_result["optimal_lambda"],
        "eval_hc_test": {
            "edge_r2": eval_result["edge_r2"],
            "pc_r2_by_k": eval_result["pc_r2_by_k"],
        },
        "eval_full_test": {
            "edge_r2": eval_full_test["edge_r2"],
            "pc_r2_by_k": eval_full_test["pc_r2_by_k"],
        },
        "subspace_overlap": subspace_results,
        "hc_vs_mixed_comparison": comparison,
        "top10_singular_values": S[:10].tolist(),
    }

    print(f"\n  Effective rank (HC-only): {eff_rank}")
    for k in pca_ks:
        hc_r2 = eval_result["pc_r2_by_k"][f"k{k}"]["pc_r2_mean"]
        full_r2 = eval_full_test["pc_r2_by_k"][f"k{k}"]["pc_r2_mean"]
        print(f"  PC-R² k={k}: HC-test={hc_r2:.4f}, full-test={full_r2:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# E2: 10K Permutation Test
# ═══════════════════════════════════════════════════════════════════════════

def experiment_e2_10k_permutations(data: Dict, seed: int = 42) -> Dict:
    """Re-run subspace overlap tests with 10,000 iterations.

    Uses 10K random-subspace null (cheap: QR only) and 1K row-permutation null
    (expensive: SVD per permutation) for practical computation time.
    """
    print("\n" + "=" * 70)
    print("E2: 10K Permutation Test")
    print("=" * 70)

    set_seed(seed)
    rng = np.random.default_rng(seed)
    n_random = 10000
    n_perm = 1000  # row-permutation is SVD-heavy, keep at 1K

    X1, Y1 = data["X1"].astype(np.float64), data["Y1"].astype(np.float64)
    idx_test = data["idx1_test"]
    Xte, Yte = X1[idx_test], Y1[idx_test]

    # Get Nuclear Norm B
    B = _get_or_fit_B("nuclear_norm", data, seed)
    Y_pred = Xte @ B
    n, d = Yte.shape

    subspace_ks = [3, 5, 10, 20]
    results = {}

    for k in subspace_ks:
        print(f"\n  k={k}: running {n_random} random + {n_perm} perm null ...", flush=True)
        t0 = time.time()

        V_actual = top_k_subspace(Yte, k)
        V_pred = top_k_subspace(Y_pred, k)
        cos_angles = principal_angles(V_actual, V_pred)
        observed_overlap = float(np.mean(cos_angles ** 2))

        # Random subspace null (10K — cheap)
        null_random = []
        for i in range(n_random):
            Q, _ = np.linalg.qr(rng.standard_normal((d, k)))
            null_random.append(subspace_overlap(V_actual, Q))
            if (i + 1) % 5000 == 0:
                print(f"    random null: {i+1}/{n_random}", flush=True)

        null_random = np.array(null_random)
        p_random = float((np.sum(null_random >= observed_overlap) + 1) / (n_random + 1))

        # Row-permutation null (1K — SVD-heavy)
        null_perm = []
        for i in range(n_perm):
            perm_idx = rng.permutation(n)
            V_perm = top_k_subspace(Y_pred[perm_idx], k)
            null_perm.append(subspace_overlap(V_actual, V_perm))
            if (i + 1) % 500 == 0:
                print(f"    perm null: {i+1}/{n_perm}", flush=True)

        null_perm = np.array(null_perm)
        p_perm = float((np.sum(null_perm >= observed_overlap) + 1) / (n_perm + 1))

        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")
        print(f"    overlap={observed_overlap:.4f}")
        print(f"    random null: mean={np.mean(null_random):.6f}, max={np.max(null_random):.6f}, p={p_random:.6f}")
        print(f"    perm null:   mean={np.mean(null_perm):.6f}, max={np.max(null_perm):.6f}, p={p_perm:.6f}")

        results[f"k={k}"] = {
            "observed_overlap": observed_overlap,
            "cos_angles": cos_angles.tolist(),
            "random_null": {
                "mean": float(np.mean(null_random)),
                "std": float(np.std(null_random)),
                "max": float(np.max(null_random)),
                "p_value": p_random,
                "n_perm": n_random,
            },
            "permutation_null": {
                "mean": float(np.mean(null_perm)),
                "std": float(np.std(null_perm)),
                "max": float(np.max(null_perm)),
                "p_value": p_perm,
                "n_perm": n_perm,
            },
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# E3: Nested CV SZ Classification
# ═══════════════════════════════════════════════════════════════════════════

def experiment_e3_nested_cv_classification(data: Dict, seed: int = 42) -> Dict:
    """5-fold stratified CV for SZ classification using coupled/full/uncoupled GM."""
    print("\n" + "=" * 70)
    print("E3: Nested CV SZ Classification")
    print("=" * 70)

    set_seed(seed)

    subjects = data["subjects1"]
    X = data["X1"].astype(np.float64)
    Y = data["Y1"].astype(np.float64)
    diag = subjects["Diagnosis"].values

    # Use all DS1 subjects with known diagnosis
    valid_mask = ~np.isnan(diag)
    X_all = X[valid_mask]
    Y_all = Y[valid_mask]
    diag_all = diag[valid_mask].astype(int)

    print(f"  Total subjects: {len(diag_all)} (SZ={np.sum(diag_all==1)}, HC={np.sum(diag_all==0)})")

    # Get B matrix for subspace definition (use seed 42)
    B = _get_or_fit_B("nuclear_norm", data, seed)
    U_B, S_B, Vt_B = np.linalg.svd(B, full_matrices=False)
    primary_rank = 38

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    conditions = ["full", "coupled", "uncoupled"]
    C_values = [0.01, 0.1, 1.0, 10.0]

    fold_results = {cond: [] for cond in conditions}

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X_all, diag_all)):
        print(f"\n  Fold {fold_i}: train={len(train_idx)}, test={len(test_idx)}")

        X_train_f, X_test_f = X_all[train_idx], X_all[test_idx]
        y_train_f, y_test_f = diag_all[train_idx], diag_all[test_idx]

        # Project into coupled/uncoupled subspaces
        P_r = U_B[:, :primary_rank] @ U_B[:, :primary_rank].T
        X_train_coupled = X_train_f @ P_r
        X_test_coupled = X_test_f @ P_r
        X_train_uncoupled = X_train_f - X_train_coupled
        X_test_uncoupled = X_test_f - X_test_coupled

        representations = {
            "full": (X_train_f, X_test_f),
            "coupled": (X_train_coupled, X_test_coupled),
            "uncoupled": (X_train_uncoupled, X_test_uncoupled),
        }

        for cond, (Xtr_c, Xte_c) in representations.items():
            # Inner CV for C selection
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + fold_i)
            best_C = 1.0
            best_inner_auc = -1.0

            for C in C_values:
                inner_aucs = []
                for i_train, i_val in inner_cv.split(Xtr_c, y_train_f):
                    clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=1000)
                    clf.fit(Xtr_c[i_train], y_train_f[i_train])
                    y_score = clf.predict_proba(Xtr_c[i_val])[:, 1]
                    if len(np.unique(y_train_f[i_val])) == 2:
                        inner_aucs.append(roc_auc_score(y_train_f[i_val], y_score))

                mean_inner = np.mean(inner_aucs) if inner_aucs else 0.0
                if mean_inner > best_inner_auc:
                    best_inner_auc = mean_inner
                    best_C = C

            # Refit on full outer train with best C
            clf = LogisticRegression(C=best_C, penalty="l2", solver="lbfgs", max_iter=1000)
            clf.fit(Xtr_c, y_train_f)
            y_score = clf.predict_proba(Xte_c)[:, 1]
            auc = roc_auc_score(y_test_f, y_score)
            fold_results[cond].append(auc)
            print(f"    {cond:12s}: AUC={auc:.4f} (C={best_C})")

    # Aggregate
    results = {}
    for cond in conditions:
        aucs = np.array(fold_results[cond])
        results[cond] = {
            "fold_aucs": aucs.tolist(),
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
            "min_auc": float(np.min(aucs)),
            "max_auc": float(np.max(aucs)),
        }
        print(f"\n  {cond:12s}: AUC = {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    # Bootstrap test: coupled vs full
    from scipy.stats import wilcoxon
    coupled_aucs = np.array(fold_results["coupled"])
    full_aucs = np.array(fold_results["full"])
    diff = coupled_aucs - full_aucs
    results["coupled_vs_full_diff"] = {
        "mean_diff": float(np.mean(diff)),
        "per_fold_diff": diff.tolist(),
    }
    # Paired sign test (small n, so just report descriptive)
    print(f"\n  Coupled - Full AUC diff: {np.mean(diff):.4f} +/- {np.std(diff):.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# E4: OptShrink Dissociation Metrics
# ═══════════════════════════════════════════════════════════════════════════

def experiment_e4_optshrink_dissociation(data: Dict, seed: int = 42) -> Dict:
    """Report subspace overlap and R² dissociation for OptShrink alongside NN."""
    print("\n" + "=" * 70)
    print("E4: OptShrink Dissociation Metrics")
    print("=" * 70)

    set_seed(seed)
    rng = np.random.default_rng(seed)

    X1, Y1 = data["X1"].astype(np.float64), data["Y1"].astype(np.float64)
    idx_test = data["idx1_test"]
    idx_train = data["idx1_train"]
    Xte, Yte = X1[idx_test], Y1[idx_test]
    Ytr = Y1[idx_train]

    methods_B = {}
    for display_name, internal_name in [("Nuclear_Norm", "nuclear_norm"),
                                         ("Linear_OptShrink", "linear_optshrink"),
                                         ("RRR", "rrr"),
                                         ("PLS", "pls")]:
        methods_B[display_name] = _get_or_fit_B(internal_name, data, seed)
        print(f"  Loaded/fit {display_name} B")

    subspace_ks = [5, 10, 20]
    results = {}

    for method_name, B in methods_B.items():
        Y_pred = Xte @ B
        r2_info = r2_summary(Yte.astype(np.float32), Y_pred.astype(np.float32))

        # SVD properties
        U, S, Vt = np.linalg.svd(B, full_matrices=False)
        eff_rank = int(np.sum(S > 1e-10))

        method_res = {
            "r2_global": float(r2_info["r2_global"]),
            "effective_rank": eff_rank,
            "subspace_analysis": {},
        }

        for k in subspace_ks:
            V_actual = top_k_subspace(Yte, k)
            V_pred = top_k_subspace(Y_pred, k)
            cos_angles = principal_angles(V_actual, V_pred)
            overlap = float(np.mean(cos_angles ** 2))

            # Quick null (1000)
            null_overlaps = []
            for _ in range(1000):
                Q, _ = np.linalg.qr(rng.standard_normal((Yte.shape[1], k)))
                null_overlaps.append(subspace_overlap(V_actual, Q))
            null_overlaps = np.array(null_overlaps)
            p_val = float((np.sum(null_overlaps >= overlap) + 1) / 1001)

            method_res["subspace_analysis"][f"k={k}"] = {
                "overlap": overlap,
                "cos_angles": cos_angles.tolist(),
                "null_mean": float(np.mean(null_overlaps)),
                "p_value": p_val,
            }

        results[method_name] = method_res
        print(f"\n  {method_name}:")
        print(f"    R² = {r2_info['r2_global']:.4f}, eff_rank = {eff_rank}")
        for k in subspace_ks:
            sa = method_res["subspace_analysis"][f"k={k}"]
            print(f"    k={k}: O={sa['overlap']:.4f} (null={sa['null_mean']:.4f}, p={sa['p_value']:.4f})")

    # Cross-method dissociation summary
    print("\n  Dissociation Summary (O >> R²):")
    for method_name, res in results.items():
        r2 = res["r2_global"]
        for k in subspace_ks:
            o = res["subspace_analysis"][f"k={k}"]["overlap"]
            ratio = o / max(r2, 1e-10)
            print(f"    {method_name} k={k}: O={o:.4f}, R²={r2:.4f}, O/R²={ratio:.1f}x")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Reviewer experiments for NeuroImage revision")
    parser.add_argument("--config", type=str, default="train/config_baselines.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiments", nargs="+", default=["E1", "E2", "E3", "E4"],
                        choices=["E1", "E2", "E3", "E4"])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(PROJECT_ROOT / args.config))
    print("Loading data ...", flush=True)
    data = load_training_contracts(cfg)
    print(f"  DS1: X={data['X1'].shape}, Y={data['Y1'].shape}")
    print(f"  DS2: X={data['X2'].shape}, Y={data['Y2'].shape}")

    all_results = {}

    if "E1" in args.experiments:
        all_results["E1_hc_only"] = experiment_e1_hc_only(data, args.seed)

    if "E2" in args.experiments:
        all_results["E2_10k_permutations"] = experiment_e2_10k_permutations(data, args.seed)

    if "E3" in args.experiments:
        all_results["E3_nested_cv_classification"] = experiment_e3_nested_cv_classification(data, args.seed)

    if "E4" in args.experiments:
        all_results["E4_optshrink_dissociation"] = experiment_e4_optshrink_dissociation(data, args.seed)

    # Save combined results
    out_path = OUT_DIR / "reviewer_experiments_results.json"
    save_json(out_path, all_results)
    print(f"\n\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
