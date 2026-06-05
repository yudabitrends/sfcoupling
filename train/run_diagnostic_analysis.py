#!/usr/bin/env python3
"""
Comprehensive diagnostic analysis: why proposed methods don't beat MLP.

Implements three investigation directions:
  A. Residualization ablation: raw vs residualized signal
  B. Multi-k evaluation:  k=5,7,10,20 for ALL methods
  E. Unified comparison table with MLP baseline + generalization (DS1 vs DS2)

Also produces per-PC R² breakdown and paired t-tests vs MLP.

Usage:
    python train/run_diagnostic_analysis.py \
        --config train/config_baselines.yaml \
        --seeds 42 43 44 45 46 47 48 \
        --pca_ks 5 7 10 20 \
        --max_rank 30 \
        --n_boot 10000 \
        --out_dir results/diagnostic_analysis
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.baselines import MLPRegressorTorch, fit_ridge_grid, train_mlp_regressor
from models.metrics import fit_pca_on_train, pc_space_r2_from_pca, r2_summary
from models.utils import load_config, load_training_contracts, save_json, set_seed
from train.run_multivariate_methods import fit_nuclear_norm, fit_pls, fit_rrr
from train.statistical_analysis import bootstrap_bca_ci, paired_t_test

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
# Direction A: Residualization ablation
# ---------------------------------------------------------------------------

def run_signal_check(
    X_raw: np.ndarray,
    Y_raw: np.ndarray,
    X_resid: np.ndarray,
    Y_resid: np.ndarray,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    pca_ks: List[int],
    alpha: float = 1.0,
    seed: int = 42,
) -> Dict:
    """Compare Ridge R² before and after residualization at multiple k."""
    results = {}
    for k in pca_ks:
        # Raw (pre-residualization)
        pca_raw = PCA(
            n_components=min(k, Y_raw.shape[1], len(idx_train)),
            random_state=seed,
        )
        pc_train_raw = pca_raw.fit_transform(Y_raw[idx_train])
        pc_test_raw = pca_raw.transform(Y_raw[idx_test])
        ridge_raw = Ridge(alpha=alpha, random_state=seed)
        ridge_raw.fit(X_raw[idx_train], pc_train_raw)
        pc_pred_raw = ridge_raw.predict(X_raw[idx_test])
        per_pc_raw = r2_score(pc_test_raw, pc_pred_raw, multioutput="raw_values")
        per_pc_raw = np.where(np.isfinite(per_pc_raw), per_pc_raw, 0.0)

        # Residualized
        pca_res = PCA(
            n_components=min(k, Y_resid.shape[1], len(idx_train)),
            random_state=seed,
        )
        pc_train_res = pca_res.fit_transform(Y_resid[idx_train])
        pc_test_res = pca_res.transform(Y_resid[idx_test])
        ridge_res = Ridge(alpha=alpha, random_state=seed)
        ridge_res.fit(X_resid[idx_train], pc_train_res)
        pc_pred_res = ridge_res.predict(X_resid[idx_test])
        per_pc_res = r2_score(pc_test_res, pc_pred_res, multioutput="raw_values")
        per_pc_res = np.where(np.isfinite(per_pc_res), per_pc_res, 0.0)

        results[f"k{k}"] = {
            "raw": {
                "pc_r2_mean": float(np.mean(per_pc_raw)),
                "pc_r2_per_pc": [float(x) for x in per_pc_raw],
                "explained_var_ratio_sum": float(np.sum(pca_raw.explained_variance_ratio_)),
            },
            "residualized": {
                "pc_r2_mean": float(np.mean(per_pc_res)),
                "pc_r2_per_pc": [float(x) for x in per_pc_res],
                "explained_var_ratio_sum": float(np.sum(pca_res.explained_variance_ratio_)),
            },
            "signal_loss": float(np.mean(per_pc_raw) - np.mean(per_pc_res)),
        }
    return results


# ---------------------------------------------------------------------------
# Direction B+E: Run all methods and evaluate at multiple k
# ---------------------------------------------------------------------------

def run_all_methods_single_seed(
    cfg: Dict,
    seed: int,
    pca_ks: List[int],
    max_rank: int = 30,
    mlp_epochs: int = 200,
    mlp_patience: int = 20,
) -> Dict:
    """Run Ridge, MLP, RRR, PLS, Nuclear Norm for a single seed.

    Returns per-method metrics at all pca_ks on both DS1 and DS2.
    """
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

    results = {"seed": seed}

    def _evaluate_predictions(Y_pred_ds1, Y_pred_ds2, method_name):
        """Evaluate predictions at all pca_ks."""
        method_res = {"dataset1_test": {}, "dataset2_external": {}}
        for k in pca_ks:
            pca = fit_pca_on_train(Ytr, k=k, seed=seed)
            pc_ds1 = pc_space_r2_from_pca(Yte, Y_pred_ds1, pca)
            pc_ds2 = pc_space_r2_from_pca(Yext, Y_pred_ds2, pca)
            method_res["dataset1_test"][f"k{k}"] = pc_ds1
            method_res["dataset2_external"][f"k{k}"] = pc_ds2
        # Edge-space R²
        method_res["dataset1_test"]["edge_r2"] = r2_summary(Yte, Y_pred_ds1)
        method_res["dataset2_external"]["edge_r2"] = r2_summary(Yext, Y_pred_ds2)
        return method_res

    # --- 1. Ridge ---
    print(f"  [seed {seed}] Ridge ...", flush=True)
    ridge_model, ridge_info = fit_ridge_grid(Xtr64, Ytr64, Xva64, Yva64, ridge_alphas)
    Y_pred_ds1_ridge = ridge_model.predict(Xte64).astype(np.float32)
    Y_pred_ds2_ridge = ridge_model.predict(Xext64).astype(np.float32)
    results["ridge"] = _evaluate_predictions(Y_pred_ds1_ridge, Y_pred_ds2_ridge, "ridge")
    results["ridge"]["best_alpha"] = ridge_info["best_alpha"]

    # --- 2. MLP ---
    print(f"  [seed {seed}] MLP ...", flush=True)
    mlp_cfg = cfg.get("mlp", {})
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_cuda", False) else "cpu")
    model = MLPRegressorTorch(
        in_dim=data["dx"],
        out_dim=data["dy"],
        hidden_dims=mlp_cfg.get("hidden_dims", [256, 128]),
        dropout=float(mlp_cfg.get("dropout", 0.1)),
        activation=str(mlp_cfg.get("activation", "relu")),
    ).to(device)
    train_info = train_mlp_regressor(
        model=model, X_train=Xtr, Y_train=Ytr, X_val=Xva, Y_val=Yva,
        epochs=mlp_epochs,
        batch_size=int(mlp_cfg.get("batch_size", 64)),
        lr=float(mlp_cfg.get("lr", 1e-3)),
        weight_decay=float(mlp_cfg.get("weight_decay", 1e-5)),
        patience=mlp_patience, seed=seed, device=device,
    )
    model.eval()
    with torch.no_grad():
        Y_pred_ds1_mlp = model(torch.tensor(Xte, dtype=torch.float32, device=device)).cpu().numpy()
        Y_pred_ds2_mlp = model(torch.tensor(Xext, dtype=torch.float32, device=device)).cpu().numpy()
    results["mlp"] = _evaluate_predictions(Y_pred_ds1_mlp, Y_pred_ds2_mlp, "mlp")
    results["mlp"]["best_val_loss"] = train_info["best_val_loss"]
    results["mlp"]["final_train_loss"] = train_info["logs"][-1]["train_loss"] if train_info["logs"] else None
    results["mlp"]["n_epochs_trained"] = len(train_info["logs"])

    # --- 3. RRR ---
    print(f"  [seed {seed}] RRR ...", flush=True)
    rrr = fit_rrr(Xtr64, Ytr64, Xva64, Yva64, max_rank=max_rank, ridge_alphas=ridge_alphas)
    Y_pred_ds1_rrr = (Xte64 @ rrr["B"]).astype(np.float32)
    Y_pred_ds2_rrr = (Xext64 @ rrr["B"]).astype(np.float32)
    results["rrr"] = _evaluate_predictions(Y_pred_ds1_rrr, Y_pred_ds2_rrr, "rrr")
    results["rrr"]["optimal_rank"] = rrr["optimal_rank"]

    # --- 4. PLS ---
    print(f"  [seed {seed}] PLS ...", flush=True)
    pls = fit_pls(Xtr64, Ytr64, Xva64, Yva64, max_components=max_rank)
    Y_pred_ds1_pls = (Xte64 @ pls["B"]).astype(np.float32)
    Y_pred_ds2_pls = (Xext64 @ pls["B"]).astype(np.float32)
    results["pls"] = _evaluate_predictions(Y_pred_ds1_pls, Y_pred_ds2_pls, "pls")
    results["pls"]["optimal_n"] = pls["optimal_n"]

    # --- 5. Nuclear Norm ---
    print(f"  [seed {seed}] Nuclear Norm ...", flush=True)
    nn = fit_nuclear_norm(Xtr64, Ytr64, Xva64, Yva64)
    Y_pred_ds1_nn = (Xte64 @ nn["B"]).astype(np.float32)
    Y_pred_ds2_nn = (Xext64 @ nn["B"]).astype(np.float32)
    results["nuclear_norm"] = _evaluate_predictions(Y_pred_ds1_nn, Y_pred_ds2_nn, "nuclear_norm")
    results["nuclear_norm"]["optimal_lambda"] = nn["optimal_lambda"]

    return results


# ---------------------------------------------------------------------------
# Aggregation with paired comparisons vs MLP
# ---------------------------------------------------------------------------

def aggregate_all_methods(
    all_results: List[Dict],
    pca_ks: List[int],
    n_boot: int = 10000,
    boot_seed: int = 42,
) -> Dict:
    """Aggregate per-seed results, compute paired t-tests vs MLP at each k."""
    methods = ["ridge", "mlp", "rrr", "pls", "nuclear_norm"]
    seeds = [r["seed"] for r in all_results]
    summary = {"n_seeds": len(seeds), "seeds": seeds}

    # Collect per-seed metrics by method and k
    per_seed_data = {}  # method -> k -> dataset -> [values]
    for method in methods:
        per_seed_data[method] = {}
        for k in pca_ks:
            kstr = f"k{k}"
            d1_vals = []
            d2_vals = []
            d1_per_pc = []
            for r in all_results:
                d1 = r[method]["dataset1_test"][kstr]
                d2 = r[method]["dataset2_external"][kstr]
                d1_vals.append(d1["pc_r2_mean"])
                d2_vals.append(d2["pc_r2_mean"])
                d1_per_pc.append(d1["pc_r2_all"])
            per_seed_data[method][kstr] = {
                "d1": np.array(d1_vals),
                "d2": np.array(d2_vals),
                "d1_per_pc": d1_per_pc,
            }

    # Summary per method per k
    for method in methods:
        summary[method] = {}
        for k in pca_ks:
            kstr = f"k{k}"
            d1 = per_seed_data[method][kstr]["d1"]
            d2 = per_seed_data[method][kstr]["d2"]
            summary[method][kstr] = {
                "ds1_pc_r2": _stats(d1),
                "ds2_pc_r2": _stats(d2),
                "ds1_minus_ds2": _stats(d1 - d2),
            }
            # Bootstrap CI
            if len(d1) > 1 and n_boot > 0:
                summary[method][kstr]["ds1_bootstrap_ci"] = bootstrap_bca_ci(
                    d1, n_boot=n_boot, seed=boot_seed)

        # Edge R² (not k-dependent)
        edge_d1 = [r[method]["dataset1_test"]["edge_r2"]["r2_edge_mean"]
                    for r in all_results]
        edge_d2 = [r[method]["dataset2_external"]["edge_r2"]["r2_edge_mean"]
                    for r in all_results]
        summary[method]["edge_r2_d1"] = _stats(edge_d1)
        summary[method]["edge_r2_d2"] = _stats(edge_d2)

    # Paired t-tests: each method vs MLP at each k
    summary["paired_vs_mlp"] = {}
    for method in ["ridge", "rrr", "pls", "nuclear_norm"]:
        summary["paired_vs_mlp"][method] = {}
        for k in pca_ks:
            kstr = f"k{k}"
            mlp_vals = per_seed_data["mlp"][kstr]["d1"]
            method_vals = per_seed_data[method][kstr]["d1"]
            test = paired_t_test(method_vals, mlp_vals)
            summary["paired_vs_mlp"][method][kstr] = test
        # DS2 at primary k=20
        mlp_d2 = per_seed_data["mlp"]["k20"]["d2"]
        method_d2 = per_seed_data[method]["k20"]["d2"]
        summary["paired_vs_mlp"][method]["k20_ds2"] = paired_t_test(method_d2, mlp_d2)

    # Per-PC R² analysis at k=20 (mean across seeds)
    summary["per_pc_analysis"] = {}
    for method in methods:
        all_per_pc = per_seed_data[method]["k20"]["d1_per_pc"]
        if all_per_pc and all(len(pc) == len(all_per_pc[0]) for pc in all_per_pc):
            arr = np.array(all_per_pc)  # (n_seeds, n_pcs)
            mean_per_pc = np.mean(arr, axis=0).tolist()
            std_per_pc = np.std(arr, axis=0, ddof=1).tolist() if arr.shape[0] > 1 else [0.0] * arr.shape[1]
            summary["per_pc_analysis"][method] = {
                "mean_per_pc_r2": [float(x) for x in mean_per_pc],
                "std_per_pc_r2": [float(x) for x in std_per_pc],
                "n_positive_pcs": int(np.sum(np.array(mean_per_pc) > 0)),
                "n_significant_pcs": int(np.sum(np.array(mean_per_pc) > 0.01)),
            }

    return summary


def print_unified_table(summary: Dict, pca_ks: List[int]) -> None:
    """Print unified comparison table for all methods at all k values."""
    methods = ["ridge", "mlp", "rrr", "pls", "nuclear_norm"]
    method_labels = {
        "ridge": "Ridge",
        "mlp": "MLP (baseline)",
        "rrr": "RRR",
        "pls": "PLS",
        "nuclear_norm": "Nuclear Norm",
    }

    print("\n" + "=" * 100)
    print("UNIFIED METHOD COMPARISON (pc_r2_mean)")
    print("=" * 100)

    # Header
    header = f"{'Method':<20s}"
    for k in pca_ks:
        header += f" {'DS1 k=' + str(k):>14s}"
    header += f" {'DS2 k=20':>14s} {'Edge R2 DS1':>14s}"
    print(header)
    print("-" * 100)

    for method in methods:
        s = summary[method]
        row = f"{method_labels[method]:<20s}"
        for k in pca_ks:
            kstr = f"k{k}"
            d1 = s[kstr]["ds1_pc_r2"]
            if np.isnan(d1.get("ci95", float("nan"))):
                row += f" {d1['mean']:>14.4f}"
            else:
                row += f" {d1['mean']:.4f}+/-{d1['ci95']:.4f}"
        d2 = s["k20"]["ds2_pc_r2"]
        if np.isnan(d2.get("ci95", float("nan"))):
            row += f" {d2['mean']:>14.4f}"
        else:
            row += f" {d2['mean']:.4f}+/-{d2['ci95']:.4f}"
        e1 = s["edge_r2_d1"]
        if np.isnan(e1.get("ci95", float("nan"))):
            row += f" {e1['mean']:>14.4f}"
        else:
            row += f" {e1['mean']:.4f}+/-{e1['ci95']:.4f}"
        print(row)

    print("=" * 100)

    # Paired t-tests vs MLP
    print("\nPaired t-tests vs MLP (DS1):")
    print(f"{'Method':<20s}", end="")
    for k in pca_ks:
        print(f" {'k=' + str(k) + ' p-val':>14s}", end="")
    print(f" {'k20 DS2 p-val':>14s}")
    print("-" * 80)

    for method in ["ridge", "rrr", "pls", "nuclear_norm"]:
        row = f"{method_labels[method]:<20s}"
        for k in pca_ks:
            kstr = f"k{k}"
            t = summary["paired_vs_mlp"][method][kstr]
            p = t["p_value"]
            diff = t["mean_diff"]
            sig = "*" if p < 0.05 else " "
            row += f" {diff:+.4f} p={p:.3f}{sig}"
        t_ds2 = summary["paired_vs_mlp"][method]["k20_ds2"]
        p = t_ds2["p_value"]
        diff = t_ds2["mean_diff"]
        sig = "*" if p < 0.05 else " "
        row += f" {diff:+.4f} p={p:.3f}{sig}"
        print(row)

    print()

    # Per-PC R² breakdown
    if "per_pc_analysis" in summary:
        print("Per-PC R² analysis (k=20, DS1, mean across seeds):")
        print(f"{'Method':<20s} {'#PC>0':>6s} {'#PC>0.01':>8s}  PC1    PC2    PC3    PC4    PC5")
        print("-" * 85)
        for method in methods:
            if method in summary["per_pc_analysis"]:
                pa = summary["per_pc_analysis"][method]
                pcs = pa["mean_per_pc_r2"][:5]
                row = (f"{method_labels[method]:<20s} "
                       f"{pa['n_positive_pcs']:>6d} "
                       f"{pa['n_significant_pcs']:>8d}  ")
                row += "  ".join(f"{v:.3f}" for v in pcs)
                print(row)
        print()

    # Generalization gap analysis
    print("Generalization gap (DS1 - DS2 at k=20):")
    for method in methods:
        s = summary[method]["k20"]
        gap = s["ds1_minus_ds2"]
        d1 = s["ds1_pc_r2"]["mean"]
        d2 = s["ds2_pc_r2"]["mean"]
        gen_ratio = d2 / d1 if d1 > 0 else float("nan")
        print(f"  {method_labels[method]:<20s}: DS1={d1:.4f}  DS2={d2:.4f}  "
              f"gap={gap['mean']:+.4f}  DS2/DS1={gen_ratio:.2f}")


def print_signal_check(signal_results: Dict) -> None:
    """Print Direction A signal check results."""
    print("\n" + "=" * 80)
    print("DIRECTION A: RESIDUALIZATION SIGNAL CHECK")
    print("=" * 80)
    print(f"{'k':<6s} {'Raw pc_r2':>12s} {'Resid pc_r2':>12s} {'Signal loss':>12s} {'Raw var%':>10s} {'Resid var%':>10s}")
    print("-" * 70)
    for kstr in sorted(signal_results.keys(), key=lambda x: int(x[1:])):
        k = int(kstr[1:])
        r = signal_results[kstr]
        print(f"k={k:<4d} {r['raw']['pc_r2_mean']:>12.4f} "
              f"{r['residualized']['pc_r2_mean']:>12.4f} "
              f"{r['signal_loss']:>12.4f} "
              f"{r['raw']['explained_var_ratio_sum']:>9.1%} "
              f"{r['residualized']['explained_var_ratio_sum']:>9.1%}")
    print()

    # Per-PC comparison at k=20
    if "k20" in signal_results:
        print("Per-PC R² at k=20 (raw vs residualized):")
        raw_pcs = signal_results["k20"]["raw"]["pc_r2_per_pc"]
        res_pcs = signal_results["k20"]["residualized"]["pc_r2_per_pc"]
        n_show = min(len(raw_pcs), len(res_pcs), 10)
        print(f"  {'PC':<5s} {'Raw':>8s} {'Resid':>8s} {'Loss':>8s}")
        for i in range(n_show):
            print(f"  PC{i+1:<3d} {raw_pcs[i]:>8.4f} {res_pcs[i]:>8.4f} "
                  f"{raw_pcs[i] - res_pcs[i]:>+8.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive diagnostic: why proposed methods don't beat MLP",
    )
    parser.add_argument("--config", type=str, default="train/config_baselines.yaml")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 43, 44, 45, 46, 47, 48])
    parser.add_argument("--pca_ks", type=int, nargs="+", default=[5, 7, 10, 20])
    parser.add_argument("--max_rank", type=int, default=30)
    parser.add_argument("--mlp_epochs", type=int, default=200)
    parser.add_argument("--mlp_patience", type=int, default=20)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--out_dir", type=str,
                        default="results/diagnostic_analysis")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMPREHENSIVE DIAGNOSTIC ANALYSIS")
    print(f"  Seeds:   {args.seeds}")
    print(f"  PCA ks:  {args.pca_ks}")
    print(f"  MLP:     {args.mlp_epochs} epochs, patience={args.mlp_patience}")
    print(f"  Output:  {out_dir}")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Direction A: Residualization signal check
    # -----------------------------------------------------------------------
    print("\n[Direction A] Signal check: raw vs residualized ...")
    data = load_training_contracts(cfg)
    base_dir = Path(cfg["paths"]["aligned_features_dir"])
    raw_x_path = base_dir / "dataset1_X_raw.npy"
    raw_y_path = base_dir / "dataset1_Y_raw.npy"

    signal_results = None
    if raw_x_path.exists() and raw_y_path.exists():
        X_raw = np.load(raw_x_path)
        Y_raw = np.load(raw_y_path)
        X_resid = data["X1"]
        Y_resid = data["Y1"]
        signal_results = run_signal_check(
            X_raw, Y_raw, X_resid, Y_resid,
            data["idx1_train"], data["idx1_test"],
            pca_ks=args.pca_ks,
        )
        save_json(out_dir / "signal_check.json", signal_results)
        print_signal_check(signal_results)
    else:
        print("  WARNING: Raw arrays not found. Skipping signal check.")
        print(f"  Expected: {raw_x_path}, {raw_y_path}")

    # -----------------------------------------------------------------------
    # Direction B+E: Run all methods across seeds
    # -----------------------------------------------------------------------
    print("\n[Direction B+E] Running all methods across seeds ...")
    all_results = []
    for i, seed in enumerate(args.seeds):
        t0 = time.time()
        print(f"\n[{i+1}/{len(args.seeds)}] Seed={seed}", flush=True)
        result = run_all_methods_single_seed(
            cfg, seed=seed, pca_ks=args.pca_ks,
            max_rank=args.max_rank,
            mlp_epochs=args.mlp_epochs,
            mlp_patience=args.mlp_patience,
        )
        elapsed = time.time() - t0

        # Quick printout at k=20
        k20_str = "k20" if 20 in args.pca_ks else f"k{args.pca_ks[-1]}"
        methods_brief = ["ridge", "mlp", "rrr", "pls", "nuclear_norm"]
        vals = []
        for m in methods_brief:
            v = result[m]["dataset1_test"][k20_str]["pc_r2_mean"]
            vals.append(f"{m}={v:.4f}")
        print(f"  {', '.join(vals)}  [{elapsed:.1f}s]", flush=True)

        all_results.append(result)
        save_json(out_dir / f"seed_{seed}.json", result)

    # Aggregate
    summary = aggregate_all_methods(
        all_results, pca_ks=args.pca_ks,
        n_boot=args.n_boot, boot_seed=42,
    )
    save_json(out_dir / "summary.json", summary)

    # Print unified table
    print_unified_table(summary, args.pca_ks)

    # -----------------------------------------------------------------------
    # Key findings summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Finding 1: Signal ceiling
    if signal_results and "k5" in signal_results:
        loss_k5 = signal_results["k5"]["signal_loss"]
        loss_k20 = signal_results.get("k20", {}).get("signal_loss", float("nan"))
        print(f"\n1. SIGNAL CEILING (Direction A):")
        print(f"   Residualization removes {loss_k5:.4f} R² at k=5, "
              f"{loss_k20:.4f} R² at k=20")
        raw_k5 = signal_results["k5"]["raw"]["pc_r2_mean"]
        res_k5 = signal_results["k5"]["residualized"]["pc_r2_mean"]
        print(f"   Raw signal: {raw_k5:.4f}  |  After residualization: {res_k5:.4f}")

    # Finding 2: k-dependent comparison
    print(f"\n2. K-DEPENDENT COMPARISON (Direction B):")
    best_k_for_nn = None
    best_diff = -np.inf
    for k in args.pca_ks:
        kstr = f"k{k}"
        if kstr in summary.get("nuclear_norm", {}) and kstr in summary.get("mlp", {}):
            nn_mean = summary["nuclear_norm"][kstr]["ds1_pc_r2"]["mean"]
            mlp_mean = summary["mlp"][kstr]["ds1_pc_r2"]["mean"]
            diff = nn_mean - mlp_mean
            print(f"   k={k:>2d}: NN={nn_mean:.4f}  MLP={mlp_mean:.4f}  "
                  f"diff={diff:+.4f}  {'NN wins' if diff > 0 else 'MLP wins'}")
            if diff > best_diff:
                best_diff = diff
                best_k_for_nn = k
    if best_k_for_nn is not None:
        print(f"   -> Best k for Nuclear Norm: k={best_k_for_nn} (diff={best_diff:+.4f})")

    # Finding 3: Generalization
    print(f"\n3. GENERALIZATION (Direction E):")
    for method in ["mlp", "nuclear_norm"]:
        d1 = summary[method]["k20"]["ds1_pc_r2"]["mean"]
        d2 = summary[method]["k20"]["ds2_pc_r2"]["mean"]
        label = "MLP" if method == "mlp" else "Nuclear Norm"
        print(f"   {label:<15s}: DS1={d1:.4f}  DS2={d2:.4f}  "
              f"retention={d2/d1:.1%}" if d1 > 0 else f"   {label}: DS1={d1:.4f}  DS2={d2:.4f}")

    # Finding 4: Effective signal dimensionality
    if "per_pc_analysis" in summary:
        print(f"\n4. EFFECTIVE SIGNAL DIMENSIONALITY:")
        for method in ["mlp", "nuclear_norm"]:
            if method in summary["per_pc_analysis"]:
                pa = summary["per_pc_analysis"][method]
                label = "MLP" if method == "mlp" else "Nuclear Norm"
                print(f"   {label:<15s}: {pa['n_positive_pcs']} PCs with R²>0, "
                      f"{pa['n_significant_pcs']} PCs with R²>0.01")

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
