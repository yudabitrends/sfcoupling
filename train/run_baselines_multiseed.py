#!/usr/bin/env python3
"""
Multi-seed baseline training for Ridge and MLP regressors.

Runs Ridge (with PCA-k grid) and MLP baselines across multiple seeds,
producing aggregated summary statistics .

Usage:
    python train/run_baselines_multiseed.py \
        --config train/config_baselines.yaml \
        --seeds 42 43 44 45 46 \
        --pca_ks 5 10 20 50 \
        --mlp_epochs 100 --mlp_patience 15
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.baselines import fit_ridge_grid, MLPRegressorTorch, train_mlp_regressor
from models.metrics import fit_pca_on_train, pc_space_r2_from_pca, r2_summary
from models.utils import load_config, load_training_contracts, save_json, set_seed

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


def run_single_seed(cfg: Dict, seed: int, pca_ks: List[int],
                    mlp_epochs: int, mlp_patience: int) -> Dict:
    """Run Ridge (multi pca_k) and MLP for a single seed."""
    import torch
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

    results = {"seed": seed, "ridge": {}, "mlp": {}}

    # Ridge with PCA-k grid
    ridge_cfg = cfg.get("ridge", {})
    alphas = ridge_cfg.get("alphas", [1e-3, 1e-2, 1e-1, 1, 10, 100])

    for k in pca_ks:
        label = f"pca_k{k}" if k > 0 else "no_pca"
        if k > 0:
            pca = fit_pca_on_train(Ytr, k=k, seed=seed)
            Ytr_t = pca.transform(Ytr)
            Yva_t = pca.transform(Yva)
        else:
            pca = None
            Ytr_t, Yva_t = Ytr, Yva

        ridge_model, ridge_info = fit_ridge_grid(Xtr, Ytr_t, Xva, Yva_t, alphas)
        yte_pred = ridge_model.predict(Xte)
        yext_pred = ridge_model.predict(Xext)

        if pca is not None:
            yte_full = pca.inverse_transform(yte_pred)
            yext_full = pca.inverse_transform(yext_pred)
            eval_pca = fit_pca_on_train(Ytr, k=20, seed=seed)
            ds1 = {
                "edge_r2": r2_summary(Yte, yte_full),
                "pc_r2": pc_space_r2_from_pca(Yte, yte_full, eval_pca),
            }
            ds2 = {
                "edge_r2": r2_summary(Yext, yext_full),
                "pc_r2": pc_space_r2_from_pca(Yext, yext_full, eval_pca),
            }
        else:
            eval_pca = fit_pca_on_train(Ytr, k=20, seed=seed)
            ds1 = {
                "edge_r2": r2_summary(Yte, yte_pred),
                "pc_r2": pc_space_r2_from_pca(Yte, yte_pred, eval_pca),
            }
            ds2 = {
                "edge_r2": r2_summary(Yext, yext_pred),
                "pc_r2": pc_space_r2_from_pca(Yext, yext_pred, eval_pca),
            }
        results["ridge"][label] = {
            "best_alpha": ridge_info["best_alpha"],
            "dataset1_test": ds1,
            "dataset2_external": ds2,
        }

    # MLP baseline
    mlp_cfg = cfg.get("mlp", {})
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_cuda", False) else "cpu")
    hidden_dims = mlp_cfg.get("hidden_dims", [256, 128])
    model = MLPRegressorTorch(
        in_dim=data["dx"],
        out_dim=data["dy"],
        hidden_dims=hidden_dims,
        dropout=float(mlp_cfg.get("dropout", 0.1)),
        activation=str(mlp_cfg.get("activation", "relu")),
    ).to(device)
    train_mlp_regressor(
        model=model, X_train=Xtr, Y_train=Ytr, X_val=Xva, Y_val=Yva,
        epochs=mlp_epochs,
        batch_size=int(mlp_cfg.get("batch_size", 64)),
        lr=float(mlp_cfg.get("lr", 1e-3)),
        weight_decay=float(mlp_cfg.get("weight_decay", 1e-5)),
        patience=mlp_patience, seed=seed, device=device,
    )
    model.eval()
    with torch.no_grad():
        yte_pred_m = model(torch.tensor(Xte, dtype=torch.float32, device=device)).cpu().numpy()
        yext_pred_m = model(torch.tensor(Xext, dtype=torch.float32, device=device)).cpu().numpy()
    eval_pca = fit_pca_on_train(Ytr, k=20, seed=seed)
    results["mlp"] = {
        "dataset1_test": {
            "edge_r2": r2_summary(Yte, yte_pred_m),
            "pc_r2": pc_space_r2_from_pca(Yte, yte_pred_m, eval_pca),
        },
        "dataset2_external": {
            "edge_r2": r2_summary(Yext, yext_pred_m),
            "pc_r2": pc_space_r2_from_pca(Yext, yext_pred_m, eval_pca),
        },
    }
    return results


def aggregate_results(all_results: List[Dict]) -> Dict:
    """Aggregate per-seed results into summary statistics."""
    summary = {"n_seeds": len(all_results), "seeds": [r["seed"] for r in all_results]}

    # Ridge per pca_k
    ridge_keys = set()
    for r in all_results:
        ridge_keys.update(r["ridge"].keys())

    summary["ridge"] = {}
    for rk in sorted(ridge_keys):
        vals_d1 = [r["ridge"][rk]["dataset1_test"]["pc_r2"]["pc_r2_mean"]
                    for r in all_results if rk in r["ridge"]]
        vals_d2 = [r["ridge"][rk]["dataset2_external"]["pc_r2"]["pc_r2_mean"]
                    for r in all_results if rk in r["ridge"]]
        edge_d1 = [r["ridge"][rk]["dataset1_test"]["edge_r2"]["r2_edge_mean"]
                    for r in all_results if rk in r["ridge"]]
        edge_d2 = [r["ridge"][rk]["dataset2_external"]["edge_r2"]["r2_edge_mean"]
                    for r in all_results if rk in r["ridge"]]
        summary["ridge"][rk] = {
            "pc_r2_mean_d1": _stats(vals_d1),
            "pc_r2_mean_d2": _stats(vals_d2),
            "edge_r2_mean_d1": _stats(edge_d1),
            "edge_r2_mean_d2": _stats(edge_d2),
        }

    # MLP
    mlp_d1 = [r["mlp"]["dataset1_test"]["pc_r2"]["pc_r2_mean"] for r in all_results]
    mlp_d2 = [r["mlp"]["dataset2_external"]["pc_r2"]["pc_r2_mean"] for r in all_results]
    mlp_edge_d1 = [r["mlp"]["dataset1_test"]["edge_r2"]["r2_edge_mean"] for r in all_results]
    mlp_edge_d2 = [r["mlp"]["dataset2_external"]["edge_r2"]["r2_edge_mean"] for r in all_results]
    summary["mlp"] = {
        "pc_r2_mean_d1": _stats(mlp_d1),
        "pc_r2_mean_d2": _stats(mlp_d2),
        "edge_r2_mean_d1": _stats(mlp_edge_d1),
        "edge_r2_mean_d2": _stats(mlp_edge_d2),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-seed baseline training")
    parser.add_argument("--config", type=str, default="train/config_baselines.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--pca_ks", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--mlp_epochs", type=int, default=100)
    parser.add_argument("--mlp_patience", type=int, default=15)
    parser.add_argument("--out_dir", type=str, default="results/baselines_multiseed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for i, seed in enumerate(args.seeds):
        t0 = time.time()
        print(f"\n[{i+1}/{len(args.seeds)}] Seed={seed}", flush=True)
        result = run_single_seed(
            cfg, seed=seed, pca_ks=args.pca_ks,
            mlp_epochs=args.mlp_epochs, mlp_patience=args.mlp_patience,
        )
        elapsed = time.time() - t0
        # Quick printout
        best_ridge_key = max(
            result["ridge"].keys(),
            key=lambda k: result["ridge"][k]["dataset1_test"]["pc_r2"]["pc_r2_mean"],
        )
        best_ridge = result["ridge"][best_ridge_key]["dataset1_test"]["pc_r2"]["pc_r2_mean"]
        mlp_val = result["mlp"]["dataset1_test"]["pc_r2"]["pc_r2_mean"]
        print(f"  Ridge best ({best_ridge_key}): pc_r2_mean={best_ridge:.4f}  "
              f"MLP: pc_r2_mean={mlp_val:.4f}  [{elapsed:.1f}s]", flush=True)
        all_results.append(result)
        save_json(out_dir / f"seed_{seed}.json", result)

    summary = aggregate_results(all_results)
    save_json(out_dir / "summary.json", summary)
    print(f"\nSummary saved to {out_dir / 'summary.json'}")

    # Print summary table
    print("\n" + "=" * 70)
    print("BASELINE SUMMARY (DS1 test pc_r2_mean)")
    print("=" * 70)
    for rk, rv in summary["ridge"].items():
        s = rv["pc_r2_mean_d1"]
        print(f"  Ridge {rk:>12s}: {s['mean']:.4f} +/- {s['ci95']:.4f}")
    s = summary["mlp"]["pc_r2_mean_d1"]
    print(f"  MLP             : {s['mean']:.4f} +/- {s['ci95']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
