#!/usr/bin/env python3
"""
Diagnostic script: compare Ridge R² (PCA space) before and after residualization.
Detects whether covariate residualization removes useful GM->FNC signal.

Usage:
    python train/check_signal.py --config preprocess/config_converted.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.utils import load_config, load_training_contracts, set_seed


def ridge_pca_r2(X_train, Y_train, X_test, Y_test, pca_k=20, alpha=1.0, seed=42):
    """Fit Ridge in PCA space and return per-PC and mean R²."""
    pca = PCA(n_components=min(pca_k, Y_train.shape[1], Y_train.shape[0]), random_state=seed)
    pc_train = pca.fit_transform(Y_train)
    pc_test = pca.transform(Y_test)

    ridge = Ridge(alpha=alpha, random_state=seed)
    ridge.fit(X_train, pc_train)
    pc_pred = ridge.predict(X_test)

    per_pc = r2_score(pc_test, pc_pred, multioutput="raw_values")
    per_pc = np.where(np.isfinite(per_pc), per_pc, 0.0)
    return {
        "pc_r2_mean": float(np.mean(per_pc)),
        "pc_r2_median": float(np.median(per_pc)),
        "pc_r2_first5": [float(x) for x in per_pc[:5]],
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }


def main():
    parser = argparse.ArgumentParser(description="Check GM->FNC signal before/after residualization")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--pca_k", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    data = load_training_contracts(cfg)

    idx_tr = data["idx1_train"]
    idx_te = data["idx1_test"]

    X_tr = data["X1"][idx_tr]
    Y_tr = data["Y1"][idx_tr]
    X_te = data["X1"][idx_te]
    Y_te = data["Y1"][idx_te]

    result_post = ridge_pca_r2(X_tr, Y_tr, X_te, Y_te, pca_k=args.pca_k, alpha=args.alpha, seed=args.seed)

    # Check for raw (pre-residualized) arrays
    base_dir = Path(cfg["paths"]["aligned_features_dir"])
    raw_x_path = base_dir / "dataset1_X_raw.npy"
    raw_y_path = base_dir / "dataset1_Y_raw.npy"

    result = {"post_residualization": result_post}

    if raw_x_path.exists() and raw_y_path.exists():
        X_raw = np.load(raw_x_path)
        Y_raw = np.load(raw_y_path)
        X_raw_tr = X_raw[idx_tr]
        Y_raw_tr = Y_raw[idx_tr]
        X_raw_te = X_raw[idx_te]
        Y_raw_te = Y_raw[idx_te]
        result_pre = ridge_pca_r2(X_raw_tr, Y_raw_tr, X_raw_te, Y_raw_te, pca_k=args.pca_k, alpha=args.alpha, seed=args.seed)
        result["pre_residualization"] = result_pre
        delta = result_pre["pc_r2_mean"] - result_post["pc_r2_mean"]
        result["signal_loss_from_residualization"] = float(delta)
        if delta > 0.02:
            result["warning"] = (
                f"Residualization removed {delta:.4f} R² signal. "
                "Consider whether covariates confound GM->FNC relationship."
            )
    else:
        result["note"] = (
            "Raw arrays not found. Re-run preprocessing with save_intermediate_arrays: true "
            "to compare pre/post residualization."
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
