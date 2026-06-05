#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.baselines import MLPRegressorTorch, fit_ridge_grid, train_mlp_regressor
from models.metrics import fit_pca_on_train, pc_space_r2_from_pca, r2_summary
from models.utils import (
    append_log_row,
    build_results_dirs,
    ddp_cleanup,
    ddp_setup,
    load_training_contracts,
    load_config,
    save_json,
    set_seed,
)


def _select(data: Dict, idx: np.ndarray):
    return data["X1"][idx], data["Y1"][idx]


def run(cfg: Dict, exp_name: str) -> None:
    ddp_state = ddp_setup(cfg)
    rank = ddp_state["rank"]
    is_main = ddp_state["is_main"]
    seed = int(cfg.get("seed", 42))
    set_seed(seed + rank)
    data = load_training_contracts(cfg)

    Xtr, Ytr = _select(data, data["idx1_train"])
    Xva, Yva = _select(data, data["idx1_val"])
    Xte, Yte = _select(data, data["idx1_test"])
    Xext = data["X2"][data["idx2_external"]]
    Yext = data["Y2"][data["idx2_external"]]
    if np.max(data["idx1_train"]) >= data["X1"].shape[0]:
        raise RuntimeError("dataset1_train indices out of bounds for dataset1 arrays.")

    result_root = Path(cfg.get("results_dir", "results")) / exp_name
    dirs = build_results_dirs(result_root)
    if is_main:
        save_json(dirs["exp"] / "config.json", cfg)

    # Keep baseline training rank-0 only when DDP is enabled.
    if not is_main:
        ddp_cleanup()
        return

    target_cfg = cfg.get("target_pca", {})
    target_pca_enabled = bool(target_cfg.get("enabled", False))
    pca = None
    if target_pca_enabled:
        k = int(target_cfg.get("k", 20))
        pca = fit_pca_on_train(Ytr, k=k, seed=seed)
        Ytr_t = pca.transform(Ytr)
        Yva_t = pca.transform(Yva)
        Yte_t = pca.transform(Yte)
        Yext_t = pca.transform(Yext)
    else:
        Ytr_t, Yva_t, Yte_t, Yext_t = Ytr, Yva, Yte, Yext

    # Baseline-0 Ridge
    ridge_cfg = cfg.get("ridge", {})
    alphas = ridge_cfg.get("alphas", [1e-3, 1e-2, 1e-1, 1, 10, 100])
    ridge_model, ridge_info = fit_ridge_grid(Xtr, Ytr_t, Xva, Yva_t, alphas)
    yte_pred_r = ridge_model.predict(Xte)
    yext_pred_r = ridge_model.predict(Xext)
    if target_pca_enabled:
        yte_pred_full = pca.inverse_transform(yte_pred_r)
        yext_pred_full = pca.inverse_transform(yext_pred_r)
        ridge_m1 = {
            "edge_space_r2": r2_summary(Yte, yte_pred_full),
            "pc_space_r2": pc_space_r2_from_pca(Yte, yte_pred_full, pca),
        }
        ridge_m2 = {
            "edge_space_r2": r2_summary(Yext, yext_pred_full),
            "pc_space_r2": pc_space_r2_from_pca(Yext, yext_pred_full, pca),
        }
    else:
        ridge_m1 = r2_summary(Yte, yte_pred_r)
        ridge_m2 = r2_summary(Yext, yext_pred_r)
    with open(dirs["ckpt"] / "baseline_ridge.pkl", "wb") as f:
        pickle.dump(ridge_model, f)
    append_log_row(dirs["exp"] / "logs.csv", {"model": "ridge", "split_role": "dataset1_val", **ridge_info})

    # Baseline-1 MLP
    mlp_cfg = cfg.get("mlp", {})
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_cuda", False) else "cpu")
    model = MLPRegressorTorch(
        in_dim=data["dx"],
        out_dim=Ytr_t.shape[1],
        hidden_dims=mlp_cfg.get("hidden_dims", [512, 256]),
        dropout=float(mlp_cfg.get("dropout", 0.1)),
        activation=str(mlp_cfg.get("activation", "relu")),
    ).to(device)
    train_info = train_mlp_regressor(
        model=model,
        X_train=Xtr,
        Y_train=Ytr_t,
        X_val=Xva,
        Y_val=Yva_t,
        epochs=int(mlp_cfg.get("epochs", 100)),
        batch_size=int(mlp_cfg.get("batch_size", 64)),
        lr=float(mlp_cfg.get("lr", 1e-3)),
        weight_decay=float(mlp_cfg.get("weight_decay", 1e-5)),
        patience=int(mlp_cfg.get("patience", 10)),
        seed=seed,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        yte_pred_m = model(torch.tensor(Xte, dtype=torch.float32, device=device)).cpu().numpy()
        yext_pred_m = model(torch.tensor(Xext, dtype=torch.float32, device=device)).cpu().numpy()
    if target_pca_enabled:
        yte_pred_full = pca.inverse_transform(yte_pred_m)
        yext_pred_full = pca.inverse_transform(yext_pred_m)
        mlp_m1 = {
            "edge_space_r2": r2_summary(Yte, yte_pred_full),
            "pc_space_r2": pc_space_r2_from_pca(Yte, yte_pred_full, pca),
        }
        mlp_m2 = {
            "edge_space_r2": r2_summary(Yext, yext_pred_full),
            "pc_space_r2": pc_space_r2_from_pca(Yext, yext_pred_full, pca),
        }
    else:
        mlp_m1 = r2_summary(Yte, yte_pred_m)
        mlp_m2 = r2_summary(Yext, yext_pred_m)
    torch.save({"model_state": model.state_dict(), "cfg": mlp_cfg, "dx": data["dx"], "dy": data["dy"]}, dirs["ckpt"] / "baseline_mlp.pt")
    for row in train_info["logs"]:
        append_log_row(
            dirs["exp"] / "logs.csv",
            {"model": "mlp", "train_split_role": "dataset1_train", "val_split_role": "dataset1_val", **row},
        )

    out1 = {
        "dataset": "dataset1_test",
        "target_pca_enabled": target_pca_enabled,
        "target_pca_k": int(pca.n_components_) if pca is not None else None,
        "split_sizes": {"train": int(len(Xtr)), "val": int(len(Xva)), "test": int(len(Xte))},
        "ridge": ridge_m1,
        "mlp": mlp_m1,
    }
    out2 = {
        "dataset": "dataset2_external",
        "target_pca_enabled": target_pca_enabled,
        "target_pca_k": int(pca.n_components_) if pca is not None else None,
        "external_n": int(len(Xext)),
        "ridge": ridge_m2,
        "mlp": mlp_m2,
    }
    save_json(dirs["exp"] / "metrics_dataset1.json", out1)
    save_json(dirs["exp"] / "metrics_dataset2.json", out2)

    print(json.dumps({"exp_dir": str(dirs["exp"]), "dataset1_test": out1, "dataset2_external": out2}, indent=2))
    ddp_cleanup()


def main():
    parser = argparse.ArgumentParser(description="Train baseline Ridge/MLP models on preprocessed GM-sFNC pairs")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml/json config")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name under results/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if "paths" not in cfg:
        raise ValueError("Config must contain paths.aligned_features_dir and paths.splits_dir")
    run(cfg, args.exp_name)


if __name__ == "__main__":
    main()

