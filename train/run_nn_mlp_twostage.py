#!/usr/bin/env python3
"""
Two-stage Nuclear Norm + MLP method (Direction C).

Bridges the ~0.003 R² nonlinearity gap between Nuclear Norm and MLP:
  Stage 1: Fit Nuclear Norm B (linear) -> Y_struct_lin = X @ B
  Stage 2: Train MLP on X -> residual (Y - Y_struct_lin), capturing nonlinear effects
  Final:   Y_pred = X @ B + MLP(X)

Three variants:
  A. NN + MLP(residual): MLP predicts what the linear model missed
  B. NN-init MLP: Initialize MLP output layer with B, fine-tune end-to-end
  C. NN-subspace MLP: Project X into NN-selected subspace, then MLP on subspace

Usage:
    python train/run_nn_mlp_twostage.py \
        --config train/config_baselines.yaml \
        --seeds 42 43 44 45 46 47 48 \
        --pca_ks 5 7 10 20 \
        --out_dir results/nn_mlp_twostage
"""
import argparse
import copy
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.baselines import MLPRegressorTorch, fit_ridge_grid, train_mlp_regressor
from models.metrics import fit_pca_on_train, pc_space_r2_from_pca, r2_summary
from models.mlp import build_mlp
from models.utils import load_config, load_training_contracts, save_json, set_seed
from train.run_multivariate_methods import fit_nuclear_norm
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
# Variant A: NN + MLP(residual)
# ---------------------------------------------------------------------------

class ResidualMLP(nn.Module):
    """MLP that predicts the residual after linear Nuclear Norm prediction."""

    def __init__(
        self,
        B_nn: np.ndarray,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.register_buffer(
            "B_nn", torch.tensor(B_nn, dtype=torch.float32),
        )
        self.residual_net = build_mlp(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_linear = x @ self.B_nn
        y_residual = self.residual_net(x)
        return y_linear + y_residual


def train_residual_mlp(
    model: ResidualMLP,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int,
    device: torch.device,
) -> Dict:
    """Train ResidualMLP end-to-end."""
    opt = torch.optim.Adam(model.residual_net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(Y_train, dtype=torch.float32, device=device)
    xval = torch.tensor(X_val, dtype=torch.float32, device=device)
    yval = torch.tensor(Y_val, dtype=torch.float32, device=device)

    best_state = None
    best_val = float("inf")
    wait = 0
    logs = []
    rng = np.random.default_rng(seed)

    for ep in range(epochs):
        model.train()
        idx = np.arange(len(X_train))
        rng.shuffle(idx)
        batch_losses = []
        for i in range(0, len(idx), batch_size):
            b = idx[i: i + batch_size]
            xb = xtr[b]
            yb = ytr[b]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            batch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_pred = model(xval)
            val_loss = float(loss_fn(val_pred, yval).cpu())
        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        logs.append({"epoch": ep + 1, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_loss": best_val, "logs": logs, "n_epochs": len(logs)}


# ---------------------------------------------------------------------------
# Variant B: NN-init MLP (B as output layer init)
# ---------------------------------------------------------------------------

class NNInitMLP(nn.Module):
    """MLP initialized with Nuclear Norm B in the output layer."""

    def __init__(
        self,
        B_nn: np.ndarray,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        # Build hidden layers only
        act_map = {"relu": nn.ReLU, "gelu": nn.GELU, "elu": nn.ELU}
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_map.get(activation, nn.ReLU)())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev, out_dim)

        # Initialize output layer with NN B projected through random hidden
        # Actually: use skip connection so output = hidden(x) + x @ B_nn
        self.register_buffer(
            "B_nn", torch.tensor(B_nn, dtype=torch.float32),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))  # learned blending

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_linear = x @ self.B_nn
        y_nonlinear = self.output(self.hidden(x))
        alpha = torch.sigmoid(self.alpha)
        return alpha * y_nonlinear + (1 - alpha) * y_linear


# ---------------------------------------------------------------------------
# Single seed runner
# ---------------------------------------------------------------------------

def evaluate_predictions(
    Y_pred_ds1: np.ndarray,
    Y_pred_ds2: np.ndarray,
    Yte: np.ndarray,
    Yext: np.ndarray,
    Ytr: np.ndarray,
    pca_ks: List[int],
    seed: int,
) -> Dict:
    """Evaluate at multiple k values."""
    results = {"dataset1_test": {}, "dataset2_external": {}}
    for k in pca_ks:
        pca = fit_pca_on_train(Ytr, k=k, seed=seed)
        results["dataset1_test"][f"k{k}"] = pc_space_r2_from_pca(Yte, Y_pred_ds1, pca)
        results["dataset2_external"][f"k{k}"] = pc_space_r2_from_pca(Yext, Y_pred_ds2, pca)
    results["dataset1_test"]["edge_r2"] = r2_summary(Yte, Y_pred_ds1)
    results["dataset2_external"]["edge_r2"] = r2_summary(Yext, Y_pred_ds2)
    return results


def run_single_seed(
    cfg: Dict,
    seed: int,
    pca_ks: List[int],
    mlp_epochs: int = 200,
    mlp_patience: int = 20,
) -> Dict:
    """Run all variants for a single seed."""
    set_seed(seed)
    data = load_training_contracts(cfg)

    idx_tr = data["idx1_train"]
    idx_val = data["idx1_val"]
    idx_te = data["idx1_test"]
    idx_ext = data["idx2_external"]

    Xtr = data["X1"][idx_tr]
    Ytr = data["Y1"][idx_tr]
    Xva = data["X1"][idx_val]
    Yva = data["Y1"][idx_val]
    Xte = data["X1"][idx_te]
    Yte = data["Y1"][idx_te]
    Xext = data["X2"][idx_ext]
    Yext = data["Y2"][idx_ext]

    Xtr64 = Xtr.astype(np.float64)
    Ytr64 = Ytr.astype(np.float64)
    Xva64 = Xva.astype(np.float64)
    Yva64 = Yva.astype(np.float64)

    mlp_cfg = cfg.get("mlp", {})
    hidden_dims = mlp_cfg.get("hidden_dims", [256, 128])
    dropout = float(mlp_cfg.get("dropout", 0.1))
    lr = float(mlp_cfg.get("lr", 1e-3))
    weight_decay = float(mlp_cfg.get("weight_decay", 1e-5))
    batch_size = int(mlp_cfg.get("batch_size", 64))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_cuda", False) else "cpu")

    results = {"seed": seed}

    # --- Stage 1: Fit Nuclear Norm ---
    print(f"  [seed {seed}] Stage 1: Nuclear Norm ...", flush=True)
    t0 = time.time()
    nn_result = fit_nuclear_norm(Xtr64, Ytr64, Xva64, Yva64)
    B_nn = nn_result["B"]
    results["nuclear_norm_info"] = {
        "optimal_lambda": nn_result["optimal_lambda"],
        "fit_time_s": round(time.time() - t0, 2),
    }

    # Evaluate standalone NN
    Y_pred_nn_ds1 = (Xte.astype(np.float64) @ B_nn).astype(np.float32)
    Y_pred_nn_ds2 = (Xext.astype(np.float64) @ B_nn).astype(np.float32)
    results["nuclear_norm"] = evaluate_predictions(
        Y_pred_nn_ds1, Y_pred_nn_ds2, Yte, Yext, Ytr, pca_ks, seed,
    )

    # --- Standalone MLP baseline ---
    print(f"  [seed {seed}] Standalone MLP ...", flush=True)
    t0 = time.time()
    mlp_model = MLPRegressorTorch(
        in_dim=data["dx"], out_dim=data["dy"],
        hidden_dims=hidden_dims, dropout=dropout, activation="relu",
    ).to(device)
    mlp_info = train_mlp_regressor(
        model=mlp_model, X_train=Xtr, Y_train=Ytr, X_val=Xva, Y_val=Yva,
        epochs=mlp_epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, patience=mlp_patience,
        seed=seed, device=device,
    )
    mlp_model.eval()
    with torch.no_grad():
        Y_pred_mlp_ds1 = mlp_model(torch.tensor(Xte, dtype=torch.float32, device=device)).cpu().numpy()
        Y_pred_mlp_ds2 = mlp_model(torch.tensor(Xext, dtype=torch.float32, device=device)).cpu().numpy()
    results["mlp"] = evaluate_predictions(
        Y_pred_mlp_ds1, Y_pred_mlp_ds2, Yte, Yext, Ytr, pca_ks, seed,
    )
    results["mlp"]["train_time_s"] = round(time.time() - t0, 2)

    # --- Variant A: NN + MLP(residual) ---
    print(f"  [seed {seed}] Variant A: NN + MLP(residual) ...", flush=True)
    t0 = time.time()
    model_a = ResidualMLP(
        B_nn=B_nn.astype(np.float32),
        in_dim=data["dx"], out_dim=data["dy"],
        hidden_dims=hidden_dims, dropout=dropout,
    ).to(device)
    info_a = train_residual_mlp(
        model_a, Xtr, Ytr, Xva, Yva,
        epochs=mlp_epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, patience=mlp_patience,
        seed=seed, device=device,
    )
    model_a.eval()
    with torch.no_grad():
        Y_pred_a_ds1 = model_a(torch.tensor(Xte, dtype=torch.float32, device=device)).cpu().numpy()
        Y_pred_a_ds2 = model_a(torch.tensor(Xext, dtype=torch.float32, device=device)).cpu().numpy()
    results["nn_plus_mlp_residual"] = evaluate_predictions(
        Y_pred_a_ds1, Y_pred_a_ds2, Yte, Yext, Ytr, pca_ks, seed,
    )
    results["nn_plus_mlp_residual"]["train_time_s"] = round(time.time() - t0, 2)
    results["nn_plus_mlp_residual"]["n_epochs"] = info_a["n_epochs"]

    # --- Variant B: NN-init MLP ---
    print(f"  [seed {seed}] Variant B: NN-init MLP ...", flush=True)
    t0 = time.time()
    model_b = NNInitMLP(
        B_nn=B_nn.astype(np.float32),
        in_dim=data["dx"], out_dim=data["dy"],
        hidden_dims=hidden_dims, dropout=dropout,
    ).to(device)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(Ytr, dtype=torch.float32, device=device)
    xval_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    yval_t = torch.tensor(Yva, dtype=torch.float32, device=device)

    best_state_b = None
    best_val_b = float("inf")
    wait_b = 0
    rng_b = np.random.default_rng(seed)
    n_epochs_b = 0

    for ep in range(mlp_epochs):
        model_b.train()
        idx = np.arange(len(Xtr))
        rng_b.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            b_idx = idx[i: i + batch_size]
            opt_b.zero_grad()
            pred = model_b(xtr_t[b_idx])
            loss = loss_fn(pred, ytr_t[b_idx])
            loss.backward()
            opt_b.step()

        model_b.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model_b(xval_t), yval_t).cpu())
        n_epochs_b = ep + 1

        if val_loss < best_val_b:
            best_val_b = val_loss
            best_state_b = copy.deepcopy(model_b.state_dict())
            wait_b = 0
        else:
            wait_b += 1
            if wait_b >= mlp_patience:
                break

    if best_state_b is not None:
        model_b.load_state_dict(best_state_b)
    model_b.eval()
    with torch.no_grad():
        Y_pred_b_ds1 = model_b(torch.tensor(Xte, dtype=torch.float32, device=device)).cpu().numpy()
        Y_pred_b_ds2 = model_b(torch.tensor(Xext, dtype=torch.float32, device=device)).cpu().numpy()
        alpha_val = float(torch.sigmoid(model_b.alpha).cpu())
    results["nn_init_mlp"] = evaluate_predictions(
        Y_pred_b_ds1, Y_pred_b_ds2, Yte, Yext, Ytr, pca_ks, seed,
    )
    results["nn_init_mlp"]["train_time_s"] = round(time.time() - t0, 2)
    results["nn_init_mlp"]["n_epochs"] = n_epochs_b
    results["nn_init_mlp"]["learned_alpha"] = alpha_val

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(
    all_results: List[Dict],
    pca_ks: List[int],
    n_boot: int = 10000,
) -> Dict:
    methods = ["nuclear_norm", "mlp", "nn_plus_mlp_residual", "nn_init_mlp"]
    seeds = [r["seed"] for r in all_results]
    summary = {"n_seeds": len(seeds), "seeds": seeds}

    per_seed = {}
    for method in methods:
        per_seed[method] = {}
        for k in pca_ks:
            kstr = f"k{k}"
            d1 = [r[method]["dataset1_test"][kstr]["pc_r2_mean"] for r in all_results]
            d2 = [r[method]["dataset2_external"][kstr]["pc_r2_mean"] for r in all_results]
            per_seed[method][kstr] = {"d1": np.array(d1), "d2": np.array(d2)}

    for method in methods:
        summary[method] = {}
        for k in pca_ks:
            kstr = f"k{k}"
            d1 = per_seed[method][kstr]["d1"]
            d2 = per_seed[method][kstr]["d2"]
            summary[method][kstr] = {
                "ds1_pc_r2": _stats(d1),
                "ds2_pc_r2": _stats(d2),
            }
            if len(d1) > 1 and n_boot > 0:
                summary[method][kstr]["ds1_bootstrap_ci"] = bootstrap_bca_ci(
                    d1, n_boot=n_boot, seed=42)
        edge_d1 = [r[method]["dataset1_test"]["edge_r2"]["r2_edge_mean"] for r in all_results]
        edge_d2 = [r[method]["dataset2_external"]["edge_r2"]["r2_edge_mean"] for r in all_results]
        summary[method]["edge_r2_d1"] = _stats(edge_d1)
        summary[method]["edge_r2_d2"] = _stats(edge_d2)

    # Paired tests: each proposed method vs standalone MLP
    summary["paired_vs_mlp"] = {}
    for method in ["nuclear_norm", "nn_plus_mlp_residual", "nn_init_mlp"]:
        summary["paired_vs_mlp"][method] = {}
        for k in pca_ks:
            kstr = f"k{k}"
            summary["paired_vs_mlp"][method][kstr] = paired_t_test(
                per_seed[method][kstr]["d1"],
                per_seed["mlp"][kstr]["d1"],
            )
        # DS2
        summary["paired_vs_mlp"][method]["k20_ds2"] = paired_t_test(
            per_seed[method]["k20"]["d2"],
            per_seed["mlp"]["k20"]["d2"],
        )

    # Paired tests: two-stage vs standalone NN
    summary["paired_vs_nn"] = {}
    for method in ["nn_plus_mlp_residual", "nn_init_mlp"]:
        summary["paired_vs_nn"][method] = {}
        for k in pca_ks:
            kstr = f"k{k}"
            summary["paired_vs_nn"][method][kstr] = paired_t_test(
                per_seed[method][kstr]["d1"],
                per_seed["nuclear_norm"][kstr]["d1"],
            )

    # Alpha analysis for NN-init MLP
    alphas = [r["nn_init_mlp"]["learned_alpha"] for r in all_results]
    summary["nn_init_mlp_alpha"] = _stats(alphas)

    return summary


def print_summary(summary: Dict, pca_ks: List[int]) -> None:
    methods = ["nuclear_norm", "mlp", "nn_plus_mlp_residual", "nn_init_mlp"]
    labels = {
        "nuclear_norm": "Nuclear Norm (lin)",
        "mlp": "MLP (baseline)",
        "nn_plus_mlp_residual": "NN + MLP(resid)",
        "nn_init_mlp": "NN-init MLP",
    }

    print("\n" + "=" * 95)
    print("TWO-STAGE METHODS COMPARISON (pc_r2_mean)")
    print("=" * 95)

    header = f"{'Method':<22s}"
    for k in pca_ks:
        header += f" {'DS1 k=' + str(k):>14s}"
    header += f" {'DS2 k=20':>14s}"
    print(header)
    print("-" * 95)

    for method in methods:
        s = summary[method]
        row = f"{labels[method]:<22s}"
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
        print(row)

    print("=" * 95)

    # Paired tests vs MLP
    print("\nPaired t-tests vs standalone MLP (DS1):")
    for method in ["nn_plus_mlp_residual", "nn_init_mlp"]:
        print(f"  {labels[method]}:")
        for k in pca_ks:
            kstr = f"k{k}"
            t = summary["paired_vs_mlp"][method][kstr]
            sig = "*" if t["p_value"] < 0.05 else " "
            print(f"    k={k:>2d}: diff={t['mean_diff']:+.4f}  p={t['p_value']:.3f} {sig}")

    # Paired tests vs NN
    print("\nPaired t-tests vs standalone Nuclear Norm (DS1):")
    for method in ["nn_plus_mlp_residual", "nn_init_mlp"]:
        print(f"  {labels[method]}:")
        for k in pca_ks:
            kstr = f"k{k}"
            t = summary["paired_vs_nn"][method][kstr]
            sig = "*" if t["p_value"] < 0.05 else " "
            print(f"    k={k:>2d}: diff={t['mean_diff']:+.4f}  p={t['p_value']:.3f} {sig}")

    # Alpha analysis
    alpha_s = summary["nn_init_mlp_alpha"]
    print(f"\nNN-init MLP learned alpha (nonlinear weight): "
          f"{alpha_s['mean']:.3f} +/- {alpha_s.get('ci95', 0):.3f}")
    print(f"  (alpha=0 -> pure linear NN, alpha=1 -> pure nonlinear MLP)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Two-stage Nuclear Norm + MLP (Direction C)",
    )
    parser.add_argument("--config", type=str, default="train/config_baselines.yaml")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 43, 44, 45, 46, 47, 48])
    parser.add_argument("--pca_ks", type=int, nargs="+", default=[5, 7, 10, 20])
    parser.add_argument("--mlp_epochs", type=int, default=200)
    parser.add_argument("--mlp_patience", type=int, default=20)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--out_dir", type=str,
                        default="results/nn_mlp_twostage")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TWO-STAGE NUCLEAR NORM + MLP (Direction C)")
    print(f"  Seeds:   {args.seeds}")
    print(f"  PCA ks:  {args.pca_ks}")
    print(f"  MLP:     {args.mlp_epochs} epochs, patience={args.mlp_patience}")
    print(f"  Output:  {out_dir}")
    print("=" * 80)

    all_results = []
    for i, seed in enumerate(args.seeds):
        t0 = time.time()
        print(f"\n[{i+1}/{len(args.seeds)}] Seed={seed}", flush=True)
        result = run_single_seed(
            cfg, seed=seed, pca_ks=args.pca_ks,
            mlp_epochs=args.mlp_epochs, mlp_patience=args.mlp_patience,
        )
        elapsed = time.time() - t0

        k20 = "k20" if 20 in args.pca_ks else f"k{args.pca_ks[-1]}"
        vals = []
        for m in ["nuclear_norm", "mlp", "nn_plus_mlp_residual", "nn_init_mlp"]:
            v = result[m]["dataset1_test"][k20]["pc_r2_mean"]
            vals.append(f"{m.split('_')[0]}={v:.4f}")
        print(f"  {', '.join(vals)}  [{elapsed:.1f}s]", flush=True)

        all_results.append(result)
        save_json(out_dir / f"seed_{seed}.json", result)

    summary = aggregate_results(all_results, args.pca_ks, n_boot=args.n_boot)
    save_json(out_dir / "summary.json", summary)

    print_summary(summary, args.pca_ks)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
