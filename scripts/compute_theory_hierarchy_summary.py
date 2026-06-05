#!/usr/bin/env python3
"""
Compute compact diagnostics for Section 4.2.3 theoretical validation and
hierarchy-position reporting.

Inputs:
- results/bootstrap_sv/sv_stats.json
- results/multivariate_methods/decompositions/nuclear_norm_seed42_B.npy
- results/multivariate_methods/decompositions/nuclear_norm_seed42_Y_resid_ds1.npy
- train/config_baselines.yaml
- Optional: ROI-level Margulies gradient vector for hierarchy analysis
  (plain text/CSV/NumPy array with 99 entries).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from models.utils import load_config, load_training_contracts


def load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def estimate_overlap_theory(
    s_vals: np.ndarray,
    gamma: float,
    sigma2: float,
    r: int = 20,
) -> Dict[str, float]:
    """
    Spiked-model plug-in using:
      O ≈ (1/r) Σ_{k=1}^r (1 - γ / ℓ_k^4) 1(ℓ_k > ℓ_c)
      R2 ≈ Σ σ_k^2 Var(x^T u_k) / (Σ σ_k^2 Var(x^T u_k) + qσ²)
    with ℓ_k = σ_k(B)/σ.
    """
    s_r = np.asarray(s_vals[:r], dtype=float)
    lc = gamma ** (0.25)  # BBP detectability edge = gamma^{+1/4} (where 1 - gamma/ell^4 > 0)
    l = s_r / np.sqrt(sigma2)
    o_terms = np.where(l > lc, 1.0 - gamma / np.maximum(l ** 4, np.finfo(float).eps), 0.0)
    o_theory = float(np.mean(o_terms))
    return {
        "r": int(r),
        "gamma": float(gamma),
        "lc": float(lc),
        "sigma2": float(sigma2),
        "O_theory": float(o_theory),
        "l_min_used": float(l.min()),
        "l_max_used": float(l.max()),
        "n_recovered": int(np.sum(l > lc)),
        "l2_threshold": float(np.mean(l > lc)),
        "l_values": [float(x) for x in l],
        "lc_mask": [bool(x > lc) for x in l],
    }


def estimate_r2_theory(
    B: np.ndarray,
    sigma2: float,
    X_train: np.ndarray,
    q: int,
    r: int = 20,
) -> Dict[str, float]:
    U, S, _ = np.linalg.svd(B, full_matrices=False)
    U_r = U[:, :r]
    S_r = S[:r]

    xproj = X_train @ U_r
    varxu = np.var(xproj, axis=0, ddof=0)
    numerator = float(np.sum((S_r ** 2) * varxu))
    denominator = float(np.sum((S_r ** 2) * varxu) + float(q * sigma2))
    r2_theory = numerator / denominator if denominator > 0 else float("nan")

    return {
        "r": int(r),
        "sigma2": float(sigma2),
        "R2_theory": float(r2_theory),
        "R2_numerator": numerator,
        "R2_denominator": denominator,
        "varxu_first_5": [float(v) for v in varxu[:5]],
        "varxu_mean": float(float(varxu.mean())),
    }


def estimate_hierarchy_positions(
    B: np.ndarray,
    grad_path: Optional[Path],
    bootstrap_U_path: Optional[Path] = None,
) -> Dict[str, object]:
    out: Dict[str, object] = {
        "available": False,
        "status": "Margulies gradient input not provided.",
        "positions_mode1_to_3": None,
    }

    if grad_path is None:
        return out
    if not grad_path.exists():
        out["status"] = f"Gradient file not found: {grad_path}"
        return out

    try:
        g = np.loadtxt(grad_path, delimiter=",")
    except Exception:
        out["status"] = f"Failed to read gradient file: {grad_path}"
        return out

    g = np.asarray(g).ravel()
    if g.size != B.shape[0]:
        out["status"] = f"Expected 99 ROI values; got {g.size}"
        return out

    U, _, _ = np.linalg.svd(B, full_matrices=False)
    pos = []
    for m in range(3):
        u = np.abs(U[:, m])
        denom = np.sum(u)
        if denom <= 0:
            pos.append(float("nan"))
            continue
        pos.append(float(np.sum(u * g) / denom))
    bootstrap_positions = None
    if bootstrap_U_path is not None and bootstrap_U_path.exists():
        try:
            U_boot = np.load(bootstrap_U_path)
            U_boot = np.asarray(U_boot)
            if U_boot.ndim == 3 and U_boot.shape[-1] == B.shape[1]:
                U_boot = np.swapaxes(U_boot, 1, 2)
            if U_boot.ndim == 3 and U_boot.shape[1] == B.shape[0]:
                # expected (n_boot, p, r)
                n_boot, p_boot, r_boot = U_boot.shape
                r_use = min(3, r_boot, U.shape[1])
                boot_pos = np.zeros((n_boot, r_use), dtype=float)
                for bi in range(n_boot):
                    for m in range(r_use):
                        u = np.abs(U_boot[bi, :, m])
                        denom = np.sum(u)
                        if denom > 0:
                            boot_pos[bi, m] = np.sum(u * g) / denom
                bootstrap_positions = {
                    f"mode{m+1}": [
                        float(np.percentile(boot_pos[:, m], 2.5)),
                        float(np.percentile(boot_pos[:, m], 97.5)),
                    ]
                    for m in range(r_use)
                }
        except Exception:
            bootstrap_positions = {"status": "bootstrap U load failed"}

    pos_out: Dict[str, object] = {
        "mode1": pos[0],
        "mode2": pos[1],
        "mode3": pos[2],
    }
    if bootstrap_positions:
        pos_out["ci95"] = bootstrap_positions

    # Optional percentile intervals are reported as requested.
    out.update({
        "available": True,
        "status": "ok",
        "positions_mode1_to_3": {
            "mode1": pos[0],
            "mode2": pos[1],
            "mode3": pos[2],
        },
    })
    if bootstrap_positions:
        out["positions_mode1_to_3"]["ci95"] = bootstrap_positions
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute theory + hierarchy checks")
    parser.add_argument("--config", default="train/config_baselines.yaml")
    parser.add_argument("--b-path", default="results/multivariate_methods/decompositions/nuclear_norm_seed42_B.npy")
    parser.add_argument("--sv-stats", default="results/bootstrap_sv/sv_stats.json")
    parser.add_argument("--resid-ds1", default="results/multivariate_methods/decompositions/nuclear_norm_seed42_Y_resid_ds1.npy")
    parser.add_argument("--out-json", default="results/paper_theory_hierarchy_check.json")
    parser.add_argument("--out-tsv", default=None)
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--margulies-gradient", default=None,
                        help="Optional 99-entry ROI-level gradient file (space/comma separated or 1-col CSV).")
    parser.add_argument("--bootstrap-u",
                        default=None,
                        help="Optional bootstrap V-mode U tensor for hierarchy CI (n_boot x p x r).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = load_training_contracts(cfg)
    X_train = splits["X1"][splits["idx1_train"]]
    n_train, p = X_train.shape
    q = splits["Y1"].shape[1]

    B = np.load(args.b_path)
    residual = np.load(args.resid_ds1)
    sigma2 = float(np.mean(residual ** 2))

    sv_stats = load_json(Path(args.sv_stats))
    n_train_s = int(sv_stats.get("n_train", n_train))
    dx = int(sv_stats.get("dx", p))
    gamma = dx / n_train_s

    # Also collect empirical overlap from subspace stats if available.
    subspace_path = Path("results/subspace_analysis/subspace_stats.json")
    overlap_k = {}
    r2_empirical = None
    if subspace_path.exists():
        sub = load_json(subspace_path)
        nn = sub.get("Nuclear_Norm", {})
        for key in ["k=5", "k=10", "k=20"]:
            sa = nn.get("subspace_analysis", {}).get(key, {})
            if "overlap_mean" in sa:
                overlap_k[key] = float(sa["overlap_mean"])
        if "r2_global" in nn:
            r2_empirical = float(nn["r2_global"])
        elif "r2_global_mean" in nn:
            r2_empirical = float(nn["r2_global_mean"])

    sv_ci = {}
    if "ci95_lo" in sv_stats and "ci95_hi" in sv_stats:
        ci_lo = sv_stats["ci95_lo"]
        ci_hi = sv_stats["ci95_hi"]
        for j in range(min(args.rank, len(ci_lo), len(ci_hi))):
            sv_ci[f"mode{j+1}"] = {
                "lo": float(ci_lo[j]),
                "hi": float(ci_hi[j]),
            }

    empirical = {
        "k=5": overlap_k.get("k=5"),
        "k=10": overlap_k.get("k=10"),
        "k=20": overlap_k.get("k=20"),
        "R2_global": r2_empirical,
    }
    overlap_model = estimate_overlap_theory(np.linalg.svd(B, full_matrices=False)[1], gamma=gamma, sigma2=sigma2, r=args.rank)
    r2_model = estimate_r2_theory(B, sigma2=sigma2, X_train=X_train, q=q, r=args.rank)
    hierarchy = estimate_hierarchy_positions(
        B,
        Path(args.margulies_gradient) if args.margulies_gradient else None,
        Path(args.bootstrap_u) if args.bootstrap_u else None,
    )

    out = {
        "dataset": {
            "n_train": n_train,
            "p": p,
            "q": q,
            "dx": dx,
            "sv_n_train": n_train_s,
            "rank": args.rank,
        },
        "empirical_overlap": empirical,
        "theory": {
            "overlap": overlap_model,
            "r2": r2_model,
        },
        "bootstrap_sv_ci_mode1_mode20": sv_ci,
        "hierarchy": hierarchy,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[OK] Wrote {out_path}")
    print(f"gamma = {gamma:.6f}, lc = {overlap_model['lc']:.4f}, rank={args.rank}")
    print(f"O_theory = {overlap_model['O_theory']:.6f}, observed k20 = {empirical.get('k=20', float('nan')):.4f}")
    print(f"R2_theory = {r2_model['R2_theory']:.6f}, empirical R2 = {empirical.get('R2_global', float('nan')):.4f}")

    if args.out_tsv is not None:
        tsv_path = Path(args.out_tsv)
        with open(tsv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "n_train", "p", "q", "dx", "gamma", "lc", "rank", "O_theory", "O_empirical_k20", "R2_theory", "R2_empirical", "n_recovered"])
            writer.writerow([
                "DS1",
                n_train,
                p,
                q,
                dx,
                f"{gamma:.6f}",
                f"{overlap_model['lc']:.6f}",
                args.rank,
                f"{overlap_model['O_theory']:.6f}",
                f"{empirical.get('k=20', float('nan')):.6f}",
                f"{r2_model['R2_theory']:.6f}",
                f"{empirical.get('R2_global', float('nan')):.6f}",
                overlap_model["n_recovered"],
            ])
        print(f"[OK] Wrote {tsv_path}")

if __name__ == "__main__":
    main()
