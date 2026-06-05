#!/usr/bin/env python3
"""
Subspace analysis: compare predicted vs actual FNC subspaces.

Quantifies whether GM constrains the *geometry* (directional structure) of
FNC variation rather than its amplitude, by measuring:
  1. Principal angle analysis between predicted and actual FNC subspaces
  2. Subspace overlap (mean cos²θ) at varying subspace dimensions k
  3. Permutation null distribution for statistical testing
  4. Variance-geometry decomposition (R² vs subspace overlap)

Methods analyzed (seed=42): Nuclear Norm, RRR, PLS, OptShrink, Ridge (refit).

Usage:
    python train/run_subspace_analysis.py \
        --config train/config_baselines.yaml \
        --seed 42 \
        --subspace_ks 5 10 20 30 \
        --n_perm 1000 \
        --out_dir results/subspace_analysis
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import Ridge

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.baselines import fit_ridge_grid
from models.metrics import r2_summary
from models.utils import load_config, load_training_contracts, set_seed


def principal_angles(V1: np.ndarray, V2: np.ndarray) -> np.ndarray:
    """Compute principal angles between two subspaces.

    Parameters
    ----------
    V1, V2 : (d, k) orthonormal basis matrices

    Returns
    -------
    cos_angles : (k,) array of cos(θ_i) in descending order
    """
    M = V1.T @ V2  # (k, k)
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    return np.clip(s, 0.0, 1.0)


def subspace_overlap(V1: np.ndarray, V2: np.ndarray) -> float:
    """Mean cos²(θ) between two subspaces — 1.0 = perfect alignment."""
    cos_angles = principal_angles(V1, V2)
    return float(np.mean(cos_angles ** 2))


def top_k_subspace(Y: np.ndarray, k: int) -> np.ndarray:
    """Return top-k right singular vectors of Y (n x d) as (d, k) matrix."""
    _, _, Vt = np.linalg.svd(Y, full_matrices=False)
    return Vt[:k].T  # (d, k)


def analyze_method(
    method_name: str,
    B: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    Y_train: np.ndarray,
    subspace_ks: List[int],
    n_perm: int,
    rng: np.random.Generator,
    null_model: str,
) -> Dict:
    """Run subspace analysis for one method.

    The default null is a random k-dimensional subspace in R^d_y, which asks
    whether the predicted subspace aligns with the observed FNC subspace better
    than an arbitrary subspace of the same dimensionality.
    """
    Y_pred = X_test @ B
    n, d = Y_test.shape

    # Variance explained (R²)
    r2_info = r2_summary(Y_test.astype(np.float32), Y_pred.astype(np.float32))
    r2_global = r2_info["r2_global"]

    results = {
        "method": method_name,
        "r2_global": float(r2_global),
        "subspace_analysis": {},
    }

    for k in subspace_ks:
        if k > min(n, d):
            continue

        V_actual = top_k_subspace(Y_test, k)
        V_pred = top_k_subspace(Y_pred, k)

        cos_angles = principal_angles(V_actual, V_pred)
        overlap = float(np.mean(cos_angles ** 2))

        # Null model for the subspace overlap statistic.
        null_overlaps = []
        for _ in range(n_perm):
            if null_model == "random_subspace":
                Q, _ = np.linalg.qr(rng.standard_normal((d, k)))
                V_null = Q
            elif null_model == "permute_rows":
                perm_idx = rng.permutation(n)
                V_null = top_k_subspace(Y_pred[perm_idx], k)
            else:
                raise ValueError(f"Unknown null_model: {null_model}")
            null_overlaps.append(subspace_overlap(V_actual, V_null))

        null_overlaps = np.array(null_overlaps)
        count_ge = int(np.sum(null_overlaps >= overlap))
        p_value = float((count_ge + 1) / (n_perm + 1))

        results["subspace_analysis"][f"k={k}"] = {
            "cos_angles": cos_angles.tolist(),
            "subspace_overlap": overlap,
            "null_mean": float(np.mean(null_overlaps)),
            "null_std": float(np.std(null_overlaps)),
            "chance_level_random_subspace": float(k / d),
            "null_model": null_model,
            "p_value": p_value,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Subspace analysis")
    parser.add_argument("--config", type=str, default="train/config_baselines.yaml")
    parser.add_argument("--seed", type=int, nargs="+", default=[42])
    parser.add_argument("--subspace_ks", type=int, nargs="+", default=[5, 10, 20, 30])
    parser.add_argument("--n_perm", type=int, default=1000)
    parser.add_argument(
        "--null_model",
        type=str,
        default="random_subspace",
        choices=["random_subspace", "permute_rows"],
        help="Null model for overlap significance",
    )
    parser.add_argument("--out_dir", type=str, default="results/subspace_analysis")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    seeds = args.seed

    print("Loading data ...", flush=True)
    data = load_training_contracts(cfg)
    idx_tr = data["idx1_train"]
    idx_val = data["idx1_val"]
    idx_te = data["idx1_test"]

    Xtr = data["X1"][idx_tr].astype(np.float64)
    Ytr = data["Y1"][idx_tr].astype(np.float64)
    Xva = data["X1"][idx_val].astype(np.float64)
    Yva = data["Y1"][idx_val].astype(np.float64)
    Xte = data["X1"][idx_te].astype(np.float64)
    Yte = data["Y1"][idx_te].astype(np.float64)

    base = Path("/home/users/ybi3/sfcoupling")
    mv_dec = base / "results" / "multivariate_methods" / "decompositions"
    ksr_dec = base / "results" / "kernel_spectral_regression" / "decompositions"

    print(f"Seeds: {seeds}")
    print(f"Test set shape: X={Xte.shape}, Y={Yte.shape}")
    print(f"Subspace dimensions: {args.subspace_ks}")
    print(f"Permutations: {args.n_perm}")
    print(f"Null model: {args.null_model}")
    print()

    # Accumulate per-seed results: method -> k_label -> metric arrays
    per_seed_overlaps: Dict[str, Dict[str, List[float]]] = {}
    per_seed_pvalues: Dict[str, Dict[str, List[float]]] = {}
    per_seed_null_means: Dict[str, Dict[str, List[float]]] = {}
    per_seed_null_stds: Dict[str, Dict[str, List[float]]] = {}
    per_seed_r2: Dict[str, List[float]] = {}
    last_seed_results = {}  # keep full results from last seed for angles

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")
        set_seed(seed)
        rng = np.random.default_rng(seed)

        methods: Dict[str, np.ndarray] = {}

        for name in ["nuclear_norm", "rrr", "pls"]:
            path = mv_dec / f"{name}_seed{seed}_B.npy"
            if path.exists():
                methods[name.replace("_", " ").title().replace(" ", "_")] = np.load(path)
                print(f"  Loaded {name} B from {path}")
            else:
                print(f"  [SKIP] {name}: {path} not found")

        los_path = ksr_dec / f"linear_optshrink_seed{seed}_B.npy"
        if los_path.exists():
            methods["Linear_OptShrink"] = np.load(los_path)
            print(f"  Loaded Linear OptShrink B from {los_path}")
        else:
            print(f"  [SKIP] Linear OptShrink: {los_path} not found")

        print("  Refitting Ridge for B matrix ...", flush=True)
        ridge_alphas = cfg.get("ridge", {}).get(
            "alphas", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        )
        ridge_model, _ = fit_ridge_grid(Xtr, Ytr, Xva, Yva, ridge_alphas)
        methods["Ridge"] = ridge_model.coef_.T

        for method_name, B in methods.items():
            print(f"  Analyzing {method_name} ...", flush=True)
            t0 = time.time()
            res = analyze_method(
                method_name, B, Xte, Yte, Ytr,
                args.subspace_ks, args.n_perm, rng, args.null_model,
            )
            elapsed = time.time() - t0
            print(f"    Done in {elapsed:.1f}s  R²={res['r2_global']:.4f}")
            for k_label, sa in res["subspace_analysis"].items():
                print(f"      {k_label}: overlap={sa['subspace_overlap']:.4f}  "
                      f"null={sa['null_mean']:.4f}±{sa['null_std']:.4f}  "
                      f"p={sa['p_value']:.4f}")

            # Accumulate
            if method_name not in per_seed_overlaps:
                per_seed_overlaps[method_name] = {}
                per_seed_pvalues[method_name] = {}
                per_seed_null_means[method_name] = {}
                per_seed_null_stds[method_name] = {}
                per_seed_r2[method_name] = []
            per_seed_r2[method_name].append(res["r2_global"])
            for k_label, sa in res["subspace_analysis"].items():
                per_seed_overlaps[method_name].setdefault(k_label, []).append(
                    sa["subspace_overlap"]
                )
                per_seed_pvalues[method_name].setdefault(k_label, []).append(
                    sa["p_value"]
                )
                per_seed_null_means[method_name].setdefault(k_label, []).append(
                    sa["null_mean"]
                )
                per_seed_null_stds[method_name].setdefault(k_label, []).append(
                    sa["null_std"]
                )
            last_seed_results[method_name] = res

    # Aggregate across seeds
    print(f"\n\n{'='*80}")
    print(f"AGGREGATE RESULTS ({len(seeds)} seeds)")
    print(f"{'='*80}")
    aggregated = {}
    for method_name in per_seed_overlaps:
        r2_arr = np.array(per_seed_r2[method_name])
        aggregated[method_name] = {
            "method": method_name,
            "n_seeds": len(seeds),
            "r2_global_mean": float(np.mean(r2_arr)),
            "r2_global_std": float(np.std(r2_arr)),
            "subspace_analysis": {},
        }
        print(f"\n{method_name}:  R² = {np.mean(r2_arr):.4f} ± {np.std(r2_arr):.4f}")
        for k_label in sorted(per_seed_overlaps[method_name]):
            ov_arr = np.array(per_seed_overlaps[method_name][k_label])
            p_arr = np.array(per_seed_pvalues[method_name][k_label])
            null_mean_arr = np.array(per_seed_null_means[method_name][k_label])
            null_std_arr = np.array(per_seed_null_stds[method_name][k_label])
            aggregated[method_name]["subspace_analysis"][k_label] = {
                "overlap_mean": float(np.mean(ov_arr)),
                "overlap_std": float(np.std(ov_arr)),
                "overlap_per_seed": ov_arr.tolist(),
                "p_value_mean": float(np.mean(p_arr)),
                "p_value_max": float(np.max(p_arr)),
                "p_value_per_seed": p_arr.tolist(),
                "null_mean_mean": float(np.mean(null_mean_arr)),
                "null_std_mean": float(np.mean(null_std_arr)),
            }
            print(f"    {k_label}: overlap = {np.mean(ov_arr):.4f} ± {np.std(ov_arr):.4f}")

    # Also store the last seed's full results (angles, null stats, p-values)
    # for backward compatibility with figure generation
    for method_name, res in last_seed_results.items():
        for k_label, sa in res["subspace_analysis"].items():
            agg_sa = aggregated[method_name]["subspace_analysis"][k_label]
            agg_sa["cos_angles"] = sa["cos_angles"]
            agg_sa["null_mean"] = sa["null_mean"]
            agg_sa["null_std"] = sa["null_std"]
            agg_sa["p_value"] = sa["p_value"]
            agg_sa["chance_level_random_subspace"] = sa.get(
                "chance_level_random_subspace"
            )
            agg_sa["null_model"] = sa.get("null_model")
            # Keep last-seed values for backward compatibility; plotting code
            # should prefer overlap_mean / r2_global_mean.
            agg_sa["subspace_overlap"] = sa["subspace_overlap"]
        aggregated[method_name]["r2_global"] = res["r2_global"]

    # Save
    out_path = out_dir / "subspace_stats.json"
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save principal angles for plotting (from last seed)
    angles_path = out_dir / "principal_angles.json"
    angles_data = {}
    for method_name, res in last_seed_results.items():
        angles_data[method_name] = {
            k_label: sa["cos_angles"]
            for k_label, sa in res["subspace_analysis"].items()
        }
    with open(angles_path, "w") as f:
        json.dump(angles_data, f, indent=2)
    print(f"Principal angles saved to {angles_path}")


if __name__ == "__main__":
    main()
