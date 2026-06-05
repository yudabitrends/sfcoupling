#!/usr/bin/env python3
"""
Bootstrap stability analysis for singular values of cross-covariance X'Y/sqrt(n).

Performs 200 bootstrap resamples of the training set, computes SVD each time,
and reports mean, std, CV, and 95% CI for each singular value.

Usage:
    python train/run_bootstrap_sv.py \
        --config train/config_baselines.yaml \
        --n_boot 200 \
        --max_sv 30 \
        --out_dir results/bootstrap_sv
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.utils import load_config, load_training_contracts, save_json, set_seed


def bootstrap_sv_stability(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    n_boot: int = 200,
    max_sv: int = 30,
    seed: int = 42,
) -> Dict:
    """Bootstrap resample train set and compute SVD of X'Y/sqrt(n) each time."""
    rng = np.random.RandomState(seed)
    n, dx = X_train.shape
    dy = Y_train.shape[1]
    n_sv = min(max_sv, dx, dy, n)

    X64 = X_train.astype(np.float64)
    Y64 = Y_train.astype(np.float64)

    # Full-sample SVs
    C_full = X64.T @ Y64 / np.sqrt(n)
    _, sv_full, _ = np.linalg.svd(C_full, full_matrices=False)
    sv_full = sv_full[:n_sv]

    # Bootstrap
    sv_boot = np.zeros((n_boot, n_sv), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        Xb = X64[idx]
        Yb = Y64[idx]
        Cb = Xb.T @ Yb / np.sqrt(n)
        _, svb, _ = np.linalg.svd(Cb, full_matrices=False)
        sv_boot[b] = svb[:n_sv]
        if (b + 1) % 50 == 0:
            print(f"  Bootstrap {b+1}/{n_boot} done", flush=True)

    # Statistics
    sv_mean = np.mean(sv_boot, axis=0)
    sv_std = np.std(sv_boot, axis=0, ddof=1)
    sv_cv = sv_std / np.where(sv_mean > 0, sv_mean, 1.0)
    ci_lo = np.percentile(sv_boot, 2.5, axis=0)
    ci_hi = np.percentile(sv_boot, 97.5, axis=0)

    results = {
        "n_train": int(n),
        "dx": int(dx),
        "dy": int(dy),
        "n_boot": n_boot,
        "n_sv": int(n_sv),
        "seed": seed,
        "full_sample_sv": sv_full.tolist(),
        "bootstrap_mean": sv_mean.tolist(),
        "bootstrap_std": sv_std.tolist(),
        "bootstrap_cv": sv_cv.tolist(),
        "ci95_lo": ci_lo.tolist(),
        "ci95_hi": ci_hi.tolist(),
    }
    return results, sv_boot


def plot_sv_spectrum(results: Dict, sv_boot: np.ndarray, out_path: Path) -> None:
    """Plot SV spectrum with 95% CI shading."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_sv = results["n_sv"]
    x = np.arange(1, n_sv + 1)
    full_sv = np.array(results["full_sample_sv"])
    ci_lo = np.array(results["ci95_lo"])
    ci_hi = np.array(results["ci95_hi"])
    cv = np.array(results["bootstrap_cv"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: SV spectrum with CI
    ax = axes[0]
    ax.fill_between(x, ci_lo, ci_hi, alpha=0.3, color="steelblue", label="95% CI")
    ax.plot(x, full_sv, "o-", color="steelblue", markersize=4, label="Full sample")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title("Cross-covariance SV spectrum")
    ax.legend()
    ax.set_xlim(0.5, n_sv + 0.5)

    # Right: CV per SV
    ax = axes[1]
    ax.bar(x, cv * 100, color="coral", alpha=0.7)
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("CV (%)")
    ax.set_title("Bootstrap coefficient of variation")
    ax.set_xlim(0.5, n_sv + 0.5)
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.5, label="5% threshold")
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap SV stability analysis")
    parser.add_argument("--config", type=str, default="train/config_baselines.yaml")
    parser.add_argument("--n_boot", type=int, default=200)
    parser.add_argument("--max_sv", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="results/bootstrap_sv")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BOOTSTRAP SV STABILITY ANALYSIS")
    print(f"  n_boot:  {args.n_boot}")
    print(f"  max_sv:  {args.max_sv}")
    print(f"  output:  {out_dir}")
    print("=" * 60)

    # Load data
    data = load_training_contracts(cfg)
    idx_train = data["idx1_train"]
    X_train = data["X1"][idx_train]
    Y_train = data["Y1"][idx_train]
    print(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}")

    # Run bootstrap
    t0 = time.time()
    results, sv_boot = bootstrap_sv_stability(
        X_train, Y_train,
        n_boot=args.n_boot,
        max_sv=args.max_sv,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f"\n  Bootstrap completed in {elapsed:.1f}s")

    # Save results
    save_json(out_dir / "sv_stats.json", results)
    np.save(out_dir / "sv_boot_matrix.npy", sv_boot)

    # Print summary
    print(f"\n{'SV':>4s} {'Full':>10s} {'Mean':>10s} {'Std':>10s} {'CV%':>8s} {'95% CI':>20s}")
    print("-" * 65)
    for i in range(results["n_sv"]):
        print(f"{i+1:>4d} {results['full_sample_sv'][i]:>10.4f} "
              f"{results['bootstrap_mean'][i]:>10.4f} "
              f"{results['bootstrap_std'][i]:>10.4f} "
              f"{results['bootstrap_cv'][i]*100:>7.2f}% "
              f"[{results['ci95_lo'][i]:.4f}, {results['ci95_hi'][i]:.4f}]")

    # Plot
    plot_sv_spectrum(results, sv_boot, out_dir / "sv_spectrum_with_ci.png")

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
