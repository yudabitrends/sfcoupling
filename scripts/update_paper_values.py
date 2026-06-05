#!/usr/bin/env python3
"""
Read all summary.json files and print formatted LaTeX values for main.tex.

Usage:
    python scripts/update_paper_values.py
"""
import json
from pathlib import Path

import numpy as np

BASE = Path("/home/users/ybi3/sfcoupling")


def fmt(mean, std, decimals=4):
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def print_ukb_table():
    print("=" * 80)
    print("UKB TABLE (Tab:results_ukb) — PC-R² at k=20")
    print("=" * 80)

    # Baselines
    with open(BASE / "results/ukb/baselines_multiseed/summary.json") as f:
        bl = json.load(f)

    r = bl["ridge"]["pca_k20"]
    print(f"Ridge & {fmt(r['pc_r2_mean_d1']['mean'], r['pc_r2_mean_d1']['std'])} "
          f"& {fmt(r['pc_r2_mean_d2']['mean'], r['pc_r2_mean_d2']['std'])} "
          f"& {fmt(r['edge_r2_mean_d1']['mean'], r['edge_r2_mean_d1']['std'])} "
          f"& {fmt(r['edge_r2_mean_d2']['mean'], r['edge_r2_mean_d2']['std'])} \\\\")

    m = bl["mlp"]
    print(f"MLP & {fmt(m['pc_r2_mean_d1']['mean'], m['pc_r2_mean_d1']['std'])} "
          f"& {fmt(m['pc_r2_mean_d2']['mean'], m['pc_r2_mean_d2']['std'])} "
          f"& {fmt(m['edge_r2_mean_d1']['mean'], m['edge_r2_mean_d1']['std'])} "
          f"& {fmt(m['edge_r2_mean_d2']['mean'], m['edge_r2_mean_d2']['std'])} \\\\")

    # Multivariate
    with open(BASE / "results/ukb/multivariate_methods/summary.json") as f:
        mv = json.load(f)

    for name, label in [("rrr", "RRR"), ("pls", "PLS"), ("nuclear_norm", "Nuclear Norm")]:
        d = mv[name]
        print(f"{label} & {fmt(d['pc_r2_mean_d1']['mean'], d['pc_r2_mean_d1']['std'])} "
              f"& {fmt(d['pc_r2_mean_d2']['mean'], d['pc_r2_mean_d2']['std'])} "
              f"& {fmt(d['edge_r2_mean_d1']['mean'], d['edge_r2_mean_d1']['std'])} "
              f"& {fmt(d['edge_r2_mean_d2']['mean'], d['edge_r2_mean_d2']['std'])} \\\\")

    # Linear-OptShrink from KSR seeds
    ksr_dir = BASE / "results/ukb/kernel_spectral_regression"
    ksr_seeds = sorted(ksr_dir.glob("seed_*.json"))
    d1_pc, d2_pc, d1_e, d2_e = [], [], [], []
    for p in ksr_seeds:
        with open(p) as f:
            d = json.load(f)
        lo = d["linear_optshrink"]
        d1_pc.append(lo["dataset1_test"]["pc_r2_by_k"]["k20"]["pc_r2_mean"])
        d2_pc.append(lo["dataset2_external"]["pc_r2_by_k"]["k20"]["pc_r2_mean"])
        d1_e.append(lo["dataset1_test"]["edge_r2"]["r2_global"])
        d2_e.append(lo["dataset2_external"]["edge_r2"]["r2_global"])
    print(f"Linear-OptShrink ({len(ksr_seeds)} seeds) "
          f"& {fmt(np.mean(d1_pc), np.std(d1_pc))} "
          f"& {fmt(np.mean(d2_pc), np.std(d2_pc))} "
          f"& {fmt(np.mean(d1_e), np.std(d1_e))} "
          f"& {fmt(np.mean(d2_e), np.std(d2_e))} \\\\")

    # NN-Init MLP
    with open(BASE / "results/ukb/nn_mlp_twostage/summary.json") as f:
        ts = json.load(f)
    ni = ts["nn_init_mlp"]
    print(f"NN-Init MLP & {fmt(ni['k20']['ds1_pc_r2']['mean'], ni['k20']['ds1_pc_r2']['std'])} "
          f"& {fmt(ni['k20']['ds2_pc_r2']['mean'], ni['k20']['ds2_pc_r2']['std'])} "
          f"& {fmt(ni['edge_r2_d1']['mean'], ni['edge_r2_d1']['std'])} "
          f"& {fmt(ni['edge_r2_d2']['mean'], ni['edge_r2_d2']['std'])} \\\\")


def print_subspace_values():
    print("\n" + "=" * 80)
    print("SUBSPACE OVERLAP VALUES (Sec:results_subspace)")
    print("=" * 80)

    stats_path = BASE / "results/subspace_analysis/subspace_stats.json"
    if not stats_path.exists():
        print(f"  [NOT FOUND] {stats_path}")
        print("  Run: sbatch scripts/run_subspace_analysis.sh")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    for method, res in sorted(stats.items()):
        sa = res.get("subspace_analysis", {}).get("k=20", {})
        if "overlap_mean" in sa:
            print(f"  {method}: O = {sa['overlap_mean']:.3f} ± {sa['overlap_std']:.3f}  "
                  f"(mean p = {sa.get('p_value_mean', sa.get('p_value', 'N/A'))})")
        elif "subspace_overlap" in sa:
            print(f"  {method}: O = {sa['subspace_overlap']:.3f}  "
                  f"(null = {sa['null_mean']:.4f} ± {sa['null_std']:.4f}, "
                  f"p = {sa['p_value']:.4f})")


if __name__ == "__main__":
    print_ukb_table()
    print_subspace_values()
