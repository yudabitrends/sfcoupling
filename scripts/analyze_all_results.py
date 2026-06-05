#!/usr/bin/env python3
"""
Comprehensive analysis of all benchmark results.

Reads results from:
  - results/baselines_multiseed/summary.json  (Ridge, MLP)
  - results/multivariate_methods/summary.json (RRR, PLS, Nuclear Norm)
  - results/kernel_spectral_regression/summary.json (KSR, Linear-OptShrink, KSR-NN ablation)
  - results/nn_mlp_twostage/summary.json (two-stage hybrid)

Produces:
  - Unified comparison table (all methods × DS1 test + DS2 external)
  - Statistical significance matrix
  - Rank/component analysis
  - Final verdict on spectral regularization hypothesis
"""
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        print(f"  [MISSING] {path}")
        return None
    with open(path) as f:
        return json.load(f)


def fmt(stats: Dict, key: str = "mean") -> str:
    """Format mean ± ci95 for display."""
    if stats is None:
        return "N/A"
    m = stats.get("mean", float("nan"))
    ci = stats.get("ci95", float("nan"))
    if np.isnan(ci):
        return f"{m:>8.4f}"
    return f"{m:>7.4f}±{ci:.4f}"


def main():
    base = Path("/home/users/ybi3/sfcoupling/results")

    print("=" * 90)
    print("  COMPREHENSIVE BENCHMARK ANALYSIS: Structure-Function Coupling Methods")
    print("=" * 90)
    print()

    # ---- Load all summaries ----
    bl = load_json(base / "baselines_multiseed" / "summary.json")
    mv = load_json(base / "multivariate_methods" / "summary.json")
    ksr = load_json(base / "kernel_spectral_regression" / "summary.json")
    ts = load_json(base / "nn_mlp_twostage" / "summary.json")

    # ---- Build unified table ----
    rows = []  # (name, d1_pc_r2, d2_pc_r2, d1_edge_r2, d2_edge_r2, category)

    if bl:
        n = bl.get("n_seeds", "?")
        print(f"  Baselines multi-seed: {n} seeds loaded")
        # Ridge at each PCA-k
        for rk, rv in bl.get("ridge", {}).items():
            rows.append((
                f"Ridge ({rk})",
                rv.get("pc_r2_mean_d1"),
                rv.get("pc_r2_mean_d2"),
                rv.get("edge_r2_mean_d1"),
                rv.get("edge_r2_mean_d2"),
                "Linear",
            ))
        # MLP
        rows.append((
            "MLP",
            bl.get("mlp", {}).get("pc_r2_mean_d1"),
            bl.get("mlp", {}).get("pc_r2_mean_d2"),
            bl.get("mlp", {}).get("edge_r2_mean_d1"),
            bl.get("mlp", {}).get("edge_r2_mean_d2"),
            "Nonlinear",
        ))

    if mv:
        n = mv.get("n_seeds", "?")
        print(f"  Multivariate methods: {n} seeds loaded")
        for method in ["rrr", "pls", "nuclear_norm"]:
            m = mv.get(method, {})
            label = {"rrr": "RRR", "pls": "PLS", "nuclear_norm": "Nuclear Norm"}[method]
            cat = "Spectral" if method in ("rrr", "nuclear_norm") else "Community std"
            rows.append((
                label,
                m.get("pc_r2_mean_d1"),
                m.get("pc_r2_mean_d2"),
                m.get("edge_r2_mean_d1"),
                m.get("edge_r2_mean_d2"),
                cat,
            ))

    if ksr:
        n = ksr.get("n_seeds", "?")
        print(f"  Kernel Spectral Regression: {n} seeds loaded")
        for method_key in ["ksr_optshrink", "linear_optshrink", "ksr_nuclear_norm"]:
            m = ksr.get(method_key, {})
            if not m:
                continue
            label_map = {
                "ksr_optshrink": "KSR (OptShrink)",
                "linear_optshrink": "Linear-OptShrink",
                "ksr_nuclear_norm": "KSR (Nuc.Norm)",
            }
            cat_map = {
                "ksr_optshrink": "Nonlinear+Spectral",
                "linear_optshrink": "Spectral (RMT)",
                "ksr_nuclear_norm": "Nonlinear+Spectral",
            }
            rows.append((
                label_map.get(method_key, method_key),
                m.get("pc_r2_mean_d1"),
                m.get("pc_r2_mean_d2"),
                m.get("edge_r2_mean_d1"),
                m.get("edge_r2_mean_d2"),
                cat_map.get(method_key, "Other"),
            ))

    if ts:
        n = ts.get("n_seeds", "?")
        print(f"  Two-stage NN+MLP: {n} seeds loaded")
        for method_key in ts:
            if method_key in ("n_seeds", "seeds", "paired_vs_mlp", "paired_vs_nn",
                              "nn_init_mlp_alpha"):
                continue
            m = ts[method_key]
            if not isinstance(m, dict):
                continue
            # Handle twostage format: {k20: {ds1_pc_r2: {mean,...}}}
            if "k20" in m:
                k20 = m["k20"]
                d1 = k20.get("ds1_pc_r2")
                d2 = k20.get("ds2_pc_r2")
                edge_d1 = m.get("edge_r2_d1")
                edge_d2 = m.get("edge_r2_d2")
                rows.append((
                    f"NN+MLP ({method_key})",
                    d1, d2, edge_d1, edge_d2,
                    "Hybrid",
                ))
            elif "pc_r2_mean_d1" in m:
                rows.append((
                    f"NN+MLP ({method_key})",
                    m.get("pc_r2_mean_d1"),
                    m.get("pc_r2_mean_d2"),
                    m.get("edge_r2_mean_d1"),
                    m.get("edge_r2_mean_d2"),
                    "Hybrid",
                ))

    if not rows:
        print("\n  [ERROR] No results found. Run the SLURM jobs first.")
        sys.exit(1)

    # ---- Print unified table ----
    print()
    print("=" * 90)
    print("  TABLE 1: All Methods Comparison (PC-space R² at k=20, Edge-space R²)")
    print("=" * 90)
    header = f"{'Method':<25s} {'Category':<18s} {'DS1 pc_R²':>16s} {'DS2 pc_R²':>16s} {'DS1 edge_R²':>16s}"
    print(header)
    print("-" * 90)

    # Sort by DS2 pc R² (external validation — the gold standard)
    def sort_key(row):
        d2 = row[2]
        if d2 is None:
            return -999
        return d2.get("mean", -999)

    rows_sorted = sorted(rows, key=sort_key, reverse=True)

    for name, d1_pc, d2_pc, d1_edge, d2_edge, cat in rows_sorted:
        print(f"  {name:<23s} {cat:<18s} {fmt(d1_pc):>16s} {fmt(d2_pc):>16s} {fmt(d1_edge):>16s}")

    # ---- Best method identification ----
    print()
    print("=" * 90)
    print("  KEY FINDINGS")
    print("=" * 90)

    best_d1 = max(rows, key=lambda r: r[1].get("mean", -999) if r[1] else -999)
    best_d2 = max(rows, key=lambda r: r[2].get("mean", -999) if r[2] else -999)
    print(f"  Best on DS1 (in-sample test):  {best_d1[0]} ({best_d1[5]}) "
          f"= {fmt(best_d1[1])}")
    print(f"  Best on DS2 (external val):    {best_d2[0]} ({best_d2[5]}) "
          f"= {fmt(best_d2[2])}")

    # ---- Generalization gap ----
    print()
    print("-" * 90)
    print("  GENERALIZATION GAP (DS1 pc_R² - DS2 pc_R²)")
    print("-" * 90)
    for name, d1_pc, d2_pc, d1_edge, d2_edge, cat in rows_sorted:
        if d1_pc and d2_pc:
            gap = d1_pc["mean"] - d2_pc["mean"]
            ratio = d2_pc["mean"] / d1_pc["mean"] if d1_pc["mean"] != 0 else float("nan")
            print(f"  {name:<25s}  gap={gap:>+.4f}  retention={ratio:>.1%}")

    # ---- PLS vs spectral methods ----
    print()
    print("-" * 90)
    print("  CLAIM CHECK: PLS (community standard) vs Spectral methods")
    print("-" * 90)
    pls_row = None
    spectral_rows = []
    for row in rows:
        if row[0] == "PLS":
            pls_row = row
        elif row[5] in ("Spectral", "Spectral (RMT)"):
            spectral_rows.append(row)

    if pls_row and spectral_rows:
        pls_d2 = pls_row[2]["mean"] if pls_row[2] else float("nan")
        for sr in spectral_rows:
            sr_d2 = sr[2]["mean"] if sr[2] else float("nan")
            delta = sr_d2 - pls_d2
            print(f"  {sr[0]:>25s} - PLS = {delta:>+.4f} on external validation")
    else:
        print("  (PLS or spectral results not available)")

    # ---- Nonlinear vs linear ----
    print()
    print("-" * 90)
    print("  CLAIM CHECK: Nonlinear methods vs best linear/spectral")
    print("-" * 90)
    linear_best_d2 = -999
    linear_best_name = ""
    for row in rows:
        if row[5] in ("Linear", "Spectral", "Spectral (RMT)", "Community std"):
            d2_val = row[2]["mean"] if row[2] else -999
            if d2_val > linear_best_d2:
                linear_best_d2 = d2_val
                linear_best_name = row[0]

    nonlinear_rows = [r for r in rows if r[5] in ("Nonlinear", "Nonlinear+Spectral", "Hybrid")]
    if linear_best_name and nonlinear_rows:
        print(f"  Best linear/spectral: {linear_best_name} (DS2 pc_R²={linear_best_d2:.4f})")
        for nr in nonlinear_rows:
            nr_d2 = nr[2]["mean"] if nr[2] else float("nan")
            delta = nr_d2 - linear_best_d2
            print(f"  {nr[0]:>25s}: DS2={nr_d2:.4f}  delta={delta:>+.4f}")
    else:
        print("  (Insufficient data for comparison)")

    # ---- Statistical comparisons ----
    print()
    print("-" * 90)
    print("  STATISTICAL COMPARISONS (from multivariate_methods)")
    print("-" * 90)
    stat_comp = load_json(base / "multivariate_methods" / "statistical_comparisons.json")
    if stat_comp:
        for dataset_label, comparisons in stat_comp.items():
            print(f"\n  {dataset_label}:")
            if isinstance(comparisons, list):
                for comp in comparisons:
                    a = comp.get("method_a", "?")
                    b = comp.get("method_b", "?")
                    t = comp.get("t_stat", float("nan"))
                    p = comp.get("p_value", float("nan"))
                    diff = comp.get("mean_diff", float("nan"))
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                    print(f"    {a} vs {b}: diff={diff:>+.4f}  t={t:>6.3f}  p={p:.4f} {sig}")
            elif isinstance(comparisons, dict):
                for pair_key, comp in comparisons.items():
                    if isinstance(comp, dict):
                        t = comp.get("t_stat", float("nan"))
                        p = comp.get("p_value", float("nan"))
                        diff = comp.get("mean_diff", float("nan"))
                        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                        print(f"    {pair_key}: diff={diff:>+.4f}  t={t:>6.3f}  p={p:.4f} {sig}")
    else:
        print("  (No statistical comparisons file found)")

    # ---- Rank / component analysis ----
    print()
    print("-" * 90)
    print("  EFFECTIVE DIMENSIONALITY")
    print("-" * 90)
    if mv:
        for method in ["rrr", "pls", "nuclear_norm"]:
            m = mv.get(method, {})
            ed = m.get("effective_dim")
            if ed:
                label = {"rrr": "RRR rank", "pls": "PLS components", "nuclear_norm": "NN eff. rank"}[method]
                print(f"  {label:<25s}: {ed['mean']:.1f} ± {ed.get('ci95', 0):.1f}")

    # ---- Permutation test for rank significance ----
    print()
    print("-" * 90)
    print("  RANK DIMENSION SIGNIFICANCE (RRR permutation test)")
    print("-" * 90)
    # Check per-seed files
    seed_files = sorted((base / "multivariate_methods").glob("seed_*.json"))
    if seed_files:
        last_seed = load_json(seed_files[-1])
        if last_seed and "rrr" in last_seed:
            perm = last_seed["rrr"].get("perm_test_rank_dims", [])
            if perm:
                n_sig = sum(1 for p in perm if p["significant"])
                print(f"  Tested up to rank {len(perm)}, {n_sig} significant dimensions (p<0.05)")
                for entry in perm:
                    sig = "*" if entry["significant"] else " "
                    print(f"    dim {entry['rank_dim']:2d}: "
                          f"incr_R²={entry['observed_incr_r2']:>+.6f}  "
                          f"p={entry['p_value']:.4f} {sig}")
            else:
                print("  No permutation test data found")
    else:
        print("  No per-seed files found")

    # ---- Absolute R² context ----
    print()
    print("=" * 90)
    print("  R² MAGNITUDE CONTEXT")
    print("=" * 90)
    print("  Note: R² values of ~2-6% on external validation are TYPICAL for")
    print("  multivariate GM→FNC prediction in neuroimaging. This reflects:")
    print("    - High dimensionality of FNC (1378+ edges)")
    print("    - Weak but reproducible structure-function coupling")
    print("    - Noise in both GM and FNC measurements")
    print("  The key question is RELATIVE performance between methods,")
    print("  not absolute R² magnitude.")

    # ---- Final verdict ----
    print()
    print("=" * 90)
    print("  FINAL VERDICT FOR NEUROIMAGE SUBMISSION")
    print("=" * 90)

    all_available = sum(1 for x in [bl, mv, ksr, ts] if x is not None)
    print(f"  Results available: {all_available}/4 experiment blocks")

    if best_d2[5] in ("Spectral", "Spectral (RMT)"):
        print("  ✓ Spectral regularization wins on external validation")
    elif best_d2[5] in ("Nonlinear", "Nonlinear+Spectral"):
        print("  ✗ Nonlinear method wins — challenges spectral-only narrative")
    else:
        print(f"  ? Best method category: {best_d2[5]} — nuanced result")

    print()
    print("  Analysis complete.")
    print("=" * 90)


if __name__ == "__main__":
    main()
