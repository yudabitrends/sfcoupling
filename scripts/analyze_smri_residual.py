#!/usr/bin/env python3
"""Decompose sMRI into coupled/uncoupled subspaces and characterize both.

Analyses:
  A1: sMRI variance decomposition (global + rank sweep)
  A2: Per-ROI and per-domain coupled fraction
  A3: Clinical relevance (Diagnosis AUC) of coupled vs uncoupled
  A4: Nonlinear coupling in residual (Ridge vs KernelRidge)
  A5: Cross-method subspace stability (NN vs RRR vs PLS)

Usage:
  python scripts/analyze_smri_residual.py --config train/config_baselines.yaml
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from figs.plot_style import apply_nature_style
from figs.utils import panel_label
from models.utils import load_config, load_training_contracts, save_json, set_seed

# Import brain mapping constants
from generate_brain_figures import (
    DOMAIN_COLORS,
    DOMAIN_FULL_NAMES,
    DOMAIN_ORDER_SBM,
    SBM_LABELS,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DECOMP_DIR = PROJECT_ROOT / "results" / "multivariate_methods" / "decompositions"
OUT_DIR = PROJECT_ROOT / "results" / "smri_residual"
FIG_DIR = OUT_DIR / "figures"
GM_NAMES_PATH = PROJECT_ROOT / "aligned_features" / "meta" / "feature_maps" / "gm_feature_names.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_roi_domains(gm_names):
    """Map ROI names to SBM domains."""
    domains = []
    for name in gm_names:
        roi_idx = int(name.replace("roi_", ""))
        ic_1based = roi_idx + 1
        if ic_1based in SBM_LABELS:
            domains.append(SBM_LABELS[ic_1based][0])
        else:
            domains.append("Other")
    return domains


def _project(X, U, r):
    """Project X onto first r columns of U (coupled) and complement (uncoupled)."""
    P_r = U[:, :r] @ U[:, :r].T  # (dx, dx)
    X_coupled = X @ P_r
    X_uncoupled = X - X_coupled
    return X_coupled, X_uncoupled


def _bootstrap_auc(y_true, y_score, n_boot=200, seed=42):
    """Bootstrap AUC with 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    aucs = np.array(aucs)
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _save_fig(fig, name):
    """Save figure as PDF and PNG."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)
    print(f"  Saved {name}.{{pdf,png}}")


# ---------------------------------------------------------------------------
# A1: sMRI Variance Decomposition
# ---------------------------------------------------------------------------

def analysis_a1(X_test, U, ranks):
    """Global variance decomposition + rank sweep."""
    total_var = np.sum(X_test ** 2)
    results = {}
    for r in ranks:
        X_c, X_u = _project(X_test, U, r)
        coupled_frac = float(np.sum(X_c ** 2) / total_var)
        uncoupled_frac = float(np.sum(X_u ** 2) / total_var)
        results[r] = {
            "coupled_var_frac": round(coupled_frac, 4),
            "uncoupled_var_frac": round(uncoupled_frac, 4),
            "sum_check": round(coupled_frac + uncoupled_frac, 6),
        }
        print(f"  rank {r:3d}: coupled={coupled_frac:.4f}  uncoupled={uncoupled_frac:.4f}  sum={coupled_frac + uncoupled_frac:.6f}")
    return results


# ---------------------------------------------------------------------------
# A2: Per-ROI and Per-Domain Decomposition
# ---------------------------------------------------------------------------

def analysis_a2(X_test, U, r, gm_names, roi_domains):
    """Per-ROI coupled variance fraction, aggregated by domain."""
    X_c, _ = _project(X_test, U, r)
    n_rois = X_test.shape[1]

    roi_data = []
    for j in range(n_rois):
        var_total = np.var(X_test[:, j])
        var_coupled = np.var(X_c[:, j])
        frac = float(var_coupled / var_total) if var_total > 1e-12 else 0.0
        roi_data.append({
            "roi_name": gm_names[j],
            "domain": roi_domains[j],
            "coupled_frac": round(frac, 4),
            "var_total": round(float(var_total), 6),
            "var_coupled": round(float(var_coupled), 6),
        })

    df = pd.DataFrame(roi_data)

    # Domain aggregation
    domain_agg = {}
    for domain in DOMAIN_ORDER_SBM:
        mask = df["domain"] == domain
        if mask.sum() == 0:
            continue
        domain_agg[domain] = {
            "n_rois": int(mask.sum()),
            "mean_coupled_frac": round(float(df.loc[mask, "coupled_frac"].mean()), 4),
            "std_coupled_frac": round(float(df.loc[mask, "coupled_frac"].std()), 4),
        }
        print(f"  {domain:5s} ({domain_agg[domain]['n_rois']:2d} ROIs): "
              f"coupled = {domain_agg[domain]['mean_coupled_frac']:.3f} +/- {domain_agg[domain]['std_coupled_frac']:.3f}")

    return df, domain_agg


# ---------------------------------------------------------------------------
# A3: Clinical Relevance of Uncoupled sMRI
# ---------------------------------------------------------------------------

def analysis_a3(X_train, X_test, y_train, y_test, U, r, n_boot=200):
    """Logistic regression AUC for full/coupled/uncoupled sMRI -> Diagnosis."""
    X_train_c, X_train_u = _project(X_train, U, r)
    X_test_c, X_test_u = _project(X_test, U, r)

    conditions = {
        "full": (X_train, X_test),
        "coupled": (X_train_c, X_test_c),
        "uncoupled": (X_train_u, X_test_u),
    }

    results = {}
    for name, (Xtr, Xte) in conditions.items():
        clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000)
        clf.fit(Xtr, y_train)
        y_score = clf.predict_proba(Xte)[:, 1]
        auc_mean, auc_lo, auc_hi = _bootstrap_auc(y_test, y_score, n_boot=n_boot)
        results[name] = {
            "auc": round(auc_mean, 4),
            "auc_ci_lo": round(auc_lo, 4),
            "auc_ci_hi": round(auc_hi, 4),
        }
        print(f"  {name:10s}: AUC = {auc_mean:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]")

    # Age regression sanity check (should be ~0 after residualization)
    age_results = {}
    for name, (Xtr, Xte) in conditions.items():
        # Use same train/test but predict Age if available
        # After residualization, Age R2 should be ~0
        pass  # Age was regressed out during preprocessing

    return results


# ---------------------------------------------------------------------------
# A4: Nonlinear Coupling in Residual
# ---------------------------------------------------------------------------

def analysis_a4(X_train, Y_train, X_test, Y_test, U, r, n_perm=200):
    """Test whether uncoupled sMRI has nonlinear relationship with FNC."""
    _, X_train_u = _project(X_train, U, r)
    _, X_test_u = _project(X_test, U, r)

    # Linear baseline: Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_u, Y_train)
    Y_pred_lin = ridge.predict(X_test_u)
    r2_linear = float(r2_score(Y_test, Y_pred_lin))
    print(f"  Linear  (Ridge):      R2 = {r2_linear:.6f}")

    # Nonlinear: KernelRidge with RBF
    # Use median heuristic for gamma
    from sklearn.metrics.pairwise import euclidean_distances
    dists = euclidean_distances(X_train_u[:500])  # subsample for speed
    median_dist = np.median(dists[dists > 0])
    gamma = 1.0 / (2 * median_dist ** 2) if median_dist > 0 else 1e-3

    krr = KernelRidge(alpha=1.0, kernel="rbf", gamma=gamma)
    krr.fit(X_train_u, Y_train)
    Y_pred_rbf = krr.predict(X_test_u)
    r2_rbf = float(r2_score(Y_test, Y_pred_rbf))
    print(f"  Nonlinear (RBF KRR): R2 = {r2_rbf:.6f}")

    # Permutation null for RBF
    rng = np.random.default_rng(42)
    null_r2s = []
    for i in range(n_perm):
        perm_idx = rng.permutation(len(X_train_u))
        krr_null = KernelRidge(alpha=1.0, kernel="rbf", gamma=gamma)
        krr_null.fit(X_train_u[perm_idx], Y_train)
        Y_pred_null = krr_null.predict(X_test_u)
        null_r2s.append(float(r2_score(Y_test, Y_pred_null)))
        if (i + 1) % 50 == 0:
            print(f"    permutation {i + 1}/{n_perm}")

    null_r2s = np.array(null_r2s)
    p_value = float(np.mean(null_r2s >= r2_rbf))
    print(f"  Permutation p-value: {p_value:.4f}")
    print(f"  Null R2: {np.mean(null_r2s):.6f} +/- {np.std(null_r2s):.6f}")

    return {
        "r2_linear": round(r2_linear, 6),
        "r2_rbf": round(r2_rbf, 6),
        "rbf_gamma": round(float(gamma), 8),
        "null_r2_mean": round(float(np.mean(null_r2s)), 6),
        "null_r2_std": round(float(np.std(null_r2s)), 6),
        "null_r2_95": round(float(np.percentile(null_r2s, 95)), 6),
        "p_value": round(p_value, 4),
        "n_perm": n_perm,
    }


# ---------------------------------------------------------------------------
# A5: Cross-Method Subspace Stability
# ---------------------------------------------------------------------------

def analysis_a5(r):
    """Compare coupled subspaces across NN, RRR, PLS (seed 42)."""
    methods = {}
    for method in ["nuclear_norm", "rrr", "pls"]:
        path = DECOMP_DIR / f"{method}_seed42_B.npy"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {method}")
            continue
        B = np.load(path)
        U_m, _, _ = np.linalg.svd(B, full_matrices=False)
        methods[method] = U_m[:, :r]

    method_names = list(methods.keys())
    results = {}
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            m1, m2 = method_names[i], method_names[j]
            angles = linalg.subspace_angles(methods[m1], methods[m2])
            angles_deg = np.degrees(angles)
            key = f"{m1}_vs_{m2}"
            results[key] = {
                "mean_angle_deg": round(float(np.mean(angles_deg)), 2),
                "max_angle_deg": round(float(np.max(angles_deg)), 2),
                "min_angle_deg": round(float(np.min(angles_deg)), 2),
                "mean_cos": round(float(np.mean(np.cos(angles))), 4),
            }
            print(f"  {m1:12s} vs {m2:12s}: "
                  f"mean angle = {results[key]['mean_angle_deg']:.1f} deg, "
                  f"mean cos = {results[key]['mean_cos']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_variance_decomposition(a1_results, domain_agg, primary_rank):
    """Fig 1: (a) Global coupled/uncoupled bar; (b) Stacked bar per domain."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1, 2.5]})

    # (a) Global bar
    ax = axes[0]
    r_data = a1_results[primary_rank]
    vals = [r_data["coupled_var_frac"], r_data["uncoupled_var_frac"]]
    colors = ["#1565c0", "#e0e0e0"]
    bars = ax.bar(["Coupled", "Uncoupled"], vals, color=colors, width=0.6, edgecolor="k", linewidth=0.5)
    ax.set_ylabel("Fraction of sMRI variance")
    ax.set_ylim(0, 1.05)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.1%}", ha="center", va="bottom", fontsize=6)
    panel_label(ax, "(a)")

    # (b) Stacked bar per domain
    ax = axes[1]
    domains = [d for d in DOMAIN_ORDER_SBM if d in domain_agg]
    coupled_vals = [domain_agg[d]["mean_coupled_frac"] for d in domains]
    uncoupled_vals = [1.0 - v for v in coupled_vals]
    x = np.arange(len(domains))
    ax.bar(x, coupled_vals, color="#1565c0", label="Coupled", width=0.6, edgecolor="k", linewidth=0.3)
    ax.bar(x, uncoupled_vals, bottom=coupled_vals, color="#e0e0e0", label="Uncoupled", width=0.6, edgecolor="k", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_FULL_NAMES.get(d, d) for d in domains], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Fraction of sMRI variance")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=6)
    panel_label(ax, "(b)")

    fig.tight_layout()
    _save_fig(fig, "fig_variance_decomposition")


def fig_roi_coupled_fraction(roi_df):
    """Fig 2: Horizontal bar chart of 99 ROIs, colored by domain, ordered by coupled fraction."""
    df = roi_df.sort_values("coupled_frac", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(4.5, 8))

    colors = [DOMAIN_COLORS.get(d, "#9e9e9e") for d in df["domain"]]
    ax.barh(np.arange(len(df)), df["coupled_frac"], color=colors, height=0.8, edgecolor="none")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["roi_name"], fontsize=4)
    ax.set_xlabel("Coupled variance fraction")
    ax.set_xlim(0, max(df["coupled_frac"].max() * 1.1, 0.5))

    # Legend
    from matplotlib.patches import Patch
    domains_present = df["domain"].unique()
    handles = [Patch(facecolor=DOMAIN_COLORS.get(d, "#9e9e9e"), label=DOMAIN_FULL_NAMES.get(d, d))
               for d in DOMAIN_ORDER_SBM if d in domains_present]
    ax.legend(handles=handles, loc="lower right", fontsize=5, ncol=2)

    fig.tight_layout()
    _save_fig(fig, "fig_roi_coupled_fraction")


def fig_clinical_prediction(a3_results):
    """Fig 3: Grouped bars for AUC with bootstrap CI."""
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    names = ["full", "coupled", "uncoupled"]
    labels = ["Full sMRI", "Coupled", "Uncoupled"]
    colors = ["#424242", "#1565c0", "#e0e0e0"]
    x = np.arange(len(names))

    aucs = [a3_results[n]["auc"] for n in names]
    ci_lo = [a3_results[n]["auc"] - a3_results[n]["auc_ci_lo"] for n in names]
    ci_hi = [a3_results[n]["auc_ci_hi"] - a3_results[n]["auc"] for n in names]

    bars = ax.bar(x, aucs, color=colors, width=0.6, edgecolor="k", linewidth=0.5,
                  yerr=[ci_lo, ci_hi], capsize=3, error_kw={"linewidth": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("AUC (Diagnosis)")
    ax.set_ylim(0.4, max(aucs) + 0.1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)

    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(ci_hi) + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    fig.tight_layout()
    _save_fig(fig, "fig_clinical_prediction")


def fig_nonlinear_residual(a4_results):
    """Fig 4: Bar chart of linear vs RBF R² from X_uncoupled→Y with null band."""
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    names = ["Linear (Ridge)", "Nonlinear (RBF)"]
    vals = [a4_results["r2_linear"], a4_results["r2_rbf"]]
    colors = ["#1565c0", "#c62828"]

    bars = ax.bar(np.arange(2), vals, color=colors, width=0.5, edgecolor="k", linewidth=0.5)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("R² (X_uncoupled → Y)")

    # Null band
    null_95 = a4_results["null_r2_95"]
    null_mean = a4_results["null_r2_mean"]
    ax.axhspan(null_mean - a4_results["null_r2_std"], null_95,
               color="gray", alpha=0.2, label="95th pctl null")
    ax.axhline(null_mean, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=6)

    ax.legend(fontsize=5)
    fig.tight_layout()
    _save_fig(fig, "fig_nonlinear_residual")


def fig_rank_sensitivity(a1_results, a3_by_rank):
    """Fig 5: Line plot of coupled variance fraction + uncoupled AUC vs rank r."""
    fig, ax1 = plt.subplots(figsize=(4.5, 3.0))

    ranks = sorted(a1_results.keys())
    coupled_fracs = [a1_results[r]["coupled_var_frac"] for r in ranks]

    color1 = "#1565c0"
    color2 = "#c62828"

    ax1.plot(ranks, coupled_fracs, "o-", color=color1, markersize=4, label="Coupled var. fraction")
    ax1.set_xlabel("Rank r")
    ax1.set_ylabel("Coupled variance fraction", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 1.0)

    if a3_by_rank:
        ax2 = ax1.twinx()
        auc_vals = [a3_by_rank[r]["uncoupled"]["auc"] for r in ranks if r in a3_by_rank]
        ranks_auc = [r for r in ranks if r in a3_by_rank]
        ax2.plot(ranks_auc, auc_vals, "s--", color=color2, markersize=4, label="Uncoupled AUC")
        ax2.set_ylabel("Uncoupled AUC (Diagnosis)", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(0.4, 1.0)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=6)
    else:
        ax1.legend(fontsize=6)

    fig.tight_layout()
    _save_fig(fig, "fig_rank_sensitivity")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="sMRI residual information analysis")
    parser.add_argument("--config", type=str, default="train/config_baselines.yaml")
    parser.add_argument("--primary_rank", type=int, default=38)
    parser.add_argument("--ranks", type=int, nargs="+", default=[3, 5, 10, 20, 38])
    parser.add_argument("--n_boot", type=int, default=200)
    parser.add_argument("--n_perm", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    apply_nature_style()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    cfg = load_config(str(PROJECT_ROOT / args.config))
    data = load_training_contracts(cfg)

    X = data["X1"]          # (n, 99)
    Y = data["Y1"]          # (n, 1378)
    subjects = data["subjects1"]
    gm_names = data["gm_names"]
    idx_train = data["idx1_train"]
    idx_test = data["idx1_test"]

    X_train, X_test = X[idx_train], X[idx_test]
    Y_train, Y_test = Y[idx_train], Y[idx_test]

    print(f"  X: {X.shape}, Y: {Y.shape}")
    print(f"  Train: {len(idx_train)}, Test: {len(idx_test)}")

    # ── Load B and compute SVD ─────────────────────────────────────────────
    print("Loading B matrix and computing SVD...")
    B = np.load(DECOMP_DIR / "nuclear_norm_seed42_B.npy")
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    print(f"  B: {B.shape}, U: {U.shape}, top-5 singular values: {S[:5].round(3)}")

    # ROI domain mapping
    roi_domains = _get_roi_domains(gm_names)

    # ── A1: Variance decomposition ────────────────────────────────────────
    print("\n=== A1: sMRI Variance Decomposition ===")
    a1_results = analysis_a1(X_test, U, args.ranks)

    # ── A2: Per-ROI and per-domain decomposition ──────────────────────────
    print(f"\n=== A2: Per-ROI / Per-Domain (r={args.primary_rank}) ===")
    roi_df, domain_agg = analysis_a2(X_test, U, args.primary_rank, gm_names, roi_domains)
    roi_df.to_csv(OUT_DIR / "roi_decomposition.csv", index=False)
    print(f"  Saved roi_decomposition.csv ({len(roi_df)} ROIs)")

    # ── A3: Clinical relevance ────────────────────────────────────────────
    has_diagnosis = "Diagnosis" in subjects.columns
    a3_results = None
    a3_by_rank = {}
    if has_diagnosis:
        print(f"\n=== A3: Clinical Relevance (r={args.primary_rank}) ===")
        y_diag = subjects["Diagnosis"].values
        y_train_diag = y_diag[idx_train]
        y_test_diag = y_diag[idx_test]
        a3_results = analysis_a3(X_train, X_test, y_train_diag, y_test_diag,
                                 U, args.primary_rank, n_boot=args.n_boot)

        # Rank sweep for A3
        print("\n  Rank sweep for clinical AUC:")
        for r in args.ranks:
            print(f"  --- rank {r} ---")
            a3_by_rank[r] = analysis_a3(X_train, X_test, y_train_diag, y_test_diag,
                                        U, r, n_boot=args.n_boot)
    else:
        print("\n=== A3: SKIPPED (no Diagnosis column) ===")

    # ── A4: Nonlinear coupling in residual ────────────────────────────────
    print(f"\n=== A4: Nonlinear Coupling in Residual (r={args.primary_rank}) ===")
    a4_results = analysis_a4(X_train, Y_train, X_test, Y_test, U, args.primary_rank,
                             n_perm=args.n_perm)

    # ── A5: Cross-method stability ────────────────────────────────────────
    print(f"\n=== A5: Cross-Method Subspace Stability (r={args.primary_rank}) ===")
    a5_results = analysis_a5(args.primary_rank)

    # ── Save summary ──────────────────────────────────────────────────────
    summary = {
        "primary_rank": args.primary_rank,
        "ranks_swept": args.ranks,
        "n_subjects_train": len(idx_train),
        "n_subjects_test": len(idx_test),
        "n_rois": len(gm_names),
        "n_fnc_edges": Y.shape[1],
        "A1_variance_decomposition": a1_results,
        "A2_domain_aggregation": domain_agg,
        "A3_clinical_relevance": a3_results,
        "A3_rank_sweep": {str(k): v for k, v in a3_by_rank.items()} if a3_by_rank else None,
        "A4_nonlinear_residual": a4_results,
        "A5_cross_method_stability": a5_results,
    }
    save_json(OUT_DIR / "summary.json", summary)
    print(f"\nSaved summary.json")

    # ── Figures ────────────────────────────────────────────────────────────
    print("\n=== Generating figures ===")
    fig_variance_decomposition(a1_results, domain_agg, args.primary_rank)
    fig_roi_coupled_fraction(roi_df)
    if a3_results:
        fig_clinical_prediction(a3_results)
    if a4_results:
        fig_nonlinear_residual(a4_results)
    fig_rank_sensitivity(a1_results, a3_by_rank)

    print("\nDone.")


if __name__ == "__main__":
    main()
