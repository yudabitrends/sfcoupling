#!/usr/bin/env python3
"""
Generate paper-ready figures for NeuroImage submission.

Reads results from all benchmark directories and bootstrap SV analysis.
Produces PDF/PNG figures in figures/ directory.

Figures:
  1. SV spectrum with bootstrap 95% CI
  2. Method comparison bar chart (DS1 test + DS2 external, SZ + UKB)
  3. Residualization ablation (from diagnostic analysis Direction A)
  4. Per-PC R² breakdown heatmap (top 20 PCs × methods)
  5. SZ vs UKB method ranking comparison
  6. Subspace analysis (principal angles + geometry vs amplitude)

Usage:
    python scripts/generate_paper_figures.py
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from figs.plot_style import apply_nature_style
from figs.utils import (
    panel_label,
    adjust_annotations,
    style_axes,
    style_colorbar,
    add_panel_title,
    add_baseline,
    draw_grouped_bars,
    draw_lollipop_series,
    draw_stat_heatmap,
    draw_method_scatter,
    add_identity_line,
    CMAP_STAT_DIVERGING,
    CMAP_MAGNITUDE,
    CMAP_HEAT,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_NEUTRAL,
    COLOR_DARK,
    COLOR_ACCENT,
    COLOR_HIGHLIGHT,
    COLOR_WARM,
    COLOR_COOL,
    METHOD_COLORS,
    METHOD_MARKERS,
    METHOD_NAME_MAP,
    TEXT_MUTED,
    FIG_W_DOUBLE,
)

BASE = Path("/home/users/ybi3/sfcoupling")
RESULTS = BASE / "results"
FIG_DIR = BASE / "paper" / "standalone" / "figure"


def save_panel(fig, fig_name: str, panel_name: str):
    """Save a single panel PDF into figures/{fig_name}/{panel_name}.pdf"""
    d = FIG_DIR / fig_name
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{panel_name}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fig_name}/{panel_name} saved")


def save_main_figure(fig, figure_idx: int):
    """Save a reviewer-facing main figure directly into paper/standalone/figure."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"figure{figure_idx}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure{figure_idx}.pdf saved")


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        print(f"  [SKIP] {path}")
        return None
    with open(path) as f:
        return json.load(f)


# ---- Helper: extract pc_r2 mean±ci from different summary formats ----

def extract_method_metrics_baselines(summary: Dict) -> Dict[str, Dict]:
    """Extract from baselines_multiseed format: {method: {pc_r2_mean_d1: {mean,std,ci95}, ...}}"""
    out = {}
    if summary is None:
        return out
    for method_key in ["ridge", "mlp"]:
        m = summary.get(method_key, {})
        d1 = m.get("pc_r2_mean_d1", {})
        d2 = m.get("pc_r2_mean_d2", {})
        if not d1 and "pca_k20" in m:
            d1 = m["pca_k20"].get("pc_r2_mean_d1", {})
            d2 = m["pca_k20"].get("pc_r2_mean_d2", {})
        label = {"ridge": "Ridge", "mlp": "MLP"}[method_key]
        out[label] = {"d1_mean": d1.get("mean"), "d1_ci": d1.get("ci95"),
                      "d2_mean": d2.get("mean"), "d2_ci": d2.get("ci95")}
    return out


def extract_method_metrics_mv(summary: Dict) -> Dict[str, Dict]:
    """Extract from multivariate_methods format."""
    out = {}
    if summary is None:
        return out
    for method_key, label in [("rrr", "RRR"), ("pls", "PLS"), ("nuclear_norm", "Nuclear Norm")]:
        m = summary.get(method_key, {})
        d1 = m.get("pc_r2_mean_d1", {})
        d2 = m.get("pc_r2_mean_d2", {})
        out[label] = {"d1_mean": d1.get("mean"), "d1_ci": d1.get("ci95"),
                      "d2_mean": d2.get("mean"), "d2_ci": d2.get("ci95")}
    return out


def extract_method_metrics_ksr(summary: Dict) -> Dict[str, Dict]:
    """Extract from kernel_spectral_regression format."""
    out = {}
    if summary is None:
        return out
    for method_key, label in [("linear_optshrink", "OptShrink")]:
        m = summary.get(method_key, {})
        d1 = m.get("pc_r2_mean_d1", {})
        d2 = m.get("pc_r2_mean_d2", {})
        out[label] = {"d1_mean": d1.get("mean"), "d1_ci": d1.get("ci95"),
                      "d2_mean": d2.get("mean"), "d2_ci": d2.get("ci95")}
    return out


def extract_method_metrics_twostage(summary: Dict) -> Dict[str, Dict]:
    """Extract from nn_mlp_twostage format: {method: {k20: {ds1_pc_r2: {mean,...}}}}"""
    out = {}
    if summary is None:
        return out
    for method_key, label in [("nn_init_mlp", "NN-init MLP")]:
        m = summary.get(method_key, {})
        k20 = m.get("k20", {})
        d1 = k20.get("ds1_pc_r2", {})
        d2 = k20.get("ds2_pc_r2", {})
        out[label] = {"d1_mean": d1.get("mean"), "d1_ci": d1.get("ci95"),
                      "d2_mean": d2.get("mean"), "d2_ci": d2.get("ci95")}
    return out


def collect_all_methods(prefix: str = "") -> Dict[str, Dict]:
    """Collect all method metrics from all result directories."""
    res_base = RESULTS / prefix if prefix else RESULTS
    all_m = {}
    all_m.update(extract_method_metrics_baselines(load_json(res_base / "baselines_multiseed" / "summary.json")))
    all_m.update(extract_method_metrics_mv(load_json(res_base / "multivariate_methods" / "summary.json")))
    all_m.update(extract_method_metrics_ksr(load_json(res_base / "kernel_spectral_regression" / "summary.json")))
    all_m.update(extract_method_metrics_twostage(load_json(res_base / "nn_mlp_twostage" / "summary.json")))
    # Filter out methods with no data
    return {k: v for k, v in all_m.items() if v.get("d1_mean") is not None}


# ---- Figure 1: SV Spectrum ----

def fig1_sv_spectrum():
    sv_stats = load_json(RESULTS / "bootstrap_sv" / "sv_stats.json")
    if sv_stats is None:
        print("  [SKIP] Fig 1: bootstrap_sv/sv_stats.json not found")
        return

    n_sv = sv_stats["n_sv"]
    x = np.arange(1, n_sv + 1)
    full_sv = np.array(sv_stats["full_sample_sv"])
    ci_lo = np.array(sv_stats["ci95_lo"])
    ci_hi = np.array(sv_stats["ci95_hi"])

    fig, ax = plt.subplots(figsize=(3.55, 2.6))
    ax.fill_between(x, ci_lo, ci_hi, alpha=0.22, color=COLOR_HIGHLIGHT, label="95% bootstrap CI")
    ax.plot(x, full_sv, "o-", color=COLOR_DARK, markersize=4.2, linewidth=1.25, label="Full sample", zorder=3)
    ax.set_xlabel("Singular value index")
    ax.set_ylabel(r"Singular value of $\mathbf{X}^\top\mathbf{Y} / \sqrt{n}$")
    style_axes(ax, ygrid=True, xgrid=False)
    add_panel_title(ax, "Bootstrap singular-value spectrum")
    ax.legend(fontsize=5.2, loc="upper right")
    ax.set_xlim(0.5, n_sv + 0.5)

    save_panel(fig, "fig1", "panel_a")


# ---- Figure 2: Method comparison bar chart ----

def fig2_method_comparison():
    sz = collect_all_methods("")
    ukb = collect_all_methods("ukb")

    if not sz:
        print("  [SKIP] Fig 2: no SZ results")
        return

    # 7 methods only (no KSR/KSR-NN)
    method_order = ["Ridge", "RRR", "PLS", "Nuclear Norm", "OptShrink",
                    "MLP", "NN-init MLP"]
    methods = [m for m in method_order if m in sz]

    has_ukb = len(ukb) > 0
    n_datasets = 2 if has_ukb else 1

    fig, axes = plt.subplots(
        1, n_datasets, figsize=(3.9 * n_datasets, 3.25), squeeze=False
    )

    dataset_specs = [("Clinical cohorts", sz)] + ([("UK Biobank", ukb)] if has_ukb else [])
    for di, (dataset_label, data) in enumerate(dataset_specs):
        ax = axes[0, di]
        present = [m for m in methods if m in data]
        n = len(present)

        d1_means = [data[m]["d1_mean"] or 0 for m in present]
        d1_cis = [data[m]["d1_ci"] or 0 for m in present]
        d2_means = [data[m]["d2_mean"] or 0 for m in present]
        d2_cis = [data[m]["d2_ci"] or 0 for m in present]

        draw_grouped_bars(
            ax,
            present,
            ["DS1 test", "DS2 external"],
            np.array([d1_means, d2_means]),
            errors=np.array([d1_cis, d2_cis]),
            colors=[COLOR_PRIMARY, COLOR_SECONDARY],
            ylabel="PC-space $R^2$ ($k=20$)",
        )
        ax.set_xticklabels(present, rotation=30, ha="right")
        add_baseline(ax, y=0.0)
        add_panel_title(ax, dataset_label)
        if di == 0:
            ax.legend(loc="upper left", ncol=2)
        panel_label(ax, f"({'abcdef'[di]})")

    plt.tight_layout()
    out = FIG_DIR / "fig2_method_comparison.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Fig 2 saved: {out}")


# ---- Final composite: Benchmark visualization for manuscript slot ----

def fig_benchmark_viz():
    """Two-panel benchmark figure matching the manuscript figure slot."""
    sz = collect_all_methods("")
    if not sz:
        print("  [SKIP] fig_benchmark_viz: no SZ results")
        return

    method_order = ["Ridge", "MLP", "RRR", "PLS", "Nuclear Norm", "OptShrink", "NN-init MLP"]
    present = [m for m in method_order if m in sz]
    d1_means = np.array([sz[m]["d1_mean"] or 0 for m in present], dtype=float)
    d1_cis = np.array([sz[m]["d1_ci"] or 0 for m in present], dtype=float)
    d2_means = np.array([sz[m]["d2_mean"] or 0 for m in present], dtype=float)
    d2_cis = np.array([sz[m]["d2_ci"] or 0 for m in present], dtype=float)
    retention = np.divide(d2_means, d1_means, out=np.zeros_like(d2_means), where=d1_means > 0)

    # Panel a: grouped bars
    fig_a, ax1 = plt.subplots(figsize=(4.5, 3.15))
    draw_grouped_bars(
        ax1,
        present,
        ["DS1 test", "DS2 external"],
        np.array([d1_means, d2_means]),
        errors=np.array([d1_cis, d2_cis]),
        colors=[COLOR_PRIMARY, COLOR_SECONDARY],
        ylabel="PC-space $R^2$ ($k=20$)",
    )
    ax1.set_xticklabels(
        ["Ridge", "MLP", "RRR", "PLS", "NN", "OptShrink", "NN-MLP"],
        rotation=28,
        ha="right",
    )
    add_baseline(ax1, y=0.0)
    add_panel_title(ax1, "Discovery vs external validation")
    ax1.legend(loc="upper left", ncol=2, fontsize=5.4)
    save_panel(fig_a, "fig3", "panel_a")

    # Panel b: retention lollipop
    fig_b, ax2 = plt.subplots(figsize=(3.5, 3.15))
    ret_labels = ["PLS", "MLP", "NN-MLP", "RRR", "NN", "OptShrink", "Ridge"]
    ret_map = {m: r for m, r in zip(present, retention)}
    ret_values = [ret_map.get("PLS", 0), ret_map.get("MLP", 0), ret_map.get("NN-init MLP", 0),
                  ret_map.get("RRR", 0), ret_map.get("Nuclear Norm", 0),
                  ret_map.get("OptShrink", 0), ret_map.get("Ridge", 0)]
    ret_colors = [
        METHOD_COLORS.get("PLS"),
        METHOD_COLORS.get("MLP"),
        METHOD_COLORS.get("NN-init MLP"),
        METHOD_COLORS.get("RRR"),
        METHOD_COLORS.get("Nuclear Norm"),
        METHOD_COLORS.get("OptShrink"),
        METHOD_COLORS.get("Ridge"),
    ]
    draw_lollipop_series(
        ax2,
        ret_labels,
        ret_values,
        colors=ret_colors,
        xlabel="Generalization retention (DS2/DS1)",
        fmt="{:.0%}",
    )
    add_baseline(ax2, y=1.0, horizontal=False, style="dashed", alpha=0.55)
    add_panel_title(ax2, "Retention")
    save_panel(fig_b, "fig3", "panel_b")


# ---- Figure 3: Residualization ablation ----

def fig3_residualization():
    diag = load_json(RESULTS / "diagnostic_analysis" / "signal_check.json")
    if diag is None:
        print("  [SKIP] Fig 3: diagnostic_analysis/signal_check.json not found")
        return

    ks = sorted(diag.keys(), key=lambda x: int(x[1:]))
    k_vals = [int(k[1:]) for k in ks]
    raw_r2 = [diag[k]["raw"]["pc_r2_mean"] for k in ks]
    res_r2 = [diag[k]["residualized"]["pc_r2_mean"] for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # Left: bar chart raw vs resid
    ax = axes[0]
    draw_grouped_bars(
        ax,
        [f"k={k}" for k in k_vals],
        ["Raw", "Residualized"],
        np.array([raw_r2, res_r2]),
        colors=[COLOR_SECONDARY, COLOR_PRIMARY],
        ylabel="Ridge PC-space $R^2$",
    )
    add_panel_title(ax, "Residualization effect")
    ax.legend(fontsize=5.5)
    panel_label(ax, "(a)")

    # Right: per-PC at k=20
    if "k20" in diag:
        ax = axes[1]
        raw_pcs = diag["k20"]["raw"]["pc_r2_per_pc"]
        res_pcs = diag["k20"]["residualized"]["pc_r2_per_pc"]
        n_show = min(len(raw_pcs), len(res_pcs), 20)
        x = np.arange(1, n_show + 1)
        ax.plot(x, raw_pcs[:n_show], "o-", label="Raw", color=COLOR_SECONDARY, markersize=4.5, linewidth=1.3, zorder=3)
        ax.plot(x, res_pcs[:n_show], "s-", label="Residualized", color=COLOR_PRIMARY, markersize=4.5, linewidth=1.3, zorder=3)
        ax.set_xlabel("PC index")
        ax.set_ylabel("$R^2$")
        style_axes(ax, ygrid=True, xgrid=False)
        add_baseline(ax, y=0.0)
        add_panel_title(ax, "Per-PC comparison at $k=20$")
        ax.legend(fontsize=5.5)
        panel_label(ax, "(b)")

    plt.tight_layout()
    out = FIG_DIR / "fig3_residualization.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Fig 3 saved: {out}")


# ---- Figure 4: Per-PC R² heatmap ----

def fig4_per_pc_heatmap():
    diag = load_json(RESULTS / "diagnostic_analysis" / "summary.json")
    if diag is None:
        print("  [SKIP] Fig 4: diagnostic_analysis/summary.json not found")
        return

    per_pc = diag.get("per_pc_analysis", {})
    if not per_pc:
        print("  [SKIP] Fig 4: no per_pc_analysis in summary")
        return

    method_order = ["ridge", "mlp", "rrr", "pls", "nuclear_norm"]
    labels = {"ridge": "Ridge", "mlp": "MLP", "rrr": "RRR", "pls": "PLS",
              "nuclear_norm": "Nuclear Norm"}
    present = [m for m in method_order if m in per_pc]

    n_pcs = min(20, min(len(per_pc[m]["mean_per_pc_r2"]) for m in present))
    mat = np.array([per_pc[m]["mean_per_pc_r2"][:n_pcs] for m in present])

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    im = draw_stat_heatmap(
        ax,
        mat,
        [labels[m] for m in present],
        [f"PC{i+1}" for i in range(n_pcs)],
        cmap=CMAP_STAT_DIVERGING,
        vmin=-0.02,
        vmax=max(0.18, float(np.max(mat))),
        annotate=True,
        fmt="{:.2f}",
    )
    add_panel_title(
        ax,
        "Per-component predictive structure",
        "Higher values indicate stronger recoverable\nGM-FNC coupling per FNC PC",
        title_y=1.15,
        subtitle_y=1.27,
        loc="left",
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.02)
    style_colorbar(cb, "$R^2$")

    save_panel(fig, "fig4", "panel_a")


# ---- Figure 5: SZ vs UKB ranking comparison ----

def fig5_sz_vs_ukb():
    sz = collect_all_methods("")
    ukb = collect_all_methods("ukb")

    # Exclude KSR/KSR-NN
    excluded = {"KSR", "KSR-NN"}
    sz = {k: v for k, v in sz.items() if k not in excluded}
    ukb = {k: v for k, v in ukb.items() if k not in excluded}

    common = sorted(set(sz.keys()) & set(ukb.keys()))
    if len(common) < 2:
        print(f"  [SKIP] Fig 5: only {len(common)} methods in common between SZ and UKB")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))

    for di, metric_key, _title in [(0, "d2_mean", "External validation (DS2)"),
                                    (1, "d1_mean", "In-sample test (DS1)")]:
        ax = axes[di]
        sz_vals = [sz[m][metric_key] or 0 for m in common]
        ukb_vals = [ukb[m][metric_key] or 0 for m in common]
        colors = [METHOD_COLORS.get(m, COLOR_PRIMARY) for m in common]
        draw_method_scatter(
            ax,
            sz_vals,
            ukb_vals,
            common,
            colors=colors,
            xlabel="Clinical-cohort PC-$R^2$ ($k=20$)",
            ylabel="UKB PC-$R^2$ ($k=20$)",
        )
        add_panel_title(
            ax,
            "External ranking" if metric_key == "d2_mean" else "In-sample ranking",
        )
        panel_label(ax, f"({'ab'[di]})")

    plt.tight_layout()
    out = FIG_DIR / "fig5_sz_vs_ukb.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Fig 5 saved: {out}")


# ---- Figure 6: Subspace Analysis ----

def fig6_subspace_analysis():
    """Panel A: cos²θ vs dimension; Panel B: R² vs subspace overlap scatter."""
    stats_path = RESULTS / "subspace_analysis" / "subspace_stats.json"
    angles_path = RESULTS / "subspace_analysis" / "principal_angles.json"
    if not stats_path.exists():
        print("  [SKIP] Fig 6: subspace_stats.json not found")
        return
    with open(stats_path) as f:
        stats = json.load(f)
    with open(angles_path) as f:
        angles = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.15))

    # Panel A: Principal angles at k=20
    k_label = "k=20"
    for method_name, angle_data in angles.items():
        if k_label not in angle_data:
            continue
        cos_vals = np.array(angle_data[k_label])
        label = METHOD_NAME_MAP.get(method_name, method_name)
        color = METHOD_COLORS.get(label, "gray")
        marker = METHOD_MARKERS.get(label, "o")
        ax1.plot(range(1, len(cos_vals) + 1), cos_vals ** 2,
                 marker=marker, markersize=4.5, label=label, color=color,
                 linewidth=1.3, zorder=3)

    ax1.set_xlabel("Principal angle index")
    ax1.set_ylabel(r"$\cos^2(\theta_i)$")
    style_axes(ax1, ygrid=True, xgrid=False)
    add_panel_title(ax1, "Principal-angle decay", "Top dimensions stay aligned even when amplitude prediction is weak")
    ax1.legend(fontsize=5.5, loc="upper right")
    ax1.set_ylim(-0.05, 1.05)
    panel_label(ax1, "(a)")

    # Panel B: R² vs subspace overlap scatter
    texts = []
    for method_name, res in stats.items():
        r2 = res.get("r2_global_mean", res.get("r2_global"))
        sa = res["subspace_analysis"].get(k_label, {})
        overlap = sa.get("overlap_mean", sa.get("subspace_overlap"))
        if overlap is None:
            continue
        label = METHOD_NAME_MAP.get(method_name, method_name)
        color = METHOD_COLORS.get(label, "gray")
        ax2.scatter(r2, overlap, s=58, color=color, zorder=3, edgecolors="white", linewidth=0.8)
        t = ax2.annotate(label, (r2, overlap), fontsize=6,
                         xytext=(5, 5), textcoords="offset points")
        texts.append(t)

    adjust_annotations(ax2, texts)

    ax2.set_xlabel(r"Variance explained ($R^2$)")
    ax2.set_ylabel(f"Subspace overlap ({k_label})")
    style_axes(ax2, ygrid=True, xgrid=True)
    add_panel_title(ax2, "Geometry exceeds amplitude", "Methods cluster high in overlap despite modest predictive $R^2$")
    panel_label(ax2, "(b)")

    plt.tight_layout()
    out = FIG_DIR / "fig6_subspace_analysis.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Fig 6 saved: {out}")


# ---- Composite: Low-Rank + Geometric (Merge 1) ----

def fig_lowrank_geometric():
    """3-panel composite: (a) SV spectrum, (b) principal angles, (c) R² vs overlap."""
    sv_stats = load_json(RESULTS / "bootstrap_sv" / "sv_stats.json")
    stats_path = RESULTS / "subspace_analysis" / "subspace_stats.json"
    angles_path = RESULTS / "subspace_analysis" / "principal_angles.json"

    if sv_stats is None or not stats_path.exists() or not angles_path.exists():
        print("  [SKIP] fig_lowrank_geometric: missing data")
        return

    with open(stats_path) as f:
        stats = json.load(f)
    with open(angles_path) as f:
        angles = json.load(f)

    k_label = "k=20"

    # Panel a: SV spectrum with bootstrap CI
    fig_a, ax1 = plt.subplots(figsize=(3.0, 2.8))
    n_sv = sv_stats["n_sv"]
    x = np.arange(1, n_sv + 1)
    full_sv = np.array(sv_stats["full_sample_sv"])
    ci_lo = np.array(sv_stats["ci95_lo"])
    ci_hi = np.array(sv_stats["ci95_hi"])
    ax1.fill_between(x, ci_lo, ci_hi, alpha=0.18, color="#56B4E9", label="95% bootstrap CI")
    ax1.plot(x, full_sv, "o-", color=COLOR_DARK, markersize=2.5, linewidth=1.0, label="Full sample", zorder=3)
    ax1.set_xlabel("Singular value index")
    ax1.set_ylabel(r"Singular value")
    style_axes(ax1, ygrid=True, xgrid=False)
    add_panel_title(ax1, "Spectral concentration")
    ax1.legend(fontsize=5, loc="upper right")
    ax1.set_xlim(0.5, n_sv + 0.5)
    save_panel(fig_a, "fig5", "panel_a")

    # Panel b: Principal angles at k=20
    fig_b, ax2 = plt.subplots(figsize=(3.0, 2.8))
    for method_name, angle_data in angles.items():
        if k_label not in angle_data:
            continue
        cos_vals = np.array(angle_data[k_label])
        label = METHOD_NAME_MAP.get(method_name, method_name)
        color = METHOD_COLORS.get(label, "gray")
        marker = METHOD_MARKERS.get(label, "o")
        ax2.plot(range(1, len(cos_vals) + 1), cos_vals ** 2,
                 marker=marker, markersize=3.5, label=label, color=color, linewidth=1.2, zorder=3)
    ax2.set_xlabel("Principal angle index")
    ax2.set_ylabel(r"$\cos^2(\theta_i)$")
    style_axes(ax2, ygrid=True, xgrid=False)
    add_panel_title(ax2, "Directional alignment")
    ax2.legend(fontsize=4.5, ncol=1, loc="upper right")
    ax2.set_ylim(-0.05, 1.05)
    save_panel(fig_b, "fig5", "panel_b")

    # Panel c: R^2 vs subspace overlap scatter
    fig_c, ax3 = plt.subplots(figsize=(3.0, 2.8))
    texts = []
    for method_name, res in stats.items():
        r2 = res.get("r2_global_mean", res.get("r2_global"))
        sa = res["subspace_analysis"].get(k_label, {})
        overlap = sa.get("overlap_mean", sa.get("subspace_overlap"))
        if overlap is None:
            continue
        label = METHOD_NAME_MAP.get(method_name, method_name)
        color = METHOD_COLORS.get(label, "gray")
        ax3.scatter(r2, overlap, s=52, color=color, zorder=3, edgecolors="white", linewidth=0.85)
        t = ax3.annotate(label, (r2, overlap), fontsize=5,
                         xytext=(5, 5), textcoords="offset points")
        texts.append(t)
    adjust_annotations(ax3, texts)
    ax3.set_xlabel(r"Variance explained ($R^2$)")
    ax3.set_ylabel(f"Subspace overlap ({k_label})")
    style_axes(ax3, ygrid=True, xgrid=True)
    add_panel_title(ax3, "Geometric constraint")
    save_panel(fig_c, "fig5", "panel_c")


# ---- Composite: Linearity Test (Merge 2) ----

def fig_linearity_test():
    """2-panel composite: (a) DS1 vs DS2 bars for 4 models, (b) retention ratio."""
    summary = load_json(RESULTS / "nn_mlp_twostage" / "summary.json")
    if summary is None:
        print("  [SKIP] fig_linearity_test: nn_mlp_twostage/summary.json not found")
        return

    model_keys = ["nuclear_norm", "mlp", "nn_plus_mlp_residual", "nn_init_mlp"]
    model_labels = ["Nuclear\nNorm", "MLP", "NN+MLP\nresidual", "NN-init\nMLP"]

    ds1_means, ds1_cis, ds2_means, ds2_cis = [], [], [], []
    for mk in model_keys:
        m = summary.get(mk, {})
        k20 = m.get("k20", {})
        d1 = k20.get("ds1_pc_r2", {})
        d2 = k20.get("ds2_pc_r2", {})
        ds1_means.append(d1.get("mean", 0))
        ds1_cis.append(d1.get("ci95", 0))
        ds2_means.append(d2.get("mean", 0))
        ds2_cis.append(d2.get("ci95", 0))

    # Panel b: DS1 vs DS2 grouped bars
    fig_b, ax1 = plt.subplots(figsize=(4.5, 3.05))
    draw_grouped_bars(
        ax1,
        model_labels,
        ["DS1 test", "DS2 external"],
        np.array([ds1_means, ds2_means]),
        errors=np.array([ds1_cis, ds2_cis]),
        colors=[COLOR_PRIMARY, COLOR_SECONDARY],
        ylabel="PC-space $R^2$ ($k=20$)",
    )
    ax1.set_xticklabels(model_labels, fontsize=6)
    add_baseline(ax1, y=0.0)
    add_panel_title(
        ax1,
        "Linearity benchmark",
        "In-sample gains do not cleanly transfer to external data",
        title_y=1.15,
    )
    ax1.legend(fontsize=5.5, ncol=2, loc="upper left")
    save_panel(fig_b, "fig4", "panel_b")

    # Panel c: Retention ratio (DS2/DS1)
    retention = []
    for d1, d2 in zip(ds1_means, ds2_means):
        retention.append(d2 / d1 if d1 > 0 else 0)
    fig_c, ax2 = plt.subplots(figsize=(3.5, 3.05))
    colors_ret = [METHOD_COLORS.get("Nuclear Norm") if r >= 0.7 else COLOR_SECONDARY for r in retention]
    draw_lollipop_series(
        ax2,
        model_labels,
        retention,
        colors=colors_ret,
        xlabel="Generalization retention (DS2/DS1)",
        fmt="{:.0%}",
    )
    add_baseline(ax2, y=1.0, horizontal=False, style="dashed", alpha=0.55)
    add_panel_title(ax2, "Retention", title_y=1.15)
    save_panel(fig_c, "fig4", "panel_c")


# ---- Composite: Clinical Relevance (Merge 3) ----

def fig_clinical_composite():
    """3-panel composite: (a) coupled var fraction vs rank, (b) AUC vs rank, (c) AUC bars at primary rank."""
    summary = load_json(RESULTS / "smri_residual" / "summary.json")
    if summary is None:
        print("  [SKIP] fig_clinical_composite: smri_residual/summary.json not found")
        return

    a1 = summary["A1_variance_decomposition"]
    a3_primary = summary["A3_clinical_relevance"]
    a3_sweep = summary.get("A3_rank_sweep", {})

    ranks = sorted(int(r) for r in a1.keys())
    coupled_fracs = [a1[str(r)]["coupled_var_frac"] for r in ranks]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.2, 2.9),
                                         gridspec_kw={"width_ratios": [1.2, 1.2, 1]})

    # (a) Coupled variance fraction vs rank
    ax1.plot(ranks, coupled_fracs, "o-", color=COLOR_PRIMARY, markersize=4.5, linewidth=1.45, zorder=3)
    ax1.set_xlabel("Rank $r$")
    ax1.set_ylabel("Coupled variance fraction")
    ax1.set_ylim(0, 1.0)
    style_axes(ax1, ygrid=True, xgrid=False)
    add_panel_title(ax1, "Subspace occupancy")
    panel_label(ax1, "(a)")

    # (b) AUC vs rank (coupled + uncoupled lines)
    if a3_sweep:
        sweep_ranks = sorted(int(r) for r in a3_sweep.keys())
        coupled_aucs = [a3_sweep[str(r)]["coupled"]["auc"] for r in sweep_ranks]
        uncoupled_aucs = [a3_sweep[str(r)]["uncoupled"]["auc"] for r in sweep_ranks]
        ax2.plot(sweep_ranks, coupled_aucs, "o-", color=COLOR_PRIMARY, markersize=5,
                 linewidth=1.4, label="Coupled", zorder=3)
        ax2.plot(sweep_ranks, uncoupled_aucs, "s--", color=COLOR_NEUTRAL, markersize=5,
                 linewidth=1.3, label="Uncoupled", zorder=3)
        # Full FNC reference line
        full_auc = a3_primary["full"]["auc"]
        ax2.axhline(full_auc, color=COLOR_DARK, linestyle=":", linewidth=0.8,
                     label=f"Full FNC ({full_auc:.3f})")
        ax2.set_xlabel("Rank $r$")
        ax2.set_ylabel("AUC (SZ classification)")
        ax2.set_ylim(0.45, 0.90)
        ax2.legend(fontsize=5, loc="lower right")
    style_axes(ax2, ygrid=True, xgrid=False)
    add_baseline(ax2, y=0.5, style="dashed", alpha=0.5)
    add_panel_title(ax2, "Clinical signal across rank")
    panel_label(ax2, "(b)")

    # (c) AUC bars at primary rank
    if a3_primary:
        names = ["full", "coupled", "uncoupled"]
        labels = ["Full\nGM", "Coupled", "Uncoupled"]
        colors = [COLOR_DARK, COLOR_PRIMARY, COLOR_NEUTRAL]
        aucs = [a3_primary[n]["auc"] for n in names]
        ci_lo = [a3_primary[n]["auc"] - a3_primary[n]["auc_ci_lo"] for n in names]
        ci_hi = [a3_primary[n]["auc_ci_hi"] - a3_primary[n]["auc"] for n in names]
        x = np.arange(len(names))
        bars = ax3.bar(x, aucs, color=colors, width=0.6, edgecolor="white", linewidth=0.5,
                       yerr=[ci_lo, ci_hi], capsize=3, error_kw={"linewidth": 1.0})
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, fontsize=6)
        ax3.set_ylabel("AUC")
        ax3.set_ylim(0.5, max(aucs) + 0.12)
        style_axes(ax3, ygrid=True, xgrid=False)
        add_baseline(ax3, y=0.5, style="dashed", alpha=0.5)
        for bar, v in zip(bars, aucs):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(ci_hi) + 0.01,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=5.5)
    add_panel_title(ax3, "Primary-rank comparison")
    panel_label(ax3, "(c)")

    plt.tight_layout()
    out = FIG_DIR / "fig_clinical.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  fig_clinical_composite saved: {out}")


# ---- Composite: Robustness (Merge 4: Residualization + SZ vs UKB) ----

def fig_robustness_composite():
    """4-panel 2×2: (a,b) residualization, (c,d) SZ vs UKB."""
    diag = load_json(RESULTS / "diagnostic_analysis" / "signal_check.json")
    sz = collect_all_methods("")
    ukb = collect_all_methods("ukb")

    if diag is None and not sz:
        print("  [SKIP] fig_robustness_composite: no data")
        return

    # ── Panel a: Residualization bars ──
    if diag is not None:
        ks = sorted(diag.keys(), key=lambda x: int(x[1:]))
        k_vals = [int(k[1:]) for k in ks]
        raw_r2 = [diag[k]["raw"]["pc_r2_mean"] for k in ks]
        res_r2 = [diag[k]["residualized"]["pc_r2_mean"] for k in ks]

        fig_a, ax = plt.subplots(figsize=(3.5, 2.8))
        draw_grouped_bars(
            ax,
            [f"k={k}" for k in k_vals],
            ["Raw", "Residualized"],
            np.array([raw_r2, res_r2]),
            errors=None,
            colors=[COLOR_SECONDARY, COLOR_PRIMARY],
            ylabel="Ridge PC-space $R^2$",
        )
        add_panel_title(ax, "Residualization robustness")
        ax.legend(fontsize=5.5, loc="upper left")
        save_panel(fig_a, "figS2", "panel_a")

        # ── Panel b: Per-PC at k=20 ──
        if "k20" in diag:
            fig_b, ax = plt.subplots(figsize=(3.5, 2.8))
            raw_pcs = diag["k20"]["raw"]["pc_r2_per_pc"]
            res_pcs = diag["k20"]["residualized"]["pc_r2_per_pc"]
            n_show = min(len(raw_pcs), len(res_pcs), 20)
            x = np.arange(1, n_show + 1)
            ax.plot(x, raw_pcs[:n_show], "o-", label="Raw", color=COLOR_SECONDARY, markersize=4.5, linewidth=1.3, zorder=3)
            ax.plot(x, res_pcs[:n_show], "s-", label="Residualized", color=COLOR_PRIMARY, markersize=4.5, linewidth=1.3, zorder=3)
            ax.set_xlabel("PC index")
            ax.set_ylabel("$R^2$")
            style_axes(ax, ygrid=True, xgrid=False)
            add_baseline(ax, y=0.0)
            add_panel_title(ax, "Per-PC stability at $k=20$")
            ax.legend(fontsize=5.5)
            save_panel(fig_b, "figS2", "panel_b")

    # ── Panel c & d: SZ vs UKB ──
    excluded = {"KSR", "KSR-NN"}
    sz_f = {k: v for k, v in sz.items() if k not in excluded}
    ukb_f = {k: v for k, v in ukb.items() if k not in excluded}
    common = sorted(set(sz_f.keys()) & set(ukb_f.keys()))

    if len(common) >= 2:
        for di, metric_key, _title in [(0, "d2_mean", "External validation (DS2)"),
                                        (1, "d1_mean", "In-sample test (DS1)")]:
            fig_panel, ax = plt.subplots(figsize=(3.5, 2.8))
            sz_vals = [sz_f[m][metric_key] or 0 for m in common]
            ukb_vals = [ukb_f[m][metric_key] or 0 for m in common]
            draw_method_scatter(
                ax,
                sz_vals,
                ukb_vals,
                common,
                colors=[METHOD_COLORS.get(m, COLOR_PRIMARY) for m in common],
                xlabel="Clinical-cohort PC-$R^2$ ($k=20$)",
                ylabel="UKB PC-$R^2$ ($k=20$)",
            )
            add_panel_title(ax, "External transfer" if metric_key == "d2_mean" else "In-sample transfer")
            panel_name = "panel_c" if di == 0 else "panel_d"
            save_panel(fig_panel, "figS2", panel_name)


# ---- Composite: Biological (Fig 5 in paper) ----

def build_roi_coupled_nifti(roi_csv_path, atlas_path, gm_names_path):
    """Build a NIfTI volume where each ROI's voxels = coupled_frac from roi_decomposition.csv."""
    import pandas as pd
    import nibabel as nib

    df = pd.read_csv(roi_csv_path)
    gm_names = Path(gm_names_path).read_text().strip().split("\n")
    roi_indices = [int(n.replace("roi_", "")) for n in gm_names]
    # Map roi_name -> coupled_frac
    coupled_map = dict(zip(df["roi_name"], df["coupled_frac"]))

    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()
    composite = np.zeros(atlas_data.shape[:3], dtype=np.float64)

    for i, roi_idx in enumerate(roi_indices):
        roi_name = gm_names[i]
        frac = coupled_map.get(roi_name, 0.0)
        vol = np.abs(atlas_data[:, :, :, roi_idx])
        pos = vol[vol > 0]
        if len(pos) == 0:
            continue
        mu, sigma = pos.mean(), pos.std()
        if sigma < 1e-10:
            continue
        z = (vol - mu) / sigma
        mask = z > 2.0
        composite[mask] = np.maximum(composite[mask], frac)

    composite = np.clip(composite, 0.0, 1.0)
    return nib.Nifti1Image(composite, atlas_img.affine)


def build_mode_weighted_nifti(U, mode_idx, roi_indices, atlas_path):
    """Build NIfTI from |U[:,m]| weighted atlas volumes using max-weighted-projection."""
    import nibabel as nib

    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()
    composite = np.zeros(atlas_data.shape[:3], dtype=np.float64)

    w = np.abs(U[:, mode_idx])
    w_norm = w / w.max() if w.max() > 0 else w

    for i, roi_idx in enumerate(roi_indices):
        vol = np.abs(atlas_data[:, :, :, roi_idx])
        pos = vol[vol > 0]
        if len(pos) == 0:
            continue
        mu, sigma = pos.mean(), pos.std()
        if sigma < 1e-10:
            continue
        z = (vol - mu) / sigma
        z[z < 2.0] = 0
        weighted = w_norm[i] * z
        composite = np.maximum(composite, weighted)

    if composite.max() > 0:
        composite /= composite.max()

    return nib.Nifti1Image(composite, atlas_img.affine)


_MNI_BG_CACHE = None


def _get_mni_bg():
    """Lazily load MNI152 background image used for all brain montages."""
    global _MNI_BG_CACHE
    if _MNI_BG_CACHE is None:
        from nilearn import datasets
        _MNI_BG_CACHE = datasets.load_mni152_template(resolution=1)
    return _MNI_BG_CACHE


def _render_montage_to_array(
    nifti_img,
    cmap,
    threshold,
    cut_coords,
    colorbar=True,
    title=None,
    figsize=(5.0, 1.8),
    vmax=None,
    dpi=200,
    canonical=False,
    flip_lr=False,
    radiological=False,
    resample_to_mni=False,
):
    """Render a nilearn plot_stat_map to a numpy RGBA array for embedding in composites."""
    from nilearn import plotting
    import io
    from PIL import Image
    if canonical:
        # Keep compatibility with previous API; avoid altering orientation here unless
        # a caller explicitly opts in.
        pass
    if flip_lr:
        try:
            # Flip in world left-right without disturbing colormap/colorbar logic.
            # Use nibabel slicing so affine is updated consistently.
            nifti_img = nifti_img.slicer[::-1, :, :]
        except Exception:
            try:
                import nibabel as nib
                data = np.asarray(nifti_img.dataobj)
                affine = nifti_img.affine.copy()
                affine[0, 0] *= -1
                data = data[::-1, :, :]
                nifti_img = nib.Nifti1Image(data, affine, header=nifti_img.header.copy())
            except Exception:
                pass

    bg_img = _get_mni_bg()
    if resample_to_mni:
        try:
            from nilearn import image
            nifti_img = image.resample_to_img(
                nifti_img, bg_img, interpolation="continuous",
                force_resample=True, copy_header=True
            )
        except Exception:
            pass

    # Temporarily override image.origin to avoid flipping nilearn output
    with plt.rc_context({"image.origin": "upper"}):
        tmp_fig = plt.figure(figsize=figsize)
        plotting.plot_stat_map(
            nifti_img,
            display_mode="z",
            cut_coords=cut_coords,
            colorbar=colorbar,
            cmap=cmap,
            threshold=threshold,
            vmax=vmax,
            black_bg=False,
            annotate=True,
            radiological=radiological,
            figure=tmp_fig,
            title=title,
            bg_img=bg_img,
        )
        buf = io.BytesIO()
        tmp_fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(tmp_fig)
    buf.seek(0)
    img_arr = np.array(Image.open(buf))
    buf.close()
    return img_arr


def _render_stacked_montages(
    nifti_imgs,
    labels,
    cmap,
    threshold,
    cut_coords,
    figsize=(7.2, 2.5),
    vmax=None,
    dpi=200,
    canonical=True,
    flip_lr=False,
    radiological=False,
    resample_to_mni=False,
):
    """Render multiple nilearn montages stacked vertically with a single shared colorbar.

    Each montage is rendered to its own figure, rasterized, then stacked.
    Only the last montage gets a colorbar.
    """
    arrays = []
    target_w = None
    for i, (nii, label) in enumerate(zip(nifti_imgs, labels)):
        show_cb = (i == len(nifti_imgs) - 1)
        arr = _render_montage_to_array(
            nii, cmap=cmap, threshold=threshold, cut_coords=cut_coords,
            colorbar=show_cb, vmax=vmax, title=label, figsize=figsize, dpi=dpi,
            canonical=canonical,
            flip_lr=flip_lr,
            radiological=radiological,
            resample_to_mni=resample_to_mni,
        )
        if target_w is None:
            target_w = arr.shape[1]
        # Pad narrower arrays (no-colorbar rows) to match the widest (colorbar row)
        if arr.shape[1] < target_w:
            pad = np.full((arr.shape[0], target_w - arr.shape[1], arr.shape[2]),
                          255, dtype=np.uint8)
            arr = np.concatenate([arr, pad], axis=1)
        elif arr.shape[1] > target_w:
            target_w = arr.shape[1]
        arrays.append(arr)
    # Second pass: ensure all same width
    max_w = max(a.shape[1] for a in arrays)
    padded = []
    for arr in arrays:
        if arr.shape[1] < max_w:
            pad = np.full((arr.shape[0], max_w - arr.shape[1], arr.shape[2]),
                          255, dtype=np.uint8)
            arr = np.concatenate([arr, pad], axis=1)
        padded.append(arr)
    return np.vstack(padded)


def fig_biological_composite():
    """2 panels saved to fig6:
    (a) ROI coupled fraction montage,
    (b) per-mode montage slices.
    """
    import sys as _sys
    _sys.path.insert(0, str(BASE / "scripts"))

    summary = load_json(RESULTS / "smri_residual" / "summary.json")
    if summary is None:
        print("  [SKIP] fig_biological_composite: smri_residual/summary.json not found")
        return

    # Try to load SVD mode data
    try:
        from generate_brain_figures import (
            DOMAIN_ORDER_SBM, DOMAIN_ORDER_FNC, DOMAIN_COLORS, DOMAIN_FULL_NAMES,
            ATLAS_PATH, B_MATRIX_PATH, GM_NAMES_PATH,
            load_data, parse_fnc_edges, get_fnc_domain, get_roi_domain_and_coords,
        )
        from analyze_svd_modes import svd_decompose, mode_domain_coupling
        B, roi_indices, fnc_names = load_data()
        fnc_edges = parse_fnc_edges(fnc_names)
        roi_domains, roi_coords = get_roi_domain_and_coords(roi_indices)
        S_all = np.linalg.svd(B, compute_uv=False)
        U, S, Vt = svd_decompose(B, k=3)
        _, _, _ = mode_domain_coupling(
            U, S, Vt, roi_domains, fnc_edges, k=3
        )
        has_svd = True
    except Exception as e:
        print(f"  [WARN] Could not load SVD mode data: {e}")
        has_svd = False

    if not has_svd:
        print("  [SKIP] fig_biological_composite panel_b: no SVD data")
        return

    total_var = np.sum(S_all**2)

    # ── Panel b: ROI coupled fraction montage ──
    fig_b, ax_b = plt.subplots(figsize=(7.2, 2.5))
    try:
        roi_csv = RESULTS / "smri_residual" / "roi_decomposition.csv"
        roi_nifti = build_roi_coupled_nifti(roi_csv, ATLAS_PATH, GM_NAMES_PATH)
        cut_coords_b = [-25, -10, 5, 20, 35, 50, 65]
        brain_cmap = "coolwarm"
        img_arr = _render_montage_to_array(
            roi_nifti, cmap=brain_cmap, threshold=0.5,
            cut_coords=cut_coords_b, colorbar=True, vmax=1.0,
            title="ROI coupled variance fraction", figsize=(7.2, 2.3), dpi=260,
            canonical=False, flip_lr=False, radiological=False, resample_to_mni=False,
        )
        ax_b.imshow(img_arr)
    except Exception as e:
        print(f"  [WARN] ROI coupled fraction montage skipped: {e}")
        ax_b.text(0.5, 0.5, "ROI coupled fraction\n(not available)",
                  ha="center", va="center", fontsize=7, transform=ax_b.transAxes)
    ax_b.set_xticks([]); ax_b.set_yticks([])
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    save_panel(fig_b, "fig6", "panel_a")

    # ── Panel b: 3 mode montages stacked (remapped to panel_b output) ──
    fig_d, ax_d = plt.subplots(figsize=(7.2, 7.0))
    try:
        cut_coords_d = [-25, -10, 5, 20, 35, 50, 65]
        mode_niftis = []
        mode_labels = []
        for m in range(3):
            var_pct = S[m]**2 / total_var * 100
            mode_niftis.append(build_mode_weighted_nifti(U, m, roi_indices, ATLAS_PATH))
            mode_labels.append(f"Mode {m+1} GM loading ({var_pct:.1f}% var)")

        stacked_arr = _render_stacked_montages(
            mode_niftis, mode_labels, cmap="coolwarm", threshold=0.10,
            cut_coords=cut_coords_d, figsize=(7.2, 2.25), dpi=260,
            canonical=False, flip_lr=False, radiological=False, resample_to_mni=False,
        )
        ax_d.imshow(stacked_arr)
    except Exception as e:
        print(f"  [WARN] Montage slice panels skipped: {e}")
        ax_d.text(0.5, 0.5, "Mode montages\n(not available)",
                  ha="center", va="center", fontsize=7, transform=ax_d.transAxes)
    ax_d.set_xticks([]); ax_d.set_yticks([])
    for spine in ax_d.spines.values():
        spine.set_visible(False)
    save_panel(fig_d, "fig6", "panel_b")


def _add_box(ax, xy, width, height, text, facecolor="#f6f7fb", edgecolor="#1f2937", fontsize=8):
    """Draw a rounded text box in axis coordinates."""
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        transform=ax.transAxes,
    )


def paper_fig1_overview():
    """Figure 1: study overview + SVD interpretation + geometry-vs-amplitude dissociation."""
    geo = load_json(RESULTS / "geometry_robustness" / "geometry_robustness_summary.json")
    if geo is None:
        print("  [SKIP] paper_fig1_overview: geometry summary not found")
        return

    nn_k20 = geo["ds1"]["methods"]["Nuclear_Norm"]["by_k"]["k=20"]
    overlap = nn_k20["predicted_test_subspace"]["mean"]
    r2_global = geo["ds1"]["methods"]["Nuclear_Norm"]["r2_global"]["mean"]
    chance = nn_k20["null"]["chance_k_over_d"]

    fig = plt.figure(figsize=(11.2, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 0.9], wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_axis_off()
    add_panel_title(ax1, "Study workflow")
    _add_box(ax1, (0.02, 0.62), 0.26, 0.20, "Gray-matter\nROI features\n(99 ROIs)", facecolor="#e8f1fb")
    _add_box(ax1, (0.37, 0.62), 0.26, 0.20, "Learn GM→FNC\nmap $B$", facecolor="#eef8ea")
    _add_box(ax1, (0.72, 0.62), 0.26, 0.20, "Static FNC\n(1,378 edges)", facecolor="#fbf0e7")
    _add_box(ax1, (0.09, 0.20), 0.22, 0.18, "Lens selection", facecolor="#f7f7f7")
    _add_box(ax1, (0.39, 0.20), 0.22, 0.18, "Geometry metrics\n($\\mathcal{O}$, angles)", facecolor="#f7f7f7")
    _add_box(ax1, (0.69, 0.20), 0.22, 0.18, "Biology +\nclinical utility", facecolor="#f7f7f7")
    ax1.annotate("", xy=(0.37, 0.72), xytext=(0.28, 0.72), xycoords=ax1.transAxes, textcoords=ax1.transAxes,
                 arrowprops=dict(arrowstyle="->", lw=1.4, color=COLOR_DARK))
    ax1.annotate("", xy=(0.72, 0.72), xytext=(0.63, 0.72), xycoords=ax1.transAxes, textcoords=ax1.transAxes,
                 arrowprops=dict(arrowstyle="->", lw=1.4, color=COLOR_DARK))
    for x0 in [0.20, 0.50, 0.80]:
        ax1.annotate("", xy=(x0, 0.38), xytext=(x0, 0.57), xycoords=ax1.transAxes, textcoords=ax1.transAxes,
                     arrowprops=dict(arrowstyle="->", lw=1.2, color=COLOR_DARK))
    ax1.text(
        0.02,
        0.03,
        "Central question: does GM recover the shared directions of FNC variation\n"
        "more reliably than subject-specific amplitude coordinates?",
        fontsize=7.2,
        transform=ax1.transAxes,
    )
    panel_label(ax1, "(a)")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_axis_off()
    add_panel_title(ax2, "Interpretable decomposition")
    ax2.text(0.50, 0.82, r"$B = U \Sigma V^\top$", ha="center", va="center", fontsize=17, transform=ax2.transAxes)
    _add_box(ax2, (0.03, 0.38), 0.25, 0.20, "$U$\nGM loadings", facecolor="#e8f1fb")
    _add_box(ax2, (0.37, 0.38), 0.25, 0.20, "$\\Sigma$\nsoft rank /\ncoupling strength", facecolor="#eef8ea")
    _add_box(ax2, (0.71, 0.38), 0.25, 0.20, "$V$\nFNC directions", facecolor="#fbf0e7")
    ax2.text(
        0.50,
        0.12,
        "Interpretation target: the right-singular subspace of $B$ defines\n"
        "an anatomy-aligned functional basis, even when amplitudes stay weak.",
        ha="center",
        fontsize=7.2,
        transform=ax2.transAxes,
    )
    panel_label(ax2, "(b)")

    ax3 = fig.add_subplot(gs[0, 2])
    vals = [overlap, r2_global, chance]
    colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_NEUTRAL]
    labels = [r"$\mathcal{O}(\hat{Y},Y)$", r"$R^2_{\mathrm{global}}$", "Chance"]
    bars = ax3.bar(np.arange(3), vals, color=colors, width=0.58, edgecolor="white", linewidth=0.4)
    ax3.set_xticks(np.arange(3))
    ax3.set_xticklabels(labels, rotation=0, ha="center")
    ax3.set_ylabel("Magnitude")
    style_axes(ax3, ygrid=True, xgrid=False)
    add_panel_title(ax3, "Geometry vs amplitude")
    ax3.set_ylim(0, max(vals) * 1.22)
    ax3.text(0.97, 0.95, "DS1, $k$=20", fontsize=6, transform=ax3.transAxes, ha="right", va="top", color=TEXT_MUTED)
    panel_label(ax3, "(c)")

    plt.tight_layout()
    save_main_figure(fig, 1)


def paper_fig2_lens_selection():
    """Figure 2: analytical lens selection and linearity evidence."""
    methods = collect_all_methods("")
    ukb = collect_all_methods("ukb")
    nn_summary = load_json(RESULTS / "nn_mlp_twostage" / "summary.json")
    if not methods or nn_summary is None:
        print("  [SKIP] paper_fig2_lens_selection: missing benchmark data")
        return

    order = ["Ridge", "MLP", "RRR", "PLS", "Nuclear Norm", "OptShrink", "NN-init MLP"]
    present = [m for m in order if m in methods]
    d1 = np.array([methods[m]["d1_mean"] for m in present], dtype=float)
    d1_ci = np.array([methods[m]["d1_ci"] for m in present], dtype=float)
    d2 = np.array([methods[m]["d2_mean"] for m in present], dtype=float)
    d2_ci = np.array([methods[m]["d2_ci"] for m in present], dtype=float)
    ret = np.divide(d2, d1, out=np.zeros_like(d2), where=d1 > 0)

    fig = plt.figure(figsize=(11.6, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 1.0, 1.15], wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    draw_grouped_bars(
        ax1,
        present,
        ["DS1 test", "DS2 external"],
        np.array([d1, d2]),
        errors=np.array([d1_ci, d2_ci]),
        colors=[COLOR_PRIMARY, COLOR_SECONDARY],
        ylabel="PC-space $R^2$ ($k=20$)",
    )
    ax1.set_xticklabels(["Ridge", "MLP", "RRR", "PLS", "NN", "OptShrink", "NN-MLP"], rotation=28, ha="right")
    add_baseline(ax1, y=0.0)
    add_panel_title(ax1, "Analytical lens selection")
    ax1.legend(loc="upper left", ncol=2, fontsize=5.5)
    panel_label(ax1, "(a)")

    ax2 = fig.add_subplot(gs[0, 1])
    ret_labels = ["PLS", "MLP", "NN-MLP", "RRR", "NN", "OptShrink", "Ridge"]
    ret_map = {m: r for m, r in zip(present, ret)}
    ret_values = [
        ret_map.get("PLS", 0.0),
        ret_map.get("MLP", 0.0),
        ret_map.get("NN-init MLP", 0.0),
        ret_map.get("RRR", 0.0),
        ret_map.get("Nuclear Norm", 0.0),
        ret_map.get("OptShrink", 0.0),
        ret_map.get("Ridge", 0.0),
    ]
    ret_colors = [
        METHOD_COLORS.get("PLS"),
        METHOD_COLORS.get("MLP"),
        METHOD_COLORS.get("NN-init MLP"),
        METHOD_COLORS.get("RRR"),
        METHOD_COLORS.get("Nuclear Norm"),
        METHOD_COLORS.get("OptShrink"),
        METHOD_COLORS.get("Ridge"),
    ]
    draw_lollipop_series(
        ax2,
        ret_labels,
        ret_values,
        colors=ret_colors,
        xlabel="DS2 / DS1 retention",
        fmt="{:.0%}",
    )
    add_baseline(ax2, y=1.0, horizontal=False, style="dashed", alpha=0.55)
    add_panel_title(ax2, "External retention")
    panel_label(ax2, "(b)")

    ax3 = fig.add_subplot(gs[0, 2])
    model_keys = ["nuclear_norm", "mlp", "nn_plus_mlp_residual", "nn_init_mlp"]
    model_labels = ["NN", "MLP", "NN+MLP\nresid.", "NN-MLP"]
    ds1_means = []
    ds2_means = []
    for mk in model_keys:
        m = nn_summary.get(mk, {})
        k20 = m.get("k20", {})
        ds1_means.append(k20.get("ds1_pc_r2", {}).get("mean", 0.0))
        ds2_means.append(k20.get("ds2_pc_r2", {}).get("mean", 0.0))
    draw_grouped_bars(
        ax3,
        model_labels,
        ["DS1", "DS2"],
        np.array([ds1_means, ds2_means]),
        colors=[COLOR_PRIMARY, COLOR_SECONDARY],
        ylabel="PC-space $R^2$ ($k=20$)",
    )
    ax3.set_xticklabels(model_labels, fontsize=6)
    add_baseline(ax3, y=0.0)
    add_panel_title(ax3, "Linearity check")
    ax3.legend(loc="upper left", fontsize=5.5)
    panel_label(ax3, "(c)")

    plt.tight_layout()
    save_main_figure(fig, 2)


def paper_fig3_core_dissociation():
    """Figure 3: spectral concentration, angle decay, and geometry-vs-amplitude scatter."""
    sv_stats = load_json(RESULTS / "bootstrap_sv" / "sv_stats.json")
    geo = load_json(RESULTS / "geometry_robustness" / "geometry_robustness_summary.json")
    angles = load_json(RESULTS / "subspace_analysis" / "principal_angles.json")
    if sv_stats is None or geo is None or angles is None:
        print("  [SKIP] paper_fig3_core_dissociation: missing data")
        return

    fig = plt.figure(figsize=(11.6, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0], wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(1, sv_stats["n_sv"] + 1)
    full_sv = np.array(sv_stats["full_sample_sv"])
    ci_lo = np.array(sv_stats["ci95_lo"])
    ci_hi = np.array(sv_stats["ci95_hi"])
    ax1.fill_between(x, ci_lo, ci_hi, alpha=0.18, color="#56B4E9", label="95% bootstrap CI")
    ax1.plot(x, full_sv, "o-", color=COLOR_DARK, markersize=2.5, linewidth=1.0, label="Full sample", zorder=3)
    ax1.set_xlabel("Singular value index")
    ax1.set_ylabel("Singular value")
    style_axes(ax1, ygrid=True, xgrid=False)
    add_panel_title(ax1, "Spectral concentration")
    ax1.legend(fontsize=5.2, loc="upper right")
    panel_label(ax1, "(a)")

    ax2 = fig.add_subplot(gs[0, 1])
    for method_name in ["Nuclear_Norm", "Linear_OptShrink", "Rrr", "Pls"]:
        if method_name not in angles or "k=20" not in angles[method_name]:
            continue
        label = METHOD_NAME_MAP.get(method_name, method_name)
        color = METHOD_COLORS.get(label, COLOR_PRIMARY)
        marker = METHOD_MARKERS.get(label, "o")
        cos_vals = np.array(angles[method_name]["k=20"]) ** 2
        ax2.plot(
            np.arange(1, len(cos_vals) + 1),
            cos_vals,
            marker=marker,
            markersize=3.6,
            linewidth=1.2,
            color=color,
            label=label,
            zorder=3,
        )
    ax2.set_xlabel("Principal angle index")
    ax2.set_ylabel(r"$\cos^2(\theta_i)$")
    ax2.set_ylim(-0.03, 1.03)
    style_axes(ax2, ygrid=True, xgrid=False)
    add_panel_title(ax2, "Directional recovery")
    ax2.legend(fontsize=5.0, loc="upper right")
    panel_label(ax2, "(b)")

    ax3 = fig.add_subplot(gs[0, 2])
    points = []
    for method_name, stats in geo["ds1"]["methods"].items():
        if "k=20" not in stats["by_k"]:
            continue
        label = METHOD_NAME_MAP.get(method_name, method_name)
        r2 = stats["r2_global"]["mean"]
        overlap = stats["by_k"]["k=20"]["predicted_test_subspace"]["mean"]
        color = METHOD_COLORS.get(label, COLOR_PRIMARY)
        ax3.scatter(r2, overlap, s=64, color=color, edgecolors="white", linewidth=0.8, zorder=3)
        points.append((label, r2, overlap))
    for label, r2, overlap in points:
        ax3.annotate(label, (r2, overlap), xytext=(5, 5), textcoords="offset points", fontsize=5.8)
    chance = geo["ds1"]["methods"]["Nuclear_Norm"]["by_k"]["k=20"]["null"]["chance_k_over_d"]
    ax3.axhline(chance, color=COLOR_NEUTRAL, linestyle="--", linewidth=0.8, alpha=0.7)
    ax3.text(0.97, 0.08, f"chance = {chance:.3f}", fontsize=5.5, color=TEXT_MUTED,
             transform=ax3.transAxes, ha="right", va="bottom")
    ax3.set_xlabel(r"$R^2_{\mathrm{global}}$")
    ax3.set_ylabel(r"$\mathcal{O}(\hat{Y},Y)$ at $k=20$")
    style_axes(ax3, ygrid=True, xgrid=True)
    add_panel_title(ax3, "Geometry exceeds amplitude")
    panel_label(ax3, "(c)")

    plt.tight_layout()
    save_main_figure(fig, 3)


def paper_fig4_robustness():
    """Figure 4: geometry-definition, seed/CV, UKB transfer, and motion robustness."""
    geo = load_json(RESULTS / "geometry_robustness" / "geometry_robustness_summary.json")
    cv = load_json(RESULTS / "cv" / "cv_summary.json")
    diag = load_json(RESULTS / "diagnostic_analysis" / "summary.json")
    if geo is None or cv is None or diag is None:
        print("  [SKIP] paper_fig4_robustness: missing data")
        return

    fig = plt.figure(figsize=(11.4, 8.2))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.30)

    # (a) Geometry-definition robustness
    ax1 = fig.add_subplot(gs[0, 0])
    methods = ["Nuclear_Norm", "Linear_OptShrink"]
    labels = ["NN", "OptShrink"]
    metric_labels = [r"$\mathcal{O}(\hat{Y},Y)$", r"$\mathcal{O}(B,Y)$", r"$\mathcal{O}(X^\top Y,Y)$"]
    width = 0.22
    x = np.arange(len(labels))
    metric_colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT]
    for idx, key in enumerate(["predicted_test_subspace", "map_right_singular_subspace", "train_crosscov_subspace"]):
        vals = [geo["ds1"]["methods"][m]["by_k"]["k=20"][key]["mean"] for m in methods]
        ax1.bar(x + (idx - 1) * width, vals, width=width, color=metric_colors[idx], label=metric_labels[idx])
    chance = geo["ds1"]["methods"]["Nuclear_Norm"]["by_k"]["k=20"]["null"]["chance_k_over_d"]
    ax1.axhline(chance, color=COLOR_NEUTRAL, linestyle="--", linewidth=1.0)
    ax1.text(-0.25, chance + 0.012, f"chance = {chance:.3f}", fontsize=6.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Overlap")
    style_axes(ax1, ygrid=True, xgrid=False)
    add_panel_title(ax1, "Geometry-definition robustness")
    ax1.legend(fontsize=5.0, loc="upper right")
    panel_label(ax1, "(a)")

    # (b) Full split vs CV stability
    ax2 = fig.add_subplot(gs[0, 1])
    full_ds1 = diag["nuclear_norm"]["k20"]["ds1_pc_r2"]["mean"]
    full_ds2 = diag["nuclear_norm"]["k20"]["ds2_pc_r2"]["mean"]
    full_ds1_ci = diag["nuclear_norm"]["k20"]["ds1_pc_r2"]["ci95"]
    full_ds2_ci = diag["nuclear_norm"]["k20"]["ds2_pc_r2"]["ci95"]
    cv_ds1 = cv["nuclear_norm"]["pc_r2_mean_d1"]["cv_mean"]
    cv_ds2 = cv["nuclear_norm"]["pc_r2_mean_d2"]["cv_mean"]
    cv_ds1_ci = cv["nuclear_norm"]["pc_r2_mean_d1"]["cv_ci95"]
    cv_ds2_ci = cv["nuclear_norm"]["pc_r2_mean_d2"]["cv_ci95"]
    draw_grouped_bars(
        ax2,
        ["DS1", "DS2"],
        ["Seed-mean split", "5-fold CV"],
        np.array([[full_ds1, full_ds2], [cv_ds1, cv_ds2]]),
        errors=np.array([[full_ds1_ci, full_ds2_ci], [cv_ds1_ci, cv_ds2_ci]]),
        colors=[COLOR_PRIMARY, COLOR_SECONDARY],
        ylabel="PC-space $R^2$ ($k=20$)",
    )
    add_baseline(ax2, y=0.0)
    style_axes(ax2, ygrid=True, xgrid=False)
    add_panel_title(ax2, "Seed and fold stability")
    ax2.legend(fontsize=5.2, loc="upper right")
    panel_label(ax2, "(b)")

    # (c) UKB external dissociation
    ax3 = fig.add_subplot(gs[1, 0])
    ukb_methods = ["Nuclear_Norm", "Linear_OptShrink"]
    ukb_labels = ["NN", "OptShrink"]
    ukb_overlap = [geo["ukb"]["methods"][m]["by_k"]["k=20"]["predicted_test_subspace"]["mean"] for m in ukb_methods]
    ukb_r2 = [geo["ukb"]["methods"][m]["r2_global"]["mean"] for m in ukb_methods]
    x = np.arange(len(ukb_labels))
    ax3.bar(x - 0.16, ukb_overlap, width=0.32, color=COLOR_PRIMARY, label=r"$\mathcal{O}(\hat{Y},Y)$")
    ax3.bar(x + 0.16, ukb_r2, width=0.32, color=COLOR_SECONDARY, label=r"$R^2_{\mathrm{global}}$")
    ukb_chance = geo["ukb"]["methods"]["Nuclear_Norm"]["by_k"]["k=20"]["chance_k_over_d"]
    ax3.axhline(ukb_chance, color=COLOR_NEUTRAL, linestyle="--", linewidth=1.0)
    ax3.text(-0.35, ukb_chance + 0.01, f"chance = {ukb_chance:.3f}", fontsize=6.3)
    ax3.set_xticks(x)
    ax3.set_xticklabels(ukb_labels)
    ax3.set_ylabel("Magnitude")
    style_axes(ax3, ygrid=True, xgrid=False)
    add_panel_title(ax3, "UKB external dissociation")
    ax3.legend(fontsize=5.2, loc="upper right")
    panel_label(ax3, "(c)")

    # (d) UKB motion sensitivity for the selected lens
    ax4 = fig.add_subplot(gs[1, 1])
    cond_order = ["all", "low_motion_median", "low_motion_fd020", "motion_residualized"]
    cond_labels = ["All", "FD≤median", "FD≤0.2", "FD resid."]
    nn_motion = geo["ukb_motion"]["methods"]["Nuclear_Norm"]["by_condition"]
    opt_motion = geo["ukb_motion"]["methods"]["Linear_OptShrink"]["by_condition"]
    nn_overlap = [nn_motion[c]["O_predY"]["mean"] for c in cond_order]
    opt_overlap = [opt_motion[c]["O_predY"]["mean"] for c in cond_order]
    nn_r2 = [nn_motion[c]["r2_global"]["mean"] for c in cond_order]
    x = np.arange(len(cond_labels))
    ax4.plot(x, nn_overlap, "o-", color=COLOR_PRIMARY, linewidth=1.5, markersize=5, label="NN overlap")
    ax4.plot(x, opt_overlap, "s--", color=COLOR_ACCENT, linewidth=1.3, markersize=5, label="OptShrink overlap")
    ax4.plot(x, nn_r2, "d:", color=COLOR_SECONDARY, linewidth=1.3, markersize=4.5, label="NN $R^2$")
    ax4.set_xticks(x)
    ax4.set_xticklabels(cond_labels, rotation=15, ha="right")
    ax4.set_ylabel("Magnitude")
    style_axes(ax4, ygrid=True, xgrid=False)
    add_panel_title(ax4, "Motion sensitivity")
    ax4.legend(fontsize=5.0, loc="upper right")
    panel_label(ax4, "(d)")

    plt.tight_layout()
    save_main_figure(fig, 4)


def paper_fig5_biology():
    """Figure 5: ROI map, mode snapshots, and hierarchy-resolved dissociation."""
    summary = load_json(RESULTS / "smri_residual" / "summary.json")
    hierarchy = load_json(RESULTS / "hierarchy_analysis" / "hierarchy_resolved_metrics.json")
    if summary is None or hierarchy is None:
        print("  [SKIP] paper_fig5_biology: missing biological summaries")
        return

    fig = plt.figure(figsize=(11.6, 8.1))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.05], height_ratios=[1.0, 0.95], wspace=0.20, hspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    try:
        from generate_brain_figures import ATLAS_PATH, GM_NAMES_PATH
        roi_csv = RESULTS / "smri_residual" / "roi_decomposition.csv"
        roi_nifti = build_roi_coupled_nifti(roi_csv, ATLAS_PATH, GM_NAMES_PATH)
        img_arr = _render_montage_to_array(
            roi_nifti,
            cmap="coolwarm",
            threshold=0.5,
            cut_coords=[-25, -10, 5, 20, 35, 50, 65],
            colorbar=True,
            vmax=1.0,
            title="ROI coupled variance fraction",
            figsize=(5.3, 2.4),
            dpi=250,
        )
        ax1.imshow(img_arr, origin="upper")
    except Exception as e:
        ax1.text(0.5, 0.5, f"ROI map unavailable\n{e}", ha="center", va="center",
                 fontsize=7, color=TEXT_MUTED, transform=ax1.transAxes, style="italic")
        ax1.set_facecolor("#fafafa")
    ax1.set_xticks([]); ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)
    add_panel_title(ax1, "Anatomical heterogeneity")
    panel_label(ax1, "(a)")

    ax2 = fig.add_subplot(gs[0, 1])
    try:
        import sys as _sys
        _sys.path.insert(0, str(BASE / "scripts"))
        from generate_brain_figures import ATLAS_PATH, B_MATRIX_PATH, GM_NAMES_PATH, load_data
        from analyze_svd_modes import svd_decompose
        B, roi_indices, _ = load_data()
        U, S, _ = svd_decompose(B, k=3)
        total_var = np.sum(np.linalg.svd(B, compute_uv=False) ** 2)
        mode_niftis = [build_mode_weighted_nifti(U, m, roi_indices, ATLAS_PATH) for m in range(3)]
        mode_labels = [f"Mode {m+1} ({(S[m] ** 2 / total_var * 100):.1f}% var)" for m in range(3)]
        stacked = _render_stacked_montages(
            mode_niftis,
            mode_labels,
            cmap="coolwarm",
            threshold=0.10,
            cut_coords=[-25, -10, 5, 20, 35, 50, 65],
            figsize=(5.4, 2.1),
            dpi=250,
        )
        ax2.imshow(stacked)
    except Exception:
        ax2.text(0.5, 0.5,
                 "Mode loading maps require B matrix\n(regenerate after model training)",
                 ha="center", va="center", fontsize=7, color=TEXT_MUTED,
                 transform=ax2.transAxes, style="italic")
        ax2.set_facecolor("#fafafa")
    ax2.set_xticks([]); ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)
    add_panel_title(ax2, "Mode 1\u20133 simplified maps")
    panel_label(ax2, "(b)")

    ax3 = fig.add_subplot(gs[1, :])
    primary = hierarchy["summary"]["primary"]
    tiers = ["sensorimotor", "heteromodal", "transmodal"]
    x = np.arange(len(tiers))
    overlap = [primary[t]["O_mean"] for t in tiers]
    r2 = [primary[t]["R2_mean"] for t in tiers]
    width = 0.34
    ax3.bar(x - width / 2, overlap, width=width, color=COLOR_PRIMARY, label=r"$\mathcal{O}$")
    ax3.bar(x + width / 2, r2, width=width, color=COLOR_SECONDARY, label=r"$R^2$")
    for idx, tier in enumerate(tiers):
        chance = primary[tier]["chance_mean"]
        ax3.hlines(chance, idx - 0.45, idx + 0.45, colors="#999999", linestyles="--", linewidth=0.8, zorder=4)
    ax3.set_xticks(x)
    ax3.set_xticklabels(["Sensorimotor", "Heteromodal", "Transmodal"])
    ax3.set_ylabel("Magnitude")
    style_axes(ax3, ygrid=True, xgrid=False)
    add_panel_title(ax3, "Hierarchy-resolved dissociation")
    ax3.legend(fontsize=5.5, loc="upper right")
    panel_label(ax3, "(c)")

    plt.tight_layout()
    save_main_figure(fig, 5)


def paper_fig6_utility():
    """Figure 6: rank sweep, occupancy-vs-prediction, and clinical utility."""
    summary = load_json(RESULTS / "smri_residual" / "summary.json")
    geo = load_json(RESULTS / "geometry_robustness" / "geometry_robustness_summary.json")
    if summary is None or geo is None:
        print("  [SKIP] paper_fig6_utility: missing utility summaries")
        return

    a1 = summary["A1_variance_decomposition"]
    a3_primary = summary["A3_clinical_relevance"]
    a3_sweep = summary["A3_rank_sweep"]
    ranks = sorted(int(r) for r in a1.keys())
    coupled = [a1[str(r)]["coupled_var_frac"] for r in ranks]
    r2_global = geo["ds1"]["methods"]["Nuclear_Norm"]["r2_global"]["mean"]

    fig = plt.figure(figsize=(11.4, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 0.95], wspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ranks, coupled, "o-", color=COLOR_PRIMARY, linewidth=1.5, markersize=5)
    ax1.axhline(r2_global, color=COLOR_SECONDARY, linestyle="--", linewidth=1.2, label=r"$R^2_{\mathrm{global}}$")
    ax1.set_xlabel("Rank $r$")
    ax1.set_ylabel("GM coupled variance fraction")
    ax1.set_ylim(0, 1.0)
    style_axes(ax1, ygrid=True, xgrid=False)
    add_panel_title(ax1, "GM occupancy exceeds prediction")
    ax1.legend(fontsize=5.5, loc="upper left")
    panel_label(ax1, "(a)")

    ax2 = fig.add_subplot(gs[0, 1])
    sweep_ranks = sorted(int(r) for r in a3_sweep.keys())
    coupled_auc = [a3_sweep[str(r)]["coupled"]["auc"] for r in sweep_ranks]
    uncoupled_auc = [a3_sweep[str(r)]["uncoupled"]["auc"] for r in sweep_ranks]
    full_auc = a3_primary["full"]["auc"]
    ax2.plot(sweep_ranks, coupled_auc, "o-", color=COLOR_PRIMARY, linewidth=1.5, markersize=5, label="Coupled")
    ax2.plot(sweep_ranks, uncoupled_auc, "s--", color=COLOR_NEUTRAL, linewidth=1.3, markersize=5, label="Uncoupled")
    ax2.axhline(full_auc, color=COLOR_DARK, linestyle=":", linewidth=1.0, label=f"Full GM ({full_auc:.3f})")
    ax2.set_xlabel("Rank $r$")
    ax2.set_ylabel("AUC")
    ax2.set_ylim(0.5, 0.9)
    style_axes(ax2, ygrid=True, xgrid=False)
    add_baseline(ax2, y=0.5, style="dashed", alpha=0.45)
    add_panel_title(ax2, "Clinical utility across rank")
    ax2.legend(fontsize=5.2, loc="lower right")
    panel_label(ax2, "(b)")

    ax3 = fig.add_subplot(gs[0, 2])
    names = ["full", "coupled", "uncoupled"]
    labels = ["Full\nGM", "Coupled", "Uncoupled"]
    vals = [a3_primary[n]["auc"] for n in names]
    ci_lo = [a3_primary[n]["auc"] - a3_primary[n]["auc_ci_lo"] for n in names]
    ci_hi = [a3_primary[n]["auc_ci_hi"] - a3_primary[n]["auc"] for n in names]
    colors = [COLOR_DARK, COLOR_PRIMARY, COLOR_NEUTRAL]
    bars = ax3.bar(np.arange(3), vals, color=colors, width=0.62, yerr=[ci_lo, ci_hi],
                   capsize=3, edgecolor="white", linewidth=0.7, error_kw={"linewidth": 1.0})
    ax3.set_xticks(np.arange(3))
    ax3.set_xticklabels(labels, fontsize=6.2)
    ax3.set_ylabel("AUC")
    ax3.set_ylim(0.5, 0.9)
    style_axes(ax3, ygrid=True, xgrid=False)
    add_baseline(ax3, y=0.5, style="dashed", alpha=0.45)
    for bar, value in zip(bars, vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}",
                 ha="center", va="bottom", fontsize=6.2)
    add_panel_title(ax3, "Primary-rank utility check")
    panel_label(ax3, "(c)")

    plt.tight_layout()
    save_main_figure(fig, 6)


# ---- Main ----

def main():
    apply_nature_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING ZERO-SUPPLEMENT MAIN FIGURES")
    print(f"  Output: {FIG_DIR}")
    print("=" * 60)

    paper_fig1_overview()
    paper_fig2_lens_selection()
    paper_fig3_core_dissociation()
    paper_fig4_robustness()
    paper_fig5_biology()
    paper_fig6_utility()

    # Supplementary figures
    print("\n--- Supplementary figures ---")
    try:
        fig_robustness_composite()
        print("  figS2 panels saved")
    except Exception as e:
        print(f"  figS2 skipped: {e}")

    print("\nDone. Figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
