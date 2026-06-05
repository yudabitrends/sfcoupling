"""
Figure A: Method benchmark and geometric structure (6 panels).

Panel A: PC-R^2 at k=20 for 7 methods x 3 cohorts (grouped bars)
Panel B: Edge-R^2 vs PC-R^2 on DS2 (scatter; highlights the metric pivot)
Panel C: PC-R^2 vs k sweep for spectral methods on DS1
Panel D: Bootstrap singular value spectrum with 95% CI and BBP threshold
Panel E: Principal angle cosines decay for 5 methods (k=20)
Panel F: Subspace overlap vs PC-R^2 bar (geometry-amplitude dissociation)

Data sources:
  results/bootstrap_sv/sv_stats.json
  results/subspace_analysis/principal_angles.json
  Manuscript Table 1 (hardcoded; values verified against tex)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/users/ybi3/sfcoupling")
sys.path.insert(0, str(ROOT / "scripts" / "reviewer_revision_figures"))
from _style import (METHOD_COLORS, COHORT_COLORS, FIG_DOUBLE_TALL, OKABE_ITO,
                     apply_style, panel_label, save_figure)

# --- Table 1 values (hardcoded from paper_vince_framing.tex:493-500 + revision updates) ---
# mean PC-R^2 at k=20 with std (seven seeds)
TABLE1 = {
    "Ridge":        {"DS1": (0.025, 0.001), "DS2": (0.025, 0.001), "UKB": (0.058, 0.000)},
    "MLP":          {"DS1": (0.060, 0.005), "DS2": (0.033, 0.006), "UKB": (0.056, 0.000)},
    "RRR":          {"DS1": (0.037, 0.000), "DS2": (0.029, 0.000), "UKB": (0.056, 0.000)},
    "PLS":          {"DS1": (0.045, 0.001), "DS2": (0.020, 0.000), "UKB": (0.050, 0.000)},
    "Nuclear Norm": {"DS1": (0.056, 0.001), "DS2": (0.041, 0.001), "UKB": (0.058, 0.000)},
    "OptShrink":    {"DS1": (0.051, 0.000), "DS2": (0.040, 0.000), "UKB": (0.058, 0.000)},
    "NN-Init MLP":  {"DS1": (0.066, 0.002), "DS2": (0.041, 0.004), "UKB": (0.059, 0.000)},
}

# Edge R^2 from Table 1 (same rows) — DS2 column, used to show Edge vs PC on DS2
TABLE1_EDGE_DS2 = {
    "Ridge":        -0.006,
    "MLP":          -0.007,
    "RRR":          -0.007,
    "PLS":          -0.014,
    "Nuclear Norm": +0.001,
    "OptShrink":    +0.001,
    "NN-Init MLP":  -0.003,
}

# PC-R^2 k-sweep from Table 3 (tab:all_pca_k_ds1) — DS1 test only
K_SWEEP_DS1 = {
    "Ridge":        {5: 0.0163, 10: 0.0200, 20: 0.0255, 50: 0.0255},
    "RRR":          {5: 0.0825, 10: 0.0550, 20: 0.0372, 50: 0.0088},
    "PLS":          {5: 0.0965, 10: 0.0712, 20: 0.0447, 50: 0.0183},
    "Nuclear Norm": {5: 0.0915, 10: 0.0710, 20: 0.0559, 50: 0.0269},
    "OptShrink":    {5: 0.0896, 10: 0.0667, 20: 0.0508, 50: 0.0229},
}

# Subspace overlap at k=20 (from the tex, sec:methods_bootstrap item 9)
DISSOC = {
    "Nuclear Norm": {"O": 0.387, "R2": 0.054},
    "OptShrink":    {"O": 0.400, "R2": 0.035},
    "RRR":          {"O": 0.436, "R2": 0.041},
    "PLS":          {"O": 0.294, "R2": 0.045},
}

# BBP detectability threshold for this design:
#   gamma = 99/805 ≈ 0.123, ell_c = gamma^{+1/4} ≈ 0.592 (the value where the overlap
#   factor 1 - gamma/ell^4 turns positive); 18 bootstrap-stable modes lie above it.
# Expressed in raw singular-value units requires the noise scale sigma. Using
# sigma^2 ≈ 1.04 (from paper sec:results_dissociation_rmt) gives sigma ≈ 1.02,
# so the BBP threshold in raw SV units is approximately ell_c * sqrt(n) * sigma.
# Since the bootstrap SVs are from Ridge on standardized data with N=805, the
# cleanest thing is to report the signal-to-noise threshold directly. We draw
# a horizontal line at the empirical noise floor estimated from the last 10
# bootstrap mean SVs (which are the tail).


def load_data():
    sv = json.load(open(ROOT / "results" / "bootstrap_sv" / "sv_stats.json"))
    pa = json.load(open(ROOT / "results" / "subspace_analysis" / "principal_angles.json"))
    return sv, pa


def panel_a_benchmark(ax):
    methods = list(TABLE1.keys())
    cohorts = ["DS1", "DS2", "UKB"]
    n_methods = len(methods)
    n_cohorts = len(cohorts)
    x = np.arange(n_methods)
    bar_w = 0.25
    offsets = [-bar_w, 0, bar_w]

    for i, cohort in enumerate(cohorts):
        means = [TABLE1[m][cohort][0] for m in methods]
        errs = [TABLE1[m][cohort][1] for m in methods]
        ax.bar(x + offsets[i], means, bar_w, yerr=errs,
               label=cohort, color=COHORT_COLORS[cohort],
               edgecolor="white", linewidth=0.5,
               error_kw=dict(lw=0.8, capsize=1.5, ecolor=OKABE_ITO["black"]))
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel(r"PC-$R^2$ at $k{=}20$")
    ax.set_ylim(0, 0.080)
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(title="Cohort", loc="upper left",
              handlelength=1, handletextpad=0.4, borderpad=0.25,
              frameon=True, framealpha=0.92, edgecolor="none",
              facecolor="white", ncol=1)
    ax.set_title("Method benchmark across 3 cohorts")


def panel_b_edge_vs_pc(ax):
    methods = list(TABLE1_EDGE_DS2.keys())
    pc = np.array([TABLE1[m]["DS2"][0] for m in methods])
    edge = np.array([TABLE1_EDGE_DS2[m] for m in methods])
    # Carefully tuned offsets so no labels overlap each other
    offsets = {
        "Ridge":        (6, -9),
        "MLP":          (6, 9),
        "RRR":          (-6, -9),
        "PLS":          (7, 2),
        "Nuclear Norm": (-7, 9),
        "OptShrink":    (8, 0),
        "NN-Init MLP":  (-7, -9),
    }
    for i, m in enumerate(methods):
        ax.scatter(edge[i], pc[i], s=60,
                   color=METHOD_COLORS[m], edgecolor="black", linewidth=0.6,
                   zorder=4)
        dx, dy = offsets[m]
        ha = "left" if dx > 0 else "right"
        ax.annotate(m, (edge[i], pc[i]), xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=6.2, va="center", ha=ha, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor="none",
                              alpha=0.85))
    ax.axhline(0, color=OKABE_ITO["grey"], lw=0.5, ls="--", alpha=0.6)
    ax.axvline(0, color=OKABE_ITO["grey"], lw=0.5, ls="--", alpha=0.6)
    # Shaded "overfitting quadrant"
    ax.axvspan(-0.028, 0, alpha=0.07, color=OKABE_ITO["vermillion"], zorder=1)
    ax.text(-0.025, 0.050, "overfits in\nedge space",
            fontsize=6, color=OKABE_ITO["vermillion"],
            ha="left", va="top", fontweight="bold")
    ax.set_xlabel(r"Edge-$R^2$ on DS2")
    ax.set_ylabel(r"PC-$R^2$ on DS2 ($k{=}20$)")
    ax.set_xlim(-0.028, 0.013)
    ax.set_ylim(-0.008, 0.058)
    ax.set_title("Edge vs. PC metrics disagree on DS2")


def panel_c_k_sweep(ax):
    ks = [5, 10, 20, 50]
    for m, vals in K_SWEEP_DS1.items():
        ys = [vals[k] for k in ks]
        ax.plot(ks, ys, "o-", color=METHOD_COLORS[m], label=m,
                lw=1.5, markersize=4.5, markeredgecolor="white",
                markeredgewidth=0.4)
    ax.set_xlabel(r"PC-space dimensionality $k$")
    ax.set_ylabel(r"DS1 test PC-$R^2$")
    ax.set_xticks(ks)
    ax.set_ylim(0, 0.115)
    ax.set_title(r"Leading-block vs. long-tail tradeoff")
    ax.legend(loc="upper right", handlelength=1.4, handletextpad=0.4,
              frameon=True, framealpha=0.92, edgecolor="none",
              facecolor="white", fontsize=6.5)


def panel_d_bootstrap_sv(ax, sv_stats):
    n_show = 20
    means = np.array(sv_stats["bootstrap_mean"])[:n_show]
    lo = np.array(sv_stats["ci95_lo"])[:n_show]
    hi = np.array(sv_stats["ci95_hi"])[:n_show]
    x = np.arange(1, n_show + 1)

    # Shade the BBP-detectable region first (so errorbars sit on top)
    ax.axvspan(0.5, 18.5, alpha=0.12, color=OKABE_ITO["green"], zorder=1)

    ax.errorbar(x, means, yerr=[means - lo, hi - means],
                fmt="o", color=OKABE_ITO["blue"],
                ecolor=OKABE_ITO["blue"], elinewidth=1, capsize=2,
                markersize=4, markeredgecolor="white", markeredgewidth=0.3,
                zorder=3)
    # Connect with a thin line
    ax.plot(x, means, color=OKABE_ITO["blue"], lw=0.6, alpha=0.4, zorder=2)
    # Label the BBP region near the top right of the shaded area
    ax.text(9.0, means[0] * 1.08,
            "BBP-detectable\n(18 modes)", fontsize=6,
            color=OKABE_ITO["green"], ha="center", va="bottom",
            fontweight="bold")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Bootstrap singular value")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(0, means[0] * 1.3)
    ax.set_title("Ridge bootstrap spectrum (200 resamples)", pad=10)


def panel_e_principal_angles(ax, pa_data):
    methods = ["Nuclear_Norm", "OptShrink", "Rrr", "Pls", "Ridge"]
    display = {
        "Nuclear_Norm": "Nuclear Norm",
        "OptShrink": "OptShrink",
        "Linear_OptShrink": "OptShrink",
        "Rrr": "RRR",
        "Pls": "PLS",
        "Ridge": "Ridge",
    }
    json_keys = {"Nuclear_Norm": "Nuclear_Norm",
                 "OptShrink": "Linear_OptShrink",
                 "Rrr": "Rrr", "Pls": "Pls", "Ridge": "Ridge"}
    for m in methods:
        jk = json_keys[m]
        angles = pa_data[jk]["k=20"]
        disp = display[jk]
        ax.plot(range(1, 21), angles, "o-",
                color=METHOD_COLORS[disp], label=disp,
                lw=1.4, markersize=3.2, markeredgecolor="white",
                markeredgewidth=0.3)
    # Leading-block threshold line
    ax.axhline(0.93, ls="--", color=OKABE_ITO["grey"], lw=0.6, alpha=0.8)
    ax.text(1.5, 0.97, r"leading block ($\cos\theta > 0.93$)",
            fontsize=6, color=OKABE_ITO["grey"], ha="left", va="center",
            fontweight="bold")
    ax.set_xlabel("Principal angle index $i$")
    ax.set_ylabel(r"$\cos\theta_i$")
    ax.set_ylim(-0.05, 1.12)
    ax.set_xticks([1, 5, 10, 15, 20])
    # Legend in lower left where curves have already decayed
    ax.legend(loc="lower left", handlelength=1.4, handletextpad=0.4,
              fontsize=6.5, ncol=1,
              framealpha=0.92, frameon=True,
              edgecolor="none", facecolor="white")
    ax.set_title("Leading-block alignment across methods")


def panel_f_dissociation(ax):
    methods = list(DISSOC.keys())
    x = np.arange(len(methods))
    bar_w = 0.38
    os = [d["O"] for d in DISSOC.values()]
    r2s = [d["R2"] for d in DISSOC.values()]
    bars_o = ax.bar(x - bar_w/2, os, bar_w,
                     label=r"Subspace overlap $\mathcal{O}$",
                     color=OKABE_ITO["blue"],
                     edgecolor="white", linewidth=0.5)
    bars_r = ax.bar(x + bar_w/2, r2s, bar_w, label=r"PC-$R^2$",
                     color=OKABE_ITO["vermillion"],
                     edgecolor="white", linewidth=0.5)
    # Value labels above each bar
    for b in bars_o:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                f"{b.get_height():.2f}", ha="center", va="bottom",
                fontsize=6.5, fontweight="bold",
                color=OKABE_ITO["blue"])
    for b in bars_r:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                f"{b.get_height():.3f}", ha="center", va="bottom",
                fontsize=6, color=OKABE_ITO["vermillion"])
    # Chance line and label (centered above the axis)
    ax.axhline(0.0145, ls="--", lw=0.7, color=OKABE_ITO["grey"], alpha=0.8)
    ax.text(0.05, 0.045,
            r"null $\mathcal{O}_{\rm chance}\approx 0.0145$",
            fontsize=6, color=OKABE_ITO["grey"],
            ha="left", va="center", fontweight="bold",
            transform=ax.get_yaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=18, ha="right")
    ax.set_ylabel(r"Value at $k{=}20$")
    ax.set_ylim(0, 0.62)
    ax.legend(loc="upper right", handlelength=1.2, handletextpad=0.4,
              frameon=True, framealpha=0.92, edgecolor="none",
              facecolor="white")
    ax.set_title(r"Directional overlap vs.\ amplitude recovery across methods",
                 fontsize=8.5)


def main():
    apply_style()
    sv_stats, pa_data = load_data()

    fig = plt.figure(figsize=(7.5, 9.2))
    gs = fig.add_gridspec(3, 2, hspace=0.95, wspace=0.40,
                           left=0.09, right=0.97, top=0.965, bottom=0.05)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])

    panel_a_benchmark(ax_a)
    panel_b_edge_vs_pc(ax_b)
    panel_c_k_sweep(ax_c)
    panel_d_bootstrap_sv(ax_d, sv_stats)
    panel_e_principal_angles(ax_e, pa_data)
    panel_f_dissociation(ax_f)

    for ax, lab in zip((ax_a, ax_b, ax_c, ax_d, ax_e, ax_f),
                       "ABCDEF"):
        panel_label(ax, lab, x=-0.18, y=1.20)

    outputs = save_figure(fig, "figure_benchmark_geometry", out_dir=ROOT / "IMAG" / "figure")
    print("Saved:")
    for p in outputs:
        print(f"  {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
