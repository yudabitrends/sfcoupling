"""
Figure C: Linearity, clinical utility, and theory (4 panels).

Panel A: Nonlinearity test - NN vs NN+MLP residual vs MLP at k=5/10/20 on DS2
Panel B: Clinical schizophrenia AUC bars (Coupled / Full / Uncoupled)
Panel C: RMT simulation - ordering holds across 20 draws
Panel D: Subspace variance occupancy vs rank (GM occupancy, FNC occupancy,
         PCA-maximum reference)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/users/ybi3/sfcoupling")
sys.path.insert(0, str(ROOT / "scripts" / "reviewer_revision_figures"))
from _style import (METHOD_COLORS, FIG_DOUBLE_COL, OKABE_ITO,
                     apply_style, panel_label, save_figure)


# From tab:nn_mlp_all_k in manuscript (Supplementary, lines 1305-1320 of revised tex)
# Format: (mean, std) across 7 seeds
NONLIN_DS1 = {
    "NN-Init MLP": {5: (0.1176, 0.0040), 10: (0.0923, 0.0014), 20: (0.0660, 0.0016)},
    "MLP":         {5: (0.1180, 0.0058), 10: (0.0934, 0.0046), 20: (0.0595, 0.0046)},
}
NONLIN_DS2 = {
    "NN-Init MLP": {5: (0.0035, 0.0092), 10: (0.0413, 0.0063), 20: (0.0407, 0.0043)},
    "MLP":         {5: (0.0007, 0.0127), 10: (0.0406, 0.0078), 20: (0.0328, 0.0060)},
}
# From tab:all_pca_k_ds1 - we also want NN alone at k=20 (DS1=0.0559±0.0008)
NN_ONLY_DS2 = {5: 0.0393, 10: 0.0384, 20: 0.0413}  # from Table 1 and text

# Clinical AUCs (from tex §4.5: coupled 0.735±0.014, full 0.724±0.013, uncoupled 0.638±0.016)
CLINICAL = {
    "Coupled":    (0.735, 0.014),
    "Full GM":    (0.724, 0.013),
    "Uncoupled":  (0.638, 0.016),
}

# Subspace variance occupancy from manuscript sec:results_supporting_lowrank (line 543-545)
# accumulated at ranks 3/5/10/20/38
RANKS = [3, 5, 10, 20, 38]
OCC = {
    "GM predicted":  [0.016, 0.045, 0.111, 0.277, 0.795],
    "FNC observed":  [0.280, 0.312, 0.370, 0.420, 0.470],
    "FNC PCA (reference)": [0.293, 0.339, 0.418, 0.491, 0.572],
}


def load_rmt():
    return json.load(open(ROOT / "results" / "reviewer_revision"
                           / "M5_rmt_simulation.json"))


def panel_a_nonlinearity(ax):
    """Bar: DS2 PC-R^2 for NN vs NN+MLP vs MLP at k=5,10,20."""
    ks = [5, 10, 20]
    x = np.arange(len(ks))
    bar_w = 0.25
    offsets = [-bar_w, 0, bar_w]

    methods_order = ["MLP", "NN-Init MLP", "Nuclear Norm"]
    colors_row = [METHOD_COLORS["MLP"], METHOD_COLORS["NN-Init MLP"],
                  METHOD_COLORS["Nuclear Norm"]]

    # NN alone on DS2 from NONLIN_DS2? Actually NN_ONLY is different — we need
    # NN alone values too. From tab:all_pca_k_ds1 Nuclear Norm: k=5=0.0915±0
    # (DS1). On DS2: from main Table 1, NN at k=20 DS2 is 0.041. For k=5 and
    # k=10 on DS2 we use the tab:all_pca_k_ds1 which only reports DS1 and the
    # tab:nn_mlp_all_k which reports NN-Init MLP and MLP. Direct NN on DS2
    # at k=5 and k=10 is not in the tables; we therefore restrict panel A to
    # k=20 comparison, which is the crucial one anyway.

    # Actually, restructure to: bar plot comparing three methods at k=5,10,20 on DS2
    for i, method in enumerate(methods_order):
        if method == "Nuclear Norm":
            vals = [None, None, 0.041]  # only k=20 on DS2 from Table 1
            errs = [0, 0, 0.001]
        else:
            vals = [NONLIN_DS2[method][k][0] for k in ks]
            errs = [NONLIN_DS2[method][k][1] for k in ks]
        # Plot only non-None values
        for j, k in enumerate(ks):
            if vals[j] is None:
                continue
            ax.bar(x[j] + offsets[i], vals[j], bar_w,
                   yerr=errs[j],
                   color=colors_row[i],
                   label=method if j == (2 if method == "Nuclear Norm" else 0) else None,
                   edgecolor="white", linewidth=0.5,
                   error_kw=dict(lw=0.6, capsize=1.5,
                                 ecolor=OKABE_ITO["black"]))

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$k{{=}}{k}$" for k in ks])
    ax.set_ylabel(r"DS2 external PC-$R^2$")
    ax.set_ylim(-0.012, 0.062)
    ax.legend(loc="upper left", handlelength=1.2, handletextpad=0.4,
              frameon=True, framealpha=0.92, edgecolor="none",
              facecolor="white", fontsize=6.5)
    # Annotate k=20 agreement — arrow below the bars
    ax.annotate("NN = NN-Init MLP\nat $k{=}20$", xy=(x[2], 0.041),
                xytext=(x[2], 0.056),
                ha="center", fontsize=6, color=OKABE_ITO["green"],
                fontweight="bold",
                arrowprops=dict(arrowstyle="-[,widthB=0.8",
                                 color=OKABE_ITO["green"], lw=0.7))
    ax.set_title("Linear spectral fit captures the dominant DS2 signal",
                 fontsize=8.5)


def panel_b_clinical(ax):
    """Clinical AUC bar chart."""
    names = list(CLINICAL.keys())
    means = np.array([CLINICAL[n][0] for n in names])
    errs = np.array([CLINICAL[n][1] for n in names])
    colors = [OKABE_ITO["blue"], OKABE_ITO["grey"], OKABE_ITO["vermillion"]]

    x = np.arange(len(names))
    bars = ax.bar(x, means, 0.55, yerr=errs, color=colors,
                   edgecolor="white", linewidth=0.5,
                   error_kw=dict(lw=1, capsize=3, ecolor=OKABE_ITO["black"]))
    # Value labels
    for b, m, e in zip(bars, means, errs):
        ax.text(b.get_x() + b.get_width()/2, m + e + 0.008,
                f"{m:.3f}", ha="center", va="bottom",
                fontsize=7, fontweight="bold")

    # Chance line
    ax.axhline(0.5, color="grey", lw=0.6, ls="--")
    ax.text(len(names) - 0.5, 0.505, "chance",
            fontsize=6, color="grey", ha="right", va="bottom")

    # Highlight coupled-vs-uncoupled gap
    delta_coupled_uncoupled = means[0] - means[2]
    ax.annotate("", xy=(2, means[2]), xytext=(0, means[0]),
                arrowprops=dict(arrowstyle="<->",
                                 color=OKABE_ITO["vermillion"], lw=1.2,
                                 connectionstyle="arc3,rad=0.15"))
    ax.text(1, (means[0] + means[2]) / 2,
            f"$\\Delta$AUC\n$\\approx {delta_coupled_uncoupled:.2f}$",
            ha="center", va="center", fontsize=7,
            color=OKABE_ITO["vermillion"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=OKABE_ITO["vermillion"], lw=0.6))

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.45, 0.83)
    ax.set_ylabel("SZ vs. HC AUC (5-fold nested CV)")
    ax.set_title("Coupled subspace concentrates diagnostic signal",
                 fontsize=8.5)


def panel_c_rmt(ax, rmt):
    """Scatter: per-draw O vs R^2 from RMT simulation, vs observed."""
    O_sim = np.array(rmt["O_values"])
    R2_sim = np.array(rmt["R2_values"])

    # Simulated points
    ax.scatter(R2_sim, O_sim, s=30, color=OKABE_ITO["skyblue"],
               edgecolor=OKABE_ITO["blue"], linewidth=0.6,
               label=f"Simulated (n={len(O_sim)})", zorder=3)
    # Observed
    ax.scatter(0.058, 0.391, s=90, color=OKABE_ITO["vermillion"],
               edgecolor="black", linewidth=0.8, marker="*",
               label="Observed", zorder=4)
    # Theory prediction
    ax.scatter(0.077, 0.299, s=70, color=OKABE_ITO["green"],
               edgecolor="black", linewidth=0.5, marker="s",
               label="Spiked-matrix theory", zorder=4)

    # O = R^2 diagonal
    xmin, xmax = -0.20, 0.12
    ax.plot([xmin, xmax], [xmin, xmax], color="grey", lw=0.6, ls=":",
            label=r"$\mathcal{O} = R^2$")

    ax.set_xlabel(r"$R^2$ at $k{=}20$")
    ax.set_ylabel(r"$\mathcal{O}$ at $k{=}20$")
    ax.set_xlim(-0.23, 0.13)
    ax.set_ylim(-0.08, 0.52)
    ax.legend(loc="lower right", handlelength=1.2, handletextpad=0.4,
              fontsize=6.5, frameon=True, framealpha=0.92,
              edgecolor="none", facecolor="white")
    ax.set_title("RMT ordering holds; magnitudes do not", fontsize=8.5)


def panel_d_occupancy(ax):
    """Line: subspace variance occupancy vs rank."""
    colors = {
        "GM predicted": OKABE_ITO["blue"],
        "FNC observed": OKABE_ITO["vermillion"],
        "FNC PCA (reference)": OKABE_ITO["grey"],
    }
    markers = {
        "GM predicted": "o",
        "FNC observed": "s",
        "FNC PCA (reference)": "D",
    }
    ls_map = {
        "GM predicted": "-",
        "FNC observed": "-",
        "FNC PCA (reference)": "--",
    }
    for name, vals in OCC.items():
        ax.plot(RANKS, vals, marker=markers[name], linestyle=ls_map[name],
                color=colors[name], label=name, lw=1.5, markersize=5,
                markeredgecolor="white", markeredgewidth=0.4)

    # Highlight the leading block and effective rank
    ax.axvline(3, color=OKABE_ITO["green"], lw=0.5, ls=":", alpha=0.6)
    ax.axvline(38, color=OKABE_ITO["orange"], lw=0.5, ls=":", alpha=0.6)
    ax.text(3.3, 0.05, "leading\nblock", fontsize=6,
            color=OKABE_ITO["green"], ha="left", va="bottom")
    ax.text(36.5, 0.05, "$r_{\\rm eff}$", fontsize=6,
            color=OKABE_ITO["orange"], ha="right", va="bottom")

    ax.set_xlabel(r"Rank $r$")
    ax.set_ylabel(r"Subspace variance fraction")
    ax.set_xticks(RANKS)
    ax.set_ylim(0, 0.90)
    ax.legend(loc="upper left", handlelength=2.0, handletextpad=0.4,
              fontsize=6.5, frameon=True, framealpha=0.92,
              edgecolor="none", facecolor="white")
    ax.set_title("Occupancy vs. rank reveals coupling geometry",
                 fontsize=8.5)


def main():
    apply_style()

    # 3-panel layout: linearity, clinical, occupancy.
    # The random-matrix simulation scatter (former panel C) is moved out of the
    # main narrative to the Supplement (Appendix S.D), per reviewer request.
    fig = plt.figure(figsize=(11.0, 3.6))
    gs = fig.add_gridspec(1, 3, wspace=0.40,
                          left=0.07, right=0.985, top=0.86, bottom=0.16)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_a_nonlinearity(ax_a)
    panel_b_clinical(ax_b)
    panel_d_occupancy(ax_c)

    for ax, lab in zip((ax_a, ax_b, ax_c), "ABC"):
        panel_label(ax, lab, x=-0.16, y=1.16)

    out_dir = ROOT / "IMAG" / "figure"
    outputs = save_figure(fig, "figure_linearity_clinical", out_dir=out_dir)
    print("Saved:")
    for p in outputs:
        print(f"  {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
