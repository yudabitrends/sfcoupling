"""
Figure B: Revision analyses (4 panels).

Panel A: Residualization sensitivity sweep (DS1 and DS2, 4 confound sets × 3 k)
Panel B: Scrambled-GM null distribution vs observed (3 k values, DS1 + DS2)
Panel C: Subject-level bootstrap CIs on DS2 PC-R^2 (NN / PLS / RRR / Ridge)
Panel D: NN - PLS paired DS2 difference distribution (bootstrap histogram)

All data comes from the reviewer_revision experiment JSONs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/users/ybi3/sfcoupling")
sys.path.insert(0, str(ROOT / "scripts" / "reviewer_revision_figures"))
from _style import (METHOD_COLORS, COHORT_COLORS, FIG_DOUBLE_COL, OKABE_ITO,
                     apply_style, panel_label, save_figure)


def load_data():
    resid = json.load(open(ROOT / "results" / "reviewer_revision"
                            / "M1_resid_sensitivity.json"))
    null = json.load(open(ROOT / "results" / "reviewer_revision"
                           / "M4_scrambled_null.json"))
    # Use the seed-averaged bootstrap (R2.3 fix) so panel C matches Tab S.C
    boot = json.load(open(ROOT / "results" / "reviewer_revision"
                           / "M3_retention_bootstrap_7seeds.json"))
    return resid, null, boot


def panel_a_residualization(ax, resid):
    """Grouped bars: DS1 PC-R^2 as function of confound removal, for k=5,10,20."""
    mode_labels = {
        "none": "raw",
        "age": "+age",
        "age_sex": "+age, sex",
        "age_sex_totgm": "+age, sex,\ntotal-GM",
    }
    modes = ["none", "age", "age_sex", "age_sex_totgm"]
    ks = [5, 10, 20]
    k_colors = [OKABE_ITO["green"], OKABE_ITO["skyblue"], OKABE_ITO["blue"]]

    x = np.arange(len(modes))
    bar_w = 0.25
    offsets = [-bar_w, 0, bar_w]

    row_by_mode = {r["mode"]: r for r in resid}
    for i, k in enumerate(ks):
        means = [row_by_mode[m][f"DS1_pc_k{k}_mean"] for m in modes]
        stds = [row_by_mode[m][f"DS1_pc_k{k}_std"] for m in modes]
        ax.bar(x + offsets[i], means, bar_w, yerr=stds,
               color=k_colors[i], label=rf"$k{{=}}{k}$",
               edgecolor="white", linewidth=0.5,
               error_kw=dict(lw=0.6, capsize=1.5, ecolor=OKABE_ITO["black"]))

    ax.set_xticks(x)
    ax.set_xticklabels([mode_labels[m] for m in modes], fontsize=6.5)
    ax.set_ylabel(r"DS1 test PC-$R^2$")
    ax.set_ylim(0, 0.050)
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(loc="upper right", handlelength=1.2, handletextpad=0.4,
              title="PC target", fontsize=6.5,
              frameon=True, framealpha=0.92, edgecolor="none",
              facecolor="white")

    # Arrow annotating the drop — placed on left side to avoid legend
    raw_k20 = row_by_mode["none"]["DS1_pc_k20_mean"]
    full_k20 = row_by_mode["age_sex_totgm"]["DS1_pc_k20_mean"]
    drop_pct = (raw_k20 - full_k20) / raw_k20 * 100
    ax.annotate(f"$-${drop_pct:.0f}%",
                xy=(3 + bar_w, full_k20),
                xytext=(0.8, raw_k20 + 0.007),
                fontsize=7, color=OKABE_ITO["vermillion"],
                ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->",
                                color=OKABE_ITO["vermillion"], lw=0.8))
    ax.set_title(r"Residualization discards $\sim$37--44% of raw PC-$R^2$")


def panel_b_scrambled_null(ax, null):
    """Bar with error: observed vs null for each k (DS1 only, the cleaner case)."""
    ks = [5, 10, 20]
    x = np.arange(len(ks))
    bar_w = 0.35

    obs = [null["observed"][f"k={k}"]["DS1"] for k in ks]
    null_means = [null["null"][f"k={k}"]["DS1_null_mean"] for k in ks]
    null_stds = [null["null"][f"k={k}"]["DS1_null_std"] for k in ks]

    bars_obs = ax.bar(x - bar_w/2, obs, bar_w,
                       color=OKABE_ITO["blue"], label="Observed (Ridge)",
                       edgecolor="white", linewidth=0.5)
    bars_null = ax.bar(x + bar_w/2, null_means, bar_w, yerr=null_stds,
                        color=OKABE_ITO["grey"], label="Scrambled-GM null",
                        edgecolor="white", linewidth=0.5,
                        error_kw=dict(lw=0.7, capsize=2,
                                      ecolor=OKABE_ITO["black"]))

    # Add sigma separation annotation
    for i, k in enumerate(ks):
        n_sigma = (obs[i] - null_means[i]) / null_stds[i]
        ax.text(i, max(obs[i], 0.005) + 0.005,
                f"{n_sigma:.1f}$\\sigma$",
                ha="center", fontsize=6.5, fontweight="bold",
                color=OKABE_ITO["vermillion"])

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$k{{=}}{k}$" for k in ks])
    ax.set_ylabel(r"DS1 test PC-$R^2$")
    ax.set_ylim(-0.100, 0.045)
    ax.legend(loc="lower left", handlelength=1.2, handletextpad=0.4,
              frameon=True, framealpha=0.92, edgecolor="none",
              facecolor="white")
    ax.set_title("Scrambled-GM null is far below observed")


def panel_c_subject_bootstrap(ax, boot):
    """Error bar plot: DS2 PC-R^2 for each method with subject-level 95% CI."""
    methods_json = ["nuclear_norm", "pls", "rrr", "ridge"]
    method_display = {
        "nuclear_norm": "Nuclear Norm",
        "pls": "PLS",
        "rrr": "RRR",
        "ridge": "Ridge",
    }
    y = np.arange(len(methods_json))

    for i, m in enumerate(methods_json):
        data = boot["methods"][m]
        point = data["point_ext"]
        lo, hi = data["boot_ext_ci"]
        disp = method_display[m]
        ax.errorbar(point, i, xerr=[[point - lo], [hi - point]],
                    fmt="o", color=METHOD_COLORS[disp],
                    ecolor=OKABE_ITO["black"], elinewidth=1,
                    capsize=3, markersize=7,
                    markeredgecolor="white", markeredgewidth=0.6,
                    zorder=3)

    ax.axvline(0, color=OKABE_ITO["grey"], lw=0.6, ls="--", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([method_display[m] for m in methods_json])
    ax.set_xlabel(r"DS2 external PC-$R^2$ ($k{=}20$)")
    ax.set_xlim(-0.035, 0.100)
    ax.invert_yaxis()

    # Annotate NN - PLS significance
    nn_point = boot["methods"]["nuclear_norm"]["point_ext"]
    pls_point = boot["methods"]["pls"]["point_ext"]
    diff_data = boot["paired_nn_vs_pls"]
    ax.annotate("", xy=(nn_point, -0.4), xytext=(pls_point, -0.4),
                arrowprops=dict(arrowstyle="<->",
                                 color=OKABE_ITO["vermillion"], lw=1))
    ax.text((nn_point + pls_point) / 2, -0.65,
            f"$\\Delta={diff_data['ext_diff_mean']:+.3f}$, $p<0.001$",
            ha="center", fontsize=6.5, color=OKABE_ITO["vermillion"],
            fontweight="bold")
    ax.set_ylim(3.7, -1.0)  # reverse, with room above for the annotation
    ax.set_title("Subject-level bootstrap (2000 DS2 resamples)")


def panel_d_retention_instability(ax, boot):
    """Visualize retention ratio CIs for NN and PLS — both very wide."""
    methods_json = ["nuclear_norm", "pls", "rrr"]
    method_display = {
        "nuclear_norm": "Nuclear Norm",
        "pls": "PLS",
        "rrr": "RRR",
    }
    y = np.arange(len(methods_json))

    for i, m in enumerate(methods_json):
        data = boot["methods"][m]
        point = data["point_retention"]
        lo, hi = data["boot_ret_ci"]
        disp = method_display[m]
        # Clip RRR retention to reasonable range for display
        display_hi = min(hi, 2.5)
        clipped = hi > 2.5
        ax.errorbar(point, i,
                    xerr=[[point - lo], [display_hi - point]],
                    fmt="D", color=METHOD_COLORS[disp],
                    ecolor=OKABE_ITO["black"], elinewidth=1,
                    capsize=3, markersize=7,
                    markeredgecolor="white", markeredgewidth=0.6,
                    zorder=3)
        if clipped:
            ax.annotate(f"(CI reaches {hi:.1f})",
                        xy=(display_hi, i), xytext=(5, 0),
                        textcoords="offset points",
                        fontsize=6, color=OKABE_ITO["grey"], va="center")

    # Reference: "100% retention" line
    ax.axvline(1.0, color=OKABE_ITO["green"], lw=0.7, ls="--",
               label="100% retention", alpha=0.85)
    ax.axvline(0, color=OKABE_ITO["grey"], lw=0.6, ls="--", alpha=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels([method_display[m] for m in methods_json])
    ax.set_xlabel(r"DS2/DS1 retention ratio (bootstrap 95% CI)")
    ax.set_xlim(-0.7, 2.8)
    ax.invert_yaxis()
    ax.legend(loc="upper right", handlelength=1.5, handletextpad=0.4,
              frameon=True, framealpha=0.92, edgecolor="none",
              facecolor="white")
    ax.set_title(r"Retention ratios are too unstable for headline use")


def main():
    apply_style()
    resid, null, boot = load_data()

    fig = plt.figure(figsize=(7.5, 6.5))
    gs = fig.add_gridspec(2, 2, hspace=0.80, wspace=0.40,
                           left=0.10, right=0.98, top=0.94, bottom=0.08)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_a_residualization(ax_a, resid)
    panel_b_scrambled_null(ax_b, null)
    panel_c_subject_bootstrap(ax_c, boot)
    panel_d_retention_instability(ax_d, boot)

    for ax, lab in zip((ax_a, ax_b, ax_c, ax_d), "ABCD"):
        panel_label(ax, lab, x=-0.18, y=1.22)

    outputs = save_figure(fig, "figure_revision_analyses", out_dir=ROOT / "IMAG" / "figure")
    print("Saved:")
    for p in outputs:
        print(f"  {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
