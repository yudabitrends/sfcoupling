#!/usr/bin/env python3
"""Fig — Cross-cohort Mode-1 alignment (horizontal dot plot).

Compares Mode-1 |r| with the canonical Tier-3 reference across: within-UKB
sub-threshold anchors (Tier 1, five external N=805 R1 subsets + mean), the
clinical cross-cohort fits (DS1-RSCM, DS1-NN), and the same-cohort cross-method
ceiling. The R1 baseline mean +-1 SD is shown as a shaded band; the cross-cohort
points sit well below it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from plot_style import apply_nature_strict
from utils import style_axes

OUT_PDF = Path(__file__).parent / "fig_cohort_anchor.pdf"
C_UKB = "#0072B2"
C_UKB_L = "#88B0D6"
C_CROSS = "#D55E00"
C_CEIL = "#7F7F7F"


def main() -> None:
    apply_nature_strict()
    r1 = json.loads((PROJECT_ROOT / "results/reviewer_revision/ukb_downsample_N805_mode1_control_disjoint.json").read_text())
    cross = json.loads((PROJECT_ROOT / "results/reviewer_revision/cross_cohort_mode_procrustes.json").read_text())
    rscm = cross["comparisons"]["same_method_DS1RSCM_vs_UKBRSCM"]["mode1_abs_r_mean"]
    nn = cross["comparisons"]["cross_method_DS1NN_vs_UKBRSCM"]["mode1_abs_r_mean"]
    ceil = cross["comparisons"]["same_cohort_baseline_DS1NN_vs_DS1RSCM"]["mode1_abs"]
    r1m, r1s = r1["mode1_r_mean"], r1["mode1_r_std"]
    subs = [s["mode1_abs_r_vs_tier3"] for s in r1["per_subset"]]

    # rows: (y, label, value, color, kind)
    rows = [
        (9.0, "Tier 1  (UKB, $N{=}1{,}079$)", 0.872, C_UKB, "dot"),
        (8.0, "R1 mean  ($N{=}805$, external)", r1m, C_UKB, "mean"),
        (7.0, "R1 subsets  ($\\times5$)", None, C_UKB_L, "strip"),
        (5.3, "DS1-NN $\\leftrightarrow$ DS1-RSCM  (same-cohort ceiling)", ceil, C_CEIL, "dot"),
        (3.6, "DS1-NN  (clinical, cross-cohort)", nn, C_CROSS, "dot"),
        (2.6, "DS1-RSCM  (clinical, cross-cohort)", rscm, C_CROSS, "dot"),
    ]

    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    plt.subplots_adjust(left=0.40, right=0.97, top=0.90, bottom=0.13)

    # R1 baseline band (vertical, since x = correlation)
    ax.axvspan(r1m - r1s, r1m + r1s, color=C_UKB, alpha=0.10, zorder=0)
    ax.axvline(r1m, color=C_UKB, ls="--", lw=0.6, alpha=0.7, zorder=1)

    for y, label, val, color, kind in rows:
        if kind == "strip":
            ys = y + np.linspace(-0.28, 0.28, len(subs))
            ax.scatter(subs, ys, s=16, color=color, edgecolor="white",
                       linewidth=0.4, zorder=3)
        elif kind == "mean":
            ax.errorbar(val, y, xerr=r1s, fmt="o", color=color, ms=6,
                        capsize=2.5, elinewidth=1.0, mec="white", mew=0.5, zorder=4)
            ax.text(val, y + 0.40, f"{val:.3f}", ha="center", fontsize=6, color=color)
        else:
            ax.scatter(val, y, s=42, color=color, edgecolor="white",
                       linewidth=0.5, zorder=3)
            ax.text(val, y + 0.40, f"{val:.3f}", ha="center", fontsize=6, color=color)
        ax.text(-0.02, y, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=6.6, color="#222")

    # baseline label + cross-cohort gap
    gap = (r1m - rscm) / r1s
    ax.text(r1m, 9.95, f"R1 baseline {r1m:.3f}$\\pm${r1s:.3f}", ha="center",
            fontsize=6, color=C_UKB)
    ax.annotate("", xy=(rscm, 2.6), xytext=(r1m - r1s, 2.6),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=0.6))
    ax.text((rscm + r1m) / 2, 2.0, f"$\\approx${gap:.0f}$\\sigma$ below baseline",
            ha="center", fontsize=6.3, color="#444")

    # category separators
    for ysep in (6.15, 4.45):
        ax.axhline(ysep, color="#dddddd", lw=0.6, zorder=0)

    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(1.6, 10.4)
    ax.set_yticks([])
    ax.set_xlabel("Mode-1 $|r|$ with Tier-3 reference ($N{=}37{,}775$)")
    for sp in ("left", "top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color="#cccccc", alpha=0.4, lw=0.4)
    ax.set_axisbelow(True)

    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_PDF.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
