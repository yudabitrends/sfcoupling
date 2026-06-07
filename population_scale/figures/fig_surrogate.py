#!/usr/bin/env python3
"""Fig — Cross-cohort coupling-subspace conservation requires a surrogate floor.

The cautionary main figure for the A+B-hybrid manuscript. Nature-grade layout:
no overlapping labels, consistent fonts, generous spacing.

(a) Null ladder: per-cohort real O6 vs the same-pipeline surrogate floor, with
    the matched-N ceiling and the Haar-random null as references.
(b) Two-metric paradox: the same data reads r=0.03 (naive vector) or 0.93
    (subspace cosine); both misleading.
(c) Within-UKB vs cross-cohort, one metric: within-cohort overlap is far above
    its floor; cross-cohort overlap sits at its floor.
(d) The robust primary: the coupling is low-dimensional (participation ratio
    far below the Marchenko-Pastur null; log scale).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from nature_style import apply_neuroimage_style, nature_panel_label  # noqa: E402

DISC = ROOT / "results/discovery"
OUT = Path(__file__).parent / "fig_surrogate.pdf"
SITES = ["COBRE", "FBIRN", "PK_MPRC", "ChineseSZ"]
LAB = {"COBRE": "COBRE", "FBIRN": "FBIRN", "PK_MPRC": "MPRC", "ChineseSZ": "ChineseSZ"}
CC = {"COBRE": "#2166AC", "FBIRN": "#4393C3", "PK_MPRC": "#4393C3", "ChineseSZ": "#B2182B"}
C_FLOOR = "#B2182B"
C_CEIL = "#D9D9D9"
C_RAND = "#9A9A9A"


def main():
    apply_neuroimage_style()
    plt.rcParams.update({"axes.titlesize": 8, "axes.titleweight": "bold",
                         "axes.titlepad": 7, "font.size": 7})
    r2 = json.loads((DISC / "universality_round2_controls.json").read_text())
    ps = json.loads((DISC / "universality_persite_replication.json").read_text())
    pooled = json.loads((DISC / "universality_modefree_ds1_ukb.json").read_text())
    tiers = json.loads((DISC / "universality_within_ukb_tiers.json").read_text())

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7))
    plt.subplots_adjust(left=0.085, right=0.975, top=0.90, bottom=0.11,
                        hspace=0.62, wspace=0.32)
    (axA, axB), (axC, axD) = axes

    # ── (a) null ladder per cohort ──
    x = np.arange(len(SITES)); w = 0.62
    real = [ps["sites"][S]["vs_ukb_resid"]["k6"]["overlap"] for S in SITES]
    floor = [r2["same_pipeline_surrogate_floor_O6"][S]["surrogate_p95"] for S in SITES]
    ceil = [ps["sites"][S]["matched_N_ceiling"]["6"][0] for S in SITES]
    rb = ps["random_baseline"]["k6"][0]
    axA.bar(x, ceil, w, color=C_CEIL, zorder=1)
    axA.bar(x, real, w * 0.5, color=[CC[S] for S in SITES], zorder=3)
    for xi, f in zip(x, floor):
        axA.plot([xi - w / 2, xi + w / 2], [f, f], color=C_FLOOR, lw=1.8, solid_capstyle="butt", zorder=4)
    axA.axhline(rb, color=C_RAND, ls=":", lw=0.9, zorder=2)
    axA.set_xticks(x); axA.set_xticklabels([LAB[S] for S in SITES], rotation=18, ha="right")
    axA.set_ylabel("subspace overlap $O_6$"); axA.set_ylim(0, max(ceil) * 1.18)
    axA.set_xlim(-0.6, len(SITES) - 0.4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_CEIL), plt.Rectangle((0, 0), 1, 1, color="#2166AC"),
               Line2D([0], [0], color=C_FLOOR, lw=1.8), Line2D([0], [0], color=C_RAND, ls=":", lw=0.9)]
    axA.legend(handles, ["matched-$N$ ceiling", "real $O_6$", "surrogate floor (p95)", "Haar null"],
               fontsize=5.3, loc="upper center", ncol=2, columnspacing=1.0, handlelength=1.4,
               borderpad=0.3, frameon=False)
    axA.set_title("Real overlap sits at the surrogate floor")
    nature_panel_label(axA, "a")

    # ── (b) two-metric paradox ──
    naive = r2["degeneracy_check"]["what_drives_the_collapse"]["rscm_operator_per_cohort_basepoint_r"]
    subcos = r2["degeneracy_check"]["what_drives_the_collapse"]["model_free_no_basepoint_leadcos"]
    bx = np.arange(2)
    axB.bar(bx, [naive, subcos], 0.55, color=["#BDBDBD", "#2166AC"], edgecolor="#333", lw=0.5)
    for xi, v in zip(bx, [naive, subcos]):
        axB.text(xi, v + 0.025, f"{v:.2f}", ha="center", fontsize=8)
    axB.set_xticks(bx); axB.set_xticklabels(["naive Mode-1\nvector", "subspace\nleading cosine"])
    axB.set_ylim(0, 1.12); axB.set_ylabel("DS1$\\leftrightarrow$UKB reading")
    axB.set_xlim(-0.6, 1.6)
    axB.text(0.0, 0.62, "both readings\nmisleading", transform=axB.transData,
             ha="center", va="center", fontsize=6, color="#555", style="italic")
    axB.set_title("Same data, opposite readings")
    nature_panel_label(axB, "b")

    # ── (c) within-UKB vs cross-cohort (one metric) ──
    within = tiers["T2_vs_T3"]["k6"]["overlap"]
    within_floor = 0.085
    cross = pooled["ds1_vs_ukb_resid"]["k6"]["overlap"]
    cross_floor = float(np.mean(floor))
    cats = [("within-UKB\n(T2$\\leftrightarrow$T3)", within, within_floor, "#5AAE61", "$\\gg$ floor"),
            ("cross-cohort\n(DS1$\\leftrightarrow$UKB)", cross, cross_floor, "#B2182B", "at floor")]
    cx = np.arange(2)
    axC.bar(cx, [c[1] for c in cats], 0.5, color=[c[3] for c in cats], zorder=3)
    for xi, c in zip(cx, cats):
        axC.plot([xi - 0.30, xi + 0.30], [c[2], c[2]], color="#222", lw=1.6, solid_capstyle="butt", zorder=5)
        axC.text(xi, c[1] + 0.03, f"{c[1]:.2f}", ha="center", fontsize=8, zorder=6)
        # floor annotation to the right of the bar, never on the value
        axC.annotate(c[4], xy=(xi + 0.30, c[2]), xytext=(xi + 0.36, c[2]),
                     fontsize=5.6, color="#222", va="center", ha="left")
    axC.set_xticks(cx); axC.set_xticklabels([c[0] for c in cats])
    axC.set_ylim(0, 1.15); axC.set_ylabel("subspace overlap $O_6$")
    axC.set_xlim(-0.6, 1.75)
    axC.legend([Line2D([0], [0], color="#222", lw=1.6)], ["surrogate floor"],
               fontsize=5.6, loc="upper right", frameon=False, handlelength=1.4)
    axC.set_title("Within-cohort real; cross-cohort at floor")
    nature_panel_label(axC, "c")

    # ── (d) robust low dimensionality ──
    pr = r2["participation_ratio"]
    # plot the OBSERVED PR (matches the text \prObs) with the bootstrap 95% CI as error bars
    vals = [pr["observed"], pr["mp_null_mean"]]
    errs = [[pr["observed"] - pr["bootstrap_ci95"][0], pr["mp_null_mean"] - pr["mp_null_ci95"][0]],
            [pr["bootstrap_ci95"][1] - pr["observed"], pr["mp_null_ci95"][1] - pr["mp_null_mean"]]]
    dx = np.arange(2)
    axD.bar(dx, vals, 0.5, color=["#2166AC", "#BDBDBD"], edgecolor="#333", lw=0.5,
            yerr=errs, capsize=3, error_kw=dict(lw=0.8))
    axD.set_yscale("log")
    for xi, v in zip(dx, vals):
        axD.text(xi, v * 1.35, f"{v:.1f}", ha="center", fontsize=8)
    axD.set_xticks(dx); axD.set_xticklabels(["observed\nPR", "Marchenko–Pastur\nnull PR"])
    axD.set_ylabel("participation ratio"); axD.set_ylim(1, 400)
    axD.set_xlim(-0.6, 1.6)
    axD.set_title("Robust: the coupling is low-dimensional")
    nature_panel_label(axD, "d")

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
