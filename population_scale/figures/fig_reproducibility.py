#!/usr/bin/env python3
"""Figure — Within-UKB reproducibility of the leading coupling mode.

a  Mode-1 alignment vs N (within-UKB climb toward the Tier-3 reference).
b  Per-mode (1-3) x tier-pair reproducibility heatmap.
c  100-bootstrap stability of Mode-1/2/3 alignment (raincloud).

(Cross-cohort conservation is shown separately in fig_universality, using the
rotation-invariant model-free subspace overlap rather than vector alignment.)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from nature_style import apply_nature_strict, nature_panel_label
from utils import style_axes

OUT = Path(__file__).parent / "fig_reproducibility.pdf"
RR = PROJECT_ROOT / "results/reviewer_revision"
BOOT = PROJECT_ROOT / "results/rscm_ukb37775_bootstrap100"


def panel_traj(ax):
    within = json.loads((RR / "within_ukb_tier_mode1_alignment.json").read_text())
    r1 = json.loads((RR / "ukb_downsample_N805_mode1_control_disjoint.json").read_text())
    def pair(p): return next(x for x in within["pairs"] if x["pair"] == p)["mode1_abs_r"]
    Ns = np.array([805, 1079, 11820, 37775])
    rs = np.array([r1["mode1_r_mean"], pair("tier1_N1079__tier3_N37775"), pair("tier2_N11820__tier3_N37775"), 1.0])
    ax.errorbar(Ns, rs, yerr=[r1["mode1_r_std"], 0, 0, 0], fmt="o-", color="#2166AC",
                ms=4, lw=1.0, capsize=2, label="UKB Mode-1 vs Tier 3")
    ax.axvline(1000, ls="--", color="#888", lw=0.5)
    ax.text(1080, 0.10, "Helmer\n$N{\\approx}1{,}000$", fontsize=5, color="#666")
    ax.set_xscale("log"); ax.set_ylim(-0.05, 1.06); ax.set_xlim(500, 60000)
    ax.set_xlabel("Sample size $N$"); ax.set_ylabel("Mode-1 $|r|$ vs Tier 3")
    ax.legend(frameon=False, fontsize=5, loc="lower right")
    style_axes(ax, ygrid=True)


def panel_heatmap(ax):
    within = json.loads((RR / "within_ukb_tier_mode1_alignment.json").read_text())
    order = ["tier1_N1079__tier2_N11820", "tier1_N1079__tier3_N37775", "tier2_N11820__tier3_N37775"]
    HM = np.zeros((3, 3))
    for j, p in enumerate(order):
        d = next(x for x in within["pairs"] if x["pair"] == p)
        HM[:, j] = [d["mode1_abs_r"], d["mode2_abs_r"], d["mode3_abs_r"]]
    im = ax.imshow(HM, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{HM[i, j]:.2f}", ha="center", va="center",
                    color="white" if HM[i, j] < 0.5 else "black", fontsize=6)
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["T1$\\leftrightarrow$T2", "T1$\\leftrightarrow$T3", "T2$\\leftrightarrow$T3"], fontsize=6)
    ax.set_yticklabels(["Mode 1", "Mode 2", "Mode 3"], fontsize=6)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=5)


def panel_boot(ax):
    boots = [json.loads(p.read_text()) for p in sorted(BOOT.glob("per_boot_*.json"))]
    rng = np.random.default_rng(0); colors = ["#2166AC", "#5AAE61", "#B2182B"]
    if boots:
        data = [np.array([b[f"mode{m}_abs_r_vs_main"] for b in boots]) for m in (1, 2, 3)]
        for i, (vals, color) in enumerate(zip(data, colors)):
            pos = i + 1
            vp = ax.violinplot(vals, positions=[pos], widths=0.7, showextrema=False, points=150)
            for body in vp["bodies"]:
                v = body.get_paths()[0].vertices; v[:, 0] = np.clip(v[:, 0], pos, np.inf)
                body.set_facecolor(color); body.set_alpha(0.25); body.set_edgecolor(color); body.set_linewidth(0.5)
            ax.scatter(pos - 0.10 - rng.uniform(0, 0.16, len(vals)), vals, s=4, color=color, alpha=0.5, edgecolor="none", zorder=3)
            lo, hi = np.percentile(vals, [2.5, 97.5])
            ax.errorbar(pos - 0.02, vals.mean(), yerr=[[vals.mean() - lo], [hi - vals.mean()]], fmt="o",
                        color=color, ms=4, capsize=2, elinewidth=0.9, mec="white", mew=0.4, zorder=4)
    else:
        # raw per-bootstrap draws unavailable on this machine (big-data purge); show the
        # published 100-bootstrap mean +/- 95% CI (identical numbers to the manuscript macros).
        summ = {1: (0.9951, 0.9922, 0.9970), 2: (0.9959, 0.9919, 0.9977), 3: (0.9850, 0.9600, 0.9935)}
        for i, color in enumerate(colors):
            pos = i + 1; mean, lo, hi = summ[i + 1]
            ax.errorbar(pos, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", color=color, ms=5,
                        capsize=2.5, elinewidth=1.0, mec="white", mew=0.5, zorder=4)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["Mode 1", "Mode 2", "Mode 3"]); ax.set_xlim(0.5, 3.6)
    ax.set_ylim(0.94, 1.002); ax.set_ylabel("Bootstrap $|r|$")
    style_axes(ax, ygrid=True)


def panel_cohort(ax):
    r1 = json.loads((RR / "ukb_downsample_N805_mode1_control_disjoint.json").read_text())
    cr = json.loads((RR / "cross_cohort_mode_procrustes.json").read_text())
    rscm = cr["comparisons"]["same_method_DS1RSCM_vs_UKBRSCM"]["mode1_abs_r_mean"]
    nn = cr["comparisons"]["cross_method_DS1NN_vs_UKBRSCM"]["mode1_abs_r_mean"]
    ceil = cr["comparisons"]["same_cohort_baseline_DS1NN_vs_DS1RSCM"]["mode1_abs"]
    r1m, r1s = r1["mode1_r_mean"], r1["mode1_r_std"]; subs = [s["mode1_abs_r_vs_tier3"] for s in r1["per_subset"]]
    rows = [(6.0, "Tier 1  (UKB)", 0.872, "#2166AC", "dot"),
            (5.0, "R1 mean  ($N{=}805$, ext.)", r1m, "#2166AC", "mean"),
            (4.0, "R1 subsets ($\\times5$)", None, "#92C5DE", "strip"),
            (2.7, "DS1-NN$\\leftrightarrow$DS1-RSCM (ceiling)", ceil, "#7F7F7F", "dot"),
            (1.6, "DS1-NN (cross-cohort)", nn, "#B2182B", "dot"),
            (0.8, "DS1-RSCM (cross-cohort)", rscm, "#B2182B", "dot")]
    ax.axvspan(r1m - r1s, r1m + r1s, color="#2166AC", alpha=0.10, zorder=0)
    ax.axvline(r1m, color="#2166AC", ls="--", lw=0.6, alpha=0.7, zorder=1)
    for y, label, val, color, kind in rows:
        if kind == "strip":
            ax.scatter(subs, y + np.linspace(-0.18, 0.18, len(subs)), s=12, color=color, edgecolor="white", linewidth=0.3, zorder=3)
        elif kind == "mean":
            ax.errorbar(val, y, xerr=r1s, fmt="o", color=color, ms=5, capsize=2, elinewidth=0.9, mec="white", mew=0.4, zorder=4)
            ax.text(val, y + 0.30, f"{val:.3f}", ha="center", fontsize=5.5, color=color)
        else:
            ax.scatter(val, y, s=34, color=color, edgecolor="white", linewidth=0.4, zorder=3)
            ax.text(val, y + 0.30, f"{val:.3f}", ha="center", fontsize=5.5, color=color)
        ax.text(-0.015, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6, color="#222")
    gap = (r1m - rscm) / r1s
    ax.text((rscm + r1m) / 2, 0.2, f"$\\approx${gap:.0f}$\\sigma$ below baseline", ha="center", fontsize=5.8, color="#444")
    ax.set_xlim(-0.02, 1.0); ax.set_ylim(0.2, 6.7); ax.set_yticks([])
    ax.set_xlabel("Mode-1 $|r|$ with Tier-3 reference")
    for sp in ("left", "top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color="#ccc", alpha=0.4, lw=0.4); ax.set_axisbelow(True)


def main():
    apply_nature_strict()
    fig = plt.figure(figsize=(7.2, 2.5))
    gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.08, right=0.97, top=0.88, bottom=0.20)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1]); axC = fig.add_subplot(gs[0, 2])
    panel_traj(axA); panel_heatmap(axB); panel_boot(axC)
    nature_panel_label(axA, "a", x=-0.22, y=1.10)
    nature_panel_label(axB, "b", x=-0.18, y=1.10)
    nature_panel_label(axC, "c", x=-0.20, y=1.10)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
