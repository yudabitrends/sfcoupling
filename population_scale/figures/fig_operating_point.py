#!/usr/bin/env python3
"""Figure 3 — The nuclear-norm rank is an operating point; dual regularization.

a  Effective rank vs N under fixed lambda=0.3 (descends) vs lambda re-tuned per N (rises) --
   the two regularization regimes move the rank in opposite directions.
b  Effective rank vs lambda at fixed Tier-3 N (rank is a monotone function of lambda).
c  Prediction tradeoff: pc-R^2 at lambda=0.3 (rank-revealing) vs lambda=0.1 (prediction).
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

OUT = Path(__file__).parent / "fig_operating_point.pdf"
RR = PROJECT_ROOT / "results/reviewer_revision"
C_REV, C_PRED = "#5AAE61", "#762A83"


def panel_pern(ax):
    L = json.loads((RR / "per_n_lambda_ladder.json").read_text())
    fx = L["fixed_lambda_reference"]
    ax.plot(fx["N"], fx["eff_rank_ladder"], "o-", color="#2166AC", ms=4.5, lw=1.2,
            label="fixed $\\lambda=0.3$")
    pn = L["per_n_tuned"]
    ax.plot([r["N_total"] for r in pn], [r["eff_rank_eps1e-4"] for r in pn], "s--",
            color="#B2182B", ms=4.5, lw=1.2, label="$\\lambda$ re-tuned per $N$")
    ax.set_xscale("log"); ax.set_xlabel("Sample size $N$")
    ax.set_ylabel("Effective rank of $\\hat{B}$")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    ax.xaxis.set_major_locator(FixedLocator([1000, 2000, 4000, 8000]))
    ax.xaxis.set_major_formatter(FixedFormatter(["1k", "2k", "4k", "8k"]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    style_axes(ax, ygrid=True)


def panel_rankvslam(ax):
    rl = json.loads((RR / "rank_vs_lambda.json").read_text())
    lam = np.array([r["lambda"] for r in rl["rank_vs_lambda"]])
    rank = np.array([r["eff_rank"] for r in rl["rank_vs_lambda"]])
    ax.semilogx(lam, rank, "o-", color="#053061", ms=4.5, lw=1.2, zorder=3)
    for L, rk in zip(lam, rank):
        ax.annotate(str(rk), (L, rk), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6, color="#053061")
    ax.axvline(0.3, ls="--", color=C_REV, lw=0.7, alpha=0.8)
    ax.text(0.31, 23, "$\\lambda{=}0.3$\nrank-revealing", fontsize=5.5, color=C_REV, ha="left", va="top")
    ax.axvline(0.1, ls="--", color=C_PRED, lw=0.7, alpha=0.8)
    ax.text(0.103, 14.5, "$\\lambda{=}0.1$\nprediction", fontsize=5.5, color=C_PRED, ha="left", va="top")
    ax.set_xlim(0.085, 1.2); ax.set_ylim(0, 27)
    ax.set_xlabel("Nuclear-norm $\\lambda$"); ax.set_ylabel("Effective rank of $\\hat{B}$")
    style_axes(ax, ygrid=True)


def panel_predtradeoff(ax):
    pcr = {0.3: [0.0418, 0.0449], 0.1: [0.0605, 0.0647]}
    groups = ["DS1 test", "DS2 ext."]; x = np.arange(2); w = 0.36
    b1 = ax.bar(x - w / 2, pcr[0.3], w, color=C_REV, label="$\\lambda{=}0.3$ (rank)")
    b2 = ax.bar(x + w / 2, pcr[0.1], w, color=C_PRED, label="$\\lambda{=}0.1$ (pred.)")
    for bars in (b1, b2):
        for rect in bars:
            ax.annotate(f"{rect.get_height():.3f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        textcoords="offset points", xytext=(0, 2), ha="center", fontsize=5.5, color="#333")
    ax.text(0, pcr[0.1][0] + 0.006, "+45%", ha="center", fontsize=6, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("pc-$R^2$ at $k{=}20$"); ax.set_ylim(0, 0.092)
    ax.legend(frameon=False, fontsize=5.5, loc="upper center", ncol=1)
    style_axes(ax, ygrid=True)


def main():
    apply_nature_strict()
    fig = plt.figure(figsize=(7.2, 2.5))
    gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.07, right=0.985, top=0.92, bottom=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    panel_pern(axes[0]); panel_rankvslam(axes[1]); panel_predtradeoff(axes[2])
    for ax, lab in zip(axes, "abc"):
        nature_panel_label(ax, lab, x=-0.17, y=1.08)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
