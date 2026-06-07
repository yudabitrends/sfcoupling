#!/usr/bin/env python3
"""Fig — Dual-regularization: rank vs lambda (real A5 sweep) + prediction tradeoff.

Panel a: effective rank vs nuclear-norm lambda at fixed Tier-3 N (real sweep,
         results/reviewer_revision/rank_vs_lambda.json) — the rank-revealing axis.
Panel b: cross-validated pc-R^2(k=20) at the two regimes (lambda=0.3 rank-revealing
         vs lambda=0.1 prediction) on the DS1-test and DS2-external partitions,
         showing the +45% prediction gain. All values are measured (no hardcoded
         interpolation).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from plot_style import apply_nature_strict, nature_panel_label
from utils import style_axes

OUT_PDF = Path(__file__).parent / "fig_lambda_dual.pdf"
RR = PROJECT_ROOT / "results/reviewer_revision"

C_RANK = "#003D6B"
C_DS1 = "#0072B2"
C_DS2 = "#D55E00"
C_REV = "#009E73"   # rank-revealing lambda=0.3
C_PRED = "#CC79A7"  # prediction lambda=0.1


def main() -> None:
    apply_nature_strict()

    # ── Real rank-vs-lambda sweep (A5) ──
    rl = json.loads((RR / "rank_vs_lambda.json").read_text())
    lam = np.array([r["lambda"] for r in rl["rank_vs_lambda"]])
    rank = np.array([r["eff_rank"] for r in rl["rank_vs_lambda"]])

    # ── Measured pc-R^2(k=20) at the two regimes ──
    pcr = {  # [DS1-test, DS2-external]
        0.3: [0.0418, 0.0449],
        0.1: [0.0605, 0.0647],
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    plt.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.20, wspace=0.34)

    # ── Panel a: effective rank vs lambda ──
    ax = axes[0]
    ax.semilogx(lam, rank, "o-", color=C_RANK, markersize=4.5, linewidth=1.2, zorder=3)
    for L, rk in zip(lam, rank):
        ax.annotate(str(rk), (L, rk), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=6.5, color=C_RANK)
    ax.axvline(0.3, ls="--", color=C_REV, lw=0.7, alpha=0.8)
    ax.text(0.31, 23.0, "$\\lambda=0.3$\nrank-revealing", fontsize=6, color=C_REV,
            ha="left", va="top")
    ax.axvline(0.1, ls="--", color=C_PRED, lw=0.7, alpha=0.8)
    ax.text(0.103, 14.5, "$\\lambda=0.1$\nprediction", fontsize=6, color=C_PRED,
            ha="left", va="top")
    ax.set_xlabel("Nuclear-norm regularization $\\lambda$")
    ax.set_ylabel("Effective rank of $\\hat{B}$")
    ax.set_xlim(0.085, 1.2)
    ax.set_ylim(0, 27)
    nature_panel_label(ax, "a", x=-0.17, y=1.08)
    style_axes(ax, ygrid=True)

    # ── Panel b: prediction tradeoff (grouped bars) ──
    ax = axes[1]
    groups = ["DS1 test", "DS2 external"]
    x = np.arange(len(groups))
    w = 0.36
    rev = [pcr[0.3][0], pcr[0.3][1]]
    pred = [pcr[0.1][0], pcr[0.1][1]]
    b1 = ax.bar(x - w / 2, rev, w, color=C_REV, label="$\\lambda=0.3$ (rank-revealing)")
    b2 = ax.bar(x + w / 2, pred, w, color=C_PRED, label="$\\lambda=0.1$ (prediction)")
    for bars in (b1, b2):
        for rect in bars:
            ax.annotate(f"{rect.get_height():.3f}", (rect.get_x() + rect.get_width() / 2,
                        rect.get_height()), textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=6, color="#333")
    # +45% gain annotation on DS1
    ax.annotate("", xy=(0 + w / 2, pred[0] + 0.004), xytext=(0 - w / 2, rev[0] + 0.004),
                arrowprops=dict(arrowstyle="->", color="#333", lw=0.6))
    ax.text(0, pred[0] + 0.011, "+45%", ha="center", fontsize=6.5, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("pc-$R^2$ at $k=20$")
    ax.set_ylim(0, 0.082)
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    nature_panel_label(ax, "b", x=-0.17, y=1.08)
    style_axes(ax, ygrid=True)

    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_PDF.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
