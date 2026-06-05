"""
Figure 1: Study workflow schematic — three panels in a horizontal row (1x3).

Panel A: Three cohort cards (DS1, DS2, UKB) stacked vertically within the column
Panel B: Vertical spectral-mapping flow  GM -> B = U Sigma V^T -> FNC
Panel C: Directional overlap O vs amplitude recovery R^2 (tall bars, not flat)

Layout rules:
  - Landscape figure, 3 columns side by side -> panel C is tall/narrow (fixes the
    previous "too flat" wide-short bar panel).
  - Schematic panels A and B use set_aspect('equal') so boxes/cards never distort;
    portrait canvases (tall, narrow) match the column shape.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path("/home/users/ybi3/sfcoupling")
sys.path.insert(0, str(ROOT / "scripts" / "reviewer_revision_figures"))
from _style import (COHORT_COLORS, OKABE_ITO, apply_style, panel_label,
                    save_figure)


def _portrait_axis(ax, w, h):
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal", adjustable="box", anchor="N")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


def panel_a_cohorts(ax):
    """Three cohort cards stacked vertically (portrait column)."""
    W, H = 5.0, 8.4
    _portrait_axis(ax, W, H)

    cohorts = [
        dict(name="DS1", subtitle="Discovery", N=r"$N{=}1{,}151$",
             l1="601 SZ / 550 HC", l2=r"age $34.6{\pm}12.3$ $\cdot$ 42% F",
             color=COHORT_COLORS["DS1"]),
        dict(name="DS2", subtitle="External validation", N=r"$N{=}102$",
             l1="49 SZ / 53 HC", l2=r"age $35.8{\pm}13.1$ $\cdot$ 44% F",
             color=COHORT_COLORS["DS2"]),
        dict(name="UKB", subtitle="UK Biobank", N=r"$N{\approx}37{,}775$",
             l1="primarily healthy", l2=r"age $55.0{\pm}7.5$ $\cdot$ 53% F",
             color=COHORT_COLORS["UKB"]),
    ]

    cx, bw, bh, gap = 2.5, 4.7, 2.35, 0.35
    top0 = H - 0.15
    for i, c in enumerate(cohorts):
        y_top = top0 - i * (bh + gap)
        y0 = y_top - bh
        ax.add_patch(FancyBboxPatch(
            (cx - bw / 2, y0), bw, bh,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            facecolor=c["color"], edgecolor="none", alpha=0.95))
        ax.text(cx - bw / 2 + 0.25, y_top - 0.5, c["name"], ha="left",
                va="center", fontsize=13.5, fontweight="bold", color="white")
        ax.text(cx + bw / 2 - 0.25, y_top - 0.5, c["N"], ha="right",
                va="center", fontsize=11, fontweight="bold", color="white")
        ax.text(cx, y_top - 1.07, c["subtitle"], ha="center", va="center",
                fontsize=8.3, color="white", style="italic")
        ax.text(cx, y_top - 1.62, c["l1"], ha="center", va="center",
                fontsize=8.3, color="white")
        ax.text(cx, y_top - 2.08, c["l2"], ha="center", va="center",
                fontsize=8.3, color="white")

    ax.set_title("Cohorts", pad=4)


def panel_b_mapping(ax):
    """Vertical mapping flow GM -> B = U Sigma V^T -> FNC (portrait column)."""
    W, H = 5.0, 8.4
    _portrait_axis(ax, W, H)

    cx, bw, bh = 2.5, 3.7, 1.45
    rows = [
        dict(cy=6.55, col=OKABE_ITO["skyblue"], title="GM ROIs",
             sub=r"$X \in \mathbb{R}^{N \times 99}$"),
        dict(cy=4.05, col=OKABE_ITO["orange"], title=r"$B = U\Sigma V^{\!\top}$",
             sub=r"$\in \mathbb{R}^{99 \times 1{,}378}$"),
        dict(cy=1.55, col=OKABE_ITO["green"], title="FNC",
             sub=r"$Y \in \mathbb{R}^{N \times 1{,}378}$"),
    ]
    for b in rows:
        ax.add_patch(FancyBboxPatch(
            (cx - bw / 2, b["cy"] - bh / 2), bw, bh,
            boxstyle="round,pad=0.02,rounding_size=0.16",
            facecolor=b["col"], edgecolor="none", alpha=0.92))
        ax.text(cx, b["cy"] + 0.32, b["title"], ha="center", va="center",
                fontsize=11.5, fontweight="bold", color="black")
        ax.text(cx, b["cy"] - 0.34, b["sub"], ha="center", va="center",
                fontsize=9, color="black")

    arrow_kw = dict(arrowstyle="-|>,head_length=0.5,head_width=0.32",
                    color=OKABE_ITO["black"], lw=2.0,
                    mutation_scale=12, shrinkA=0, shrinkB=0)
    ax.add_patch(FancyArrowPatch((cx, 5.82), (cx, 4.78), **arrow_kw))
    ax.add_patch(FancyArrowPatch((cx, 3.32), (cx, 2.28), **arrow_kw))
    ax.text(cx + 0.35, 5.30, "fit", ha="left", va="center", fontsize=9,
            color=OKABE_ITO["grey"], style="italic")
    ax.text(cx + 0.35, 2.80, "predict", ha="left", va="center", fontsize=9,
            color=OKABE_ITO["grey"], style="italic")

    ax.text(cx, 7.78,
            "Seven methods: Ridge, MLP, RRR,\n"
            "PLS, Nuclear Norm, OptShrink, NN-Init MLP",
            ha="center", va="center", fontsize=7.2, color=OKABE_ITO["grey"])
    ax.text(cx, 0.45,
            r"$V$: FNC directions (geometry)" "\n"
            r"$\Sigma$: coupling strength per mode",
            ha="center", va="center", fontsize=8, color=OKABE_ITO["black"])
    ax.set_title("Spectral mapping", pad=4, fontsize=8.5)


def panel_c_dissociation(ax):
    """Two tall bars: directional overlap O vs amplitude recovery R^2."""
    names = [r"Directional" "\n" r"overlap $\mathcal{O}$",
             r"Amplitude" "\n" r"recovery PC-$R^2$"]
    values = [0.39, 0.058]
    colors = [OKABE_ITO["blue"], OKABE_ITO["vermillion"]]

    x = np.arange(len(values))
    bars = ax.bar(x, values, 0.62, color=colors, edgecolor="white",
                  linewidth=0.8, zorder=3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008,
                f"{v:.2f}" if v >= 0.1 else f"{v:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(0.0145, ls="--", lw=0.9, color=OKABE_ITO["grey"], zorder=2)
    ax.text(0.97, 0.95,
            "dashed: random-\nsubspace null\n"
            r"$\mathcal{O}_{\mathrm{chance}}{\approx}0.0145$"
            "\n" r"($p<10^{-4}$)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.4,
            color=OKABE_ITO["grey"])

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.3)
    ax.set_ylabel(r"value at $k{=}20$ on DS1")
    ax.set_ylim(0, 0.46)
    ax.set_xlim(-0.65, 1.65)
    ax.set_title("Directional vs. amplitude", pad=6, fontsize=8.5)


def main():
    apply_style()

    fig = plt.figure(figsize=(7.2, 3.9))
    gs = fig.add_gridspec(
        1, 3, wspace=0.30,
        left=0.055, right=0.985, top=0.90, bottom=0.11,
        width_ratios=[1.0, 1.0, 0.92],
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_a_cohorts(ax_a)
    panel_b_mapping(ax_b)
    panel_c_dissociation(ax_c)

    panel_label(ax_a, "A", x=-0.05, y=1.05)
    panel_label(ax_b, "B", x=-0.05, y=1.05)
    panel_label(ax_c, "C", x=-0.22, y=1.05)

    outputs = save_figure(fig, "figure_workflow", out_dir=ROOT / "IMAG" / "figure")
    print("Saved:")
    for p in outputs:
        print(f"  {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
