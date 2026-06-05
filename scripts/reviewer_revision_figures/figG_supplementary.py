"""
Supplementary figures S1 (mode stability) and S2 (robustness).

Each supplementary figure is aggregated from its original panel PDFs via
PyMuPDF embedding, so the submission package has no loose panel PDFs.

S1: Cross-method mode agreement + singular value spectrum
S2: Residualization robustness + per-PC stability + two more panels
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("/home/users/ybi3/sfcoupling")
sys.path.insert(0, str(ROOT / "scripts" / "reviewer_revision_figures"))
from _style import apply_style, embed_pdf, panel_label, save_figure

S1_A = ROOT / "paper/standalone/figure/figS1/panel_a.pdf"
S1_B = ROOT / "paper/standalone/figure/figS1/panel_b.pdf"

S2_A = ROOT / "paper/standalone/figure/figS2/panel_a.pdf"
S2_B = ROOT / "paper/standalone/figure/figS2/panel_b.pdf"
S2_C = ROOT / "paper/standalone/figure/figS2/panel_c.pdf"
S2_D = ROOT / "paper/standalone/figure/figS2/panel_d.pdf"


def make_s1():
    apply_style()

    fig = plt.figure(figsize=(7.2, 3.4))
    gs = fig.add_gridspec(1, 2, wspace=0.15,
                           left=0.04, right=0.98, top=0.92, bottom=0.06,
                           width_ratios=[1.15, 0.9])

    ax_a = fig.add_subplot(gs[0, 0])
    embed_pdf(ax_a, S1_A, dpi=450)

    ax_b = fig.add_subplot(gs[0, 1])
    embed_pdf(ax_b, S1_B, dpi=450)

    panel_label(ax_a, "A", x=0.005, y=1.08)
    panel_label(ax_b, "B", x=0.005, y=1.08)

    outputs = save_figure(fig, "figure_supp_mode_stability", out_dir=ROOT / "IMAG" / "figure")
    print("S1 saved:")
    for p in outputs:
        print(f"  {p}")
    plt.close(fig)


def make_s2():
    apply_style()

    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.18,
                           left=0.04, right=0.98, top=0.95, bottom=0.04)

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    panels = [S2_A, S2_B, S2_C, S2_D]
    labels = "ABCD"

    axes = []
    for (r, c), pdf, lab in zip(positions, panels, labels):
        ax = fig.add_subplot(gs[r, c])
        embed_pdf(ax, pdf, dpi=450)
        panel_label(ax, lab, x=0.005, y=1.08)
        axes.append(ax)

    outputs = save_figure(fig, "figure_supp_robustness", out_dir=ROOT / "IMAG" / "figure")
    print("S2 saved:")
    for p in outputs:
        print(f"  {p}")
    plt.close(fig)


if __name__ == "__main__":
    make_s1()
    make_s2()
