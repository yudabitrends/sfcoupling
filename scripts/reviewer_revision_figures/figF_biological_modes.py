"""
Figure 5: Biological organization of the coupled subspace (aggregated).

Four panels embedded from NeuroMark3 visualization outputs:
  Panel A: ROI coupled variance fraction across 7 axial slices (z=-25 to 65)
  Panel B: SVD mode 1 dorsal GM loadings + FNC loadings matrix (36.5% var)
  Panel C: SVD mode 2 dorsal GM loadings + FNC loadings matrix (9.2% var)
  Panel D: SVD mode 3 dorsal GM loadings + FNC loadings matrix (7.9% var)

All source panel PDFs are rasterized via PyMuPDF and embedded as images,
producing a single aggregated PDF with no loose panel files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("/home/users/ybi3/sfcoupling")
sys.path.insert(0, str(ROOT / "scripts" / "reviewer_revision_figures"))
from _style import (apply_style, embed_pdf, panel_label, save_figure)

PANEL_ROI = ROOT / "paper/standalone/figure/fig6/panel_a.pdf"
PANEL_MODE1 = ROOT / "paper/standalone/figure/fig7/panel_mode1.pdf"
PANEL_MODE2 = ROOT / "paper/standalone/figure/fig7/panel_mode2.pdf"
PANEL_MODE3 = ROOT / "paper/standalone/figure/fig7/panel_mode3.pdf"


def main():
    apply_style()

    fig = plt.figure(figsize=(7.5, 9.5))
    gs = fig.add_gridspec(
        4, 1, hspace=0.22,
        left=0.04, right=0.96, top=0.965, bottom=0.02,
        height_ratios=[1.0, 1.1, 1.1, 1.1],
    )

    ax_a = fig.add_subplot(gs[0, 0])
    embed_pdf(ax_a, PANEL_ROI, dpi=500)

    ax_b = fig.add_subplot(gs[1, 0])
    embed_pdf(ax_b, PANEL_MODE1, dpi=450)

    ax_c = fig.add_subplot(gs[2, 0])
    embed_pdf(ax_c, PANEL_MODE2, dpi=450)

    ax_d = fig.add_subplot(gs[3, 0])
    embed_pdf(ax_d, PANEL_MODE3, dpi=450)

    panel_label(ax_a, "A", x=0.005, y=1.05)
    panel_label(ax_b, "B", x=0.005, y=1.04)
    panel_label(ax_c, "C", x=0.005, y=1.04)
    panel_label(ax_d, "D", x=0.005, y=1.04)

    outputs = save_figure(fig, "figure_biological_modes", out_dir=ROOT / "IMAG" / "figure")
    print("Saved:")
    for p in outputs:
        print(f"  {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
