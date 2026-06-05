"""
Figure 2: Spatial overview of the GM -> FNC coupling map (aggregated).

Three panels embedded from the original NeuroMark3 visualization outputs:
  Panel A: Glass-brain GM anatomical contribution map (3 views)
  Panel B: GM -> FNC domain-flow chord diagram
  Panel C: Domain-sorted FNC coupling matrix

All panels are rasterized from the source PDF files via PyMuPDF and embedded
as images in a single aggregated matplotlib figure so there are no loose
panel PDFs anywhere in the submission package.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("/home/users/ybi3/sfcoupling")
sys.path.insert(0, str(ROOT / "scripts" / "reviewer_revision_figures"))
from _style import (OKABE_ITO, apply_style, despine_all, embed_pdf,
                     panel_label, save_figure)

# Panel sources — these are the authoritative, high-quality renderings from
# the TReNDS visualization pipeline. We embed them as rasterized images in
# the aggregated figure below.
PANEL_A = ROOT / "paper/standalone/figure/fig2/panel_a.pdf"
PANEL_B = ROOT / "paper/standalone/figure/fig2/panel_b.pdf"
PANEL_C = ROOT / "paper/standalone/figure/fig2/panel_c.pdf"


def main():
    apply_style()

    # Source mapping (verified via get_text):
    #   panel_a.pdf -> glass brain anatomical contribution map
    #   panel_b.pdf -> domain-sorted FNC coupling matrix
    #   panel_c.pdf -> GM-to-FNC domain chord diagram

    fig = plt.figure(figsize=(7.5, 7.0))
    gs = fig.add_gridspec(
        2, 2, hspace=0.12, wspace=0.04,
        left=0.04, right=0.98, top=0.94, bottom=0.03,
        height_ratios=[1.05, 1.20],
        width_ratios=[1.0, 1.0],
    )

    # Panel A: Top row spans both columns (widest — the 3-view glass brain)
    ax_a = fig.add_subplot(gs[0, :])
    embed_pdf(ax_a, PANEL_A, dpi=450)

    # Panel B: Bottom left — chord diagram (panel_c.pdf is the chord)
    ax_b = fig.add_subplot(gs[1, 0])
    embed_pdf(ax_b, PANEL_C, dpi=450)

    # Panel C: Bottom right — coupling matrix (panel_b.pdf is the matrix)
    ax_c = fig.add_subplot(gs[1, 1])
    embed_pdf(ax_c, PANEL_B, dpi=450)

    # Panel labels placed well outside axes; no matplotlib titles because
    # the rasterized PDFs already carry their own (small, neat) titles.
    panel_label(ax_a, "A", x=0.005, y=1.03)
    panel_label(ax_b, "B", x=0.005, y=1.03)
    panel_label(ax_c, "C", x=0.005, y=1.03)

    outputs = save_figure(fig, "figure_spatial_overview", out_dir=ROOT / "IMAG" / "figure")
    print("Saved:")
    for p in outputs:
        print(f"  {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
