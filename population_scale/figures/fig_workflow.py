#!/usr/bin/env python3
"""Fig 1 — Workflow + N-ladder + RSCM pipeline + 4 rescue experiments.

Nature double-column (7.2" x 3.0"), 3 panels (a / b / c).

  Panel a — Three-tier UKB N-ladder data flow (nested tiers + 80/20 split)
  Panel b — RSCM analytical pipeline (GM/FNC → tangent space → nuclear-norm regression → SVD)
  Panel c — Four rescue experiments (R1 downsample, R2 cross-tier, B1 bootstrap, B2 λ-sweep)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from plot_style import (
    NATURE_DOUBLE,
    apply_nature_strict,
    nature_panel_label,
)

OUT_PDF = Path(__file__).parent / "fig_workflow.pdf"
BRAIN_PNG = Path(__file__).parent / "brain_panel.png"
NEUROMARK_NII = (
    "/data/qneuromark/Network_templates/NeuroMark1/Structural/"
    "Matched_template/Neuromark_v01_sMRI_high_100.nii"
)


def ensure_brain_panel() -> None:
    """Render (and cache) a decorative glass-brain of the NeuroMark sMRI
    gray-matter ICA-network coverage (max |IC| across the 100 components).

    This is a DECORATIVE input visual only — it shows the gray-matter network
    coverage of the parcellation, NOT any Mode-1 loading or per-region result.
    """
    if BRAIN_PNG.exists():
        return
    import nibabel as nib
    import numpy as np
    from nilearn import plotting

    img = nib.load(NEUROMARK_NII)
    arr = img.dataobj
    nx, ny, nz, nc = img.shape
    comp = np.zeros((nx, ny, nz), dtype=np.float32)
    for c in range(nc):
        np.maximum(comp, np.abs(np.asarray(arr[..., c], dtype=np.float32)), out=comp)
    comp_img = nib.Nifti1Image(comp, img.affine)
    thr = float(np.percentile(comp[comp > 0], 80))
    bfig = plt.figure(figsize=(2.6, 1.3))
    plotting.plot_glass_brain(
        comp_img, threshold=thr, display_mode="xz", colorbar=False,
        cmap="cividis", alpha=0.9, plot_abs=True, figure=bfig,
    )
    bfig.savefig(BRAIN_PNG, dpi=300, transparent=True,
                 bbox_inches="tight", pad_inches=0.0)
    plt.close(bfig)

# Okabe-Ito palette anchors
C_BLUE_DARK = "#003D6B"
C_BLUE = "#0072B2"
C_BLUE_LIGHT = "#88B0D6"
C_ORANGE = "#D55E00"
C_GREEN = "#009E73"
C_PURPLE = "#CC79A7"
C_GRAY = "#666666"
C_FILL_LIGHT = "#F2F4F7"


def _box(ax, x, y, w, h, text, color, fill=None, fontsize=6.5,
         text_color="black", boxstyle="round,pad=0.02"):
    rect = FancyBboxPatch((x, y), w, h, boxstyle=boxstyle,
                          edgecolor=color, facecolor=fill or "white",
                          linewidth=0.7)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color)


def _arrow(ax, x0, y0, x1, y1, color="#444", lw=0.7):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->", mutation_scale=8,
                        color=color, linewidth=lw)
    ax.add_patch(a)


# ──────────────────────────────────────────────────────────────────────
# Panel a — N-ladder
# ──────────────────────────────────────────────────────────────────────
def draw_panel_a(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    nature_panel_label(ax, "a", x=-0.02, y=1.05, fontsize=10)
    ax.set_title("Three-tier UKB N-ladder", fontsize=8, pad=2)

    # Three nested tier rectangles — Tier 1 ⊂ Tier 2 ⊂ Tier 3
    # Tier 3 outer
    _box(ax, 0.4, 0.6, 9.2, 5.8, "", C_BLUE_DARK, fill="#F0F4F8")
    ax.text(5.0, 6.05, "Tier 3 — N = 37,775", ha="center", va="center",
            fontsize=7.5, color=C_BLUE_DARK, fontweight="bold")

    # Tier 2 middle
    _box(ax, 1.4, 1.2, 6.6, 4.4, "", C_BLUE, fill="#E7EEF6")
    ax.text(4.7, 5.25, "Tier 2 — N = 11,820", ha="center", va="center",
            fontsize=7, color=C_BLUE, fontweight="bold")

    # Tier 1 inner
    _box(ax, 2.5, 2.0, 4.2, 2.6, "", C_BLUE_LIGHT, fill="#DCE6F2")
    ax.text(4.6, 4.25, "Tier 1 — N = 1,079", ha="center", va="center",
            fontsize=7, color=C_BLUE_DARK)
    ax.text(4.6, 3.2, "DS1≈1,079 × 0.8\n+ DS2 split", ha="center", va="center",
            fontsize=6, color="#222")

    # Right-side split sketch: UKB-DS1 (80%) / UKB-DS2 (20%)
    _box(ax, 8.2, 4.0, 1.4, 1.5, "UKB-DS1\n80%\ntrain+val+test",
         C_GREEN, fill="#E1F0EA", fontsize=5.5)
    _box(ax, 8.2, 2.0, 1.4, 1.5, "UKB-DS2\n20%\nheld-out",
         C_ORANGE, fill="#FBE5D7", fontsize=5.5)
    _arrow(ax, 7.0, 5.4, 8.2, 4.75, color="#444")
    _arrow(ax, 7.0, 2.4, 8.2, 2.75, color="#444")
    ax.text(7.7, 6.0, "80/20 within-UKB", fontsize=5.5, color="#666")


# ──────────────────────────────────────────────────────────────────────
# Panel b — RSCM pipeline
# ──────────────────────────────────────────────────────────────────────
def draw_panel_b(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    nature_panel_label(ax, "b", x=-0.02, y=1.05, fontsize=10)
    ax.set_title("RSCM analytical pipeline", fontsize=8, pad=2)

    # Decorative glass-brain: gray-matter ICA-network coverage (the GM input).
    if BRAIN_PNG.exists():
        brain = mpimg.imread(BRAIN_PNG)
        oi = OffsetImage(brain, zoom=0.105)
        ab = AnnotationBbox(oi, (5.0, 6.30), frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    ax.text(5.0, 5.05, "Gray-matter morphometry (99 networks)\n+ FNC (1,378 edges)",
            ha="center", va="center", fontsize=6.2, color=C_BLUE_DARK)

    # 4-step vertical pipeline beneath the GM/FNC input
    steps = [
        ("Fisher-$z$ $\\rightarrow$ SPD$_{53\\times53}$", C_BLUE),
        ("LogE tangent space\n+ Age/Gender residualization", C_GREEN),
        ("Nuclear-norm regression\n$\\min\\,\\|T - XB\\|_F^2 + \\lambda\\|B\\|_*$",
         C_ORANGE),
        ("SVD $B = U\\Sigma V^\\top$\n$\\rightarrow$ Mode-$k$ analysis",
         C_PURPLE),
    ]
    h, pad, x0, w = 0.86, 0.30, 1.6, 6.8
    y = 4.30
    _arrow(ax, 5.0, 4.72, 5.0, y + h, color="#666")
    for i, (text, color) in enumerate(steps):
        _box(ax, x0, y, w, h, text, color, fill="#FFFFFF", fontsize=6.5)
        if i < len(steps) - 1:
            _arrow(ax, x0 + w / 2, y, x0 + w / 2, y - pad, color="#666")
        y -= h + pad


# ──────────────────────────────────────────────────────────────────────
# Panel c — 4 rescue experiments
# ──────────────────────────────────────────────────────────────────────
def draw_panel_c(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    nature_panel_label(ax, "c", x=-0.02, y=1.05, fontsize=10)
    ax.set_title("Four targeted experiments", fontsize=8, pad=2)

    # Central Tier 3 anchor
    _box(ax, 3.6, 3.0, 2.8, 1.4,
         "Tier 3 anchor\nN = 37,775\n$\\lambda=0.3$ canonical",
         C_BLUE_DARK, fill="#E7EEF6", fontsize=6.5)

    # Four orbiting experiments
    # R1 — top-left
    _box(ax, 0.4, 5.2, 3.2, 1.5,
         "R1 — UKB-N=805\n$\\times$5 random subsets\n(same-method control)",
         C_GREEN, fill="#E1F0EA", fontsize=6)
    _arrow(ax, 2.0, 5.2, 4.0, 4.4, color="#666")

    # R2 — top-right
    _box(ax, 6.4, 5.2, 3.2, 1.5,
         "R2 — within-UKB\ncross-tier Mode-1\nProcrustes",
         C_PURPLE, fill="#F3E2EC", fontsize=6)
    _arrow(ax, 8.0, 5.2, 6.0, 4.4, color="#666")

    # B1 — bottom-left
    _box(ax, 0.4, 0.5, 3.2, 1.5,
         "B1 — 100-bootstrap\nMode-$k$ stability\nat Tier 3",
         C_ORANGE, fill="#FBE5D7", fontsize=6)
    _arrow(ax, 2.0, 2.0, 4.0, 3.0, color="#666")

    # B2 — bottom-right
    _box(ax, 6.4, 0.5, 3.2, 1.5,
         "B2 — $\\lambda$-sweep at\nN=37,775 via val-MSE\n($\\{0.1, 0.2, 0.3, 0.5, 1.0\\}$)",
         "#666666", fill="#EEEEEE", fontsize=6)
    _arrow(ax, 8.0, 2.0, 6.0, 3.0, color="#666")


# ──────────────────────────────────────────────────────────────────────
# Main figure
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ensure_brain_panel()
    apply_nature_strict()
    fig, axes = plt.subplots(1, 3, figsize=NATURE_DOUBLE)
    plt.subplots_adjust(left=0.02, right=0.99, top=0.92, bottom=0.04, wspace=0.10)

    draw_panel_a(axes[0])
    draw_panel_b(axes[1])
    draw_panel_c(axes[2])

    # NOTE: do NOT use bbox_inches='tight' here — schematic axes have
    # axis('off'), and tight bbox would shrink vertically because there are
    # no spines to anchor the bounds. Use the explicit figsize instead.
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PDF.with_suffix(".png"), dpi=300)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
