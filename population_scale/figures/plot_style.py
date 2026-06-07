"""NeuroImage-grade publication figure style.

Designed for maximum clarity and accessibility:
- Pure white backgrounds (mandatory for NeuroImage)
- Sans-serif fonts (Arial/Helvetica) at sizes legible after reduction
- Colorblind-safe palette (Okabe-Ito inspired + muted editorial tones)
- Vector-first output (PDF fonttype 42 for TrueType embedding)

NeuroImage figure widths:
  single column  =  85 mm  ≈ 3.35 in
  1.5 column     = 114 mm  ≈ 4.49 in
  double column  = 178 mm  ≈ 7.01 in
"""
import matplotlib as mpl
from cycler import cycler


def apply_neuroimage_style():
    """Configure rcParams for NeuroImage submission figures."""

    # ── Sans-serif font stack (NeuroImage requirement) ──
    sans_fonts = [
        "Arial",
        "Helvetica Neue",
        "Helvetica",
        "DejaVu Sans",
        "Liberation Sans",
        "sans-serif",
    ]

    # ── Colorblind-safe color cycle ──
    # Inspired by Okabe-Ito, tuned for print contrast
    color_cycle = [
        "#0072B2",   # blue (strong)
        "#D55E00",   # vermillion
        "#009E73",   # bluish green
        "#E69F00",   # orange
        "#56B4E9",   # sky blue
        "#CC79A7",   # reddish purple
        "#F0E442",   # yellow
        "#000000",   # black
    ]

    params = {
        # ── Typography ──
        "font.family":         "sans-serif",
        "font.sans-serif":     sans_fonts,
        "font.size":           7,
        "font.weight":         "regular",
        "axes.titlesize":      8,
        "axes.titleweight":    "bold",
        "axes.titlepad":       6,
        "axes.labelsize":      7,
        "axes.labelpad":       3,
        "axes.labelweight":    "regular",
        "xtick.labelsize":     6,
        "ytick.labelsize":     6,
        "legend.fontsize":     6,
        "figure.titlesize":    9,

        # ── Lines & markers ──
        "lines.linewidth":     1.2,
        "lines.markersize":    4,
        "lines.markeredgewidth": 0.6,
        "lines.solid_capstyle":  "round",
        "lines.solid_joinstyle": "round",
        "patch.linewidth":     0.6,
        "patch.facecolor":     "#ffffff",
        "patch.edgecolor":     "#333333",

        # ── Axes ──
        "axes.linewidth":      0.7,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.edgecolor":      "#333333",
        "axes.facecolor":      "#ffffff",
        "axes.labelcolor":     "#222222",
        "axes.grid":           False,
        "axes.axisbelow":      True,
        "axes.prop_cycle":     cycler(color=color_cycle),
        "axes.formatter.useoffset":  False,
        "axes.formatter.limits":     (-4, 6),

        # ── Ticks ──
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "xtick.major.width":   0.6,
        "ytick.major.width":   0.6,
        "xtick.minor.width":   0.4,
        "ytick.minor.width":   0.4,
        "xtick.major.size":    3,
        "ytick.major.size":    3,
        "xtick.minor.size":    1.5,
        "ytick.minor.size":    1.5,
        "xtick.major.pad":     2.5,
        "ytick.major.pad":     2.5,
        "xtick.color":         "#333333",
        "ytick.color":         "#333333",
        "xtick.top":           False,
        "ytick.right":         False,

        # ── Text ──
        "text.color":          "#222222",

        # ── Figure background ── MUST be white for NeuroImage
        "figure.facecolor":    "#ffffff",
        "savefig.facecolor":   "#ffffff",
        "savefig.transparent": False,

        # ── Output quality ──
        "figure.dpi":          150,
        "savefig.dpi":         600,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.02,
        "pdf.fonttype":        42,
        "ps.fonttype":         42,
        "svg.fonttype":        "none",

        # ── Layout ──
        "figure.autolayout":           False,
        "figure.constrained_layout.use": False,

        # ── Legend ──
        "legend.frameon":       False,
        "legend.borderpad":     0.2,
        "legend.handlelength":  1.2,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 0.8,
        "legend.labelspacing":  0.3,
        "legend.framealpha":    1.0,

        # ── Grid (off by default; enable per-panel) ──
        "grid.color":     "#cccccc",
        "grid.alpha":     0.4,
        "grid.linewidth": 0.4,
        "grid.linestyle": "-",

        # ── Image ──
        "image.interpolation": "nearest",
        "image.origin":        "lower",

        # ── Boxplot ──
        "boxplot.flierprops.linewidth":   0.6,
        "boxplot.medianprops.linewidth":  1.0,
        "boxplot.whiskerprops.linewidth": 0.6,
        "boxplot.boxprops.linewidth":     0.6,
        "boxplot.capprops.linewidth":     0.6,

        # ── Misc ──
        "hist.bins":       20,
        "path.simplify":   True,
        "axes3d.grid":     False,
    }
    mpl.rcParams.update(params)


# Backward compatibility alias
apply_nature_style = apply_neuroimage_style


# ════════════════════════════════════════════════════════════════════
# Nature-strict layer (figs/plot_style.py — appended 2026-05-31)
# Hard-locks Nature submission dimensions + lowercase panel labels.
# ════════════════════════════════════════════════════════════════════

# Nature column widths (inches)
NATURE_SINGLE = (3.5, 2.6)          #  89 mm
NATURE_ONE_AND_HALF = (4.5, 3.0)    # 115 mm
NATURE_DOUBLE = (7.2, 3.0)          # 183 mm


def apply_nature_strict():
    """Apply Nature-strict figure style on top of NeuroImage baseline.

    Builds on apply_neuroimage_style() and tightens a handful of specs to
    match Nature submission templates: title not bold, marker size 4, line
    width 1.0, panel-label tracking.
    """
    apply_neuroimage_style()
    mpl.rcParams.update({
        "axes.titleweight": "regular",  # Nature title is regular, panel label is bold
        "axes.titlesize":   8,
        "axes.labelsize":   8,
        "xtick.labelsize":  7,
        "ytick.labelsize":  7,
        "legend.fontsize":  6.5,
        "lines.linewidth":  1.0,
        "lines.markersize": 4,
        "axes.linewidth":   0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "savefig.dpi":      300,
        "figure.dpi":       100,
    })


def nature_panel_label(ax, label, x=-0.14, y=1.08, fontsize=9):
    """Place lowercase bold panel label outside axis area (Nature convention).

    Example: nature_panel_label(ax, 'a')  →  bold lowercase 'a' at top-left.
    """
    ax.text(
        x, y, label, transform=ax.transAxes,
        fontsize=fontsize, fontweight="bold", va="top", ha="left",
    )

