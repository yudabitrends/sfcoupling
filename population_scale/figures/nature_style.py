"""Shared Nature-grade figure style for the NeuroImage paper, matching the PD
project's house style (scripts/make_nature_figures.py).

Usage in a figure script:
    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import nature_style as ns
    ns.apply()
    ...
    ns.plabel(ax, "a", "Panel title")
    ax.plot(..., color=ns.PRIMARY)
    ns.refline(ax, 0)                  # dashed grey reference
    ns.save(fig, OUT_DIR, "fig_name")  # pdf (+optional png)
"""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4  # mm -> inches (PD sizes figures in mm; 180mm = double column)

# ---- restrained, colorblind-aware palette (identical hues to the PD figures) ----
PRIMARY   = "#2166AC"   # blue   (main series / observed)
ACCENT    = "#B2182B"   # red    (highlight / key result)
GREEN     = "#5AAE61"   # green  (secondary)
ORANGE    = "#E08214"   # orange (tertiary / floor)
LIGHTBLUE = "#4393C3"   # light blue (paired/control)
GREY      = "#9E9E9E"   # mid grey
NULL      = "#BDBDBD"   # null / baseline bars
REF       = "#888888"   # dashed reference lines
INK       = "#222222"   # near-black text
SUBTLE    = "#555555"   # annotation grey
# ordered cycle for multi-series panels
CYCLE = [PRIMARY, ACCENT, GREEN, ORANGE, LIGHTBLUE, GREY]


def apply():
    """Set the PD house-style rcParams globally (matches make_nature_figures.py)."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.8, "legend.frameon": False,
        # PD title convention: bold, left-aligned ("a  Title" reads as one unit)
        "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlepad": 4,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 1.2, "lines.markersize": 4,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
    })


# ---- drop-in replacements so figure scripts only change ONE import line ----
def apply_nature_strict():
    """Drop-in for figs.plot_style.apply_nature_strict, but PD house style."""
    apply()


def nature_panel_label(ax, label, x=-0.11, y=1.08, fontsize=9):
    """Drop-in for figs.plot_style.nature_panel_label. Bold letter placed OUTSIDE
    the axis at top-left so it clears the (left-aligned, bold) panel title; the
    pair then reads as PD's "a  Title"."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", va="bottom", ha="left")


# name aliases so any existing import works unchanged
apply_neuroimage_style = apply
apply_nature_style = apply


def plabel(ax, letter, title, fs=8):
    """Nature-style bold panel label + title, top-left (matches PD plabel)."""
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", fontsize=fs)


def corner_label(ax, letter, fs=9):
    """Bold panel letter at the axes top-left (for axis-off / schematic panels)."""
    ax.text(-0.01, 1.02, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=fs, ha="left", va="bottom")


def refline(ax, x=None, y=None, lw=0.7, ls="--", color=REF):
    if x is not None:
        ax.axvline(x, color=color, lw=lw, ls=ls, zorder=0)
    if y is not None:
        ax.axhline(y, color=color, lw=lw, ls=ls, zorder=0)


def annotate(ax, s, xy=(0.04, 0.96), color=SUBTLE, fs=6, **kw):
    ax.text(xy[0], xy[1], s, transform=ax.transAxes, va="top", ha="left",
            fontsize=fs, color=color, **kw)


def save(fig, out_dir, name, formats=("pdf", "png")):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        kw = {"dpi": 600} if ext in ("tiff", "tif") else {}
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"[fig] wrote {name}." + "/".join(formats))
