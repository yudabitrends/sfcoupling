"""
Shared publication-quality style module for all revision figures.

Design principles:
  - Okabe-Ito colorblind-safe palette
  - Sans-serif fonts (DejaVu Sans as Arial fallback)
  - Despined axes (no top/right spines)
  - Consistent panel labeling (bold A, B, C, ...) placed well above titles
  - Journal-standard sizes: 89 mm (single) and 183 mm (double) columns
  - 300 DPI raster, PDF primary
  - Generous margins and pad to prevent any text overflow
  - Helper for embedding rasterized PDFs (brain / chord / coupling panels)
    via PyMuPDF into matplotlib axes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --- Okabe-Ito colorblind-safe palette ---
OKABE_ITO = {
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "grey": "#6E6E6E",
    "lightgrey": "#BFBFBF",
}
OKABE_ITO_LIST = [
    OKABE_ITO["blue"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["green"],
    OKABE_ITO["orange"],
    OKABE_ITO["purple"],
    OKABE_ITO["skyblue"],
    OKABE_ITO["yellow"],
    OKABE_ITO["black"],
]

# --- Method-specific colors (locked across all figures) ---
METHOD_COLORS = {
    "Ridge": OKABE_ITO["grey"],
    "MLP": OKABE_ITO["yellow"],
    "RRR": OKABE_ITO["skyblue"],
    "PLS": OKABE_ITO["vermillion"],
    "Nuclear Norm": OKABE_ITO["blue"],
    "OptShrink": OKABE_ITO["green"],
    "NN-Init MLP": OKABE_ITO["orange"],
    "NN+MLP res.": OKABE_ITO["purple"],
}

COHORT_COLORS = {
    "DS1": OKABE_ITO["blue"],
    "DS2": OKABE_ITO["vermillion"],
    "UKB": OKABE_ITO["green"],
}

# --- Figure size presets (inches, = mm / 25.4) ---
FIG_SINGLE_COL = (3.5, 2.6)
FIG_DOUBLE_COL = (7.2, 5.0)
FIG_DOUBLE_TALL = (7.2, 8.0)
FIG_DOUBLE_SQUARE = (7.2, 7.2)
FIG_DOUBLE_WIDE = (7.2, 4.2)


def apply_style() -> None:
    """Install matplotlib rcParams for all revision figures. Idempotent."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica",
                             "Liberation Sans"],
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.labelweight": "normal",
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.titlepad": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,
        "legend.borderpad": 0.2,
        "legend.columnspacing": 0.8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO_LIST),
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 110,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def panel_label(ax: "mpl.axes.Axes", label: str,
                x: float = -0.16, y: float = 1.22,
                fontsize: float = 12) -> None:
    """Add a bold panel label placed well above the title so it never overlaps.

    Uses figure fraction outside the axes. Tuned so labels don't collide with
    titles even at small figure sizes.
    """
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top", ha="left",
            clip_on=False)


def despine_all(ax: "mpl.axes.Axes") -> None:
    """Remove all spines (useful for schematic or image-embedded panels)."""
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def save_figure(fig: "mpl.figure.Figure", name: str,
                out_dir: Path | None = None) -> list[Path]:
    """Save figure as both PDF (vector) and PNG (300 DPI) for review."""
    if out_dir is None:
        out_dir = Path("/home/users/ybi3/sfcoupling/paper/standalone/figure")
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in ("pdf", "png"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=300)
        outputs.append(path)
    return outputs


# ---------------------------------------------------------------------------
# PDF-to-image embedding helpers (used by spatial/biological aggregates)
# ---------------------------------------------------------------------------

def pdf_to_array(pdf_path: str | Path, dpi: int = 400,
                 trim: bool = True) -> np.ndarray:
    """Rasterize the first page of a PDF into a numpy RGBA array.

    Requires PyMuPDF (`pip install PyMuPDF`). Used to embed panel PDFs from
    the original fig2/, fig6/, fig7/ directories into matplotlib aggregated
    figures.

    Args:
        pdf_path: path to the PDF file.
        dpi: rasterization DPI (default 400 for print quality).
        trim: if True, crop out pure-white borders.

    Returns:
        (H, W, 4) uint8 RGBA array ready for ax.imshow.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi, alpha=True)
    # Convert to numpy: pix.samples is bytes in RGBA order
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    if pix.n == 3:
        # Add alpha channel
        alpha = np.full(
            (pix.height, pix.width, 1), 255, dtype=np.uint8)
        arr = np.concatenate([arr, alpha], axis=-1)
    doc.close()

    if trim:
        arr = _trim_white(arr)

    return arr


def _trim_white(img: np.ndarray, tol: int = 5) -> np.ndarray:
    """Crop out nearly-white borders around an RGBA image."""
    if img.ndim != 3:
        return img
    # Treat a row/column as "white" if all RGB values are > 255 - tol
    rgb = img[..., :3]
    is_white = (rgb > (255 - tol)).all(axis=-1)
    non_white_rows = np.where(~is_white.all(axis=1))[0]
    non_white_cols = np.where(~is_white.all(axis=0))[0]
    if non_white_rows.size == 0 or non_white_cols.size == 0:
        return img
    r0, r1 = non_white_rows[0], non_white_rows[-1] + 1
    c0, c1 = non_white_cols[0], non_white_cols[-1] + 1
    # Add a small pad so imshow doesn't touch the border
    pad = 4
    r0 = max(0, r0 - pad)
    c0 = max(0, c0 - pad)
    r1 = min(img.shape[0], r1 + pad)
    c1 = min(img.shape[1], c1 + pad)
    return img[r0:r1, c0:c1]


def embed_pdf(ax: "mpl.axes.Axes", pdf_path: str | Path,
              dpi: int = 400, trim: bool = True) -> None:
    """Embed a rasterized PDF as an image in the given axes and despine."""
    arr = pdf_to_array(pdf_path, dpi=dpi, trim=trim)
    ax.imshow(arr, interpolation="antialiased")
    despine_all(ax)


__all__ = [
    "OKABE_ITO", "OKABE_ITO_LIST", "METHOD_COLORS", "COHORT_COLORS",
    "FIG_SINGLE_COL", "FIG_DOUBLE_COL", "FIG_DOUBLE_TALL", "FIG_DOUBLE_SQUARE",
    "FIG_DOUBLE_WIDE",
    "apply_style", "panel_label", "despine_all", "save_figure",
    "pdf_to_array", "embed_pdf",
]
