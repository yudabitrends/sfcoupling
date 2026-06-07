"""Shared utilities for publication-grade figure scripts."""
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Figure sizing — NeuroImage specification
# ---------------------------------------------------------------------------
FIG_W_SINGLE = 3.35   # 85 mm single column
FIG_W_1HALF = 4.49    # 114 mm 1.5 column
FIG_W_DOUBLE = 7.01   # 178 mm double column
GRID_COLOR = "#cccccc"
GRID_MINOR_COLOR = "#e0e0e0"
SPINE_COLOR = "#333333"
TEXT_MUTED = "#666666"
BAR_EDGE = "#ffffff"

# ---------------------------------------------------------------------------
# Unified colour palette — colorblind-safe (Okabe-Ito based)
# ---------------------------------------------------------------------------
COLOR_DS1 = "#0072B2"       # blue
COLOR_DS2 = "#D55E00"       # vermillion
COLOR_NEUTRAL = "#999999"   # gray
COLOR_DARK = "#222222"      # near-black
COLOR_POSITIVE = "#009E73"  # bluish green
COLOR_NEGATIVE = "#CC3311"  # red
COLOR_ACCENT = "#CC79A7"    # reddish purple
COLOR_HIGHLIGHT = "#E69F00" # orange
COLOR_WARM = "#D55E00"      # vermillion
COLOR_COOL = "#0072B2"      # blue

COLOR_PRIMARY = COLOR_DS1
COLOR_SECONDARY = COLOR_DS2

# Legacy aliases
COLOR_D1 = COLOR_DARK
COLOR_D2 = COLOR_DS1
COLOR_NOID = COLOR_NEUTRAL
COLOR_WITHID = COLOR_DS1
COLOR_LEAK_G = COLOR_DS2
COLOR_LEAK_F = COLOR_ACCENT

# Colormaps — perceptually uniform, colorblind-safe
# Diverging: blue-white-red (no green, accessible)
CMAP_STAT_DIVERGING = LinearSegmentedColormap.from_list(
    "stat_diverging",
    ["#2166AC", "#4393C3", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#D6604D", "#B2182B"],
    N=256,
)
# Sequential: viridis-inspired warm (white to dark)
CMAP_MAGNITUDE = LinearSegmentedColormap.from_list(
    "magnitude_seq",
    ["#FFFFFF", "#FDE0C5", "#EB9B59", "#C25B28", "#7F2704"],
    N=256,
)
# Heat: warm sequential (for coupling strength)
CMAP_HEAT = LinearSegmentedColormap.from_list(
    "heat_seq",
    ["#FFFFFF", "#FEE5D9", "#FCAE91", "#FB6A4A", "#CB181D", "#67000D"],
    N=256,
)
# Brain: cool sequential (for brain overlays)
CMAP_BRAIN = LinearSegmentedColormap.from_list(
    "brain_seq",
    ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"],
    N=256,
)

# ---------------------------------------------------------------------------
# Unified method styling
# ---------------------------------------------------------------------------
# Method colors — Okabe-Ito derived, distinct for all color vision types
METHOD_COLORS = {
    "Nuclear Norm": "#0072B2",  # blue (primary method)
    "OptShrink":    "#56B4E9",  # sky blue
    "RRR":          "#E69F00",  # orange
    "PLS":          "#009E73",  # bluish green
    "Ridge":        "#999999",  # gray (baseline)
    "MLP":          "#D55E00",  # vermillion
    "NN-init MLP":  "#CC79A7",  # reddish purple
}

METHOD_MARKERS = {
    "Nuclear Norm": "o",
    "OptShrink": "D",
    "RRR": "s",
    "PLS": "^",
    "Ridge": "v",
    "MLP": "X",
    "NN-init MLP": "P",
}

METHOD_NAME_MAP = {
    "Nuclear_Norm": "Nuclear Norm",
    "Rrr": "RRR",
    "Pls": "PLS",
    "Linear_OptShrink": "OptShrink",
    "Ridge": "Ridge",
}

LABEL_D1 = "Dataset 1 (test)"
LABEL_D2 = "Dataset 2 (external)"

# Brain-domain palette — high-contrast, colorblind-accessible
DOMAIN_COLORS = {
    "SC": "#882255",   # wine (subcortical)
    "HP": "#44AA99",   # teal (hippocampal)
    "AUD": "#DDCC77",  # sand (auditory)
    "SM": "#0072B2",   # blue (sensorimotor)
    "VS": "#009E73",   # green (visual)
    "CC": "#CC6677",   # rose (cognitive control)
    "PA": "#AA4499",   # purple (parietal)
    "DM": "#332288",   # indigo (default mode)
    "CB": "#E69F00",   # orange (cerebellar)
    "Other": "#BBBBBB", # gray
}

# ---------------------------------------------------------------------------
# t-table for 95 % CI
# ---------------------------------------------------------------------------
_T_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    20: 2.086, 25: 2.060, 30: 2.042,
}


def _t_crit(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T_TABLE:
        return _T_TABLE[df]
    if df < 30:
        lo = max(k for k in _T_TABLE if k <= df)
        hi = min(k for k in _T_TABLE if k >= df)
        if lo == hi:
            return _T_TABLE[lo]
        frac = (df - lo) / (hi - lo)
        return _T_TABLE[lo] + frac * (_T_TABLE[hi] - _T_TABLE[lo])
    if df < 60:
        return 2.000
    return 1.960


def compute_ci(values, confidence=0.95):
    """Return (mean, ci_low, ci_high) using t-interval."""
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    mean = float(np.mean(arr))
    if n <= 1:
        return mean, float("nan"), float("nan")
    std = float(np.std(arr, ddof=1))
    t = _t_crit(n - 1)
    half = t * std / math.sqrt(n)
    return mean, mean - half, mean + half


def compute_ci_array(array_2d):
    """Per-column CI from (n_seeds, n_features) array -> (mean, lo, hi) arrays."""
    arr = np.asarray(array_2d, dtype=np.float64)
    n = arr.shape[0]
    mean = np.mean(arr, axis=0)
    if n <= 1:
        nans = np.full_like(mean, np.nan)
        return mean, nans, nans
    std = np.std(arr, axis=0, ddof=1)
    t = _t_crit(n - 1)
    half = t * std / math.sqrt(n)
    return mean, mean - half, mean + half


def load_summary_csv(path) -> pd.DataFrame:
    return pd.read_csv(path)


def _deep_get(d: dict, path: Tuple[str, ...]) -> Any:
    cur = d
    for p in path:
        cur = cur[p]
    return cur


def style_axes(ax, *, all_spines: bool = False, xgrid: bool = False, ygrid: bool = True):
    """Apply clean NeuroImage-ready axis styling: white background, minimal spines."""
    ax.set_facecolor("#ffffff")
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    if all_spines:
        for name in ["top", "right"]:
            ax.spines[name].set_visible(True)
            ax.spines[name].set_color(SPINE_COLOR)
            ax.spines[name].set_linewidth(0.5)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.tick_params(
        direction="out",
        length=2.5,
        width=0.5,
        colors=SPINE_COLOR,
        labelcolor=SPINE_COLOR,
    )
    if xgrid or ygrid:
        ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.5)
    if xgrid:
        ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.4)
    if not xgrid and not ygrid:
        ax.grid(False)
    if all_spines:
        ax.xaxis.set_tick_params(which="both", top=False)
    ax.set_rasterized(False)


def style_colorbar(cbar, label: Optional[str] = None):
    cbar.outline.set_edgecolor("#999999")
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(labelsize=5.5, length=2, width=0.4, colors=SPINE_COLOR)
    if label:
        cbar.set_label(label, fontsize=6, color=SPINE_COLOR)


def panel_label(ax, label, x=-0.12, y=1.06):
    """Add bold lowercase panel label (NeuroImage convention)."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
        color="#000000",
    )


def add_panel_title(
    ax,
    title: str,
    subtitle: Optional[str] = None,
    loc: str = "left",
    title_y: float = 1.03,
    subtitle_y: float = 1.07,
):
    ax.set_title(
        title,
        loc=loc,
        fontsize=7,
        color=COLOR_DARK,
        pad=6,
        fontweight="bold",
        y=title_y,
    )
    if subtitle:
        ax.text(
            0.0 if loc == "left" else 0.5,
            subtitle_y,
            subtitle,
            transform=ax.transAxes,
            fontsize=5.5,
            color=TEXT_MUTED,
            ha="left" if loc == "left" else "center",
            va="bottom",
        )


def add_baseline(ax, y=0.0, *, horizontal=True, style="solid", alpha=0.7):
    kwargs = dict(color="#666666", linewidth=0.5, alpha=alpha)
    if style == "dashed":
        kwargs["linestyle"] = "--"
    elif style == "dotted":
        kwargs["linestyle"] = ":"
    if horizontal:
        ax.axhline(y, **kwargs)
    else:
        ax.axvline(y, **kwargs)


def add_identity_line(ax, xs: Sequence[float], ys: Sequence[float]):
    lo = min(min(xs), min(ys))
    hi = max(max(xs), max(ys))
    margin = (hi - lo) * 0.08 if hi > lo else 0.05
    ax.plot(
        [lo - margin, hi + margin],
        [lo - margin, hi + margin],
        linestyle=(0, (3, 3)),
        color="#999999",
        linewidth=0.5,
        alpha=0.6,
        zorder=1,
    )


def method_style(label: str):
    return METHOD_COLORS.get(label, COLOR_NEUTRAL), METHOD_MARKERS.get(label, "o")


def method_legend_handles(labels: Sequence[str]):
    handles = []
    for label in labels:
        color, marker = method_style(label)
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                markersize=4.5,
                linestyle="-",
                linewidth=1.1,
                color=color,
                label=label,
            )
        )
    return handles


def draw_grouped_bars(
    ax,
    categories: Sequence[str],
    group_labels: Sequence[str],
    values: np.ndarray,
    errors: Optional[np.ndarray] = None,
    colors: Optional[Sequence[str]] = None,
    width: Optional[float] = None,
    ylabel: Optional[str] = None,
):
    """Draw compact grouped bars with understated CI styling."""
    values = np.asarray(values, dtype=float)
    n_groups, n_categories = values.shape
    x = np.arange(n_categories)
    width = width or min(0.76 / max(n_groups, 1), 0.28)
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2) * width
    colors = list(colors or [COLOR_PRIMARY, COLOR_SECONDARY])
    for gi in range(n_groups):
        yerr = None if errors is None else np.asarray(errors[gi], dtype=float)
        ax.bar(
            x + offsets[gi],
            values[gi],
            width=width * 0.88,
            yerr=yerr,
            color=colors[gi % len(colors)],
            label=group_labels[gi],
            edgecolor="white",
            linewidth=0.4,
            capsize=2 if yerr is not None else 0,
            error_kw={"linewidth": 0.6, "capthick": 0.6, "ecolor": "#333333"},
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    if ylabel:
        ax.set_ylabel(ylabel)
    style_axes(ax, ygrid=True)


def draw_lollipop_series(
    ax,
    labels: Sequence[str],
    values: Sequence[float],
    colors: Optional[Sequence[str]] = None,
    xlabel: Optional[str] = None,
    fmt: str = "{:.0%}",
):
    vals = np.asarray(values, dtype=float)
    y = np.arange(len(labels))
    colors = list(colors or [COLOR_PRIMARY] * len(labels))
    ax.hlines(y, 0, vals, color="#cccccc", linewidth=0.8, zorder=1)
    ax.scatter(vals, y, s=30, color=colors, edgecolors="white", linewidths=0.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    if xlabel:
        ax.set_xlabel(xlabel)
    for yi, val in enumerate(vals):
        ax.text(
            val + 0.01,
            yi + 0.06,
            fmt.format(val),
            va="center",
            fontsize=5.8,
            color=TEXT_MUTED,
            fontweight="medium",
        )
    style_axes(ax, ygrid=False, xgrid=True)


def draw_stat_heatmap(
    ax,
    mat: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    cmap=CMAP_STAT_DIVERGING,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annotate: bool = True,
    fmt: str = "{:.2f}",
):
    mat = np.asarray(mat, dtype=float)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=5.8)
    style_axes(ax, all_spines=True, ygrid=False, xgrid=False)
    if annotate:
        threshold = np.nanmedian([vmin if vmin is not None else np.nanmin(mat), vmax if vmax is not None else np.nanmax(mat)])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                color = "white" if val > threshold else COLOR_DARK
                ax.text(
                    j,
                    i,
                    fmt.format(val),
                    ha="center",
                    va="center",
                    fontsize=5.3,
                    color=color,
                    fontweight="medium",
                    path_effects=None,
                )
    return im


def draw_method_scatter(
    ax,
    xs: Sequence[float],
    ys: Sequence[float],
    labels: Sequence[str],
    colors: Optional[Sequence[str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
):
    colors = list(colors or [COLOR_PRIMARY] * len(labels))
    # One scatter call per method so a colour-keyed legend replaces inline labels
    # (inline text labels overlap badly when methods cluster, e.g. transfer panels).
    for x, y, label, c in zip(xs, ys, labels, colors):
        ax.scatter([x], [y], s=36, color=c, edgecolors="white", linewidths=0.5,
                   zorder=3, label=label)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    add_identity_line(ax, xs, ys)
    style_axes(ax, ygrid=True, xgrid=True)
    ax.legend(loc="best", fontsize=5.2, frameon=False, handletextpad=0.15,
              labelspacing=0.2, borderpad=0.2, markerscale=0.75)


def adjust_annotations(ax, texts, max_iter=60):
    """Greedy 2D label repulsion for scatter annotations."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for _ in range(max_iter):
        moved = False
        for i in range(len(texts)):
            bb1 = texts[i].get_window_extent(renderer)
            for j in range(i + 1, len(texts)):
                bb2 = texts[j].get_window_extent(renderer)
                if not bb1.overlaps(bb2):
                    continue
                inv = ax.transData.inverted()
                dx_px = (bb1.width + bb2.width) * 0.28
                dy_px = (bb1.height + bb2.height) * 0.55
                x0, y0 = inv.transform((0, 0))
                x1, y1 = inv.transform((dx_px, dy_px))
                shift_x = abs(x1 - x0)
                shift_y = abs(y1 - y0)
                pos_i = texts[i].get_position()
                pos_j = texts[j].get_position()
                if pos_i[0] <= pos_j[0]:
                    texts[i].set_position((pos_i[0] - shift_x, pos_i[1] - shift_y * 0.35))
                    texts[j].set_position((pos_j[0] + shift_x, pos_j[1] + shift_y * 0.35))
                else:
                    texts[i].set_position((pos_i[0] + shift_x, pos_i[1] + shift_y * 0.35))
                    texts[j].set_position((pos_j[0] - shift_x, pos_j[1] - shift_y * 0.35))
                moved = True
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                bb1 = texts[i].get_window_extent(renderer)
        if not moved:
            break


def categorical_handles(domain_order: Iterable[str], full_names: Optional[Dict[str, str]] = None):
    handles = []
    for dom in domain_order:
        handles.append(
            mpatches.Patch(
                facecolor=DOMAIN_COLORS[dom],
                edgecolor="none",
                label=(full_names or {}).get(dom, dom),
            )
        )
    return handles
