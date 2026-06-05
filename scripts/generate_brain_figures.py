#!/usr/bin/env python3
"""Generate brain visualization figures (Figs 7-9) for sfcoupling paper.

Figure 7: GM ROI importance glass brain (nilearn markers)
Figure 8: FNC coupling matrix sorted by functional domain
Figure 9: Domain-to-domain coupling chord diagram
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from figs.plot_style import apply_nature_style
from figs.utils import (
    panel_label,
    style_axes,
    style_colorbar,
    add_panel_title,
    CMAP_MAGNITUDE,
    CMAP_HEAT,
    DOMAIN_COLORS as SHARED_DOMAIN_COLORS,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ATLAS_PATH = "/data/qneuromark/Network_templates/NeuroMark3/T1.nii"
B_MATRIX_PATH = PROJECT_ROOT / "results/ukb/multivariate_methods/decompositions/nuclear_norm_seed42_B.npy"
GM_NAMES_PATH = PROJECT_ROOT / "aligned_features/meta/feature_maps/gm_feature_names.txt"
FNC_NAMES_PATH = PROJECT_ROOT / "aligned_features/meta/feature_maps/fnc_edge_names.txt"
FIG_DIR = PROJECT_ROOT / "figures"


def save_panel(fig, fig_name: str, panel_name: str):
    """Save a single panel PDF into figures/{fig_name}/{panel_name}.pdf"""
    d = FIG_DIR / fig_name
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{panel_name}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fig_name}/{panel_name} saved")


# ---------------------------------------------------------------------------
# NeuroMark3 SBM domain mapping (from NeuroMark3_Labels.docx, T1 adult SBM)
# Key: IC index (1-based) -> (domain, abbreviation, label, MNI x, y, z)
# ---------------------------------------------------------------------------
SBM_LABELS = {
    # SC (3)
    1:   ("SC", "Puta", "Putamen", -27, -6, 4),
    3:   ("SC", "Caud", "Caudate", 16, 18, 10),
    2:   ("SC", "Thal", "Thalamus", 12, -15, 10),
    # HP (1)
    68:  ("HP", "Hipp", "Hippocampus", -31, -22, -16),
    # AUD (8)
    31:  ("AUD", "ITG", "Inferior Temporal Gyrus", 60, -25, -30),
    35:  ("AUD", "MTG", "Middle Temporal Gyrus", -48, -37, 0),
    39:  ("AUD", "MTG/STG", "Middle/Superior Temporal Gyrus", -60, 1, -21),
    11:  ("AUD", "TP", "Temporal Pole", 39, 22, -31),
    47:  ("AUD", "MTG", "Middle Temporal Gyrus", 43, -54, 15),
    18:  ("AUD", "ITG/TP", "Inferior Temporal Gyrus/Temporal Pole", 31, 3, -49),
    9:   ("AUD", "L-TP", "Left Temporal Pole", -24, 12, -39),
    94:  ("AUD", "ITG/MTG", "Inferior/Middle Temporal Gyrus", -55, -69, -6),
    # SM (4)
    24:  ("SM", "SMA", "Supplementary Motor Area", -7, 1, 72),
    5:   ("SM", "ParaCG", "Paracentral Gyrus", -10, -22, 73),
    95:  ("SM", "PreCG/PostCG", "Precentral/Postcentral Gyrus", -34, -22, 60),
    8:   ("SM", "RO", "Rolandic Operculum", -37, -36, 18),
    # VS (12)
    27:  ("VS", "LingG", "Lingual Gyrus", 27, -54, 7),
    64:  ("VS", "MOG", "Middle Occipital Gyrus", 19, -99, -7),
    4:   ("VS", "Fusi", "Fusiform", 18, 1, -42),
    29:  ("VS", "CalG", "Calcarine Gyrus", 9, -81, 6),
    85:  ("VS", "MOG/SOG", "Middle/Superior Occipital Gyrus", -25, -66, 30),
    84:  ("VS", "Fusi", "Fusiform", -30, -49, -6),
    15:  ("VS", "IOG/MOG", "Inferior/Middle Occipital Gyrus", -40, -64, -1),
    52:  ("VS", "SOG/Cuneus", "Superior Occipital Gyrus/Cuneus", 15, -96, 22),
    34:  ("VS", "LingG/IOG", "Lingual/Inferior Occipital Gyrus", 22, -84, -6),
    88:  ("VS", "L-CalG", "Left Calcarine Gyrus", -9, -99, -13),
    48:  ("VS", "MOG/SOG", "Middle/Superior Occipital Gyrus", 27, -78, 19),
    43:  ("VS", "Fusi", "Fusiform", 43, -19, -34),
    # CC (13)
    17:  ("CC", "SMFG", "Superior Medial Frontal Gyrus", 33, 61, -6),
    42:  ("CC", "SMOFG", "Superior Medial Orbital Frontal Gyrus", 1, 30, -28),
    13:  ("CC", "IFG", "Inferior Frontal Gyrus", -36, 16, 24),
    20:  ("CC", "MFG", "Middle Frontal Gyrus", -34, 4, 31),
    7:   ("CC", "Insu/RO", "Insula/Rolandic Operculum", 33, 21, 10),
    38:  ("CC", "Insu", "Insula", -45, 10, -4),
    55:  ("CC", "IOFG", "Inferior Orbital Frontal Gyrus", 48, 27, -13),
    49:  ("CC", "MFG", "Middle Frontal Gyrus", -31, 42, 37),
    69:  ("CC", "OC", "Olfactory Cortex", 1, 13, -22),
    62:  ("CC", "SFG", "Superior Frontal Gyrus", 27, -1, 46),
    14:  ("CC", "MFG", "Middle Frontal Gyrus", -27, 28, 33),
    10:  ("CC", "IFG/MFG", "Inferior/Middle Frontal Gyrus", -33, 15, 31),
    78:  ("CC", "IFG/MFG", "Inferior/Middle Frontal Gyrus", -36, 27, 19),
    # PA (4)
    37:  ("PA", "SPL", "Superior Parietal Lobule", -27, -57, 39),
    72:  ("PA", "IPL", "Inferior Parietal Lobule", 39, -43, 55),
    51:  ("PA", "SMG", "Supramarginal Gyrus", 33, -33, 40),
    100: ("PA", "IPL/SMG", "Inferior Parietal Lobule/Supramarginal Gyrus", 51, -48, 52),
    # DM (8)
    36:  ("DM", "ACC", "Anterior Cingulate Cortex", 1, 33, -1),
    25:  ("DM", "PreCu/PCC", "Precuneus/Posterior Cingulate Cortex", -10, -45, 33),
    56:  ("DM", "ACC/MCC", "Anterior/Middle Cingulate Cortex", -9, 7, 36),
    57:  ("DM", "PreCu", "Precuneus", -18, -58, 21),
    54:  ("DM", "PCC", "Posterior Cingulate Cortex", 1, -43, 18),
    41:  ("DM", "AG", "Angular Gyrus", 45, -45, 18),
    67:  ("DM", "PreCu", "Precuneus", 9, -75, 40),
    16:  ("DM", "PreCu", "Precuneus", 0, -55, 64),
    # CB (13)
    22:  ("CB", "CB", "Cerebellum 4_5_6/Crus 1", -30, -36, -33),
    6:   ("CB", "Vermis", "Cerebellar Vermis/Cerebellum 6", 3, -75, -16),
    91:  ("CB", "L-CB Crus", "Left Cerebellum Crus 1_2", -15, -90, -22),
    19:  ("CB", "Vermis", "Vermis/Cerebellum 4_5", 1, -46, -9),
    87:  ("CB", "CB", "Cerebellum 8_7_9/Crus 2", -37, -43, -46),
    33:  ("CB", "CB", "Cerebellum 6_7_8/Crus 1_2", 19, -72, -36),
    98:  ("CB", "CB Crus", "Cerebellum Crus 1_2", 9, -84, -28),
    32:  ("CB", "CB", "Cerebellum 8", 25, -55, -60),
    77:  ("CB", "CB", "Cerebellum 8_9", 13, -48, -61),
    53:  ("CB", "R-CB Crus", "Right Cerebellum Crus 1_2", 46, -61, -49),
    74:  ("CB", "CB Crus", "Cerebellum Crus 1_2", 46, -48, -39),
    59:  ("CB", "L-CB Crus", "Left Cerebellum Crus 1_2", -48, -55, -48),
    66:  ("CB", "CB Crus", "Cerebellum Crus 2/Cerebellum 7_8", -25, -81, -46),
}

# NeuroMark1 fMRI 53-IC domain mapping (0-indexed IC)
# From: /data/qneuromark/Network_templates/NeuroMark1/Neuromark_fMRI_1.0.txt
FNC_DOMAIN_RANGES = {
    "SC":  (0, 5),    # IC_0 to IC_4
    "AUD": (5, 7),    # IC_5 to IC_6
    "SM":  (7, 16),   # IC_7 to IC_15
    "VS":  (16, 25),  # IC_16 to IC_24
    "CC":  (25, 42),  # IC_25 to IC_41
    "DM":  (42, 49),  # IC_42 to IC_48
    "CB":  (49, 53),  # IC_49 to IC_52
}

# Domain display order and colors
DOMAIN_ORDER_SBM = ["SC", "HP", "AUD", "SM", "VS", "CC", "PA", "DM", "CB", "Other"]
DOMAIN_ORDER_FNC = ["SC", "AUD", "SM", "VS", "CC", "DM", "CB"]

DOMAIN_COLORS = SHARED_DOMAIN_COLORS

DOMAIN_FULL_NAMES = {
    "SC": "Subcortical", "HP": "Hippocampal", "AUD": "Auditory",
    "SM": "Sensorimotor", "VS": "Visual", "CC": "Cognitive Control",
    "PA": "Parietal", "DM": "Default Mode", "CB": "Cerebellar",
    "Other": "Other",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_data():
    """Load B matrix, ROI names, and FNC edge names."""
    B = np.load(B_MATRIX_PATH)
    gm_names = Path(GM_NAMES_PATH).read_text().strip().split("\n")
    fnc_names = Path(FNC_NAMES_PATH).read_text().strip().split("\n")
    roi_indices = [int(n.replace("roi_", "")) for n in gm_names]
    return B, roi_indices, fnc_names


def get_roi_domain_and_coords(roi_indices):
    """Map each ROI to its SBM domain and MNI coordinates.

    For labeled SBMs, use docx coordinates. For unlabeled ROIs, compute
    center of mass from the atlas.
    """
    import nibabel as nib
    from scipy.ndimage import center_of_mass

    img = nib.load(ATLAS_PATH)
    data = img.get_fdata()
    affine = img.affine

    domains = []
    coords = []

    for roi_idx in roi_indices:
        ic_1based = roi_idx + 1  # volume index 0-based -> IC 1-based
        if ic_1based in SBM_LABELS:
            info = SBM_LABELS[ic_1based]
            domains.append(info[0])
            coords.append(np.array([info[3], info[4], info[5]], dtype=float))
        else:
            # Compute center of mass from atlas volume
            vol = data[:, :, :, roi_idx]
            if vol.max() > 0:
                com_vox = center_of_mass(np.abs(vol))
                mni = affine @ np.array([*com_vox, 1.0])
                coords.append(mni[:3])
            else:
                coords.append(np.array([0.0, 0.0, 0.0]))
            domains.append("Other")

    return domains, np.array(coords)


def get_fnc_domain(ic_idx):
    """Return domain for a 0-indexed FNC IC."""
    for domain, (lo, hi) in FNC_DOMAIN_RANGES.items():
        if lo <= ic_idx < hi:
            return domain
    return "Other"


def parse_fnc_edges(fnc_names):
    """Parse FNC edge names to (ic_i, ic_j) pairs."""
    edges = []
    for name in fnc_names:
        parts = name.split("--")
        i = int(parts[0].replace("IC_", ""))
        j = int(parts[1].replace("IC_", ""))
        edges.append((i, j))
    return edges


# ---------------------------------------------------------------------------
# Figure 7: GM ROI Weight Map (glass brain)
# ---------------------------------------------------------------------------

def figure7_glass_brain(B, roi_indices, domains, coords):
    """Plot domain-colored ROI markers on glass brain views using plot_markers."""
    from nilearn import plotting

    w = np.linalg.norm(B, axis=1)
    w_norm = w / w.max()
    sizes = 15 + 250 * w_norm

    # Build numeric domain indices and colormap for plot_markers
    present_domains = [d for d in DOMAIN_ORDER_SBM if d in set(domains)]
    dom_to_idx = {d: i for i, d in enumerate(present_domains)}
    node_values = np.array([dom_to_idx[d] for d in domains], dtype=float)
    cmap = ListedColormap([DOMAIN_COLORS[d] for d in present_domains])
    n_dom = len(present_domains)

    fig = plt.figure(figsize=(7.2, 3.5))
    view_titles = [("l", "Left"), ("r", "Right"), ("z", "Dorsal")]

    for panel_idx, (view, title) in enumerate(view_titles):
        ax = fig.add_axes([panel_idx * 0.32 + 0.02, 0.18, 0.30, 0.72])
        plotting.plot_markers(
            node_values, coords,
            node_size=sizes, node_cmap=cmap,
            node_vmin=-0.5, node_vmax=n_dom - 0.5,
            display_mode=view, axes=ax,
            colorbar=False, annotate=False, alpha=0.9,
        )
        ax.set_title(title, fontsize=7.1, pad=4.0)

    # Legend
    handles = [mpatches.Patch(facecolor=DOMAIN_COLORS[d], label=DOMAIN_FULL_NAMES[d],
                              edgecolor="none") for d in present_domains]
    fig.legend(handles=handles, loc="lower center", ncol=min(5, len(present_domains)),
               fontsize=5.8, frameon=False, bbox_to_anchor=(0.5, 0.01),
               columnspacing=1.2)
    fig.suptitle("GM anatomical contribution map", x=0.07, y=0.98, ha="left", fontsize=8.0)

    return fig


# ---------------------------------------------------------------------------
# Figure 8: FNC Coupling Matrix
# ---------------------------------------------------------------------------

def figure8_fnc_matrix(B, fnc_names):
    """Plot domain-sorted FNC coupling strength matrix."""
    edges = parse_fnc_edges(fnc_names)
    n_ics = 53

    # Build 53x53 coupling matrix: sum |B[:, edge_idx]| for each FNC edge
    coupling = np.zeros((n_ics, n_ics))
    for edge_idx, (i, j) in enumerate(edges):
        strength = np.sum(np.abs(B[:, edge_idx]))
        coupling[i, j] = strength
        coupling[j, i] = strength

    # Sort ICs by domain
    ic_domains = [get_fnc_domain(k) for k in range(n_ics)]
    domain_order_map = {d: i for i, d in enumerate(DOMAIN_ORDER_FNC)}
    sort_key = [domain_order_map.get(d, 99) for d in ic_domains]
    sort_idx = np.argsort(sort_key, kind="stable")

    np.fill_diagonal(coupling, np.nan)
    coupling_sorted = coupling[np.ix_(sort_idx, sort_idx)]
    sorted_domains = [ic_domains[i] for i in sort_idx]

    # Find domain boundaries
    boundaries = []
    prev = sorted_domains[0]
    for k, d in enumerate(sorted_domains):
        if d != prev:
            boundaries.append(k)
            prev = d

    fig, ax = plt.subplots(figsize=(4.7, 4.1))

    style_axes(ax, all_spines=True, ygrid=False, xgrid=False)

    vmax = np.nanpercentile(coupling_sorted, 99)
    import copy
    cmap_matrix = copy.copy(CMAP_MAGNITUDE)
    cmap_matrix.set_bad("#FFFDFC")
    im = ax.imshow(coupling_sorted, cmap=cmap_matrix, aspect="equal",
                   vmin=0, vmax=vmax, interpolation="nearest")

    # Domain boundary lines
    for b in boundaries:
        ax.axhline(b - 0.5, color="white", linewidth=0.8, alpha=0.8)
        ax.axvline(b - 0.5, color="white", linewidth=0.8, alpha=0.8)

    # Domain tick labels at midpoints
    tick_positions = []
    tick_labels = []
    start = 0
    for b in boundaries + [n_ics]:
        mid = (start + b) / 2.0
        domain = sorted_domains[start]
        tick_positions.append(mid)
        tick_labels.append(domain)
        start = b

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=6)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=6)

    ax.set_xlabel("FNC component (by domain)")
    ax.set_ylabel("FNC component (by domain)")
    add_panel_title(ax, "Domain-sorted FNC coupling matrix")

    # Colorbar
    fig.subplots_adjust(right=0.85)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    style_colorbar(cbar, r"$\sum_i |B_{i,\mathrm{edge}}|$")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 9: Domain-to-Domain Chord Diagram
# ---------------------------------------------------------------------------

def _draw_chord(ax, theta1, theta2, color, alpha=0.3, lw=0):
    """Draw a quadratic Bezier chord between two arc positions."""
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    p1 = np.array([np.cos(theta1), np.sin(theta1)])
    p2 = np.array([np.cos(theta2), np.sin(theta2)])
    ctrl = 0.0 * (p1 + p2)  # control point at origin for a nice curve

    verts = [p1, ctrl, p2]
    codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
    path = MplPath(verts, codes)
    patch = PathPatch(path, facecolor="none", edgecolor=color,
                      alpha=alpha, linewidth=lw + 0.3)
    ax.add_patch(patch)


def figure9_chord(B, roi_indices, roi_domains, fnc_names):
    """Plot domain-to-domain coupling as a chord diagram.

    Aggregates B matrix contributions by GM domain -> FNC domain pairs.
    """
    edges = parse_fnc_edges(fnc_names)

    # Build domain-to-domain coupling matrix (GM domain x FNC domain)
    gm_domain_set = [d for d in DOMAIN_ORDER_SBM
                      if d in set(roi_domains) and d != "Other"]
    fnc_domain_set = DOMAIN_ORDER_FNC

    all_domains = gm_domain_set + [d for d in fnc_domain_set if d not in gm_domain_set]

    # Aggregate |B[roi, edge]| by (gm_domain, fnc_domain_of_edge)
    # For each edge, determine which FNC domain pair it connects
    # Then sum B contributions by GM domain
    n_gm_domains = len(gm_domain_set)
    n_fnc_domains = len(fnc_domain_set)

    gm_dom_idx = {d: i for i, d in enumerate(gm_domain_set)}
    fnc_dom_idx = {d: i for i, d in enumerate(fnc_domain_set)}

    coupling = np.zeros((n_gm_domains, n_fnc_domains))

    for roi_row, (roi_idx, dom) in enumerate(zip(roi_indices, roi_domains)):
        if dom not in gm_dom_idx:
            continue
        gi = gm_dom_idx[dom]
        for edge_idx, (ic_i, ic_j) in enumerate(edges):
            b_val = abs(B[roi_row, edge_idx])
            if b_val < 1e-10:
                continue
            # Attribute to both FNC domains of the edge
            dom_i = get_fnc_domain(ic_i)
            dom_j = get_fnc_domain(ic_j)
            if dom_i in fnc_dom_idx:
                coupling[gi, fnc_dom_idx[dom_i]] += b_val * 0.5
            if dom_j in fnc_dom_idx:
                coupling[gi, fnc_dom_idx[dom_j]] += b_val * 0.5

    # Normalize
    coupling_norm = coupling / coupling.max()

    # --- Draw chord diagram ---
    fig, ax = plt.subplots(figsize=(5.2, 5.2), subplot_kw={"aspect": "equal"})
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.axis("off")

    # Outer ring: GM domains (left half), FNC domains (right half)
    gm_total = coupling.sum(axis=1)
    fnc_total = coupling.sum(axis=0)

    # Assign arc sizes proportional to total coupling
    gm_sizes = gm_total / (gm_total.sum() + fnc_total.sum()) * 0.9
    fnc_sizes = fnc_total / (gm_total.sum() + fnc_total.sum()) * 0.9

    gap = 0.02  # gap between arcs
    total_gap = gap * (len(gm_domain_set) + len(fnc_domain_set))
    scale = (2 * np.pi - total_gap)

    gm_sizes_rad = gm_sizes * scale
    fnc_sizes_rad = fnc_sizes * scale

    # Place GM domains on left (pi/2 to 3pi/2), FNC on right
    theta_gm_starts = []
    theta_gm_mids = []
    theta = np.pi / 2 + gap / 2
    for i, sz in enumerate(gm_sizes_rad):
        theta_gm_starts.append(theta)
        theta_gm_mids.append(theta + sz / 2)
        theta += sz + gap

    theta_fnc_starts = []
    theta_fnc_mids = []
    # Start FNC from the right side, going counter-clockwise from bottom
    theta = np.pi / 2 - gap / 2
    for i, sz in enumerate(fnc_sizes_rad):
        theta -= sz
        theta_fnc_starts.append(theta)
        theta_fnc_mids.append(theta + sz / 2)
        theta -= gap

    # Draw outer arcs
    r_outer = 1.15
    r_inner = 1.05
    small_arc_thresh = 0.30  # radians; arcs smaller than this get pushed-out labels

    for i, dom in enumerate(gm_domain_set):
        start_deg = np.degrees(theta_gm_starts[i])
        end_deg = np.degrees(theta_gm_starts[i] + gm_sizes_rad[i])
        wedge = mpatches.Wedge((0, 0), r_outer, start_deg, end_deg,
                               width=r_outer - r_inner,
                               facecolor=DOMAIN_COLORS[dom], edgecolor="white",
                               linewidth=0.5)
        ax.add_patch(wedge)
        # Label — push farther out for small arcs
        mid = theta_gm_mids[i]
        is_small = gm_sizes_rad[i] < small_arc_thresh
        label_r = r_outer + (0.22 if is_small else 0.12)
        label_fs = 6.0 if is_small else 6.5
        lx, ly = label_r * np.cos(mid), label_r * np.sin(mid)
        ha = "right" if np.cos(mid) < 0 else "left"
        if abs(np.cos(mid)) < 0.3:
            ha = "center"
        ax.text(lx, ly, f"GM-{dom}", fontsize=label_fs, ha=ha, va="center",
                color=DOMAIN_COLORS[dom], fontweight="bold")

    for i, dom in enumerate(fnc_domain_set):
        start_deg = np.degrees(theta_fnc_starts[i])
        end_deg = np.degrees(theta_fnc_starts[i] + fnc_sizes_rad[i])
        wedge = mpatches.Wedge((0, 0), r_outer, start_deg, end_deg,
                               width=r_outer - r_inner,
                               facecolor=DOMAIN_COLORS[dom], edgecolor="white",
                               linewidth=0.5, alpha=0.7)
        ax.add_patch(wedge)
        mid = theta_fnc_mids[i]
        is_small = fnc_sizes_rad[i] < small_arc_thresh
        label_r = r_outer + (0.22 if is_small else 0.12)
        label_fs = 6.0 if is_small else 6.5
        lx, ly = label_r * np.cos(mid), label_r * np.sin(mid)
        ha = "right" if np.cos(mid) < 0 else "left"
        if abs(np.cos(mid)) < 0.3:
            ha = "center"
        ax.text(lx, ly, f"FNC-{dom}", fontsize=label_fs, ha=ha, va="center",
                color=DOMAIN_COLORS[dom], fontweight="bold")

    # Draw chords for top coupling pairs
    threshold = np.percentile(coupling, 72)
    for gi in range(n_gm_domains):
        for fi in range(n_fnc_domains):
            if coupling[gi, fi] < threshold:
                continue
            _draw_chord(
                ax,
                theta_gm_mids[gi],
                theta_fnc_mids[fi],
                color=DOMAIN_COLORS[gm_domain_set[gi]],
                alpha=min(0.62, 0.12 + 0.42 * coupling_norm[gi, fi]),
                lw=0.35 + 2.6 * coupling_norm[gi, fi],
            )

    ax.text(-1.42, 1.44, "GM domain to FNC domain flow", fontsize=7.8, color="#201C1A", fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    apply_nature_style()
    FIG_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    B, roi_indices, fnc_names = load_data()
    print(f"  B matrix: {B.shape}")
    print(f"  GM ROIs: {len(roi_indices)}, FNC edges: {len(fnc_names)}")

    print("Computing ROI domains and coordinates...")
    roi_domains, roi_coords = get_roi_domain_and_coords(roi_indices)
    domain_counts = {}
    for d in roi_domains:
        domain_counts[d] = domain_counts.get(d, 0) + 1
    print(f"  Domain distribution: {domain_counts}")

    # --- Figure 7 → fig2/panel_a ---
    print("Generating glass brain (fig2/panel_a)...")
    try:
        fig7 = figure7_glass_brain(B, roi_indices, roi_domains, roi_coords)
        save_panel(fig7, "fig2", "panel_a")
    except Exception as e:
        print(f"  Glass brain failed: {e}")
        import traceback
        traceback.print_exc()

    # --- Figure 8 → fig2/panel_b ---
    print("Generating FNC coupling matrix (fig2/panel_b)...")
    fig8 = figure8_fnc_matrix(B, fnc_names)
    save_panel(fig8, "fig2", "panel_b")

    # --- Figure 9 → fig2/panel_c ---
    print("Generating chord diagram (fig2/panel_c)...")
    fig9 = figure9_chord(B, roi_indices, roi_domains, fnc_names)
    save_panel(fig9, "fig2", "panel_c")

    print("Done!")


if __name__ == "__main__":
    main()
