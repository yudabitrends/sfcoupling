#!/usr/bin/env python3
"""Deep Brain Region Analysis via SVD Mode Decomposition of the B matrix.

Analyses:
  1. SVD decomposition of seed42 Nuclear Norm B matrix
  2. Domain fingerprint per mode (Fig 10)
  3. Top ROIs per mode (table + CSV)
  4. Mode × Domain coupling heatmap (Fig 11)
  5. Cross-seed mode stability (Fig 12)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from figs.plot_style import apply_nature_style
from figs.utils import (
    METHOD_COLORS,
    METHOD_MARKERS,
    style_axes,
    style_colorbar,
    add_panel_title,
    CMAP_HEAT,
    CMAP_MAGNITUDE,
)
from matplotlib.colors import ListedColormap
from generate_brain_figures import (
    SBM_LABELS,
    FNC_DOMAIN_RANGES,
    DOMAIN_COLORS,
    DOMAIN_FULL_NAMES,
    DOMAIN_ORDER_SBM,
    DOMAIN_ORDER_FNC,
    load_data,
    parse_fnc_edges,
    get_fnc_domain,
    get_roi_domain_and_coords,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DECOMP_DIR = PROJECT_ROOT / "results/ukb/multivariate_methods/decompositions"
FIG_DIR = PROJECT_ROOT / "figures"


def save_panel(fig, fig_name: str, panel_name: str):
    """Save a single panel PDF into figures/{fig_name}/{panel_name}.pdf"""
    d = FIG_DIR / fig_name
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{panel_name}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fig_name}/{panel_name} saved")


SEEDS = list(range(42, 49))
N_MODES = 5  # top modes to analyze
N_TOP_ROIS = 10  # top ROIs per mode table


# ---------------------------------------------------------------------------
# 1. SVD decomposition
# ---------------------------------------------------------------------------

def svd_decompose(B, k=N_MODES):
    """Compute truncated SVD of B. Returns U[:,:k], S[:k], Vt[:k,:]."""
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    return U[:, :k], S[:k], Vt[:k, :]


# ---------------------------------------------------------------------------
# 2. Domain fingerprints
# ---------------------------------------------------------------------------

def domain_fingerprint(U, Vt, roi_domains, fnc_edges, k=3):
    """Aggregate |U[:,m]| by GM domain and |V[:,m]| by FNC domain for each mode.

    Returns:
        gm_fingerprints: dict[mode_idx] -> dict[domain] -> float (normalized)
        fnc_fingerprints: dict[mode_idx] -> dict[domain] -> float (normalized)
    """
    gm_fps = {}
    fnc_fps = {}

    for m in range(k):
        # --- GM side ---
        u_abs = np.abs(U[:, m])
        gm_agg = {}
        for dom in DOMAIN_ORDER_SBM:
            gm_agg[dom] = 0.0
        for roi_row, dom in enumerate(roi_domains):
            gm_agg[dom] = gm_agg.get(dom, 0.0) + u_abs[roi_row]
        total = sum(gm_agg.values())
        if total > 0:
            gm_agg = {d: v / total for d, v in gm_agg.items()}
        gm_fps[m] = gm_agg

        # --- FNC side: aggregate |V[m, edge]| by FNC domain with 0.5 split ---
        v_abs = np.abs(Vt[m, :])
        fnc_agg = {d: 0.0 for d in DOMAIN_ORDER_FNC}
        for edge_idx, (ic_i, ic_j) in enumerate(fnc_edges):
            dom_i = get_fnc_domain(ic_i)
            dom_j = get_fnc_domain(ic_j)
            val = v_abs[edge_idx]
            if dom_i in fnc_agg:
                fnc_agg[dom_i] += val * 0.5
            if dom_j in fnc_agg:
                fnc_agg[dom_j] += val * 0.5
        total = sum(fnc_agg.values())
        if total > 0:
            fnc_agg = {d: v / total for d, v in fnc_agg.items()}
        fnc_fps[m] = fnc_agg

    return gm_fps, fnc_fps


# ---------------------------------------------------------------------------
# 3. Top ROIs per mode
# ---------------------------------------------------------------------------

def top_rois_per_mode(U, roi_indices, roi_domains, roi_coords, k=3, n_top=N_TOP_ROIS):
    """Return DataFrame of top ROIs per mode ranked by |U[:,m]|."""
    rows = []
    for m in range(k):
        u_abs = np.abs(U[:, m])
        order = np.argsort(u_abs)[::-1]
        for rank, idx in enumerate(order[:n_top]):
            ic_1based = roi_indices[idx] + 1
            info = SBM_LABELS.get(ic_1based)
            label = info[2] if info else "unlabeled"
            abbrev = info[1] if info else f"ROI_{roi_indices[idx]}"
            rows.append({
                "mode": m + 1,
                "rank": rank + 1,
                "roi_idx": roi_indices[idx],
                "domain": roi_domains[idx],
                "abbreviation": abbrev,
                "label": label,
                "mni_x": roi_coords[idx, 0],
                "mni_y": roi_coords[idx, 1],
                "mni_z": roi_coords[idx, 2],
                "loading": u_abs[idx],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Mode × Domain coupling heatmap
# ---------------------------------------------------------------------------

def mode_domain_coupling(U, S, Vt, roi_domains, fnc_edges, k=N_MODES):
    """For each mode, compute (GM domain × FNC domain) coupling matrix.

    coupling[gm_dom, fnc_dom] = s_k * sum_{roi in gm_dom} |U[roi,k]|
                                     * sum_{edge touching fnc_dom} |V[edge,k]| * 0.5
    """
    gm_doms = [d for d in DOMAIN_ORDER_SBM if d != "Other"]
    fnc_doms = DOMAIN_ORDER_FNC
    gm_idx = {d: i for i, d in enumerate(gm_doms)}
    fnc_idx = {d: i for i, d in enumerate(fnc_doms)}

    coupling_matrices = []
    for m in range(k):
        u_abs = np.abs(U[:, m])
        v_abs = np.abs(Vt[m, :])

        # GM domain loadings
        gm_load = np.zeros(len(gm_doms))
        for roi_row, dom in enumerate(roi_domains):
            if dom in gm_idx:
                gm_load[gm_idx[dom]] += u_abs[roi_row]

        # FNC domain loadings (0.5 split)
        fnc_load = np.zeros(len(fnc_doms))
        for edge_idx, (ic_i, ic_j) in enumerate(fnc_edges):
            dom_i = get_fnc_domain(ic_i)
            dom_j = get_fnc_domain(ic_j)
            if dom_i in fnc_idx:
                fnc_load[fnc_idx[dom_i]] += v_abs[edge_idx] * 0.5
            if dom_j in fnc_idx:
                fnc_load[fnc_idx[dom_j]] += v_abs[edge_idx] * 0.5

        # Outer product scaled by singular value
        mat = S[m] * np.outer(gm_load, fnc_load)
        coupling_matrices.append(mat)

    return coupling_matrices, gm_doms, fnc_doms


# ---------------------------------------------------------------------------
# 5. Cross-method mode comparison
# ---------------------------------------------------------------------------

METHODS = ["nuclear_norm", "rrr", "pls"]
METHOD_LABELS = {"nuclear_norm": "Nuclear Norm", "rrr": "RRR", "pls": "PLS"}


def load_B(method="nuclear_norm", seed=42):
    """Load a single B matrix."""
    path = DECOMP_DIR / f"{method}_seed{seed}_B.npy"
    return np.load(path)


def align_modes_signflip(U_ref, U_other, k):
    """Align modes of U_other to U_ref via greedy permutation + sign-flip."""
    U_r = U_ref[:, :k]
    U_o = U_other[:, :k]

    # Correlation matrix between ref and other modes
    C = np.abs(U_r.T @ U_o)

    # Greedy assignment
    used = set()
    perm = np.zeros(k, dtype=int)
    for m in range(k):
        row = C[m, :].copy()
        row[list(used)] = -1
        best = np.argmax(row)
        perm[m] = best
        used.add(best)

    U_aligned = U_o[:, perm]
    for m in range(k):
        if np.dot(U_r[:, m], U_aligned[:, m]) < 0:
            U_aligned[:, m] *= -1

    return U_aligned


def cross_method_stability(k=N_MODES, seed=42):
    """Compare SVD modes across methods (NN vs RRR vs PLS) after sign-flip alignment.

    Returns:
        corr_matrix: (n_methods, n_methods, k) pairwise |correlation| per mode
        all_U: dict[method] -> U[:,:k] (aligned to nuclear_norm reference)
        all_S: dict[method] -> S[:k]
    """
    all_U = {}
    all_S = {}

    # Reference: nuclear norm
    B_ref = load_B("nuclear_norm", seed)
    U_ref, S_ref, _ = svd_decompose(B_ref, k=k)
    all_U["nuclear_norm"] = U_ref
    all_S["nuclear_norm"] = S_ref

    for method in METHODS:
        if method == "nuclear_norm":
            continue
        B_m = load_B(method, seed)
        U_m, S_m, _ = svd_decompose(B_m, k=k)
        U_aligned = align_modes_signflip(U_ref, U_m, k)
        all_U[method] = U_aligned
        all_S[method] = S_m

    n = len(METHODS)
    corr_matrix = np.zeros((n, n, k))
    for i, m1 in enumerate(METHODS):
        for j, m2 in enumerate(METHODS):
            for mode in range(k):
                corr_matrix[i, j, mode] = np.abs(np.corrcoef(
                    all_U[m1][:, mode], all_U[m2][:, mode]
                )[0, 1])

    return corr_matrix, all_U, all_S


# ===========================================================================
# Figures
# ===========================================================================

def figure10_domain_fingerprints(gm_fps, fnc_fps, S, S_all, k=3):
    """Separate panels: GM domain bars (panel_a), FNC domain bars (panel_b)."""
    gm_domains = [d for d in DOMAIN_ORDER_SBM if d != "Other"]
    fnc_domains = DOMAIN_ORDER_FNC
    total_var = np.sum(S_all**2)

    # Panel a: GM domain bars (1 × k)
    fig_a, axes_a = plt.subplots(1, k, figsize=(7.2, 1.8), constrained_layout=True)
    if k == 1:
        axes_a = [axes_a]
    for m in range(k):
        var_pct = S[m]**2 / total_var * 100
        ax = axes_a[m]
        raw_vals = [gm_fps[m].get(d, 0) for d in gm_domains]
        total_gm = sum(raw_vals)
        vals = [v / total_gm for v in raw_vals] if total_gm > 0 else raw_vals
        colors = [DOMAIN_COLORS.get(d, "#999999") for d in gm_domains]
        ax.bar(range(len(gm_domains)), vals, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_xticks(range(len(gm_domains)))
        ax.set_xticklabels(gm_domains, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Proportion" if m == 0 else "")
        ax.set_title(f"Mode {m+1} ({var_pct:.1f}%)", fontsize=7)
        ax.set_ylim(0, max(vals) * 1.15 if max(vals) > 0 else 0.3)
        style_axes(ax, ygrid=True, xgrid=False)
    axes_a[0].annotate("GM\ndomains", xy=(-0.45, 0.5), xycoords="axes fraction",
                        fontsize=7, fontweight="bold", ha="center", va="center",
                        rotation=90)
    save_panel(fig_a, "fig10", "panel_a")

    # Panel b: FNC domain bars (1 × k)
    fig_b, axes_b = plt.subplots(1, k, figsize=(7.2, 1.8), constrained_layout=True)
    if k == 1:
        axes_b = [axes_b]
    for m in range(k):
        var_pct = S[m]**2 / total_var * 100
        ax = axes_b[m]
        vals = [fnc_fps[m].get(d, 0) for d in fnc_domains]
        colors = [DOMAIN_COLORS.get(d, "#999999") for d in fnc_domains]
        ax.bar(range(len(fnc_domains)), vals, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_xticks(range(len(fnc_domains)))
        ax.set_xticklabels(fnc_domains, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Proportion" if m == 0 else "")
        ax.set_ylim(0, max(vals) * 1.15 if max(vals) > 0 else 0.3)
        style_axes(ax, ygrid=True, xgrid=False)
    axes_b[0].annotate("FNC\ndomains", xy=(-0.45, 0.5), xycoords="axes fraction",
                        fontsize=7, fontweight="bold", ha="center", va="center",
                        rotation=90)
    save_panel(fig_b, "fig10", "panel_b")


def figure11_mode_domain_heatmap(coupling_matrices, gm_doms, fnc_doms, S, S_all, k=N_MODES):
    """Per-mode heatmap of (GM domain x FNC domain) coupling, saved as separate panels."""
    total_var = np.sum(S_all**2)
    vmax = max(mat.max() for mat in coupling_matrices[:k])

    for m in range(k):
        fig_m, ax = plt.subplots(figsize=(2.4, 2.8), constrained_layout=True)
        mat = coupling_matrices[m]
        var_pct = S[m]**2 / total_var * 100

        im = ax.imshow(mat, cmap=CMAP_HEAT, aspect="auto", vmin=0, vmax=vmax,
                       interpolation="nearest")

        ax.set_xticks(range(len(fnc_doms)))
        ax.set_xticklabels(fnc_doms, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(len(gm_doms)))
        ax.set_yticklabels(gm_doms, fontsize=6)
        ax.set_title(f"Mode {m+1} ({var_pct:.1f}%)", fontsize=7)
        ax.set_xlabel("FNC domain")
        ax.set_ylabel("GM domain")
        style_axes(ax, all_spines=True, ygrid=False, xgrid=False)

        cbar = fig_m.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.9)
        style_colorbar(cbar, "Coupling strength")

        save_panel(fig_m, "fig11", f"panel_mode{m+1}")


def figure12_method_stability(corr_matrix, all_S, k=N_MODES):
    """Separate panels: cross-method correlations (panel_a) and SV spectrum (panel_b)."""
    x = np.arange(k)
    width = 0.35
    nn_rrr = corr_matrix[0, 1, :k]
    nn_pls = corr_matrix[0, 2, :k]

    # Panel a: cross-method mode correlations
    fig_a, ax1 = plt.subplots(figsize=(4.5, 2.8), constrained_layout=True)
    ax1.bar(x - width/2, nn_rrr, width, label="NN vs RRR", color=METHOD_COLORS["Nuclear Norm"],
            edgecolor="white", linewidth=0.5)
    ax1.bar(x + width/2, nn_pls, width, label="NN vs PLS", color=METHOD_COLORS["PLS"],
            edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Mode {m+1}" for m in range(k)], fontsize=6)
    ax1.set_ylabel("GM loading |correlation|")
    ax1.set_ylim(0, 1.1)
    ax1.axhline(0.8, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax1.legend(fontsize=5.5, loc="lower right")
    style_axes(ax1, ygrid=True, xgrid=False)
    ax1.set_title("Cross-method mode agreement", fontsize=8)
    for i in range(k):
        ax1.text(i - width/2, nn_rrr[i] + 0.02, f"{nn_rrr[i]:.2f}",
                 ha="center", fontsize=5.5)
        ax1.text(i + width/2, nn_pls[i] + 0.02, f"{nn_pls[i]:.2f}",
                 ha="center", fontsize=5.5)
    save_panel(fig_a, "figS1", "panel_a")

    # Panel b: SV spectrum comparison
    fig_b, ax2 = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)
    for method in METHODS:
        label = METHOD_LABELS[method]
        color = METHOD_COLORS.get(label, "gray")
        marker = METHOD_MARKERS.get(label, "o")
        ax2.plot(range(1, k+1), all_S[method], marker=marker, linestyle="-",
                 markersize=4.5, color=color, label=label, linewidth=1.3, zorder=3)
    ax2.set_xlabel("Mode")
    ax2.set_ylabel("Singular value")
    ax2.set_xticks(range(1, k+1))
    ax2.legend(fontsize=5.5)
    style_axes(ax2, ygrid=True, xgrid=False)
    ax2.set_title("Singular value spectrum", fontsize=8)
    save_panel(fig_b, "figS1", "panel_b")


# ---------------------------------------------------------------------------
# Fig 13: Glass brain per SVD mode
# ---------------------------------------------------------------------------

def figure13_glass_brain_per_mode(U, S, S_all, roi_indices, roi_domains, roi_coords, k=3):
    """Per-mode glass brain with 3 views, saved as separate panels."""
    from nilearn import plotting

    total_var = np.sum(S_all**2)
    present_domains = [d for d in DOMAIN_ORDER_SBM if d in set(roi_domains)]
    dom_to_idx = {d: i for i, d in enumerate(present_domains)}
    node_values = np.array([dom_to_idx[d] for d in roi_domains], dtype=float)
    cmap = ListedColormap([DOMAIN_COLORS[d] for d in present_domains])
    n_dom = len(present_domains)

    views = [("l", "Left"), ("r", "Right"), ("z", "Dorsal")]
    handles = [mpatches.Patch(facecolor=DOMAIN_COLORS[d], label=DOMAIN_FULL_NAMES[d],
                              edgecolor="none") for d in present_domains]

    for m in range(k):
        w = np.abs(U[:, m])
        w_norm = w / w.max() if w.max() > 0 else w
        sizes = 15 + 250 * w_norm
        var_pct = S[m]**2 / total_var * 100

        fig_m = plt.figure(figsize=(7.2, 2.8))
        for col, (view, vtitle) in enumerate(views):
            ax = fig_m.add_axes([col * 0.32 + 0.02, 0.15, 0.30, 0.75])
            plotting.plot_markers(
                node_values, roi_coords,
                node_size=sizes, node_cmap=cmap,
                node_vmin=-0.5, node_vmax=n_dom - 0.5,
                display_mode=view, axes=ax,
                colorbar=False, annotate=False, alpha=0.9,
            )
            ax.set_title(vtitle, fontsize=7, pad=2)
            if col == 0:
                ax.annotate(f"Mode {m+1}\n({var_pct:.1f}%)",
                            xy=(-0.15, 0.5), xycoords="axes fraction",
                            fontsize=6, fontweight="bold", ha="right", va="center",
                            rotation=90)

        fig_m.legend(handles=handles, loc="lower center", ncol=5,
                     fontsize=6, frameon=False, bbox_to_anchor=(0.5, 0.01))
        save_panel(fig_m, "fig13", f"panel_mode{m+1}")


# ---------------------------------------------------------------------------
# Fig 14: FNC coupling matrix per SVD mode
# ---------------------------------------------------------------------------

def figure14_fnc_matrix_per_mode(Vt, fnc_names, S, S_all, k=3):
    """Per-mode 53x53 FNC coupling matrix, saved as separate panels."""
    import copy

    edges = parse_fnc_edges(fnc_names)
    n_ics = 53
    total_var = np.sum(S_all**2)

    # Domain-sort ICs (same logic as figure7)
    ic_domains = [get_fnc_domain(ic) for ic in range(n_ics)]
    domain_order_map = {d: i for i, d in enumerate(DOMAIN_ORDER_FNC)}
    sort_key = [domain_order_map.get(d, 99) for d in ic_domains]
    sort_idx = np.argsort(sort_key, kind="stable")
    sorted_domains = [ic_domains[i] for i in sort_idx]

    # Domain boundaries
    boundaries = []
    prev = sorted_domains[0]
    for idx, d in enumerate(sorted_domains):
        if d != prev:
            boundaries.append(idx)
            prev = d

    # Domain tick labels at midpoints
    tick_positions = []
    tick_labels = []
    start = 0
    for b in boundaries + [n_ics]:
        mid = (start + b) / 2.0
        tick_positions.append(mid)
        tick_labels.append(sorted_domains[start])
        start = b

    # Build per-mode matrices
    matrices = []
    for m in range(k):
        mat = np.zeros((n_ics, n_ics))
        v_abs = np.abs(Vt[m, :])
        for edge_idx, (ic_i, ic_j) in enumerate(edges):
            mat[ic_i, ic_j] = v_abs[edge_idx]
            mat[ic_j, ic_i] = v_abs[edge_idx]
        np.fill_diagonal(mat, np.nan)
        matrices.append(mat[np.ix_(sort_idx, sort_idx)])

    # Shared vmax across modes
    vmax = max(np.nanmax(m) for m in matrices)

    cmap_matrix = copy.copy(CMAP_MAGNITUDE)
    cmap_matrix.set_bad("#FFFDFC")

    for m in range(k):
        fig_m, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
        var_pct = S[m]**2 / total_var * 100

        for spine in ax.spines.values():
            spine.set_visible(True)

        im = ax.imshow(matrices[m], cmap=cmap_matrix, aspect="equal",
                       vmin=0, vmax=vmax, interpolation="nearest")

        for b in boundaries:
            ax.axhline(b - 0.5, color="white", linewidth=0.8, alpha=0.8)
            ax.axvline(b - 0.5, color="white", linewidth=0.8, alpha=0.8)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=5)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=5)
        ax.set_title(f"Mode {m+1} ({var_pct:.1f}%)", fontsize=7)

        cbar = fig_m.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.9)
        style_colorbar(cbar, "|V| loading")

        save_panel(fig_m, "fig14", f"panel_mode{m+1}")


# ---------------------------------------------------------------------------
# Fig 15: Combined glass brain + FNC matrix per mode
# ---------------------------------------------------------------------------

def figure15_combined_mode_map(U, Vt, S, S_all, roi_indices, roi_domains, roi_coords,
                                fnc_names, k=3):
    """Per-mode glass brain + FNC matrix side by side, saved as separate panels."""
    from nilearn import plotting
    import copy

    edges = parse_fnc_edges(fnc_names)
    n_ics = 53
    total_var = np.sum(S_all**2)

    # Precompute domain coloring for glass brain
    present_domains = [d for d in DOMAIN_ORDER_SBM if d in set(roi_domains)]
    dom_to_idx = {d: i for i, d in enumerate(present_domains)}
    node_values = np.array([dom_to_idx[d] for d in roi_domains], dtype=float)
    cmap_brain = ListedColormap([DOMAIN_COLORS[d] for d in present_domains])
    n_dom = len(present_domains)

    # Precompute FNC sorting
    ic_domains = [get_fnc_domain(ic) for ic in range(n_ics)]
    domain_order_map = {d: i for i, d in enumerate(DOMAIN_ORDER_FNC)}
    sort_key = [domain_order_map.get(d, 99) for d in ic_domains]
    sort_idx = np.argsort(sort_key, kind="stable")
    sorted_domains = [ic_domains[i] for i in sort_idx]

    boundaries = []
    prev = sorted_domains[0]
    for idx, d in enumerate(sorted_domains):
        if d != prev:
            boundaries.append(idx)
            prev = d

    tick_positions, tick_labels = [], []
    start = 0
    for b in boundaries + [n_ics]:
        tick_positions.append((start + b) / 2.0)
        tick_labels.append(sorted_domains[start])
        start = b

    cmap_matrix = copy.copy(CMAP_MAGNITUDE)
    cmap_matrix.set_bad("#FFFDFC")

    # Pre-compute all FNC matrices to find shared vmax
    all_mat_sorted = []
    for m in range(k):
        v_abs = np.abs(Vt[m, :])
        mat = np.zeros((n_ics, n_ics))
        for edge_idx, (ic_i, ic_j) in enumerate(edges):
            mat[ic_i, ic_j] = v_abs[edge_idx]
            mat[ic_j, ic_i] = v_abs[edge_idx]
        np.fill_diagonal(mat, np.nan)
        all_mat_sorted.append(mat[np.ix_(sort_idx, sort_idx)])
    shared_vmax = max(np.nanmax(ms) for ms in all_mat_sorted)

    handles = [mpatches.Patch(facecolor=DOMAIN_COLORS[d], label=DOMAIN_FULL_NAMES[d],
                              edgecolor="none") for d in present_domains]

    for m in range(k):
        var_pct = S[m]**2 / total_var * 100
        w = np.abs(U[:, m])
        w_norm = w / w.max() if w.max() > 0 else w
        sizes = 15 + 250 * w_norm

        fig_m = plt.figure(figsize=(7.2, 3.0))

        # Left: dorsal glass brain
        ax_brain = fig_m.add_axes([0.02, 0.12, 0.42, 0.78])
        plotting.plot_markers(
            node_values, roi_coords,
            node_size=sizes, node_cmap=cmap_brain,
            node_vmin=-0.5, node_vmax=n_dom - 0.5,
            display_mode="z", axes=ax_brain,
            colorbar=False, annotate=False, alpha=0.9,
        )
        ax_brain.set_title(f"Mode {m+1} ({var_pct:.1f}%) — GM loadings", fontsize=7, pad=2)

        # Right: FNC matrix
        ax_fnc = fig_m.add_axes([0.52, 0.12, 0.38, 0.78])
        mat_sorted = all_mat_sorted[m]

        for spine in ax_fnc.spines.values():
            spine.set_visible(True)

        im = ax_fnc.imshow(mat_sorted, cmap=cmap_matrix, aspect="equal",
                           vmin=0, vmax=shared_vmax, interpolation="nearest")
        for b in boundaries:
            ax_fnc.axhline(b - 0.5, color="white", linewidth=0.8, alpha=0.8)
            ax_fnc.axvline(b - 0.5, color="white", linewidth=0.8, alpha=0.8)
        ax_fnc.set_xticks(tick_positions)
        ax_fnc.set_xticklabels(tick_labels, fontsize=5)
        ax_fnc.set_yticks(tick_positions)
        ax_fnc.set_yticklabels(tick_labels, fontsize=5)
        ax_fnc.set_title("FNC loadings", fontsize=7, pad=2)

        cbar = fig_m.colorbar(im, ax=ax_fnc, fraction=0.046, pad=0.04, shrink=0.8)
        style_colorbar(cbar)

        fig_m.legend(handles=handles, loc="lower center", ncol=5,
                     fontsize=5.5, frameon=False, bbox_to_anchor=(0.5, 0.0))

        save_panel(fig_m, "fig7", f"panel_mode{m+1}")


# ---------------------------------------------------------------------------
# Fig: Mode Detail Composite (NEW — supplementary)
# ---------------------------------------------------------------------------

def fig_mode_detail_composite(coupling_matrices, gm_doms, fnc_doms, S, S_all,
                               Vt, fnc_names, k_heatmap=N_MODES, k_fnc=3):
    """Supplementary: heatmaps (panel_a) and FNC matrices (panel_b) as separate panels."""
    import copy

    total_var = np.sum(S_all**2)

    # ── Panel a: mode × domain coupling heatmaps ──
    vmax_heat = max(mat.max() for mat in coupling_matrices[:k_heatmap])
    fig_a = plt.figure(figsize=(7.2, 2.8))
    for m in range(k_heatmap):
        ax = fig_a.add_axes([m * 0.18 + 0.05, 0.15, 0.15, 0.75])
        mat = coupling_matrices[m]
        var_pct = S[m]**2 / total_var * 100
        im_h = ax.imshow(mat, cmap=CMAP_HEAT, aspect="auto", vmin=0, vmax=vmax_heat,
                         interpolation="nearest")
        ax.set_xticks(range(len(fnc_doms)))
        ax.set_xticklabels(fnc_doms, rotation=45, ha="right", fontsize=4)
        ax.set_yticks(range(len(gm_doms)))
        ax.set_yticklabels(gm_doms if m == 0 else [], fontsize=4)
        ax.set_title(f"Mode {m+1}\n({var_pct:.1f}%)", fontsize=5.5)
        style_axes(ax, all_spines=True, ygrid=False, xgrid=False)

    cax_top = fig_a.add_axes([0.95, 0.15, 0.015, 0.75])
    cbar_top = fig_a.colorbar(im_h, cax=cax_top)
    style_colorbar(cbar_top, "Coupling strength")
    save_panel(fig_a, "figS5", "panel_a")

    # ── Panel b: FNC matrices per mode ──
    edges = parse_fnc_edges(fnc_names)
    n_ics = 53
    ic_domains = [get_fnc_domain(ic) for ic in range(n_ics)]
    domain_order_map = {d: i for i, d in enumerate(DOMAIN_ORDER_FNC)}
    sort_key = [domain_order_map.get(d, 99) for d in ic_domains]
    sort_idx = np.argsort(sort_key, kind="stable")
    sorted_domains = [ic_domains[i] for i in sort_idx]

    boundaries = []
    prev = sorted_domains[0]
    for idx, d in enumerate(sorted_domains):
        if d != prev:
            boundaries.append(idx)
            prev = d
    tick_positions, tick_labels_fnc = [], []
    start = 0
    for b in boundaries + [n_ics]:
        tick_positions.append((start + b) / 2.0)
        tick_labels_fnc.append(sorted_domains[start])
        start = b

    cmap_matrix = copy.copy(CMAP_MAGNITUDE)
    cmap_matrix.set_bad("#FFFDFC")

    matrices = []
    for m in range(k_fnc):
        mat = np.zeros((n_ics, n_ics))
        v_abs = np.abs(Vt[m, :])
        for edge_idx, (ic_i, ic_j) in enumerate(edges):
            mat[ic_i, ic_j] = v_abs[edge_idx]
            mat[ic_j, ic_i] = v_abs[edge_idx]
        np.fill_diagonal(mat, np.nan)
        matrices.append(mat[np.ix_(sort_idx, sort_idx)])
    vmax_fnc = max(np.nanmax(m) for m in matrices)

    fig_b = plt.figure(figsize=(7.2, 2.8))
    panel_w = 0.27
    for m in range(k_fnc):
        ax = fig_b.add_axes([m * (panel_w + 0.03) + 0.07, 0.15, panel_w, 0.75])
        var_pct = S[m]**2 / total_var * 100
        style_axes(ax, all_spines=True, ygrid=False, xgrid=False)
        im_f = ax.imshow(matrices[m], cmap=cmap_matrix, aspect="equal",
                         vmin=0, vmax=vmax_fnc, interpolation="nearest")
        for b in boundaries:
            ax.axhline(b - 0.5, color="white", linewidth=0.4, alpha=0.8)
            ax.axvline(b - 0.5, color="white", linewidth=0.4, alpha=0.8)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels_fnc, fontsize=4)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels_fnc if m == 0 else [], fontsize=4)
        ax.set_title(f"Mode {m+1} ({var_pct:.1f}%)", fontsize=5.5)

    cax_bot = fig_b.add_axes([0.95, 0.15, 0.015, 0.75])
    cbar_bot = fig_b.colorbar(im_f, cax=cax_bot)
    style_colorbar(cbar_bot, "|V| loading")
    save_panel(fig_b, "figS5", "panel_b")


# ---------------------------------------------------------------------------
# 6. Per-mode text summary
# ---------------------------------------------------------------------------

def print_mode_mapping_summary(U, Vt, S, S_all, roi_indices, roi_domains, fnc_edges, k=3):
    """Print interpretive per-mode mapping summary to stdout."""
    total_var = np.sum(S_all**2)
    n_ics = 53

    for m in range(k):
        var_pct = S[m]**2 / total_var * 100
        print(f"\nMode {m+1} ({var_pct:.1f}% var):")

        # Top GM regions
        u_abs = np.abs(U[:, m])
        order = np.argsort(u_abs)[::-1]
        gm_parts = []
        for idx in order[:5]:
            ic_1based = roi_indices[idx] + 1
            info = SBM_LABELS.get(ic_1based)
            abbrev = info[1] if info else f"ROI_{roi_indices[idx]}"
            dom = roi_domains[idx]
            gm_parts.append(f"{abbrev}({dom})")
        print(f"  GM -> {', '.join(gm_parts)}")

        # Top FNC edges: build IC hub counts
        v_abs = np.abs(Vt[m, :])
        top_edges = np.argsort(v_abs)[::-1][:10]
        ic_hub_score = np.zeros(n_ics)
        for edge_idx in top_edges:
            ic_i, ic_j = fnc_edges[edge_idx]
            ic_hub_score[ic_i] += v_abs[edge_idx]
            ic_hub_score[ic_j] += v_abs[edge_idx]
        top_ics = np.argsort(ic_hub_score)[::-1][:3]

        fnc_parts = []
        for ic in top_ics:
            dom = get_fnc_domain(ic)
            # Find top connections for this IC
            ic_edges = []
            for edge_idx in top_edges:
                ic_i, ic_j = fnc_edges[edge_idx]
                if ic_i == ic or ic_j == ic:
                    partner = ic_j if ic_i == ic else ic_i
                    ic_edges.append(f"IC_{ic}-IC_{partner}({get_fnc_domain(partner)})")
            edge_str = ", ".join(ic_edges[:2])
            fnc_parts.append(f"IC_{ic}({dom}) hub: {edge_str}")
        print(f"  FNC -> {'; '.join(fnc_parts)}")

        # Interpretation: dominant GM domain -> dominant FNC domain
        gm_dom_load = {}
        for roi_row, dom in enumerate(roi_domains):
            gm_dom_load[dom] = gm_dom_load.get(dom, 0.0) + u_abs[roi_row]
        top_gm_dom = max(gm_dom_load, key=gm_dom_load.get)

        fnc_dom_load = {d: 0.0 for d in DOMAIN_ORDER_FNC}
        for edge_idx, (ic_i, ic_j) in enumerate(fnc_edges):
            val = v_abs[edge_idx]
            dom_i = get_fnc_domain(ic_i)
            dom_j = get_fnc_domain(ic_j)
            if dom_i in fnc_dom_load:
                fnc_dom_load[dom_i] += val * 0.5
            if dom_j in fnc_dom_load:
                fnc_dom_load[dom_j] += val * 0.5
        top_fnc_dom = max(fnc_dom_load, key=fnc_dom_load.get)
        print(f"  Interpretation: {DOMAIN_FULL_NAMES.get(top_gm_dom, top_gm_dom)} GM structure "
              f"-> {DOMAIN_FULL_NAMES.get(top_fnc_dom, top_fnc_dom)} functional connectivity")


# ===========================================================================
# Main
# ===========================================================================

def main():
    apply_nature_style()
    FIG_DIR.mkdir(exist_ok=True)

    # --- Load data ---
    print("Loading data...")
    B, roi_indices, fnc_names = load_data()
    fnc_edges = parse_fnc_edges(fnc_names)
    print(f"  B matrix: {B.shape}")

    print("Computing ROI domains and coordinates...")
    roi_domains, roi_coords = get_roi_domain_and_coords(roi_indices)

    # --- Analysis 1: SVD ---
    print("\n=== SVD Decomposition ===")
    S_all = np.linalg.svd(B, compute_uv=False)
    U, S, Vt = svd_decompose(B, k=N_MODES)
    total_var = np.sum(S_all**2)
    print(f"  Top {N_MODES} singular values: {[f'{s:.1f}' for s in S]}")
    print(f"  Variance explained: {[f'{s**2/total_var*100:.1f}%' for s in S]}")
    print(f"  Cumulative top-{N_MODES}: {np.sum(S**2)/total_var*100:.1f}%")

    # --- Analysis 2: Domain fingerprints ---
    print("\n=== Domain Fingerprints ===")
    gm_fps, fnc_fps = domain_fingerprint(U, Vt, roi_domains, fnc_edges, k=3)
    for m in range(3):
        top_gm = sorted(gm_fps[m].items(), key=lambda x: -x[1])[:3]
        top_fnc = sorted(fnc_fps[m].items(), key=lambda x: -x[1])[:3]
        print(f"  Mode {m+1}: GM top = {[(d, f'{v:.2f}') for d,v in top_gm]}, "
              f"FNC top = {[(d, f'{v:.2f}') for d,v in top_fnc]}")

    # --- Analysis 3: Top ROIs per mode ---
    print("\n=== Top ROIs per Mode ===")
    df_rois = top_rois_per_mode(U, roi_indices, roi_domains, roi_coords, k=3)
    for m in range(3):
        print(f"\n  Mode {m+1} (SV = {S[m]:.1f}):")
        sub = df_rois[df_rois["mode"] == m + 1].head(N_TOP_ROIS)
        for _, row in sub.iterrows():
            print(f"    {row['rank']:2d}. {row['label']:45s} [{row['domain']:5s}] "
                  f"MNI=({row['mni_x']:+5.0f},{row['mni_y']:+5.0f},{row['mni_z']:+5.0f})  "
                  f"|u|={row['loading']:.4f}")
    csv_path = FIG_DIR / "svd_top_rois.csv"
    df_rois.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n  Saved: {csv_path}")

    # --- Analysis 4: Mode × Domain coupling ---
    print("\n=== Mode × Domain Coupling ===")
    coupling_matrices, gm_doms, fnc_doms = mode_domain_coupling(
        U, S, Vt, roi_domains, fnc_edges, k=N_MODES)
    for m in range(3):
        mat = coupling_matrices[m]
        peak_gm = gm_doms[np.argmax(mat.max(axis=1))]
        peak_fnc = fnc_doms[np.argmax(mat.max(axis=0))]
        print(f"  Mode {m+1}: peak coupling = {peak_gm} GM <-> {peak_fnc} FNC")

    # --- Analysis 5: Cross-method stability ---
    print("\n=== Cross-Method Mode Stability ===")
    print("  (Note: B matrices are identical across seeds — convex NN has unique solution)")
    corr_matrix, all_U_methods, all_S_methods = cross_method_stability(k=N_MODES)
    for m in range(N_MODES):
        nn_rrr = corr_matrix[0, 1, m]
        nn_pls = corr_matrix[0, 2, m]
        print(f"  Mode {m+1}: NN-RRR r={nn_rrr:.3f}, NN-PLS r={nn_pls:.3f}")

    # --- Per-mode mapping summary ---
    print("\n=== Per-Mode Mapping Summary ===")
    print_mode_mapping_summary(U, Vt, S, S_all, roi_indices, roi_domains, fnc_edges, k=3)

    # --- Generate figures ---
    print("\n=== Generating Figures ===")

    # Fig 10: Domain fingerprints
    figure10_domain_fingerprints(gm_fps, fnc_fps, S, S_all, k=3)

    # Fig 11: Mode-domain coupling heatmaps
    figure11_mode_domain_heatmap(coupling_matrices, gm_doms, fnc_doms, S, S_all, k=N_MODES)

    # Fig 12 -> figS1: Cross-method stability
    figure12_method_stability(corr_matrix, all_S_methods, k=N_MODES)

    # Fig 13: Glass brain per SVD mode
    print("  Generating fig13 (glass brain per mode)...")
    figure13_glass_brain_per_mode(U, S, S_all, roi_indices, roi_domains, roi_coords, k=3)

    # Fig 14: FNC matrix per SVD mode
    print("  Generating fig14 (FNC matrix per mode)...")
    figure14_fnc_matrix_per_mode(Vt, fnc_names, S, S_all, k=3)

    # Fig 15 -> fig7: Combined mode maps
    print("  Generating fig7 (combined mode maps)...")
    figure15_combined_mode_map(
        U, Vt, S, S_all, roi_indices, roi_domains, roi_coords, fnc_names, k=3)

    # Fig mode detail -> figS5: heatmaps + FNC matrices (supplementary)
    print("  Generating figS5 (supplementary composite)...")
    fig_mode_detail_composite(
        coupling_matrices, gm_doms, fnc_doms, S, S_all, Vt, fnc_names,
        k_heatmap=N_MODES, k_fnc=3)

    print("\nDone!")


if __name__ == "__main__":
    main()
