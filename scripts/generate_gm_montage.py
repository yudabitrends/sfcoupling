#!/usr/bin/env python3
"""Generate GM ROI importance montage slice figure for sfcoupling paper.

Creates a composite brain map by weighting each SBM component volume
by its coupling importance (row-norm of B matrix), then displays as
axial slice montage with domain-colored ROI markers overlaid.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from nilearn import plotting, image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from figs.plot_style import apply_nature_style

# Paths
ATLAS_PATH = "/data/qneuromark/Network_templates/NeuroMark3/T1.nii"
B_MATRIX_PATH = PROJECT_ROOT / "results/ukb/multivariate_methods/decompositions/nuclear_norm_seed42_B.npy"
GM_NAMES_PATH = PROJECT_ROOT / "aligned_features/meta/feature_maps/gm_feature_names.txt"
FIG_DIR = PROJECT_ROOT / "figures"

# Domain mapping (from generate_brain_figures.py)
SBM_LABELS = {
    1: ("SC", "Putamen"), 3: ("SC", "Caudate"), 2: ("SC", "Thalamus"),
    68: ("HP", "Hippocampus"),
    31: ("AUD", "ITG"), 35: ("AUD", "MTG"), 39: ("AUD", "MTG/STG"),
    11: ("AUD", "TP"), 47: ("AUD", "MTG"), 18: ("AUD", "ITG/TP"),
    9: ("AUD", "L-TP"), 94: ("AUD", "ITG/MTG"),
    24: ("SM", "SMA"), 5: ("SM", "ParaCG"), 95: ("SM", "PreCG/PostCG"),
    8: ("SM", "RO"),
    27: ("VS", "LingG"), 64: ("VS", "MOG"), 4: ("VS", "Fusi"),
    29: ("VS", "CalG"), 85: ("VS", "MOG/SOG"), 84: ("VS", "Fusi"),
    15: ("VS", "IOG/MOG"), 52: ("VS", "SOG/Cuneus"), 34: ("VS", "LingG/IOG"),
    88: ("VS", "L-CalG"), 48: ("VS", "MOG/SOG"), 43: ("VS", "Fusi"),
    17: ("CC", "SMFG"), 42: ("CC", "SMOFG"), 13: ("CC", "IFG"),
    20: ("CC", "MFG"), 7: ("CC", "Insu/RO"), 38: ("CC", "Insu"),
    55: ("CC", "IOFG"), 49: ("CC", "MFG"), 69: ("CC", "OC"),
    62: ("CC", "SFG"), 14: ("CC", "MFG"), 10: ("CC", "IFG/MFG"),
    78: ("CC", "IFG/MFG"),
    37: ("PA", "SPL"), 72: ("PA", "IPL"), 51: ("PA", "SMG"),
    100: ("PA", "IPL/SMG"),
    36: ("DM", "ACC"), 25: ("DM", "PreCu/PCC"), 56: ("DM", "ACC/MCC"),
    57: ("DM", "PreCu"), 54: ("DM", "PCC"), 41: ("DM", "AG"),
    67: ("DM", "PreCu"), 16: ("DM", "PreCu"),
    22: ("CB", "CB"), 6: ("CB", "Vermis"), 91: ("CB", "L-CB Crus"),
    19: ("CB", "Vermis"), 87: ("CB", "CB"), 33: ("CB", "CB"),
    98: ("CB", "CB Crus"), 32: ("CB", "CB"), 77: ("CB", "CB"),
    53: ("CB", "R-CB Crus"), 74: ("CB", "CB Crus"), 59: ("CB", "L-CB Crus"),
    66: ("CB", "CB Crus"),
}

DOMAIN_COLORS = {
    "SC": "#7b2d8e", "HP": "#00897b", "AUD": "#e65100",
    "SM": "#1565c0", "VS": "#2e7d32", "CC": "#c62828",
    "PA": "#f9a825", "DM": "#d81b60", "CB": "#5d4037",
    "Other": "#9e9e9e",
}

DOMAIN_FULL_NAMES = {
    "SC": "Subcortical", "HP": "Hippocampal", "AUD": "Auditory",
    "SM": "Sensorimotor", "VS": "Visual", "CC": "Cognitive Control",
    "PA": "Parietal", "DM": "Default Mode", "CB": "Cerebellar",
    "Other": "Other",
}


def main():
    apply_nature_style()
    FIG_DIR.mkdir(exist_ok=True)

    # Load data
    B = np.load(B_MATRIX_PATH)
    gm_names = Path(GM_NAMES_PATH).read_text().strip().split("\n")
    roi_indices = [int(n.replace("roi_", "")) for n in gm_names]

    atlas_img = nib.load(ATLAS_PATH)
    atlas_data = atlas_img.get_fdata()

    # Importance weights per ROI
    w = np.linalg.norm(B, axis=1)
    w_norm = w / w.max()

    # Build composite importance map using max-weighted-projection:
    # For each voxel, keep the maximum w_i * |SBM_i(voxel)| across components.
    # This avoids the diffuse "everything lit up" problem of summing 99 maps.
    composite = np.zeros(atlas_data.shape[:3], dtype=np.float64)
    for i, roi_idx in enumerate(roi_indices):
        vol = np.abs(atlas_data[:, :, :, roi_idx])
        # Z-score within-component, keep only voxels > 2 std above mean
        pos = vol[vol > 0]
        if len(pos) == 0:
            continue
        mu, sigma = pos.mean(), pos.std()
        if sigma < 1e-10:
            continue
        z = (vol - mu) / sigma
        z[z < 2.0] = 0
        weighted = w_norm[i] * z
        composite = np.maximum(composite, weighted)

    # Normalize to [0, 1]
    composite /= composite.max()

    composite_img = nib.Nifti1Image(composite, atlas_img.affine)

    # ---- Panel A: Axial montage slices ----
    print("Generating GM montage (axial slices)...")
    fig_axial = plt.figure(figsize=(7.2, 4.0))

    cut_coords_z = [-45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65]

    display = plotting.plot_stat_map(
        composite_img,
        display_mode="z",
        cut_coords=cut_coords_z,
        colorbar=True,
        cmap="YlOrRd",
        threshold=0.10,
        black_bg=False,
        annotate=True,
        figure=fig_axial,
        title="GM ROI Importance (coupling weight)",
    )

    out_axial = FIG_DIR / "fig7b_gm_montage_axial"
    fig_axial.savefig(f"{out_axial}.png", dpi=420, bbox_inches="tight")
    fig_axial.savefig(f"{out_axial}.pdf", bbox_inches="tight")
    plt.close(fig_axial)
    print(f"  Saved {out_axial}.{{png,pdf}}")

    # ---- Panel B: Coronal montage slices ----
    print("Generating GM montage (coronal slices)...")
    fig_coronal = plt.figure(figsize=(7.2, 3.5))
    cut_coords_y = [-70, -55, -40, -25, -10, 5, 20, 35, 50]

    display = plotting.plot_stat_map(
        composite_img,
        display_mode="y",
        cut_coords=cut_coords_y,
        colorbar=True,
        cmap="YlOrRd",
        threshold=0.10,
        black_bg=False,
        annotate=True,
        figure=fig_coronal,
        title="GM ROI Importance (coupling weight)",
    )

    out_coronal = FIG_DIR / "fig7b_gm_montage_coronal"
    fig_coronal.savefig(f"{out_coronal}.png", dpi=420, bbox_inches="tight")
    fig_coronal.savefig(f"{out_coronal}.pdf", bbox_inches="tight")
    plt.close(fig_coronal)
    print(f"  Saved {out_coronal}.{{png,pdf}}")

    # ---- Panel C: Orthogonal view ----
    print("Generating GM montage (ortho view)...")
    fig_ortho = plt.figure(figsize=(7.2, 3.0))

    display = plotting.plot_stat_map(
        composite_img,
        display_mode="ortho",
        colorbar=True,
        cmap="YlOrRd",
        threshold=0.10,
        black_bg=False,
        annotate=True,
        figure=fig_ortho,
        title="GM ROI Importance (coupling weight)",
    )

    out_ortho = FIG_DIR / "fig7b_gm_montage_ortho"
    fig_ortho.savefig(f"{out_ortho}.png", dpi=420, bbox_inches="tight")
    fig_ortho.savefig(f"{out_ortho}.pdf", bbox_inches="tight")
    plt.close(fig_ortho)
    print(f"  Saved {out_ortho}.{{png,pdf}}")

    print("Done!")


if __name__ == "__main__":
    main()
