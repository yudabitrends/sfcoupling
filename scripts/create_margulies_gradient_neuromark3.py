#!/usr/bin/env python3
"""
Map Margulies et al. (2016) principal gradient to NeuroMark3 99 ROIs.

Downloads the principal gradient from NeuroVault (if not cached), loads the
NeuroMark3 atlas, and computes weighted-average gradient per ROI.

Output: results/margulies_gradient_neuromark3.csv  (99 comma-separated values)
"""
import argparse
import logging
import os
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ATLAS_PATH = Path("/data/qneuromark/Network_templates/NeuroMark3/T1.nii")
LABELS_PATH = Path("/data/qneuromark/Network_templates/NeuroMark3/T1.txt")
GRADIENT_CACHE = Path(__file__).resolve().parent.parent / "results" / "margulies_gradient_mni.nii.gz"
OUT_PATH = Path(__file__).resolve().parent.parent / "results" / "margulies_gradient_neuromark3.csv"

# NeuroVault image ID for Margulies 2016 principal gradient (fsaverage → MNI volume)
NEUROVAULT_URL = "https://neurovault.org/media/images/10426/gradient_1.nii.gz"


def download_gradient(cache_path: Path) -> None:
    """Download principal gradient NIfTI from NeuroVault."""
    import urllib.request
    logger.info("Downloading Margulies gradient from NeuroVault → %s", cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(NEUROVAULT_URL, str(cache_path))
    logger.info("Download complete")


def load_atlas(atlas_path: Path, labels_path: Path):
    """Load NeuroMark3 atlas and return 4D array + ROI label ordering."""
    import nibabel as nib
    atlas_img = nib.load(str(atlas_path))
    atlas_data = atlas_img.get_fdata()  # (x, y, z, n_components)
    labels = np.loadtxt(str(labels_path), dtype=int)
    logger.info("Atlas shape: %s, %d ROI labels", atlas_data.shape, len(labels))
    return atlas_img, atlas_data, labels


def map_gradient_to_rois(gradient_img, atlas_img, atlas_data, labels) -> np.ndarray:
    """Compute weighted-average gradient per NeuroMark3 ROI."""
    from nilearn.image import resample_to_img
    import nibabel as nib

    # Resample gradient to atlas space
    gradient_resampled = resample_to_img(
        gradient_img, atlas_img, interpolation="continuous"
    )
    grad_data = gradient_resampled.get_fdata().squeeze()
    logger.info("Gradient resampled to atlas space: %s", grad_data.shape)

    n_rois = len(labels)
    roi_gradients = np.zeros(n_rois)

    for i, label_idx in enumerate(labels):
        # atlas_data[:,:,:,label_idx-1] gives the probability map for this IC
        # (labels are 1-indexed in T1.txt)
        if label_idx < 1 or label_idx > atlas_data.shape[3]:
            logger.warning("Label %d out of range, skipping ROI %d", label_idx, i)
            continue

        prob_map = atlas_data[:, :, :, label_idx - 1]
        mask = prob_map > 0.1  # threshold probability map

        if mask.sum() == 0:
            logger.warning("ROI %d (label %d) has no voxels above threshold", i, label_idx)
            continue

        weights = prob_map[mask]
        grad_vals = grad_data[mask]

        # Weighted average
        roi_gradients[i] = np.average(grad_vals, weights=weights)

    logger.info("ROI gradients: min=%.4f, max=%.4f, mean=%.4f",
                roi_gradients.min(), roi_gradients.max(), roi_gradients.mean())
    return roi_gradients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, default=ATLAS_PATH)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--gradient-cache", type=Path, default=GRADIENT_CACHE)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    import nibabel as nib

    # Download gradient if not cached
    if not args.gradient_cache.exists():
        download_gradient(args.gradient_cache)

    # Load data
    gradient_img = nib.load(str(args.gradient_cache))
    atlas_img, atlas_data, labels = load_atlas(args.atlas, args.labels)

    # Map gradient to ROIs
    roi_gradients = map_gradient_to_rois(gradient_img, atlas_img, atlas_data, labels)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(args.output), roi_gradients.reshape(1, -1), delimiter=",", fmt="%.6f")
    logger.info("Saved %d ROI gradient values to %s", len(roi_gradients), args.output)


if __name__ == "__main__":
    main()
