"""
WM (white matter) feature extraction for sfcoupling v2.

Projects per-subject FA (or MD/RD/AD) maps onto the NeuroMark3 dMRI_FA 4D atlas
(100 z-scored signed ICA spatial components) to produce 100-dim loading vectors
via dual-regression (OLS on spatial ICs).

Input subject maps are assumed to already be in MNI152 1mm (shape 182x218x182),
i.e. the ants_FAWarped output of the qneuromark DTI pipeline; same affine as
the atlas.

Default mode is dual_regression: solve fa ≈ A · x, x = pinv(A) · fa, using only
voxels with non-zero atlas support across components. Weighted-average mode is
retained for atlases with strictly non-negative components.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from preprocess.features_gm import (
    _build_roi_weights,
    _extract_roi_vector_from_nifti,
    load_nifti_data,
)

logger = logging.getLogger(__name__)

DEFAULT_FA_ATLAS = "/data/qneuromark/Network_templates/NeuroMark3/dMRI_FA.nii"
DEFAULT_FA_LABELS = "/data/qneuromark/Network_templates/NeuroMark3/dMRI_FA.txt"


def _build_dual_regression_projector(
    atlas_path: str,
    standardize_components: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], int]:
    """
    Construct the dual-regression pseudo-inverse for a signed-ICA spatial atlas.

    Returns (P, mask_flat, spatial_shape, n_components) where:
      P                ∈ R^{K x M}   : pseudo-inverse mapping voxel vec -> K loadings
      mask_flat        ∈ {0,1}^{Nvx} : flattened in-mask indicator
      spatial_shape                    : (X,Y,Z)
      n_components     = K             : number of atlas components kept (100 typical)

    The mask is the union of voxels where any component has non-zero weight.
    If standardize_components=True, each component is centered and scaled to unit
    L2 norm before computing pinv, which makes the loadings dimensionless.
    """
    atlas = load_nifti_data(atlas_path)
    if atlas.ndim != 4:
        raise ValueError(f"Expected 4D atlas, got shape {atlas.shape}")
    spatial_shape = atlas.shape[:3]
    n_components = atlas.shape[3]
    atlas = np.nan_to_num(atlas, nan=0.0)

    mask = np.any(atlas != 0, axis=3)
    if not np.any(mask):
        raise ValueError("Atlas has empty union mask (all components zero).")
    mask_flat = mask.flatten()
    A = atlas.reshape(-1, n_components)[mask_flat].astype(np.float64)  # (M, K)

    if standardize_components:
        A = A - A.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(A, axis=0)
        norms[norms == 0] = 1.0
        A = A / norms

    # pinv(A) is (K, M); use lstsq-equivalent via SVD for numerical stability
    P = np.linalg.pinv(A, rcond=1e-8).astype(np.float32)
    logger.info(
        "Dual-regression projector built: K=%d, M=%d, mean|A|=%.4f",
        n_components,
        int(mask_flat.sum()),
        float(np.mean(np.abs(A))),
    )
    return P, mask_flat, spatial_shape, n_components


def _load_or_default_labels(labels_path: Optional[str], n_components: int) -> List[str]:
    """Load component labels from a text file, one per line, or fall back to ic_<k>."""
    if labels_path and Path(labels_path).exists():
        lines = [x.strip() for x in Path(labels_path).read_text().splitlines() if x.strip()]
        if len(lines) == n_components:
            return lines
        logger.warning(
            "Label count %d != %d components; using generic ic_<k> names.",
            len(lines),
            n_components,
        )
    return [f"ic_{k}" for k in range(n_components)]


def _project_nifti_to_ic_loadings(
    nifti_path: str,
    projector: np.ndarray,
    mask_flat: np.ndarray,
    spatial_shape: Tuple[int, int, int],
    mean_center: bool = True,
) -> np.ndarray:
    """
    Load a subject map, mask, optionally mean-center, project onto IC basis.
    Returns a 1-D vector of length K.
    """
    img = load_nifti_data(nifti_path)
    if img.ndim == 4:
        img = img[..., 0]
    if tuple(img.shape) != tuple(spatial_shape):
        raise ValueError(
            f"Subject volume shape {img.shape} != atlas shape {spatial_shape} "
            f"({nifti_path}); re-register to MNI first."
        )
    vec = img.flatten()[mask_flat].astype(np.float64)
    if mean_center:
        vec = vec - vec.mean()
    # projector is (K, M), vec is (M,)
    loadings = projector.astype(np.float64) @ vec
    return loadings.astype(np.float32)


def find_subject_fa_path(
    subject_id: str,
    bids_root: str = "/data/qneuromark/Data/UKBiobank/DTI_Data_BIDS/Raw_Data",
    visit: str = "visit1",
    modality: str = "FA",
) -> Optional[Path]:
    """
    Locate the ants-warped DTI map for a UKB subject under the qneuromark BIDS tree.

    modality in {FA, MD, RD, AD}; each maps to ants_{M}Warped.nii.gz in the xfms folder.
    Returns the path if it exists, None otherwise.
    """
    p = (
        Path(bids_root)
        / str(subject_id)
        / visit
        / "dti"
        / "dti_FA"
        / "xfms"
        / f"ants_{modality}Warped.nii.gz"
    )
    return p if p.exists() else None


def extract_wm_from_paths(
    paths: Dict[str, str],
    atlas_path: str = DEFAULT_FA_ATLAS,
    labels_path: Optional[str] = DEFAULT_FA_LABELS,
    atlas_threshold: float = 0.0,
    fail_on_missing: bool = False,
    mode: str = "dual_regression",
    mean_center_subject: bool = True,
    standardize_components: bool = True,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Extract WM features for each subject.

    paths: {subject_id -> FA nifti path}.
    mode: "dual_regression" (OLS on signed IC atlas; recommended for dMRI_FA)
          or "weighted_average" (non-negative atlas; GM-style).
    Returns (features dict, feature names).
    """
    if mode == "weighted_average":
        atlas_weights, atlas_mask_flat, feature_names = _build_roi_weights(
            roi_atlas_path=atlas_path,
            roi_labels_path=labels_path,
            roi_threshold=atlas_threshold,
        )
        atlas_shape = load_nifti_data(atlas_path).shape[:3]
        projector = None
    elif mode == "dual_regression":
        projector, atlas_mask_flat, atlas_shape, n_components = (
            _build_dual_regression_projector(
                atlas_path=atlas_path,
                standardize_components=standardize_components,
            )
        )
        feature_names = _load_or_default_labels(labels_path, n_components)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    features: Dict[str, np.ndarray] = {}
    failed: List[Tuple[str, str, str]] = []

    for sid, path in paths.items():
        sid = str(sid).strip()
        if not path or (isinstance(path, float) and np.isnan(path)):
            failed.append((sid, "", "NA path"))
            continue
        try:
            if mode == "weighted_average":
                vec = _extract_roi_vector_from_nifti(
                    str(path),
                    atlas_shape=atlas_shape,
                    atlas_weights=atlas_weights,
                    atlas_mask_flat=atlas_mask_flat,
                )
            else:
                vec = _project_nifti_to_ic_loadings(
                    str(path),
                    projector=projector,
                    mask_flat=atlas_mask_flat,
                    spatial_shape=atlas_shape,
                    mean_center=mean_center_subject,
                )
            if vec.size == 0 or np.any(np.isnan(vec)):
                failed.append((sid, str(path)[:80], "Empty/NaN output"))
                continue
            features[sid] = vec
        except FileNotFoundError as e:
            failed.append((sid, str(path)[:80], str(e)))
        except Exception as e:
            failed.append((sid, str(path)[:80], str(e)))

    if failed:
        for sid, p, err in failed[:10]:
            logger.warning("WM load failed: subject=%s path=%s error=%s", sid, p, err)
        if len(failed) > 10:
            logger.warning("... and %d more WM load failures", len(failed) - 10)
        if fail_on_missing:
            raise RuntimeError(
                f"WM extraction failed for {len(failed)} subjects; "
                f"first: subject={failed[0][0]} error={failed[0][2]}"
            )

    return features, feature_names


def extract_wm_from_subject_list(
    subject_ids: List[str],
    bids_root: str = "/data/qneuromark/Data/UKBiobank/DTI_Data_BIDS/Raw_Data",
    visit: str = "visit1",
    modality: str = "FA",
    atlas_path: str = DEFAULT_FA_ATLAS,
    labels_path: Optional[str] = DEFAULT_FA_LABELS,
    atlas_threshold: float = 0.0,
    skip_missing: bool = True,
    mode: str = "dual_regression",
    mean_center_subject: bool = True,
    standardize_components: bool = True,
) -> Tuple[Dict[str, np.ndarray], List[str], List[str]]:
    """
    Convenience wrapper: resolve BIDS paths for each subject then extract.

    Returns (features dict, feature names, list of subject_ids with missing file).
    """
    resolved: Dict[str, str] = {}
    missing: List[str] = []
    for sid in subject_ids:
        p = find_subject_fa_path(sid, bids_root=bids_root, visit=visit, modality=modality)
        if p is None:
            missing.append(str(sid))
            if skip_missing:
                continue
        else:
            resolved[str(sid)] = str(p)

    if missing:
        logger.warning(
            "%d/%d subjects missing %s map; first 5: %s",
            len(missing),
            len(subject_ids),
            modality,
            missing[:5],
        )

    features, feature_names = extract_wm_from_paths(
        resolved,
        atlas_path=atlas_path,
        labels_path=labels_path,
        atlas_threshold=atlas_threshold,
        fail_on_missing=False,
        mode=mode,
        mean_center_subject=mean_center_subject,
        standardize_components=standardize_components,
    )
    return features, feature_names, missing


def save_features_npy(
    features: Dict[str, np.ndarray],
    out_path: str,
    subject_order: Optional[List[str]] = None,
) -> List[str]:
    """
    Save a feature dict to NPY with matching subject order list.

    Returns the ordered subject ID list (can be used to write an index .tsv).
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    order = subject_order if subject_order is not None else sorted(features.keys())
    missing_in_features = [sid for sid in order if sid not in features]
    if missing_in_features:
        raise KeyError(
            f"save_features_npy: {len(missing_in_features)} subjects not in features dict, "
            f"first 5: {missing_in_features[:5]}"
        )
    arr = np.stack([features[sid] for sid in order], axis=0).astype(np.float32)
    np.save(out, arr)
    return list(order)


def audit_fa_atlas(atlas_path: str = DEFAULT_FA_ATLAS) -> Dict[str, object]:
    """
    Return quick summary of the FA atlas for preflight checks.
    """
    atlas = load_nifti_data(atlas_path)
    if atlas.ndim != 4:
        raise ValueError(f"Expected 4D atlas, got shape {atlas.shape}")
    n_comp = atlas.shape[3]
    col_sums = np.array([atlas[..., k].sum() for k in range(n_comp)])
    return dict(
        shape=tuple(atlas.shape),
        n_components=n_comp,
        zero_sum_components=int(np.sum(col_sums <= 0)),
        min_colsum=float(col_sums.min()),
        median_colsum=float(np.median(col_sums)),
        max_colsum=float(col_sums.max()),
    )
