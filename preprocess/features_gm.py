"""
GM (gray matter) feature extraction.
Supports: H5 with NIfTI paths, CSV with ROI features, NPY.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_nifti_flatten(path: str) -> np.ndarray:
    """
    Load NIfTI file and return flattened 1D array.
    Uses nibabel; raises FileNotFoundError if file missing.
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "nibabel is required for NIfTI loading. Install with: pip install nibabel"
        ) from None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    img = nib.load(str(p))
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    return data.flatten()


def load_nifti_data(path: str) -> np.ndarray:
    """Load NIfTI file and return native 3D/4D data array."""
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "nibabel is required for NIfTI loading. Install with: pip install nibabel"
        ) from None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    img = nib.load(str(p))
    return np.asarray(img.get_fdata(), dtype=np.float32)


def _resize_to_shape_nn(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Nearest-neighbor resize for 3D arrays using only numpy.
    Deterministic and dependency-light fallback when shapes differ.
    """
    sx, sy, sz = volume.shape
    tx, ty, tz = target_shape
    ix = np.clip(np.round(np.linspace(0, sx - 1, tx)).astype(int), 0, sx - 1)
    iy = np.clip(np.round(np.linspace(0, sy - 1, ty)).astype(int), 0, sy - 1)
    iz = np.clip(np.round(np.linspace(0, sz - 1, tz)).astype(int), 0, sz - 1)
    return volume[ix][:, iy][:, :, iz]


def _build_roi_weights(
    roi_atlas_path: str,
    roi_labels_path: Optional[str] = None,
    roi_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build atlas weight matrix (n_voxels x n_roi) from 4D atlas map.
    Returns (weights, valid_mask_flat, roi_names).
    """
    atlas = load_nifti_data(roi_atlas_path)
    if atlas.ndim != 4:
        raise ValueError(f"ROI atlas must be 4D (x,y,z,n_roi), got shape {atlas.shape}")
    spatial_shape = atlas.shape[:3]
    n_roi = atlas.shape[3]
    atlas = np.nan_to_num(atlas, nan=0.0)
    if roi_threshold > 0:
        atlas = np.where(atlas >= float(roi_threshold), atlas, 0.0)
    valid_mask = np.sum(atlas, axis=3) > 0
    if not np.any(valid_mask):
        raise ValueError("ROI atlas produced empty valid mask. Check roi_threshold/atlas path.")
    W = atlas[valid_mask].reshape(-1, n_roi).astype(np.float64)
    col_sums = np.sum(W, axis=0)
    keep_idx = np.where(col_sums > 0)[0]
    if keep_idx.size == 0:
        raise ValueError("ROI atlas has no valid non-zero components after masking.")
    if keep_idx.size != n_roi:
        dropped = np.where(col_sums <= 0)[0].tolist()
        logger.warning(
            "Dropping %d zero-sum ROI components: %s",
            len(dropped),
            dropped[:10],
        )
    W = W[:, keep_idx]
    col_sums = np.sum(W, axis=0)
    W /= col_sums
    if roi_labels_path:
        txt = Path(roi_labels_path).read_text().splitlines()
        labels = [x.strip() for x in txt if x.strip()]
        if len(labels) != n_roi:
            logger.warning(
                "ROI labels count %d != atlas components %d (%s). "
                "Falling back to generic roi_<index> names.",
                len(labels),
                n_roi,
                roi_atlas_path,
            )
            labels = [f"roi_{i}" for i in keep_idx.tolist()]
        else:
            labels = [labels[i] for i in keep_idx.tolist()]
    else:
        labels = [f"roi_{i}" for i in keep_idx.tolist()]
    return W, valid_mask.flatten(), labels


def _extract_roi_vector_from_nifti(
    nifti_path: str,
    atlas_shape: Tuple[int, int, int],
    atlas_weights: np.ndarray,
    atlas_mask_flat: np.ndarray,
) -> np.ndarray:
    """Compute weighted ROI vector from subject image with deterministic shape handling."""
    img = load_nifti_data(nifti_path)
    if img.ndim == 4:
        img = img[..., 0]
    if img.ndim != 3:
        raise ValueError(f"Expected 3D sMRI volume, got shape {img.shape} ({nifti_path})")
    if tuple(img.shape) != tuple(atlas_shape):
        img = _resize_to_shape_nn(img, atlas_shape)
    vals = img.flatten()[atlas_mask_flat].astype(np.float64)
    if vals.shape[0] != atlas_weights.shape[0]:
        raise ValueError(
            f"ROI mask/subject mismatch: subject voxels {vals.shape[0]} vs atlas weights {atlas_weights.shape[0]}"
        )
    roi_vec = vals @ atlas_weights
    return roi_vec.astype(np.float32)


def extract_gm_from_h5(
    df: pd.DataFrame,
    path_col: str,
    id_col: str,
    fail_on_missing: bool = False,
    representation: str = "voxel",
    roi_atlas_path: Optional[str] = None,
    roi_labels_path: Optional[str] = None,
    roi_threshold: float = 0.0,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Extract GM from H5 DataFrame. If path_col contains file paths (sMRIPath),
    load each NIfTI and flatten. Otherwise assume path_col contains arrays.
    Returns: (subject_id -> GM vector dict, feature_names)
    """
    if path_col not in df.columns:
        raise ValueError(f"Column {path_col} not found. Available: {list(df.columns)}")
    if id_col not in df.columns:
        raise ValueError(f"Column {id_col} not found. Available: {list(df.columns)}")

    gm_dict: Dict[str, np.ndarray] = {}
    feature_names: List[str] = []
    failed: List[Tuple[str, str, str]] = []  # (subject_id, path, error)
    rep = (representation or "voxel").strip().lower()
    atlas_weights = None
    atlas_mask_flat = None
    atlas_shape = None
    if rep == "roi":
        if not roi_atlas_path:
            raise ValueError("GM ROI mode requires roi_atlas_path in config.")
        W, mask_flat, roi_names = _build_roi_weights(
            roi_atlas_path=roi_atlas_path,
            roi_labels_path=roi_labels_path,
            roi_threshold=roi_threshold,
        )
        atlas_weights = W
        atlas_mask_flat = mask_flat
        atlas_shape = load_nifti_data(roi_atlas_path).shape[:3]
        feature_names = roi_names

    for idx, row in df.iterrows():
        sid = str(row[id_col]).strip()
        val = row[path_col]
        if pd.isna(val):
            failed.append((sid, "", "NA value"))
            continue
        try:
            if isinstance(val, (str, Path)) or (
                hasattr(val, "strip") and isinstance(val, str)
            ):
                path_str = str(val).strip()
                if rep == "roi":
                    assert atlas_shape is not None
                    assert atlas_weights is not None
                    assert atlas_mask_flat is not None
                    arr = _extract_roi_vector_from_nifti(
                        path_str,
                        atlas_shape=atlas_shape,
                        atlas_weights=atlas_weights,
                        atlas_mask_flat=atlas_mask_flat,
                    )
                else:
                    arr = load_nifti_flatten(path_str)
            elif isinstance(val, (np.ndarray, list)):
                arr = np.asarray(val, dtype=np.float32).flatten()
            else:
                failed.append((sid, str(val)[:100], "Unknown type"))
                continue
            if arr.size == 0:
                failed.append((sid, str(val)[:100], "Empty array"))
                continue
            gm_dict[sid] = arr.astype(np.float32)
            if not feature_names:
                feature_names = [f"gm_{i}" for i in range(arr.size)]
        except FileNotFoundError as e:
            failed.append((sid, str(val)[:80], str(e)))
        except Exception as e:
            failed.append((sid, str(val)[:80], str(e)))

    if failed:
        for sid, p, err in failed[:10]:
            logger.warning("GM load failed: subject=%s path=%s error=%s", sid, p, err)
        if len(failed) > 10:
            logger.warning("... and %d more GM load failures", len(failed) - 10)
        if fail_on_missing and failed:
            msg = (
                f"GM extraction failed for {len(failed)} subjects. "
                f"First failure: subject={failed[0][0]}, error={failed[0][2]}"
            )
            raise RuntimeError(msg)

    return gm_dict, feature_names


def extract_gm_from_csv_paths(
    path: str,
    id_col: str,
    path_col: str = "sMRIPath",
    fail_on_missing: bool = False,
    representation: str = "voxel",
    roi_atlas_path: Optional[str] = None,
    roi_labels_path: Optional[str] = None,
    roi_threshold: float = 0.0,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Extract GM from CSV with NIfTI paths. path_col gives file path per subject.
    """
    df = pd.read_csv(path, low_memory=False)
    if path_col not in df.columns or id_col not in df.columns:
        raise ValueError(
            f"Need {id_col} and {path_col}. Available: {list(df.columns)}"
        )
    gm_dict, feat_names = extract_gm_from_h5(
        df,
        path_col,
        id_col,
        fail_on_missing=fail_on_missing,
        representation=representation,
        roi_atlas_path=roi_atlas_path,
        roi_labels_path=roi_labels_path,
        roi_threshold=roi_threshold,
    )
    return gm_dict, feat_names


def extract_gm_from_csv(
    path: str, id_col: str, feature_cols: Optional[List[str]] = None
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Extract GM from CSV. Expects ROI-level or voxel-level columns.
    feature_cols: optional list of column names for GM features; if None, use all numeric except id_col.
    """
    df = pd.read_csv(path, low_memory=False)
    if id_col not in df.columns:
        raise ValueError(f"Column {id_col} not found. Available: {list(df.columns)}")

    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        cols = feature_cols
    else:
        cols = [
            c
            for c in df.columns
            if c != id_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not cols:
            raise ValueError(
                f"No numeric feature columns found in {path}. "
                "Specify feature_cols explicitly."
            )

    gm_dict = {}
    for _, row in df.iterrows():
        sid = str(row[id_col]).strip()
        vec = row[cols].values.astype(np.float32)
        if np.any(np.isnan(vec)):
            continue
        gm_dict[sid] = vec
    feature_names = list(cols)
    return gm_dict, feature_names


def extract_gm_from_npy(
    path: str, subject_ids: List[str], id_order: Optional[List[str]] = None
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Extract GM from NPY (N x dx array). subject_ids must match row order if id_order is None.
    If id_order provided, it defines row-to-subject mapping.
    """
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array from {path}, got shape {arr.shape}")
    n, dx = arr.shape
    order = id_order if id_order is not None else subject_ids
    if len(order) != n:
        raise ValueError(
            f"NPY has {n} rows but {len(order)} subject IDs. "
            "Ensure subject order matches array rows."
        )
    gm_dict = {str(sid).strip(): arr[i].astype(np.float32) for i, sid in enumerate(order)}
    feature_names = [f"gm_{i}" for i in range(dx)]
    return gm_dict, feature_names
