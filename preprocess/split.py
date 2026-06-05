"""
Train/val/test splitting with stratification.
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.model_selection import train_test_split


def create_splits(
    subject_ids: List[str],
    stratify_col: Optional[np.ndarray],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Dict[str, List[str]]:
    """
    Split subject_ids into train/val/test.
    stratify_col: array of shape (len(subject_ids),) for stratification.
    train_frac + val_frac + test_frac should equal 1; test_frac = 1 - train_frac - val_frac.
    """
    ids = np.array(subject_ids)
    n = len(ids)
    if n == 0:
        return {"train": [], "val": [], "test": []}

    test_frac = 1.0 - train_frac - val_frac
    if test_frac < 0:
        raise ValueError(
            f"train_frac ({train_frac}) + val_frac ({val_frac}) > 1. "
            "Ensure train_frac + val_frac <= 1."
        )

    if stratify_col is not None and len(stratify_col) != n:
        raise ValueError(
            f"stratify_col length {len(stratify_col)} != subject count {n}"
        )

    if stratify_col is not None:
        trainval_ids, test_ids, _, _ = train_test_split(
            ids,
            stratify_col,
            test_size=test_frac,
            random_state=seed,
            stratify=stratify_col,
        )
    else:
        trainval_ids, test_ids = train_test_split(
            ids, test_size=test_frac, random_state=seed
        )

    val_ratio = val_frac / (train_frac + val_frac) if (train_frac + val_frac) > 0 else 0
    n_trainval = len(trainval_ids)
    if n_trainval == 0:
        return {"train": [], "val": [], "test": test_ids.tolist()}

    stratify_trainval = None
    if stratify_col is not None:
        sid_to_idx = {s: i for i, s in enumerate(subject_ids)}
        stratify_trainval = np.array([stratify_col[sid_to_idx[s]] for s in trainval_ids])

    if val_ratio > 0 and val_ratio < 1:
        if stratify_trainval is not None:
            train_ids, val_ids, _, _ = train_test_split(
                trainval_ids,
                stratify_trainval,
                test_size=val_ratio,
                random_state=seed,
                stratify=stratify_trainval,
            )
        else:
            train_ids, val_ids = train_test_split(
                trainval_ids, test_size=val_ratio, random_state=seed
            )
    else:
        train_ids = trainval_ids
        val_ids = np.array([], dtype=ids.dtype)

    return {
        "train": train_ids.tolist(),
        "val": val_ids.tolist(),
        "test": test_ids.tolist(),
    }


def create_external_test_mapping(
    ds1_splits: Dict[str, List[str]],
    ds2_splits: Dict[str, List[str]],
) -> Dict:
    """Create mapping for external test: train on dataset1, test on dataset2 and vice versa."""
    def all_ids(splits: Dict) -> List[str]:
        out = []
        for k in ["train", "val", "test"]:
            out.extend(splits.get(k, []))
        return out
    return {
        "train_on_dataset1_test_on_dataset2": {
            "train": ds1_splits.get("train", []),
            "val": ds1_splits.get("val", []),
            "test": ds2_splits.get("test", all_ids(ds2_splits)),
        },
        "train_on_dataset2_test_on_dataset1": {
            "train": ds2_splits.get("train", []),
            "val": ds2_splits.get("val", []),
            "test": ds1_splits.get("test", all_ids(ds1_splits)),
        },
    }


def build_split_proof(splits: Dict[str, List[str]], sanity_only: bool = False) -> Dict:
    """Build no-overlap and coverage proof for a split dict."""
    train = set(splits.get("train", []))
    val = set(splits.get("val", []))
    test = set(splits.get("test", []))
    no_overlap = len(train & val) == 0 and len(train & test) == 0 and len(val & test) == 0
    total = len(train | val | test)
    return {
        "no_overlap": bool(no_overlap),
        "counts": {"train": len(train), "val": len(val), "test": len(test), "union": total},
        "sanity_only": bool(sanity_only),
    }
