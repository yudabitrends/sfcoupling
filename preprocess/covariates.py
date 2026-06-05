"""
Covariate residualization and z-score scaling.
Train-only fit to avoid leakage.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def _should_treat_as_numeric(s1: pd.Series, s2: pd.Series) -> bool:
    """
    Robust numeric detection for mixed-type columns (e.g., Age with int/float strings).
    Treat as numeric if >=90% non-null values can be parsed in either dataset.
    """
    a = pd.to_numeric(s1, errors="coerce")
    b = pd.to_numeric(s2, errors="coerce")
    n1 = max(int(s1.notna().sum()), 1)
    n2 = max(int(s2.notna().sum()), 1)
    r1 = float(a.notna().sum()) / n1
    r2 = float(b.notna().sum()) / n2
    return bool((r1 >= 0.9 and n1 > 0) or (r2 >= 0.9 and n2 > 0))


def build_design_matrices_consistent(
    cov1: pd.DataFrame,
    cov2: pd.DataFrame,
    sub1: List[str],
    sub2: List[str],
    covariate_cols: List[str],
    id_col: str = "SubjectID",
    train_subjects_dataset1: Optional[List[str]] = None,
    audit_out: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build design matrices for two datasets with consistent column encoding.
    Categorical mappings are learned from dataset1 train subjects only.
    Returns: (design1, design2, col_names)
    """
    need_cols = [id_col] + [c for c in covariate_cols if c in cov1.columns or c in cov2.columns]
    c1 = cov1.copy()
    c2 = cov2.copy()
    c1[id_col] = c1[id_col].astype(str).str.strip()
    c2[id_col] = c2[id_col].astype(str).str.strip()
    for c in need_cols:
        if c not in c1.columns:
            c1[c] = np.nan
        if c not in c2.columns:
            c2[c] = np.nan
    c1 = c1[need_cols].drop_duplicates(subset=id_col, keep="first").set_index(id_col)
    c2 = c2[need_cols].drop_duplicates(subset=id_col, keep="first").set_index(id_col)
    c1 = c1.reindex(sub1).reset_index()
    c2 = c2.reindex(sub2).reset_index()

    train_subjects_dataset1 = train_subjects_dataset1 or list(sub1)
    train_set = set(train_subjects_dataset1)
    train_mask1 = np.array([sid in train_set for sid in sub1], dtype=bool)

    design_parts_1 = []
    design_parts_2 = []
    col_names: List[str] = ["intercept"]
    categorical_mappings: Dict[str, List[str]] = {}
    unseen_categories: Dict[str, Dict[str, List[str]]] = {}

    available_cov = [c for c in covariate_cols if c in need_cols]
    for col in available_cov:
        s1 = c1[col]
        s2 = c2[col]
        if _should_treat_as_numeric(s1, s2):
            v_train = pd.to_numeric(s1[train_mask1], errors="coerce").values.astype(np.float64)
            med = float(np.nanmedian(v_train)) if np.any(~np.isnan(v_train)) else 0.0
            v1 = pd.to_numeric(s1, errors="coerce").fillna(med).values.astype(np.float64).reshape(-1, 1)
            v2 = pd.to_numeric(s2, errors="coerce").fillna(med).values.astype(np.float64).reshape(-1, 1)
            design_parts_1.append(v1)
            design_parts_2.append(v2)
            col_names.append(col)
            continue

        s1_str = s1.fillna("__MISSING__").astype(str)
        s2_str = s2.fillna("__MISSING__").astype(str)
        train_vals = s1_str[train_mask1]
        categories = sorted(set(train_vals.tolist()))
        categorical_mappings[col] = categories
        unseen_1 = sorted(set(v for v in s1_str.tolist() if v not in categories))
        unseen_2 = sorted(set(v for v in s2_str.tolist() if v not in categories))
        if len(categories) <= 1:
            # all baseline, no dummy columns needed for this covariate
            unseen_categories[col] = {
                "dataset1_unseen_values": unseen_1,
                "dataset2_unseen_values": unseen_2,
                "baseline_category": categories[0] if categories else "__MISSING__",
                "zero_column_handling": True,
            }
            continue
        baseline = categories[0]
        dummy_cats = categories[1:]

        arr1 = np.zeros((len(sub1), len(dummy_cats)), dtype=np.float64)
        arr2 = np.zeros((len(sub2), len(dummy_cats)), dtype=np.float64)
        cat_to_idx = {cname: i for i, cname in enumerate(dummy_cats)}
        unseen_1_set = set()
        unseen_2_set = set()
        for i, v in enumerate(s1_str.tolist()):
            if v in cat_to_idx:
                arr1[i, cat_to_idx[v]] = 1.0
            elif v not in categories:
                unseen_1_set.add(v)
        for i, v in enumerate(s2_str.tolist()):
            if v in cat_to_idx:
                arr2[i, cat_to_idx[v]] = 1.0
            elif v not in categories:
                unseen_2_set.add(v)
        unseen_categories[col] = {
            "dataset1_unseen_values": sorted(unseen_1_set),
            "dataset2_unseen_values": sorted(unseen_2_set),
            "baseline_category": baseline,
            "zero_column_handling": True,
        }
        design_parts_1.append(arr1)
        design_parts_2.append(arr2)
        col_names.extend([f"{col}_{c}" for c in dummy_cats])

    if design_parts_1:
        X1 = np.hstack(design_parts_1)
        X2 = np.hstack(design_parts_2)
    else:
        X1 = np.empty((len(sub1), 0), dtype=np.float64)
        X2 = np.empty((len(sub2), 0), dtype=np.float64)
    design1 = np.hstack([np.ones((len(sub1), 1), dtype=np.float64), X1])
    design2 = np.hstack([np.ones((len(sub2), 1), dtype=np.float64), X2])

    if audit_out is not None:
        audit_out.update(
            {
                "design_matrix_columns": col_names,
                "categorical_mappings": categorical_mappings,
                "unseen_categories": unseen_categories,
                "train_subject_count_dataset1": int(np.sum(train_mask1)),
            }
        )
    return design1, design2, col_names


def build_design_matrix(
    df: pd.DataFrame,
    covariate_cols: List[str],
    subject_ids: List[str],
    id_col: str = "SubjectID",
) -> Tuple[np.ndarray, List[str], Optional[pd.DataFrame]]:
    """
    Build design matrix for residualization, aligned to subject_ids order.
    - Numeric cols: used as-is (fillna with median)
    - Categorical cols: one-hot encoded
    Returns: (design array N x p, column names, encoded_df for reference)
    """
    df = df.copy()
    df[id_col] = df[id_col].astype(str).str.strip()
    id_to_idx = {str(s).strip(): i for i, s in enumerate(subject_ids)}
    df = df[df[id_col].isin(id_to_idx)].copy()
    df["_order"] = df[id_col].map(id_to_idx)
    df = df.sort_values("_order")
    if len(df) != len(subject_ids):
        raise ValueError(
            f"Covariates df has {len(df)} subjects but we need {len(subject_ids)}. "
            "Ensure all aligned subjects have covariate data."
        )

    available = [c for c in covariate_cols if c in df.columns]
    col_names = ["intercept"]
    design_parts = []

    for col in available:
        vals = df[col].values
        if pd.api.types.is_numeric_dtype(df[col]):
            v = np.asarray(vals, dtype=np.float64)
            med = np.nanmedian(v)
            v = np.nan_to_num(v, nan=med)
            design_parts.append(v.reshape(-1, 1))
            col_names.append(col)
        else:
            enc = pd.get_dummies(pd.Series(vals.astype(str)), drop_first=True)
            arr = enc.values.astype(np.float64)
            for j, c in enumerate(enc.columns):
                design_parts.append(arr[:, j : j + 1])
                col_names.append(f"{col}_{c}")

    if not design_parts:
        design = np.ones((len(subject_ids), 1))
        return design, ["intercept"], None

    X_design = np.hstack(design_parts)
    intercept = np.ones((len(subject_ids), 1))
    design = np.hstack([intercept, X_design])
    col_names = ["intercept"] + [c for c in col_names if c != "intercept"]
    return design, col_names, df


def residualize(
    X: np.ndarray,
    Y: np.ndarray,
    design: np.ndarray,
    train_mask: np.ndarray,
    residualize_x: bool = True,
    residualize_y: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict, Optional[LinearRegression], Optional[LinearRegression]]:
    """
    Residualize X and/or Y w.r.t. design matrix using train data only.
    Returns: (X_resid, Y_resid, params_dict, lr_x, lr_y)
    params_dict is JSON-serializable; lr_x/lr_y used for apply_residualization.
    """
    X_res = X.copy()
    Y_res = Y.copy()
    params: Dict = {"x_coef": None, "y_coef": None, "x_intercept": None, "y_intercept": None}
    lr_x, lr_y = None, None

    X_train = X[train_mask]
    Y_train = Y[train_mask]
    design_train = design[train_mask]

    if design.shape[1] <= 1:
        return X_res, Y_res, params, lr_x, lr_y

    if residualize_x:
        lr_x = LinearRegression()
        lr_x.fit(design_train, X_train)
        pred_x = lr_x.predict(design)
        X_res = X - pred_x
        params["x_coef"] = lr_x.coef_.tolist()
        params["x_intercept"] = lr_x.intercept_.tolist() if hasattr(lr_x.intercept_, "__iter__") else float(lr_x.intercept_)

    if residualize_y:
        lr_y = LinearRegression()
        lr_y.fit(design_train, Y_train)
        pred_y = lr_y.predict(design)
        Y_res = Y - pred_y
        params["y_coef"] = lr_y.coef_.tolist()
        params["y_intercept"] = lr_y.intercept_.tolist() if hasattr(lr_y.intercept_, "__iter__") else float(lr_y.intercept_)

    return X_res, Y_res, params, lr_x, lr_y


def fit_scalers_train_only(
    X: np.ndarray,
    Y: np.ndarray,
    train_mask: np.ndarray,
) -> Tuple[StandardScaler, StandardScaler]:
    """Fit StandardScaler on train subset only."""
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(X[train_mask])
    scaler_y.fit(Y[train_mask])
    return scaler_x, scaler_y


def apply_scalers(
    X: np.ndarray,
    Y: np.ndarray,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply fitted scalers to X and Y."""
    return scaler_x.transform(X), scaler_y.transform(Y)


def apply_residualization(
    X: np.ndarray,
    Y: np.ndarray,
    design: np.ndarray,
    lr_x: Optional[LinearRegression],
    lr_y: Optional[LinearRegression],
    residualize_x: bool,
    residualize_y: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply fitted residualization to new data."""
    X_res = X.copy()
    Y_res = Y.copy()
    if residualize_x and lr_x is not None:
        X_res = X - lr_x.predict(design)
    if residualize_y and lr_y is not None:
        Y_res = Y - lr_y.predict(design)
    return X_res, Y_res
