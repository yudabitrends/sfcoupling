"""
M1: Residualization sensitivity sweep for the NeuroImage revision.

Produces a table showing Ridge PC-R^2 at the MODEL PCA dimensions k=5, 10, 20
as a function of which confounds are residualized out of (Age, Gender, total_gm).

The Ridge pipeline matches run_baselines_multiseed.py exactly:
  1. Fit PCA(k) on training Y to get k-dim targets.
  2. Fit Ridge: X -> Ytr_t (k-dim PC coordinates).
  3. Predict on test: Y_pred_k (k-dim).
  4. Inverse-transform to full FNC space.
  5. Evaluate with a FIXED k=20 eval_pca fit on training Y.

Addresses Major Comment M1 that the Abstract's ~6% R^2 is post-residualization
and the raw coupling is substantially stronger.

Run from repo root:
    python scripts/reviewer_revision_M1_resid_sweep.py

Outputs:
    results/reviewer_revision/M1_resid_sensitivity.json
    results/reviewer_revision/M1_resid_sensitivity.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


REPO = Path("/home/users/ybi3/sfcoupling")
DATA = Path("/data/users1/ybi3/cVAE/aligned_features")
OUT = REPO / "results" / "reviewer_revision"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46, 47, 48]
MODEL_K_LIST = [5, 10, 20]
EVAL_K = 20
ALPHAS = [1e-3, 1e-2, 1e-1, 1, 10, 100]


def _ids_to_idx(all_ids: list[str], target_ids: list[str]) -> np.ndarray:
    lookup = {sid: i for i, sid in enumerate(all_ids)}
    return np.array([lookup[s] for s in target_ids if s in lookup], dtype=np.int64)


def residualize_fit(Z_tr: np.ndarray, C_tr: np.ndarray) -> np.ndarray:
    """Fit OLS residualization. Returns coefficient P such that R = Z - C @ P."""
    return np.linalg.lstsq(C_tr, Z_tr, rcond=None)[0]


def residualize_apply(Z: np.ndarray, C: np.ndarray, P: np.ndarray) -> np.ndarray:
    return Z - C @ P


def _safe_float(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").values
    if np.any(np.isnan(vals)):
        mean = np.nanmean(vals)
        vals = np.where(np.isnan(vals), mean, vals)
    return vals


def build_confound_matrix(meta: pd.DataFrame, X_raw: np.ndarray,
                          mode: str) -> np.ndarray:
    n = len(meta)
    cols = [np.ones(n)]
    if mode in ("age", "age_sex", "age_sex_totgm"):
        cols.append(_safe_float(meta["Age"]))
    if mode in ("age_sex", "age_sex_totgm"):
        cols.append(_safe_float(meta["Gender"]))
    if mode == "age_sex_totgm":
        cols.append(X_raw.sum(axis=1))
    return np.column_stack(cols)


def pc_r2_mean(y_true: np.ndarray, y_pred: np.ndarray, pca: PCA) -> float:
    yt = pca.transform(y_true)
    yp = pca.transform(y_pred)
    per_pc = r2_score(yt, yp, multioutput="raw_values")
    per_pc = np.asarray(per_pc, dtype=np.float64)
    per_pc = np.where(np.isfinite(per_pc), per_pc, 0.0)
    return float(per_pc.mean())


def fit_ridge_grid_pcspace(Xtr, Ytr_t, Xva, Yva_t, alphas):
    best_alpha, best_val = None, -np.inf
    for alpha in alphas:
        ridge = Ridge(alpha=alpha, random_state=0)
        ridge.fit(Xtr, Ytr_t)
        val_pred = ridge.predict(Xva)
        val_r2 = r2_score(Yva_t, val_pred, multioutput="uniform_average")
        if val_r2 > best_val:
            best_val = val_r2
            best_alpha = alpha
    ridge = Ridge(alpha=best_alpha, random_state=0)
    ridge.fit(Xtr, Ytr_t)
    return ridge, best_alpha


def run_one_mode(mode: str, X_raw: np.ndarray, Y_raw: np.ndarray,
                 meta: pd.DataFrame,
                 idx_tr: np.ndarray, idx_va: np.ndarray, idx_te: np.ndarray,
                 X2_raw: np.ndarray, Y2_raw: np.ndarray, meta2: pd.DataFrame,
                 seeds=SEEDS, model_k_list=MODEL_K_LIST, eval_k=EVAL_K) -> dict:
    # Residualization (OLS fit on training only)
    if mode == "none":
        Xr, Yr = X_raw.copy(), Y_raw.copy()
        X2r, Y2r = X2_raw.copy(), Y2_raw.copy()
    else:
        C1 = build_confound_matrix(meta, X_raw, mode)
        C2 = build_confound_matrix(meta2, X2_raw, mode)
        Px = residualize_fit(X_raw[idx_tr], C1[idx_tr])
        Py = residualize_fit(Y_raw[idx_tr], C1[idx_tr])
        Xr = residualize_apply(X_raw, C1, Px)
        Yr = residualize_apply(Y_raw, C1, Py)
        # Apply DS1-fit residualization to DS2 (since DS2 is held-out)
        X2r = residualize_apply(X2_raw, C2, Px)
        Y2r = residualize_apply(Y2_raw, C2, Py)

    # Z-score using training statistics
    mx, sx = Xr[idx_tr].mean(0), Xr[idx_tr].std(0) + 1e-8
    my, sy = Yr[idx_tr].mean(0), Yr[idx_tr].std(0) + 1e-8
    Xz = (Xr - mx) / sx
    Yz = (Yr - my) / sy
    X2z = (X2r - mx) / sx
    Y2z = (Y2r - my) / sy

    Xtr, Ytr = Xz[idx_tr], Yz[idx_tr]
    Xva, Yva = Xz[idx_va], Yz[idx_va]
    Xte, Yte = Xz[idx_te], Yz[idx_te]

    results_ds1 = {f"model_k={k}": [] for k in model_k_list}
    results_ds2 = {f"model_k={k}": [] for k in model_k_list}
    results_edge_ds1 = {f"model_k={k}": [] for k in model_k_list}
    results_edge_ds2 = {f"model_k={k}": [] for k in model_k_list}

    for seed in seeds:
        eval_pca = PCA(n_components=eval_k, svd_solver="randomized", random_state=seed)
        eval_pca.fit(Ytr)

        for model_k in model_k_list:
            pca_m = PCA(n_components=model_k, svd_solver="randomized", random_state=seed)
            pca_m.fit(Ytr)
            Ytr_t = pca_m.transform(Ytr)
            Yva_t = pca_m.transform(Yva)

            ridge, best_alpha = fit_ridge_grid_pcspace(Xtr, Ytr_t, Xva, Yva_t, ALPHAS)

            yte_pred_k = ridge.predict(Xte)
            y_ext_pred_k = ridge.predict(X2z)
            yte_full = pca_m.inverse_transform(yte_pred_k)
            yext_full = pca_m.inverse_transform(y_ext_pred_k)

            results_ds1[f"model_k={model_k}"].append(
                pc_r2_mean(Yte, yte_full, eval_pca))
            results_ds2[f"model_k={model_k}"].append(
                pc_r2_mean(Y2z, yext_full, eval_pca))
            # Also capture Edge-R^2 for completeness
            results_edge_ds1[f"model_k={model_k}"].append(
                float(r2_score(Yte, yte_full, multioutput="uniform_average")))
            results_edge_ds2[f"model_k={model_k}"].append(
                float(r2_score(Y2z, yext_full, multioutput="uniform_average")))

    summary = {"mode": mode}
    for model_k in model_k_list:
        arr1 = np.array(results_ds1[f"model_k={model_k}"])
        arr2 = np.array(results_ds2[f"model_k={model_k}"])
        earr1 = np.array(results_edge_ds1[f"model_k={model_k}"])
        earr2 = np.array(results_edge_ds2[f"model_k={model_k}"])
        summary[f"DS1_pc_k{model_k}_mean"] = float(arr1.mean())
        summary[f"DS1_pc_k{model_k}_std"] = float(arr1.std())
        summary[f"DS2_pc_k{model_k}_mean"] = float(arr2.mean())
        summary[f"DS2_pc_k{model_k}_std"] = float(arr2.std())
        summary[f"DS1_edge_k{model_k}_mean"] = float(earr1.mean())
        summary[f"DS2_edge_k{model_k}_mean"] = float(earr2.mean())
    return summary


def main():
    s1 = pd.read_csv(DATA / "meta" / "dataset1_subjects.tsv", sep="\t")
    s2 = pd.read_csv(DATA / "meta" / "dataset2_subjects.tsv", sep="\t")
    ids1 = s1["SubjectID"].astype(str).tolist()
    ids2 = s2["SubjectID"].astype(str).tolist()

    X1_raw = np.load(DATA / "dataset1_X_raw.npy")
    Y1_raw = np.load(DATA / "dataset1_Y_raw.npy")
    X2_raw = np.load(DATA / "dataset2_X_raw.npy")
    Y2_raw = np.load(DATA / "dataset2_Y_raw.npy")

    split1 = json.loads((REPO / "splits" / "dataset1_split.json").read_text())
    idx_tr = _ids_to_idx(ids1, split1["train"])
    idx_va = _ids_to_idx(ids1, split1["val"])
    idx_te = _ids_to_idx(ids1, split1["test"])

    print(f"DS1 shapes: X={X1_raw.shape} Y={Y1_raw.shape}")
    print(f"Split sizes: train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)}")
    print(f"DS2 shapes: X={X2_raw.shape} Y={Y2_raw.shape}\n")
    print("Pipeline: PCA_model(k) -> Ridge -> inverse PCA -> eval PCA(k=20)\n")

    modes = ["none", "age", "age_sex", "age_sex_totgm"]
    all_rows = []
    for mode in modes:
        print(f"=== Mode: {mode} ===")
        row = run_one_mode(mode, X1_raw, Y1_raw, s1,
                           idx_tr, idx_va, idx_te,
                           X2_raw, Y2_raw, s2)
        for model_k in MODEL_K_LIST:
            print(f"  model_k={model_k:2d}  "
                  f"DS1 PC-R^2={row[f'DS1_pc_k{model_k}_mean']:.4f}"
                  f"±{row[f'DS1_pc_k{model_k}_std']:.4f}  "
                  f"DS2={row[f'DS2_pc_k{model_k}_mean']:.4f}"
                  f"±{row[f'DS2_pc_k{model_k}_std']:.4f}")
        all_rows.append(row)
        print()

    (OUT / "M1_resid_sensitivity.json").write_text(
        json.dumps(all_rows, indent=2))
    pd.DataFrame(all_rows).to_csv(OUT / "M1_resid_sensitivity.csv", index=False)
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
