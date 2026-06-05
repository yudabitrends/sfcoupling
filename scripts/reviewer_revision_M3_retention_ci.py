"""
M3/M9: Subject-level bootstrap CI on DS2 generalization retention ratio.

The paper reports "74% retention for Nuclear Norm vs 45% for PLS" on DS2 (N=102)
as if these were stable point estimates. Reviewer M3 requests a bootstrap 95% CI
over DS2 subjects to show how tightly the retention ratio is estimated; M9 asks
for subject-level (rather than seed-level) inference for any between-method
comparison.

This script:
  1. Refits Nuclear Norm, PLS, RRR, and Ridge at seed 42 using the repo's own
     residualized+standardized arrays, so the resulting point estimates match
     the manuscript Table 1 numbers.
  2. Saves per-subject PC-space squared-error contributions for DS1-test and
     DS2-external.
  3. Bootstraps DS2 subjects (B=2000) to derive a 95% CI on
     DS2 PC-R^2 and on the DS2/DS1 retention ratio.
  4. Reports paired subject-level bootstrap p-values for NN > PLS.

Run from repo root:
    python scripts/reviewer_revision_M3_retention_ci.py

Outputs:
    results/reviewer_revision/M3_retention_bootstrap.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


REPO = Path("/home/users/ybi3/sfcoupling")
DATA = Path("/data/users1/ybi3/cVAE/aligned_features")
OUT = REPO / "results" / "reviewer_revision"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO))
from train.run_multivariate_methods import fit_nuclear_norm  # noqa: E402


SEED = 42
EVAL_K = 20
N_BOOT = 2000
RNG_SEED = 20260411


def _ids_to_idx(all_ids, target_ids):
    lookup = {sid: i for i, sid in enumerate(all_ids)}
    return np.array([lookup[s] for s in target_ids if s in lookup], dtype=np.int64)


def pc_r2_per_subject(Y_true: np.ndarray, Y_pred: np.ndarray, pca: PCA
                      ) -> np.ndarray:
    """Return per-subject contribution to PC-R^2 (averaged over k components).

    For each subject i and component j, the squared error e_{ij}^2 and the
    squared target t_{ij}^2 together determine R^2_j = 1 - sum_i e^2 / sum_i t^2.
    We return a (N, 2, k) tensor so that bootstrap resampling over subjects
    can recompute a valid R^2 per draw.
    """
    yt = pca.transform(Y_true)  # (N, k)
    yp = pca.transform(Y_pred)  # (N, k)
    err2 = (yt - yp) ** 2
    tgt2 = yt ** 2  # equivalent to (yt - mean)^2 since PCA-transformed data is already centered on training mean
    return err2, tgt2


def bootstrap_pc_r2(err2: np.ndarray, tgt2: np.ndarray,
                    rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """Subject-level bootstrap of PC-R^2.

    For each bootstrap draw, resample rows of err2/tgt2 with replacement,
    compute per-component R^2 = 1 - sum_i err2 / sum_i tgt2, and average
    over components. Returns an array of length n_boot.
    """
    n = err2.shape[0]
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        num = err2[idx].sum(axis=0)
        den = tgt2[idx].sum(axis=0) + 1e-12
        per_pc = 1.0 - num / den
        out[b] = per_pc.mean()
    return out


def fit_pls(Xtr, Ytr, Xva, Yva, component_grid):
    """Fit PLS, select n_components by validation uniform R^2."""
    best_n, best_val = None, -np.inf
    best_pls = None
    for n_comp in component_grid:
        if n_comp > min(Xtr.shape[1], Xtr.shape[0] - 1):
            continue
        pls = PLSRegression(n_components=n_comp, max_iter=1000)
        pls.fit(Xtr, Ytr)
        val_pred = pls.predict(Xva)
        # Simple aggregate R^2 across outputs
        err = np.mean((Yva - val_pred) ** 2)
        var = np.mean(Yva ** 2)  # centered Y assumption
        val_r2 = 1.0 - err / (var + 1e-12)
        if val_r2 > best_val:
            best_val = val_r2
            best_n = n_comp
            best_pls = pls
    return best_pls, best_n


def fit_ridge_alphagrid(Xtr, Ytr, Xva, Yva, alphas):
    best_a, best_v = None, -np.inf
    for a in alphas:
        r = Ridge(alpha=a, random_state=0)
        r.fit(Xtr, Ytr)
        p = r.predict(Xva)
        err = np.mean((Yva - p) ** 2)
        var = np.mean(Yva ** 2)
        v = 1.0 - err / (var + 1e-12)
        if v > best_v:
            best_v = v
            best_a = a
    r = Ridge(alpha=best_a, random_state=0)
    r.fit(Xtr, Ytr)
    return r, best_a


def main():
    s1 = pd.read_csv(DATA / "meta" / "dataset1_subjects.tsv", sep="\t")
    ids1 = s1["SubjectID"].astype(str).tolist()
    # Use the main-pipeline residualized+standardized arrays
    X1 = np.load(DATA / "dataset1_X.npy").astype(np.float64)
    Y1 = np.load(DATA / "dataset1_Y.npy").astype(np.float64)
    X2 = np.load(DATA / "dataset2_X.npy").astype(np.float64)
    Y2 = np.load(DATA / "dataset2_Y.npy").astype(np.float64)

    split1 = json.loads((REPO / "splits" / "dataset1_split.json").read_text())
    idx_tr = _ids_to_idx(ids1, split1["train"])
    idx_va = _ids_to_idx(ids1, split1["val"])
    idx_te = _ids_to_idx(ids1, split1["test"])

    Xtr, Ytr = X1[idx_tr], Y1[idx_tr]
    Xva, Yva = X1[idx_va], Y1[idx_va]
    Xte, Yte = X1[idx_te], Y1[idx_te]

    print(f"train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)} "
          f"ds2_ext={X2.shape[0]}")

    # Fit eval PCA on training Y (shared across methods)
    eval_pca = PCA(n_components=EVAL_K, svd_solver="randomized", random_state=SEED)
    eval_pca.fit(Ytr)

    methods = {}

    # ---- Nuclear Norm ----
    print("\nFitting Nuclear Norm (ISTA)...")
    t0 = time.time()
    nn = fit_nuclear_norm(Xtr, Ytr, Xva, Yva, max_iter=2000, tol=1e-6)
    B_nn = nn["B"]
    print(f"  lambda*={nn['optimal_lambda']:.4f}  "
          f"elapsed={time.time()-t0:.1f}s")
    methods["nuclear_norm"] = {
        "te_pred": Xte @ B_nn,
        "ext_pred": X2 @ B_nn,
    }

    # ---- PLS ----
    print("Fitting PLS...")
    t0 = time.time()
    pls, pls_n = fit_pls(Xtr, Ytr, Xva, Yva,
                         component_grid=list(range(1, 31)))
    print(f"  n_comp={pls_n}  elapsed={time.time()-t0:.1f}s")
    methods["pls"] = {
        "te_pred": pls.predict(Xte),
        "ext_pred": pls.predict(X2),
    }

    # ---- RRR via Ridge + SVD truncation ----
    print("Fitting RRR...")
    t0 = time.time()
    # Follow run_multivariate_methods protocol: Ridge fit then truncate SVD
    ridge_full = Ridge(alpha=100.0, random_state=0)
    ridge_full.fit(Xtr, Ytr)
    B_full = ridge_full.coef_.T  # (dx, dy)
    # Find best rank on validation set
    U, S, Vt = np.linalg.svd(B_full, full_matrices=False)
    best_r, best_v_rrr = 1, -np.inf
    for r in range(1, min(31, len(S) + 1)):
        B_r = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
        vp = Xva @ B_r
        v = 1.0 - np.mean((Yva - vp) ** 2) / (np.mean(Yva ** 2) + 1e-12)
        if v > best_v_rrr:
            best_v_rrr = v
            best_r = r
    B_rrr = U[:, :best_r] @ np.diag(S[:best_r]) @ Vt[:best_r, :]
    print(f"  best rank={best_r}  elapsed={time.time()-t0:.1f}s")
    methods["rrr"] = {
        "te_pred": Xte @ B_rrr,
        "ext_pred": X2 @ B_rrr,
    }

    # ---- Ridge baseline (PC-space, model_k=20) ----
    print("Fitting Ridge (PC-space model_k=20)...")
    t0 = time.time()
    pca_r = PCA(n_components=20, svd_solver="randomized", random_state=SEED)
    pca_r.fit(Ytr)
    ridge_pc, alpha = fit_ridge_alphagrid(Xtr, pca_r.transform(Ytr),
                                          Xva, pca_r.transform(Yva),
                                          [1e-3, 1e-2, 1e-1, 1, 10, 100])
    te_pred_pc = ridge_pc.predict(Xte)
    ext_pred_pc = ridge_pc.predict(X2)
    te_pred_r = pca_r.inverse_transform(te_pred_pc)
    ext_pred_r = pca_r.inverse_transform(ext_pred_pc)
    print(f"  alpha={alpha}  elapsed={time.time()-t0:.1f}s")
    methods["ridge"] = {
        "te_pred": te_pred_r,
        "ext_pred": ext_pred_r,
    }

    # ---- Compute per-subject err2/tgt2 for all methods ----
    print("\n--- Point estimates (PC-R^2 at eval_k=20) ---")
    per_method = {}
    for name, d in methods.items():
        err_te, tgt_te = pc_r2_per_subject(Yte, d["te_pred"], eval_pca)
        err_ext, tgt_ext = pc_r2_per_subject(Y2, d["ext_pred"], eval_pca)
        r2_te = 1.0 - err_te.sum(0) / (tgt_te.sum(0) + 1e-12)
        r2_ext = 1.0 - err_ext.sum(0) / (tgt_ext.sum(0) + 1e-12)
        per_method[name] = {
            "err_te": err_te, "tgt_te": tgt_te,
            "err_ext": err_ext, "tgt_ext": tgt_ext,
            "point_te": float(r2_te.mean()),
            "point_ext": float(r2_ext.mean()),
        }
        print(f"  {name:14s}  DS1-test={r2_te.mean():.4f}   "
              f"DS2-ext={r2_ext.mean():.4f}   "
              f"retention={r2_ext.mean()/r2_te.mean() if r2_te.mean()>0 else float('nan'):.3f}")

    # ---- Subject-level bootstrap (paired over DS2, independent over DS1) ----
    print(f"\n--- Bootstrap ({N_BOOT} draws) over test subjects ---")
    rng = np.random.default_rng(RNG_SEED)
    summary = {"n_boot": N_BOOT, "eval_k": EVAL_K, "seed": SEED, "methods": {}}

    # Keep same bootstrap indices across methods for paired comparisons
    n_te, n_ext = Yte.shape[0], Y2.shape[0]
    te_idxs = rng.integers(0, n_te, size=(N_BOOT, n_te))
    ext_idxs = rng.integers(0, n_ext, size=(N_BOOT, n_ext))

    boots = {}
    for name, m in per_method.items():
        te_arr = np.empty(N_BOOT)
        ext_arr = np.empty(N_BOOT)
        ret_arr = np.empty(N_BOOT)
        for b in range(N_BOOT):
            it = te_idxs[b]
            ie = ext_idxs[b]
            te_per = 1.0 - m["err_te"][it].sum(0) / (m["tgt_te"][it].sum(0) + 1e-12)
            ext_per = 1.0 - m["err_ext"][ie].sum(0) / (m["tgt_ext"][ie].sum(0) + 1e-12)
            te_arr[b] = te_per.mean()
            ext_arr[b] = ext_per.mean()
            ret_arr[b] = (ext_arr[b] / te_arr[b]) if te_arr[b] > 0 else np.nan
        boots[name] = {"te": te_arr, "ext": ext_arr, "ret": ret_arr}
        ci_te = np.percentile(te_arr, [2.5, 97.5])
        ci_ext = np.percentile(ext_arr, [2.5, 97.5])
        valid_ret = ret_arr[np.isfinite(ret_arr)]
        ci_ret = np.percentile(valid_ret, [2.5, 97.5])
        summary["methods"][name] = {
            "point_te": m["point_te"],
            "point_ext": m["point_ext"],
            "point_retention": m["point_ext"] / m["point_te"] if m["point_te"] > 0 else None,
            "boot_te_mean": float(te_arr.mean()),
            "boot_te_ci": [float(ci_te[0]), float(ci_te[1])],
            "boot_ext_mean": float(ext_arr.mean()),
            "boot_ext_ci": [float(ci_ext[0]), float(ci_ext[1])],
            "boot_ret_mean": float(np.nanmean(ret_arr)),
            "boot_ret_ci": [float(ci_ret[0]), float(ci_ret[1])],
            "boot_ret_frac_above_1": float(np.mean(valid_ret > 1.0)),
        }
        print(f"  {name:14s}  DS1={te_arr.mean():.4f} [{ci_te[0]:.4f}, {ci_te[1]:.4f}]"
              f"  DS2={ext_arr.mean():.4f} [{ci_ext[0]:.4f}, {ci_ext[1]:.4f}]"
              f"  retention={np.nanmean(ret_arr):.3f} "
              f"[{ci_ret[0]:.3f}, {ci_ret[1]:.3f}]")

    # Paired bootstrap comparisons
    print("\n--- Paired NN vs PLS (subject-level bootstrap) ---")
    nn_b = boots["nuclear_norm"]
    pls_b = boots["pls"]
    diff_ext = nn_b["ext"] - pls_b["ext"]
    diff_ret = nn_b["ret"] - pls_b["ret"]
    valid = np.isfinite(diff_ret)
    p_ext = float(np.mean(diff_ext <= 0))
    p_ret = float(np.mean(diff_ret[valid] <= 0))
    ci_ext_diff = np.percentile(diff_ext, [2.5, 97.5])
    ci_ret_diff = np.percentile(diff_ret[valid], [2.5, 97.5])
    summary["paired_nn_vs_pls"] = {
        "ext_diff_mean": float(diff_ext.mean()),
        "ext_diff_ci": [float(ci_ext_diff[0]), float(ci_ext_diff[1])],
        "ext_p_onesided": p_ext,
        "ret_diff_mean": float(np.nanmean(diff_ret)),
        "ret_diff_ci": [float(ci_ret_diff[0]), float(ci_ret_diff[1])],
        "ret_p_onesided": p_ret,
    }
    print(f"  NN - PLS DS2 PC-R^2 diff = {diff_ext.mean():+.4f}  "
          f"[{ci_ext_diff[0]:+.4f}, {ci_ext_diff[1]:+.4f}]   p={p_ext:.4f}")
    print(f"  NN - PLS retention diff = {np.nanmean(diff_ret):+.3f}  "
          f"[{ci_ret_diff[0]:+.3f}, {ci_ret_diff[1]:+.3f}]   p={p_ret:.4f}")

    (OUT / "M3_retention_bootstrap.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {OUT / 'M3_retention_bootstrap.json'}")


if __name__ == "__main__":
    main()
