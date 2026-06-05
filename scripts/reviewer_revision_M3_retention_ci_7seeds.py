"""
M3 (R2.3 round-2 revision, FAST version): Subject-level bootstrap with
seed-varying eval PCA.

The main pipeline's seed-to-seed variation comes primarily from the
randomized SVD solver in the evaluation PCA, not from the deterministic
fits of NN/PLS/RRR/Ridge on a fixed train set. This script exploits that:

  1. Fit Nuclear Norm, PLS, RRR, and Ridge ONCE on the deterministic
     train partition.
  2. Compute each method's test and DS2 predictions once.
  3. For each of 7 seeds in {42..48}, fit a seed-specific eval PCA and
     project the (Y_true, Y_pred) pairs into that PC space, storing
     per-subject squared-error / squared-target contributions.
  4. Average per-subject contributions ACROSS the 7 seed eval-PCAs.
  5. Bootstrap over DS2 subjects using the seed-averaged contributions.

Runtime: ~3 minutes total (NN ISTA dominates, ~2 min; others negligible).
Output: results/reviewer_revision/M3_retention_bootstrap_7seeds.json
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

ROOT = Path("/home/users/ybi3/sfcoupling")
DATA = Path("/data/users1/ybi3/cVAE/aligned_features")
OUT = ROOT / "results" / "reviewer_revision"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from train.run_multivariate_methods import fit_nuclear_norm  # noqa: E402

SEEDS = [42, 43, 44, 45, 46, 47, 48]
EVAL_K = 20
N_BOOT = 2000
BOOT_SEED = 20260412


def _ids_to_idx(all_ids, target_ids):
    lookup = {sid: i for i, sid in enumerate(all_ids)}
    return np.array([lookup[s] for s in target_ids if s in lookup],
                    dtype=np.int64)


def pc_r2_per_subject(Y_true, Y_pred, pca):
    yt = pca.transform(Y_true)
    yp = pca.transform(Y_pred)
    err2 = (yt - yp) ** 2
    tgt2 = yt ** 2
    return err2, tgt2


def main():
    # --- Load data once ---
    s1 = pd.read_csv(DATA / "meta" / "dataset1_subjects.tsv", sep="\t")
    ids1 = s1["SubjectID"].astype(str).tolist()
    X1 = np.load(DATA / "dataset1_X.npy").astype(np.float64)
    Y1 = np.load(DATA / "dataset1_Y.npy").astype(np.float64)
    X2 = np.load(DATA / "dataset2_X.npy").astype(np.float64)
    Y2 = np.load(DATA / "dataset2_Y.npy").astype(np.float64)

    split1 = json.loads((ROOT / "splits" / "dataset1_split.json").read_text())
    idx_tr = _ids_to_idx(ids1, split1["train"])
    idx_va = _ids_to_idx(ids1, split1["val"])
    idx_te = _ids_to_idx(ids1, split1["test"])

    Xtr, Ytr = X1[idx_tr], Y1[idx_tr]
    Xva, Yva = X1[idx_va], Y1[idx_va]
    Xte, Yte = X1[idx_te], Y1[idx_te]

    n_te, n_ext = Xte.shape[0], X2.shape[0]
    print(f"train={len(idx_tr)} val={len(idx_va)} test={n_te} ds2={n_ext}")

    # --- Fit each method ONCE (deterministic on fixed train split) ---
    print("\n--- fitting methods (deterministic, single fit) ---")
    preds = {}

    t0 = time.time()
    nn = fit_nuclear_norm(Xtr, Ytr, Xva, Yva,
                           max_iter=1000, tol=1e-5)
    B_nn = nn["B"]
    preds["nuclear_norm"] = (Xte @ B_nn, X2 @ B_nn)
    print(f"  NN lam*={nn['optimal_lambda']:.4f}  elapsed={time.time()-t0:.1f}s",
          flush=True)

    t0 = time.time()
    best_n, best_v_pls, best_pls = None, -np.inf, None
    for nc in range(1, 31):
        pls = PLSRegression(n_components=nc, max_iter=500)
        pls.fit(Xtr, Ytr)
        vp = pls.predict(Xva)
        v = 1.0 - np.mean((Yva - vp) ** 2) / (np.mean(Yva ** 2) + 1e-12)
        if v > best_v_pls:
            best_v_pls, best_n, best_pls = v, nc, pls
    preds["pls"] = (best_pls.predict(Xte), best_pls.predict(X2))
    print(f"  PLS n={best_n}  elapsed={time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    rr = Ridge(alpha=100.0, random_state=0)
    rr.fit(Xtr, Ytr)
    B_full = rr.coef_.T
    U, S, Vt = np.linalg.svd(B_full, full_matrices=False)
    best_r, best_v = 1, -np.inf
    for r in range(1, min(31, len(S) + 1)):
        B_r = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
        vp = Xva @ B_r
        v = 1.0 - np.mean((Yva - vp) ** 2) / (np.mean(Yva ** 2) + 1e-12)
        if v > best_v:
            best_v, best_r = v, r
    B_rrr = U[:, :best_r] @ np.diag(S[:best_r]) @ Vt[:best_r, :]
    preds["rrr"] = (Xte @ B_rrr, X2 @ B_rrr)
    print(f"  RRR r={best_r}  elapsed={time.time()-t0:.1f}s", flush=True)

    # Ridge PC-space target (match main pipeline) — needs a per-seed pca_r
    # because the Ridge model is seeded to the eval_pca. We fit once per seed.
    print(f"  Ridge: fit per-seed (PC-space target)", flush=True)

    # --- Per-seed: fit eval_pca and (for Ridge) pca_r, then collect
    #     per-subject err2/tgt2 into accumulators ---
    method_names = ["nuclear_norm", "pls", "rrr", "ridge"]
    accum = {
        m: {
            "err_te": np.zeros((n_te, EVAL_K)),
            "tgt_te": np.zeros((n_te, EVAL_K)),
            "err_ext": np.zeros((n_ext, EVAL_K)),
            "tgt_ext": np.zeros((n_ext, EVAL_K)),
        }
        for m in method_names
    }

    print("\n--- seed-varying evaluation ---")
    for seed in SEEDS:
        t0 = time.time()

        # Seed-specific eval PCA (randomized SVD)
        eval_pca = PCA(n_components=EVAL_K, svd_solver="randomized",
                        random_state=seed)
        eval_pca.fit(Ytr)

        # Seed-specific Ridge PC-space target (fit per seed to match main
        # pipeline behavior)
        pca_r = PCA(n_components=EVAL_K, svd_solver="randomized",
                     random_state=seed)
        pca_r.fit(Ytr)
        best_a, best_vv = None, -np.inf
        for a in [1e-3, 1e-2, 1e-1, 1, 10, 100]:
            rpc = Ridge(alpha=a, random_state=0)
            rpc.fit(Xtr, pca_r.transform(Ytr))
            p = rpc.predict(Xva)
            v = 1.0 - np.mean((pca_r.transform(Yva) - p) ** 2) / \
                (np.mean(pca_r.transform(Yva) ** 2) + 1e-12)
            if v > best_vv:
                best_vv, best_a = v, a
        rpc = Ridge(alpha=best_a, random_state=0)
        rpc.fit(Xtr, pca_r.transform(Ytr))
        ridge_te_pred = pca_r.inverse_transform(rpc.predict(Xte))
        ridge_ext_pred = pca_r.inverse_transform(rpc.predict(X2))
        preds_seed = dict(preds)
        preds_seed["ridge"] = (ridge_te_pred, ridge_ext_pred)

        for m in method_names:
            te_pred, ext_pred = preds_seed[m]
            err_te, tgt_te = pc_r2_per_subject(Yte, te_pred, eval_pca)
            err_ext, tgt_ext = pc_r2_per_subject(Y2, ext_pred, eval_pca)
            accum[m]["err_te"] += err_te
            accum[m]["tgt_te"] += tgt_te
            accum[m]["err_ext"] += err_ext
            accum[m]["tgt_ext"] += tgt_ext

        print(f"  seed={seed}  elapsed={time.time()-t0:.1f}s", flush=True)

    # Average across seeds
    n_seeds = len(SEEDS)
    for m in method_names:
        for key in accum[m]:
            accum[m][key] /= n_seeds

    # Seed-averaged point estimates
    print("\n--- seed-averaged point estimates ---")
    point_te, point_ext = {}, {}
    for m in method_names:
        te_pc = 1.0 - accum[m]["err_te"].sum(0) / \
                     (accum[m]["tgt_te"].sum(0) + 1e-12)
        ext_pc = 1.0 - accum[m]["err_ext"].sum(0) / \
                      (accum[m]["tgt_ext"].sum(0) + 1e-12)
        point_te[m] = float(te_pc.mean())
        point_ext[m] = float(ext_pc.mean())
        ret = point_ext[m] / point_te[m] if point_te[m] > 0 else float("nan")
        print(f"  {m:14s}  DS1={point_te[m]:.4f}  DS2={point_ext[m]:.4f}  "
              f"retention={ret:.3f}")

    # Bootstrap over subjects
    print(f"\n--- bootstrap (B={N_BOOT}) ---")
    rng = np.random.default_rng(BOOT_SEED)
    te_idxs = rng.integers(0, n_te, size=(N_BOOT, n_te))
    ext_idxs = rng.integers(0, n_ext, size=(N_BOOT, n_ext))

    summary = {
        "n_boot": N_BOOT,
        "n_seeds": n_seeds,
        "seeds": SEEDS,
        "eval_k": EVAL_K,
        "fit_strategy": "single-fit methods with seed-varying eval PCA",
        "methods": {},
    }
    boots = {}
    for m in method_names:
        te_arr = np.empty(N_BOOT)
        ext_arr = np.empty(N_BOOT)
        ret_arr = np.empty(N_BOOT)
        et = accum[m]["err_te"]; tt = accum[m]["tgt_te"]
        ee = accum[m]["err_ext"]; te_ = accum[m]["tgt_ext"]
        for b in range(N_BOOT):
            it = te_idxs[b]; ie = ext_idxs[b]
            te_per = 1.0 - et[it].sum(0) / (tt[it].sum(0) + 1e-12)
            ext_per = 1.0 - ee[ie].sum(0) / (te_[ie].sum(0) + 1e-12)
            te_arr[b] = te_per.mean()
            ext_arr[b] = ext_per.mean()
            ret_arr[b] = ext_arr[b] / te_arr[b] if te_arr[b] > 0 else np.nan
        boots[m] = {"te": te_arr, "ext": ext_arr, "ret": ret_arr}

        ci_te = np.percentile(te_arr, [2.5, 97.5])
        ci_ext = np.percentile(ext_arr, [2.5, 97.5])
        valid = ret_arr[np.isfinite(ret_arr)]
        ci_ret = (np.percentile(valid, [2.5, 97.5])
                  if valid.size else [np.nan, np.nan])

        summary["methods"][m] = {
            "point_te": point_te[m],
            "point_ext": point_ext[m],
            "point_retention": (point_ext[m] / point_te[m]
                                if point_te[m] > 0 else None),
            "boot_te_mean": float(te_arr.mean()),
            "boot_te_ci": [float(ci_te[0]), float(ci_te[1])],
            "boot_ext_mean": float(ext_arr.mean()),
            "boot_ext_ci": [float(ci_ext[0]), float(ci_ext[1])],
            "boot_ret_mean": float(np.nanmean(ret_arr)),
            "boot_ret_ci": [float(ci_ret[0]), float(ci_ret[1])],
        }
        print(f"  {m:14s}  DS1={te_arr.mean():.4f} "
              f"[{ci_te[0]:.4f},{ci_te[1]:.4f}]  "
              f"DS2={ext_arr.mean():.4f} "
              f"[{ci_ext[0]:.4f},{ci_ext[1]:.4f}]  "
              f"ret=[{ci_ret[0]:.3f},{ci_ret[1]:.3f}]")

    # Paired NN vs PLS
    diff_ext = boots["nuclear_norm"]["ext"] - boots["pls"]["ext"]
    p_ext = float(np.mean(diff_ext <= 0))
    ci_diff_ext = np.percentile(diff_ext, [2.5, 97.5])
    summary["paired_nn_vs_pls"] = {
        "ext_diff_mean": float(diff_ext.mean()),
        "ext_diff_ci": [float(ci_diff_ext[0]), float(ci_diff_ext[1])],
        "ext_p_onesided": p_ext,
    }
    print(f"\nNN-PLS DS2 diff = {diff_ext.mean():+.4f}  "
          f"[{ci_diff_ext[0]:+.4f},{ci_diff_ext[1]:+.4f}]  p={p_ext:.4f}")

    (OUT / "M3_retention_bootstrap_7seeds.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nSaved to {OUT / 'M3_retention_bootstrap_7seeds.json'}")


if __name__ == "__main__":
    main()
