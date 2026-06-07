#!/usr/bin/env python3
"""C4 rigor — proper bootstrap 95% CI for the GM<->FNC coupling participation ratio
at the canonical large N, and a formal one-sided permutation test that the coupling
PR is BELOW the break-pairing surrogate PR.

Replicates model_free_dim.py's cross-covariance construction EXACTLY (logE tangent +
Age/Gender residualization, C = X_c^T T_c / N, PR from singular spectrum), builds
(X_c, T_c) ONCE at N=30220, then:
  (1) subject bootstrap (B resamples of rows) -> PR distribution -> 95% percentile CI
      [fixes the degenerate hand-set CI 5.2/[5.2,5.8]]
  (2) break-pairing surrogate: permute the GM<->FNC subject pairing (B perms),
      recompute PR_surr -> distribution; one-sided p = P(PR_surr <= PR_obs), and the
      separation between the observed-PR bootstrap CI and the surrogate-PR CI.

Output: results/reviewer_revision/pr_bootstrap_surrogate.json
"""
from __future__ import annotations
import os
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.rsfe import RSFE, RSFEConfig                       # noqa: E402
from preprocess.harmonize import apply_tangent_residualizer    # noqa: E402

FEAT = Path(os.environ.get("ALIGNED_DIR", "data/aligned_features_ukb37775"))
OUT = PROJECT_ROOT / "results/reviewer_revision/pr_bootstrap_surrogate.json"
D = 53
N = 30220
B = 1000
SEED = 20260607


def fisher_z_to_spd(Y, d):
    n = Y.shape[0]; iu = np.triu_indices(d, k=1); fc = np.tanh(Y)
    F = np.zeros((n, d, d), np.float32)
    for i in range(n):
        m = np.zeros((d, d), np.float32); m[iu] = fc[i]
        F[i] = m + m.T + np.eye(d, dtype=np.float32)
    return F + 1e-6 * np.eye(d, dtype=np.float32)[None]


def strat(meta, n_sel, rng):
    meta = meta.copy()
    meta["ad"] = pd.qcut(meta["Age"], 10, labels=False, duplicates="drop")
    meta["s"] = meta["ad"].astype(str) + "_" + meta["Gender"].astype(int).astype(str)
    sel = []; vc = meta["s"].value_counts(); tot = vc.sum()
    for st, c in vc.items():
        sub = meta.index[meta["s"] == st].values
        k = min(max(1, int(round(n_sel * c / tot))), len(sub))
        sel.append(rng.choice(sub, k, replace=False))
    sel = np.concatenate(sel)
    if len(sel) > n_sel:
        sel = rng.choice(sel, n_sel, replace=False)
    return np.sort(sel)


def pr_from_cross(Xc, Tc):
    """Participation ratio of the cross-covariance singular spectrum (eigenvalue form)."""
    C = Xc.T @ Tc / Xc.shape[0]
    sig = np.linalg.svd(C, compute_uv=False)
    lam = sig ** 2
    return float((lam.sum() ** 2) / (lam ** 2).sum())


def main():
    t0 = time.time()
    X = np.load(FEAT / "dataset1_X.npy").astype(np.float32)
    Y = np.load(FEAT / "dataset1_Y_raw.npy").astype(np.float32)
    meta = pd.read_csv(FEAT / "meta/dataset1_subjects.tsv", sep="\t")
    rng = np.random.default_rng(SEED)
    idx = strat(meta, N, rng)
    cov = meta.loc[idx, ["Age", "Gender"]].values.astype(np.float64)
    print(f"[{time.time()-t0:.0f}s] building canonical tangent at N={len(idx)} (once)...", flush=True)
    Yspd = fisher_z_to_spd(Y[idx], D)
    m = RSFE(RSFEConfig(d=D, nn_lambda_grid=(0.3,), metric="logE")).fit(
        X[idx].astype(np.float64), Yspd, S_val=X[idx].astype(np.float64), F_val_spd=Yspd,
        covariates_train=cov, covariates_val=cov)
    T = apply_tangent_residualizer(m._to_tangent(Yspd).astype(np.float64), cov, m.harmonize_beta)
    Tc = (T - m.mean_t).astype(np.float64)
    Xc = (X[idx].astype(np.float64) - m.mean_s)
    n = Xc.shape[0]
    pr_obs = pr_from_cross(Xc, Tc)
    print(f"[{time.time()-t0:.0f}s] observed PR = {pr_obs:.3f}  (model_free_dim.json reports 5.10)", flush=True)

    # (1) subject bootstrap CI
    print(f"[{time.time()-t0:.0f}s] bootstrap CI ({B} resamples)...", flush=True)
    boot = np.empty(B)
    for b in range(B):
        r = rng.integers(0, n, n)
        boot[b] = pr_from_cross(Xc[r], Tc[r])
        if (b + 1) % 200 == 0:
            print(f"    boot {b+1}/{B}", flush=True)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    # (2) break-pairing surrogate distribution + one-sided test
    print(f"[{time.time()-t0:.0f}s] surrogate PR distribution ({B} permutations)...", flush=True)
    surr = np.empty(B)
    for b in range(B):
        perm = rng.permutation(n)
        surr[b] = pr_from_cross(Xc, Tc[perm])
        if (b + 1) % 200 == 0:
            print(f"    surr {b+1}/{B}", flush=True)
    surr_lo, surr_hi = np.percentile(surr, [2.5, 97.5])
    # one-sided p that the surrogate PR is as LOW as the observed coupling PR
    p_one_sided = float((np.sum(surr <= pr_obs) + 1) / (B + 1))

    res = {
        "design": "Subject-bootstrap 95% CI of the coupling participation ratio and a "
                  "break-pairing permutation test that coupling PR < surrogate PR, at N=30220, "
                  "same logE-tangent + Age/Gender residualized cross-covariance as model_free_dim.py.",
        "N": int(n), "B": B, "seed": SEED,
        "pr_observed": round(pr_obs, 3),
        "pr_bootstrap_ci95": [round(float(ci_lo), 3), round(float(ci_hi), 3)],
        "pr_bootstrap_mean": round(float(boot.mean()), 3),
        "surrogate_pr_mean": round(float(surr.mean()), 3),
        "surrogate_pr_ci95": [round(float(surr_lo), 3), round(float(surr_hi), 3)],
        "p_one_sided_coupling_below_surrogate": round(p_one_sided, 4),
        "separation": "observed PR CI upper bound vs surrogate PR CI lower bound",
        "obs_ci_hi": round(float(ci_hi), 3), "surr_ci_lo": round(float(surr_lo), 3),
        "cis_disjoint": bool(ci_hi < surr_lo),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"[{time.time()-t0:.0f}s] DONE")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
