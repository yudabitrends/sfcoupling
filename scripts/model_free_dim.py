#!/usr/bin/env python3
"""R2 + R4 — model-free effective dimensionality of the GM<->FNC coupling vs N.

NO nuclear-norm fit. At each N we build the residualized tangent targets T_c and centered
GM features X_c (same logE + Age/Gender pipeline), form the cross-covariance C = X_c^T T_c / n
(99 x 1431), and characterize its singular spectrum with three model-free / principled
measures:
  R2: participation ratio  PR = (sum sigma^2)^2 / sum sigma^4   (soft rank of the coupling)
      Roy-Vetterli effective rank  erank = exp(-sum p_i log p_i), p_i = sigma_i / sum sigma_j
  R4: Gavish-Donoho optimal hard threshold (unknown-noise median formula) -> GD signal rank
These are independent of the estimator, so if PR / erank / GD-rank also sit low (~ the
nuclear-norm rank-7), the low dimensionality is a property of the coupling, not of the penalty.

Output: results/reviewer_revision/model_free_dim.json
"""
from __future__ import annotations
import os
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.rsfe import RSFE, RSFEConfig  # noqa: E402
from preprocess.harmonize import apply_tangent_residualizer  # noqa: E402

FEAT = Path(os.environ.get("ALIGNED_DIR", "data/aligned_features_ukb37775"))
OUT = PROJECT_ROOT / "results/reviewer_revision/model_free_dim.json"
D = 53
N_TOTALS = [1079, 2000, 4000, 8000, 11820, 30220]


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
    if len(sel) > n_sel: sel = rng.choice(sel, n_sel, replace=False)
    return np.sort(sel)


def gd_omega(beta):
    # Gavish-Donoho 2014, unknown-noise median coefficient lambda(beta) for the
    # optimal hard threshold tau = omega(beta) * median(singular values).
    return 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43


def measures(sig):
    s = np.asarray(sig, float); s = s[s > 0]
    lam = s**2
    pr = float((lam.sum()**2) / (lam**2).sum())          # participation ratio (on eigenvalues)
    p = s / s.sum(); erank = float(np.exp(-(p * np.log(p)).sum()))  # Roy-Vetterli erank
    beta = 99.0 / 1431.0
    tau = gd_omega(beta) * np.median(s)
    gd = int((s > tau).sum())
    return pr, erank, gd


def main():
    X = np.load(FEAT / "dataset1_X.npy").astype(np.float32)
    Y = np.load(FEAT / "dataset1_Y_raw.npy").astype(np.float32)
    meta = pd.read_csv(FEAT / "meta/dataset1_subjects.tsv", sep="\t")
    print("model-free dimensionality of GM<->FNC cross-covariance vs N", flush=True)
    rows = []
    for N in N_TOTALS:
        rng = np.random.default_rng(20260603 + N)
        idx = strat(meta, N, rng)
        cov = meta.loc[idx, ["Age", "Gender"]].values.astype(np.float64)
        Yspd = fisher_z_to_spd(Y[idx], D)
        # Build the canonical tangent (logE + Age/Gender residualization) by fitting RSFE once
        m = RSFE(RSFEConfig(d=D, nn_lambda_grid=(0.3,), metric="logE")).fit(
            X[idx].astype(np.float64), Yspd, S_val=X[idx].astype(np.float64), F_val_spd=Yspd,
            covariates_train=cov, covariates_val=cov)
        T = apply_tangent_residualizer(m._to_tangent(Yspd).astype(np.float64), cov, m.harmonize_beta)
        T_c = T - m.mean_t
        X_c = X[idx].astype(np.float64) - m.mean_s
        C = X_c.T @ T_c / N                       # 99 x 1431 cross-covariance, NO fit
        sig = np.linalg.svd(C, compute_uv=False)
        pr, erank, gd = measures(sig)
        rows.append({"N_total": N, "participation_ratio": round(pr, 2),
                     "roy_vetterli_erank": round(erank, 2), "gavish_donoho_rank": gd,
                     "sig_top10": [float(x) for x in sig[:10]]})
        print(f"  N={N}: PR={pr:.1f}  erank={erank:.1f}  GD-rank={gd}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "design": "Model-free: participation ratio + Roy-Vetterli erank + Gavish-Donoho hard "
                  "threshold on the GM<->FNC cross-covariance singular spectrum, no nuclear-norm fit.",
        "nuclear_norm_rank_reference": 7, "rows": rows}, indent=2))
    print("\nMODEL-FREE DIMENSIONALITY vs N (nuclear-norm rank=7 reference):")
    print("  PR:    " + "  ".join(f"N{r['N_total']}={r['participation_ratio']}" for r in rows))
    print("  erank: " + "  ".join(f"N{r['N_total']}={r['roy_vetterli_erank']}" for r in rows))
    print("  GD:    " + "  ".join(f"N{r['N_total']}={r['gavish_donoho_rank']}" for r in rows))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
