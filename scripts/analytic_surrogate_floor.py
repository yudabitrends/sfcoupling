#!/usr/bin/env python3
"""Analytic surrogate floor for the GM-side cross-cohort coupling-subspace overlap.

Matches the manuscript's cross-cohort metric (sec:metrics): GM-side captured-variance
overlap  O_k(U_a,U_b) = (1/k)||U_a[:, :k]^T U_b[:, :k]||_F^2 , U = LEFT singular vectors
of the model-free cross-covariance C = zscore(X)^T zscore(Y)/(n-1).

THEORY (verified here)
----------------------
Under the break-pairing null, X and Y are independent within a cohort, so
  C  ~  (1/sqrt(n-1)) Sigma_X^{1/2} G Sigma_Y^{1/2},   G iid N(0,1) (d x q)   (matrix-normal)
and on the GM side
  E[C C^T] = tr(Sigma_Y) * Sigma_X,
so the GM-side null subspace U concentrates on the leading eigenvectors of the SHARED
structure covariance Sigma_X (99-network NeuroMark basis). Two cohorts sharing the
parcellation share Sigma_X => their null subspaces align => the floor is high BY
CONSTRUCTION, and is governed by the effective rank of Sigma_X (spread by tr(Sigma_Y^2)).

EXACT identity for the expected floor (two independent null cohorts a,b):
  O_k^surr = (1/k) E[ tr(P_a P_b) ] = (1/k) tr( Pbar_a Pbar_b ),
  Pbar = E[ U_{:k} U_{:k}^T ]  (mean rank-k null projector, q-> d here).
For a==b (shared covariance), O_k^surr = (1/k) tr(Pbar^2) in [k/d, 1].

Verifies on REAL UKB + clinical covariances:
  (i)   real break-pairing floor ~= matrix-normal simulation        (model correct)
  (ii)  exact projector identity tr(Pbar_a Pbar_b)/k ~= floor        (analytic object)
  (iii) floor ~independent of cohort size n                          (scale-free)
  (iv)  reproduces the manuscript cross-cohort floor (~0.39)         (clinical vs UKB)
  (v)   Sigma_X effective rank governs the floor (Sigma_Y=I probe)
contrasted with the Haar null (~k/d) the manuscript otherwise leans on.

Output: results/discovery/analytic_surrogate_floor_gmside.json
"""
from __future__ import annotations
import os
import json
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("REPO_ROOT", "."))
UKB = Path(os.environ.get("ALIGNED_DIR", "data/aligned_features_ukb37775"))
CLIN = Path(os.environ.get("CLIN_ALIGNED_DIR", "data/aligned_features"))
OUT = ROOT / "results/discovery/analytic_surrogate_floor_gmside.json"
KS = (1, 3, 6, 10)
SEED = 20260605


def corr(M):
    Z = (M - M.mean(0)) / (M.std(0) + 1e-12)
    return (Z.T @ Z / (len(M) - 1)).astype(np.float64)


def sym_sqrt(S):
    w, V = np.linalg.eigh(S)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def U_from_C(C, kmax):
    U, _, _ = np.linalg.svd(C, full_matrices=False)   # left singular vectors (d-dim)
    return U[:, :kmax]


def overlap(Ua, Ub, k):
    M = Ua[:, :k].T @ Ub[:, :k]
    return float((M ** 2).sum() / k)


def zc(M):
    return (M - M.mean(0)) / (M.std(0) + 1e-12)


def real_floor_cross(Xa, Ya, Xb, Yb, reps=100, seed=SEED):
    """Break GM<->FNC pairing in each cohort, GM-side U overlap a<->b."""
    out = {k: [] for k in KS}
    na, nb = len(Xa), len(Xb)
    for r in range(reps):
        g = np.random.default_rng(seed + r)
        Ca = zc(Xa).T @ zc(Ya)[g.permutation(na)] / (na - 1)
        Cb = zc(Xb).T @ zc(Yb)[g.permutation(nb)] / (nb - 1)
        Ua, Ub = U_from_C(Ca, max(KS)), U_from_C(Cb, max(KS))
        for k in KS:
            out[k].append(overlap(Ua, Ub, k))
    return {k: [float(np.mean(v)), float(np.std(v)), float(np.quantile(v, 0.95))]
            for k, v in out.items()}


def real_floor_self(X, Y, n_sub, reps=80, seed=SEED):
    """Two disjoint same-cohort subsamples, broken pairing, GM-side U overlap."""
    n = len(X)
    out = {k: [] for k in KS}
    for r in range(reps):
        g = np.random.default_rng(seed + r)
        idx = g.choice(n, 2 * n_sub, replace=False)
        a, b = idx[:n_sub], idx[n_sub:]
        Ca = zc(X[a]).T @ zc(Y[a])[g.permutation(n_sub)] / (n_sub - 1)
        Cb = zc(X[b]).T @ zc(Y[b])[g.permutation(n_sub)] / (n_sub - 1)
        Ua, Ub = U_from_C(Ca, max(KS)), U_from_C(Cb, max(KS))
        for k in KS:
            out[k].append(overlap(Ua, Ub, k))
    return {k: [float(np.mean(v)), float(np.std(v)), float(np.quantile(v, 0.95))]
            for k, v in out.items()}


def mn_floor_and_projector(SX, SY, d, reps=200, seed=SEED, identity_y=False):
    """Matrix-normal GM-side floor (two iid draws) AND the mean projector Pbar.
    Uses C C^T = Ax (G SY G^T) Ax to avoid the 1378x1378 sqrt; G is d x q."""
    Ax = sym_sqrt(SX)
    SYuse = np.eye(SY.shape[0]) if identity_y else SY
    g = np.random.default_rng(seed + 314)
    q = SY.shape[0]
    Pbar = np.zeros((d, d))
    Us = []
    for r in range(reps):
        G = g.standard_normal((d, q))
        M = Ax @ (G @ SYuse @ G.T) @ Ax            # d x d ~ C C^T
        w, V = np.linalg.eigh(M)
        U = V[:, ::-1][:, :max(KS)]                # top-kmax eigenvectors = U
        Us.append(U)
        Pbar += U[:, :max(KS)] @ U[:, :max(KS)].T
    Pbar /= reps
    # pairwise floor over independent draws
    out = {k: [] for k in KS}
    gp = np.random.default_rng(seed + 999)
    for _ in range(reps):
        a, b = gp.integers(0, reps, 2)
        for k in KS:
            out[k].append(overlap(Us[a], Us[b], k))
    floor = {k: float(np.mean(v)) for k, v in out.items()}
    # exact projector identity (a==b shared covariance): tr(Pbar_k^2)/k
    proj = {}
    for k in KS:
        Pk = np.zeros((d, d))
        for U in Us:
            Pk += U[:, :k] @ U[:, :k].T
        Pk /= reps
        proj[k] = float(np.trace(Pk @ Pk) / k)
    return floor, proj


def main():
    Xu = np.load(UKB / "dataset1_X_resid.npy").astype(np.float32)
    Yu = np.load(UKB / "dataset1_Y_resid.npy").astype(np.float32)
    Xc = np.load(CLIN / "dataset1_X_resid.npy").astype(np.float32)
    Yc = np.load(CLIN / "dataset1_Y_resid.npy").astype(np.float32)
    d = Xu.shape[1]
    print(f"UKB N={len(Xu)}  clinical N={len(Xc)}  d={d}  q={Yu.shape[1]}", flush=True)

    SXu, SYu = corr(Xu), corr(Yu)
    SXc, SYc = corr(Xc), corr(Yc)
    muX = np.linalg.eigvalsh(SXu)[::-1]
    prX = float((muX.sum() ** 2) / (muX ** 2).sum())
    print(f"Sigma_X (GM) participation ratio = {prX:.1f}  (d={d})", flush=True)

    res = {"design": "GM-side analytic surrogate floor (matrix-normal model + projector identity)",
           "n_ukb": len(Xu), "n_clin": len(Xc), "d": d, "q": Yu.shape[1],
           "sigma_X_participation_ratio": round(prX, 2),
           "sigma_X_eig_top8": [float(x) for x in muX[:8]],
           "haar_floor_kd": {k: round(k / d, 4) for k in KS}}

    # (iv) cross-cohort clinical(1151) vs UKB — matches the manuscript's floor
    res["real_floor_clinical_vs_ukb"] = real_floor_cross(Xc, Yc, Xu, Yu)
    # (i)+(ii) matrix-normal model + projector identity, clinical-size, real covariances
    res["mn_floor_realSXY"], res["projector_identity"] = mn_floor_and_projector(SXc, SYc, d)
    # (v) Sigma_Y = I probe
    mnI, _ = mn_floor_and_projector(SXc, SYc, d, identity_y=True)
    res["mn_floor_identitySY"] = mnI
    # (iii) self floor vs cohort size
    res["real_self_floor_vs_n"] = {str(ns): real_floor_self(Xu, Yu, ns, reps=50)
                                   for ns in (150, 500, 1500)}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))

    print("\n=== GM-side floor verification (O_k) — manuscript floor target ~0.39 @k6 ===")
    print(f"{'k':>3} | {'REAL clin-UKB':>14} | {'MNorm realSXY':>14} | {'proj tr(P^2)/k':>15} | {'MNorm SY=I':>11} | {'Haar k/d':>9}")
    for k in KS:
        rr = res["real_floor_clinical_vs_ukb"][k][0]
        mn = res["mn_floor_realSXY"][k]
        pj = res["projector_identity"][k]
        mi = res["mn_floor_identitySY"][k]
        ha = res["haar_floor_kd"][k]
        print(f"{k:>3} | {rr:>14.3f} | {mn:>14.3f} | {pj:>15.3f} | {mi:>11.3f} | {ha:>9.4f}")
    print("\n=== self floor vs cohort size n (should be ~flat) ===")
    for ns in ("150", "500", "1500"):
        row = res["real_self_floor_vs_n"][ns]
        print(f"  n={ns:>4}: " + "  ".join(f"k{k}={row[k][0]:.3f}" for k in KS))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
