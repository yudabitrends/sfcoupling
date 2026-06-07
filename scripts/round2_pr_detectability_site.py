#!/usr/bin/env python3
"""Round-2 P0 analyses C4 (low-D coupling-specific), C5 (detectability + positive
baseline + error bars), C7 (site sensitivity). All GM-side, model-free cross-covariance.

C4: PR of the REAL coupling cross-cov vs the break-pairing SURROGATE cross-cov (same
    marginals, no coupling) vs the i.i.d. Marchenko-Pastur null. If real PR < surrogate
    PR, the low dimensionality is coupling-specific, not inherited from smoothed marginals.
C5: (i) positive conservation baseline = O_6 between two UKB subsamples at clinical N
    (genuinely-shared subspace by construction) vs observed cross-cohort 0.38 vs floor 0.39;
    (ii) bootstrap sin-theta CI on cross-cohort O_6; (iii) detectability: per clinical cohort,
    real leading singular values vs break-pairing-null p95 (empirical BBP edge).
C7: site sensitivity = recompute cross-cohort O_6 after per-cohort mean-centering
    (lightweight harmonization) and report whether the at-floor verdict survives.

Outputs: results/discovery/{lowdim_vs_surrogate_pr,detectability_positive_baseline,site_sensitivity}.json
"""
from __future__ import annotations
import os
import json
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("REPO_ROOT", "."))
UKB = Path(os.environ.get("ALIGNED_DIR", "data/aligned_features_ukb37775"))
CLIN = ROOT / "aligned_features"
OUTD = ROOT / "results/discovery"
KS = (1, 3, 6, 10)
SEED = 20260606


def zc(M):
    return (M - M.mean(0)) / (M.std(0) + 1e-12)


def cross(X, Y):
    return zc(X).T @ zc(Y) / (len(X) - 1)


def svals(C):
    return np.linalg.svd(C, compute_uv=False)


def U_of(C, kmax):
    U, _, _ = np.linalg.svd(C, full_matrices=False)
    return U[:, :kmax]


def overlap(Ua, Ub, k):
    M = Ua[:, :k].T @ Ub[:, :k]
    return float((M ** 2).sum() / k)


def PR(s):
    lam = np.asarray(s, float) ** 2
    return float((lam.sum() ** 2) / (lam ** 2).sum())


def load():
    Xu = np.load(UKB / "dataset1_X_resid.npy").astype(np.float64)
    Yu = np.load(UKB / "dataset1_Y_resid.npy").astype(np.float64)
    Xc = np.load(CLIN / "dataset1_X_resid.npy").astype(np.float64)
    Yc = np.load(CLIN / "dataset1_Y_resid.npy").astype(np.float64)
    return Xu, Yu, Xc, Yc


# ---------------- C4 ----------------
def c4_lowdim(Xu, Yu, reps=40):
    d, q = Xu.shape[1], Yu.shape[1]
    real_pr = PR(svals(cross(Xu, Yu)))
    g = np.random.default_rng(SEED)
    surr = []
    for r in range(reps):
        Yp = Yu[g.permutation(len(Yu))]
        surr.append(PR(svals(cross(Xu, Yp))))
    # i.i.d. Marchenko-Pastur PR for a d x q Gaussian (rank min(d,q)=d): PR of MP spectrum
    # empirical: random Gaussian cross-cov of independent standard normals at same n
    mp = []
    n = len(Xu)
    for r in range(10):
        A = g.standard_normal((n, d)); B = g.standard_normal((n, q))
        mp.append(PR(svals(A.T @ B / (n - 1))))
    return {"real_coupling_PR": round(real_pr, 2),
            "surrogate_breakpair_PR_mean": round(float(np.mean(surr)), 2),
            "surrogate_breakpair_PR_sd": round(float(np.std(surr)), 2),
            "iid_MP_null_PR_mean": round(float(np.mean(mp)), 1),
            "interpretation": "low-D coupling-specific if real_coupling_PR < surrogate_breakpair_PR"}


# ---------------- C5 ----------------
def c5_detectability(Xu, Yu, Xc, Yc):
    d = Xu.shape[1]
    g = np.random.default_rng(SEED + 1)
    res = {}

    # (i) positive baseline: two UKB subsamples at clinical N share the subspace by construction
    pos = {}
    for N in (147, 227, 273, 504):
        ov = []
        for r in range(60):
            gg = np.random.default_rng(SEED + N + r)
            idx = gg.choice(len(Xu), 2 * N, replace=False)
            a, b = idx[:N], idx[N:]
            Ua, Ub = U_of(cross(Xu[a], Yu[a]), max(KS)), U_of(cross(Xu[b], Yu[b]), max(KS))
            ov.append(overlap(Ua, Ub, 6))
        pos[str(N)] = [round(float(np.mean(ov)), 3), round(float(np.quantile(ov, 0.05)), 3),
                       round(float(np.quantile(ov, 0.95)), 3)]
    res["positive_baseline_O6_two_ukb_subsamples_by_N"] = pos
    res["positive_baseline_note"] = ("O6 two genuinely-co-subspace UKB cohorts WOULD show at clinical N; "
                                     "compare to observed cross-cohort O6~0.38 and surrogate floor ~0.39")

    # (ii) bootstrap sin-theta CI on the clinical(pooled) vs UKB cross-cohort O6
    Uu = U_of(cross(Xu, Yu), max(KS))
    boot = []
    for r in range(300):
        gg = np.random.default_rng(SEED + 7 + r)
        bi = gg.integers(0, len(Xc), len(Xc))
        Uc = U_of(cross(Xc[bi], Yc[bi]), max(KS))
        boot.append(overlap(Uc, Uu, 6))
    res["crosscohort_O6_clinical_vs_ukb_bootstrapCI"] = [
        round(float(np.mean(boot)), 3), round(float(np.percentile(boot, 2.5)), 3),
        round(float(np.percentile(boot, 97.5)), 3)]

    # (iii) detectability: per clinical cohort, leading real singular values vs break-pairing null p95
    # cohort labels from DS1-aligned site file (built from gm_paths provenance, row order == DS1)
    import pandas as pd
    coh = pd.read_csv(OUTD / "ds1_sites.csv")["site"].values
    det = {}
    if True:
        for name in ["COBRE", "FBIRN", "PK_MPRC", "ChineseSZ"]:
            m = coh == name
            if m.sum() < 50:
                continue
            Xs, Ys = Xc[m], Yc[m]
            s_real = svals(cross(Xs, Ys))[:6]
            gg = np.random.default_rng(SEED + 99)
            null_top = []
            for r in range(60):
                Yp = Ys[gg.permutation(len(Ys))]
                null_top.append(svals(cross(Xs, Yp))[:6])
            null_p95 = np.quantile(np.array(null_top), 0.95, axis=0)
            n_detect = int((s_real > null_p95).sum())
            det[name] = {"N": int(m.sum()), "n_spikes_above_null_p95_of_6": n_detect,
                         "sigma_real_top3": [round(float(x), 4) for x in s_real[:3]],
                         "null_p95_top3": [round(float(x), 4) for x in null_p95[:3]]}
    res["detectability_per_cohort"] = det
    return res


# ---------------- C7 ----------------
def c7_site(Xu, Yu, Xc, Yc):
    import pandas as pd
    Uu = U_of(cross(Xu, Yu), max(KS))
    base = overlap(U_of(cross(Xc, Yc), max(KS)), Uu, 6)
    # site-centering: subtract per-cohort mean of GM and FNC (removes site location shift)
    coh = pd.Series(pd.read_csv(OUTD / "ds1_sites.csv")["site"].values)
    out = {"crosscohort_O6_no_harmonization": round(base, 3)}
    Xh, Yh = Xc.copy(), Yc.copy()
    for name in coh.unique():
        m = (coh == name).values
        if m.sum() > 1:
            Xh[m] = Xh[m] - Xh[m].mean(0)
            Yh[m] = Yh[m] - Yh[m].mean(0)
    harm = overlap(U_of(cross(Xh, Yh), max(KS)), Uu, 6)
    out["crosscohort_O6_site_centered"] = round(harm, 3)
    out["interpretation"] = ("at-floor verdict survives if site-centered O6 stays near the surrogate "
                             "floor (~0.39) and does not jump toward the matched-N ceiling")
    return out


def main():
    Xu, Yu, Xc, Yc = load()
    print(f"UKB {Xu.shape} clinical {Xc.shape}", flush=True)
    OUTD.mkdir(parents=True, exist_ok=True)

    c4 = c4_lowdim(Xu, Yu)
    (OUTD / "lowdim_vs_surrogate_pr.json").write_text(json.dumps(c4, indent=2))
    print("C4:", json.dumps(c4), flush=True)

    c5 = c5_detectability(Xu, Yu, Xc, Yc)
    (OUTD / "detectability_positive_baseline.json").write_text(json.dumps(c5, indent=2))
    print("C5 positive baseline:", c5["positive_baseline_O6_two_ukb_subsamples_by_N"], flush=True)
    print("C5 O6 bootstrap CI:", c5["crosscohort_O6_clinical_vs_ukb_bootstrapCI"], flush=True)
    print("C5 detectability:", {k: v["n_spikes_above_null_p95_of_6"] for k, v in c5.get("detectability_per_cohort", {}).items()}, flush=True)

    c7 = c7_site(Xu, Yu, Xc, Yc)
    (OUTD / "site_sensitivity.json").write_text(json.dumps(c7, indent=2))
    print("C7:", json.dumps(c7), flush=True)


if __name__ == "__main__":
    main()
