#!/usr/bin/env python3
"""
Phase 1.1 smoke: adapt the UK Biobank pilot aligned-feature arrays
(ukb_X_gm.npy, ukb_Y_fnc.npy, ukb_ages.npy, ukb_sexes.npy, ukb_subject_ids.tsv)
to the run_rscm.py expected layout (dataset1_* + dataset2_* + meta + splits).

Paths are configurable via environment variables (defaults are repo-relative):
  UKB_PILOT_DIR  source aligned-feature arrays   (default: data/ukb_pilot)
  ALIGNED_DIR    output aligned_features dir      (default: data/aligned_features_ukb1079)
  SPLITS_DIR     output splits dir                (default: data/splits_ukb1079)
Subject-level UK Biobank data are not distributed with this repository; obtain
them via the UK Biobank Access Management System.

Split 1,079 UKB subjects -> 880 "dataset1" (CV: 616 train / 132 val / 132 test)
+ 199 "dataset2" (all external). Deterministic via seed.

Writes z-scored (.npy), raw (.npy.raw), and tangent-space residualized (.npy.resid)
variants of X and Y so the existing runner's config flags `x_variant` /
`y_variant` / harmonization all work unmodified.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

SRC = Path(os.environ.get("UKB_PILOT_DIR", "data/ukb_pilot"))
DST_AF = Path(os.environ.get("ALIGNED_DIR", "data/aligned_features_ukb1079"))
DST_SP = Path(os.environ.get("SPLITS_DIR", "data/splits_ukb1079"))

RNG_SEED = 20260419
DS1_N = 880  # 616/132/132
DS2_N = 199


def zscore_train_apply(train: np.ndarray, *others: np.ndarray):
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True) + 1e-8
    out_train = (train - mu) / sd
    out_others = tuple(((o - mu) / sd) for o in others)
    return (out_train.astype(np.float32), *tuple(o.astype(np.float32) for o in out_others))


def residualize_ols(X_train: np.ndarray, Y_train: np.ndarray, *X_others: np.ndarray):
    """OLS residualization: Y_train_resid = Y_train - D_tr @ beta, fit on train."""
    D_tr = np.hstack([np.ones((X_train.shape[0], 1)), X_train.astype(np.float64)])
    Y_tr64 = Y_train.astype(np.float64)
    beta, *_ = np.linalg.lstsq(D_tr, Y_tr64, rcond=None)
    out = []
    for Xo in (X_train, *X_others):
        Do = np.hstack([np.ones((Xo.shape[0], 1)), Xo.astype(np.float64)])
        yo = Y_tr64 if Xo is X_train else None
        pred = Do @ beta
        if Xo is X_train:
            res = (Y_tr64 - pred).astype(np.float32)
        else:
            res = None
        out.append((pred, res))
    return beta, out  # caller matches up splits


def main():
    DST_AF.mkdir(parents=True, exist_ok=True)
    (DST_AF / "meta").mkdir(parents=True, exist_ok=True)
    (DST_AF / "meta" / "feature_maps").mkdir(parents=True, exist_ok=True)
    DST_SP.mkdir(parents=True, exist_ok=True)

    X = np.load(SRC / "ukb_X_gm.npy")      # (1079, 99)
    Y = np.load(SRC / "ukb_Y_fnc.npy")     # (1079, 1378)
    A = np.load(SRC / "ukb_ages.npy")      # (1079,)
    S = np.load(SRC / "ukb_sexes.npy")     # (1079,) {0,1}
    lines = [l.strip() for l in (SRC / "ukb_subject_ids.tsv").read_text().splitlines() if l.strip()]
    sid = lines[1:] if lines and lines[0].lower() in ("subject_id", "subjectid") else lines
    assert X.shape == (1079, 99) and Y.shape == (1079, 1378) and len(sid) == 1079, \
        f"shape mismatch: X={X.shape} Y={Y.shape} sid={len(sid)}"

    rng = np.random.default_rng(RNG_SEED)
    perm = rng.permutation(1079)
    idx_ds1 = perm[:DS1_N]
    idx_ds2 = perm[DS1_N:DS1_N + DS2_N]

    def slice_(arr, idx): return arr[idx]
    X1, Y1, A1, Sx1 = slice_(X, idx_ds1), slice_(Y, idx_ds1), slice_(A, idx_ds1), slice_(S, idx_ds1)
    sid1 = [sid[i] for i in idx_ds1]
    X2, Y2, A2, Sx2 = slice_(X, idx_ds2), slice_(Y, idx_ds2), slice_(A, idx_ds2), slice_(S, idx_ds2)
    sid2 = [sid[i] for i in idx_ds2]

    # Reproducible DS1 split (seed-independent from above RNG)
    rng_split = np.random.default_rng(RNG_SEED + 1)
    perm1 = rng_split.permutation(DS1_N)
    n_tr, n_va = 616, 132  # test is the rest (132)
    tr, va, te = perm1[:n_tr], perm1[n_tr:n_tr + n_va], perm1[n_tr + n_va:]

    # Covariates for residualization (Age + Sex, demeaned per split)
    cov1 = np.stack([A1, Sx1], axis=1)  # (880, 2)
    cov2 = np.stack([A2, Sx2], axis=1)

    # --- X variants ---
    X1_raw = X1.astype(np.float32)
    X2_raw = X2.astype(np.float32)
    # z-score fit on DS1-train
    X1_z, X2_z = zscore_train_apply(X1_raw[tr], X1_raw, X2_raw)[1:]  # outputs: tr applied to full X1 and X2
    # fix: need to re-call properly
    mu = X1_raw[tr].mean(axis=0, keepdims=True)
    sd = X1_raw[tr].std(axis=0, keepdims=True) + 1e-8
    X1_z = ((X1_raw - mu) / sd).astype(np.float32)
    X2_z = ((X2_raw - mu) / sd).astype(np.float32)
    # residualize GM on nothing (there is no prior here — X is the predictor)
    # We keep X_resid = X_z (no confound to remove from X itself in this smoke);
    # downstream thread-2 harmonization happens to Y in tangent space.
    X1_resid = X1_z.copy()
    X2_resid = X2_z.copy()

    # --- Y variants ---
    Y1_raw = Y1.astype(np.float32)
    Y2_raw = Y2.astype(np.float32)
    # z-score Y (edge-wise) on DS1-train
    muY = Y1_raw[tr].mean(axis=0, keepdims=True)
    sdY = Y1_raw[tr].std(axis=0, keepdims=True) + 1e-8
    Y1_z = ((Y1_raw - muY) / sdY).astype(np.float32)
    Y2_z = ((Y2_raw - muY) / sdY).astype(np.float32)
    # Residualize Y on [1, Age, Sex] fit on DS1-train
    D1_tr = np.hstack([np.ones((n_tr, 1)), cov1[tr].astype(np.float64)])
    betaY, *_ = np.linalg.lstsq(D1_tr, Y1_raw[tr].astype(np.float64), rcond=None)
    D1_all = np.hstack([np.ones((DS1_N, 1)), cov1.astype(np.float64)])
    D2_all = np.hstack([np.ones((DS2_N, 1)), cov2.astype(np.float64)])
    Y1_resid = (Y1_raw.astype(np.float64) - D1_all @ betaY).astype(np.float32)
    Y2_resid = (Y2_raw.astype(np.float64) - D2_all @ betaY).astype(np.float32)

    # --- Save npy ---
    def save(name, arr):
        np.save(DST_AF / name, arr)
        print(f"  wrote {name}  shape={arr.shape}  dtype={arr.dtype}")

    save("dataset1_X.npy", X1_z);          save("dataset1_X_raw.npy", X1_raw);    save("dataset1_X_resid.npy", X1_resid)
    save("dataset1_Y.npy", Y1_z);          save("dataset1_Y_raw.npy", Y1_raw);    save("dataset1_Y_resid.npy", Y1_resid)
    save("dataset2_X.npy", X2_z);          save("dataset2_X_raw.npy", X2_raw);    save("dataset2_X_resid.npy", X2_resid)
    save("dataset2_Y.npy", Y2_z);          save("dataset2_Y_raw.npy", Y2_raw);    save("dataset2_Y_resid.npy", Y2_resid)

    # --- Meta TSVs (SubjectID, Age, Gender, Diagnosis) ---
    def write_meta(path, ids, ages, sexes):
        with open(path, "w") as f:
            f.write("SubjectID\tAge\tGender\tDiagnosis\n")
            for s, a, g in zip(ids, ages, sexes):
                f.write(f"{s}\t{float(a):.2f}\t{float(g):.1f}\t0\n")
        print(f"  wrote {path}")

    write_meta(DST_AF / "meta" / "dataset1_subjects.tsv", sid1, A1, Sx1)
    write_meta(DST_AF / "meta" / "dataset2_subjects.tsv", sid2, A2, Sx2)

    # --- Feature names ---
    (DST_AF / "meta" / "feature_maps" / "gm_feature_names.txt").write_text(
        "\n".join([f"gm_{i}" for i in range(99)]) + "\n"
    )
    # 53-IC upper triangle edges: (i,j) with i<j, 53*52/2 = 1378
    edges = [f"ic{i}-ic{j}" for i in range(53) for j in range(i + 1, 53)]
    assert len(edges) == 1378
    (DST_AF / "meta" / "feature_maps" / "fnc_edge_names.txt").write_text("\n".join(edges) + "\n")

    # --- Splits ---
    split1 = {
        "train": [sid1[i] for i in tr],
        "val":   [sid1[i] for i in va],
        "test":  [sid1[i] for i in te],
    }
    (DST_SP / "dataset1_split.json").write_text(json.dumps(split1, indent=2))
    split2 = {"train": [], "val": [], "test": sid2}  # external uses all, so this is nominal
    (DST_SP / "dataset2_split.json").write_text(json.dumps(split2, indent=2))
    print(f"  wrote splits: DS1 tr/va/te = {len(tr)}/{len(va)}/{len(te)};  DS2 ext = {len(sid2)}")

    # --- Manifest ---
    manifest = {
        "source": str(SRC),
        "rng_seed": RNG_SEED,
        "ds1_n": DS1_N, "ds2_n": DS2_N,
        "split": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "dx": 99, "dy": 1378, "d_fnc_side": 53,
        "age_ds1_mean": float(A1.mean()), "age_ds2_mean": float(A2.mean()),
        "n_female_ds1": int((Sx1 == 0).sum()), "n_female_ds2": int((Sx2 == 0).sum()),
    }
    (DST_AF / "meta" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
