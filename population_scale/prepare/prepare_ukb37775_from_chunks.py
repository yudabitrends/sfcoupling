#!/usr/bin/env python3
"""
NC Phase 1 W3 — assemble UK Biobank 37,775 aligned features from 64 pre-extracted
feature chunks (chunk_0{00..63}.npz).

Each chunk ~591 subjects with 99D GM + 1378D FNC + age + sex. All chunks status=ok.

Split: 80% DS1 (~30,220) with CV 70/15/15 train/val/test; 20% DS2 (~7,555)
as external. Deterministic via RNG_SEED.

Paths are configurable via environment variables (defaults are repo-relative):
  UKB_CHUNK_DIR  source chunk_0NN.npz dir    (default: data/ukb_features/chunks)
  ALIGNED_DIR    output aligned_features dir  (default: data/aligned_features_ukb37775)
  SPLITS_DIR     output splits dir            (default: data/splits_ukb37775)
Subject-level UK Biobank data are not distributed with this repository; obtain
them via the UK Biobank Access Management System.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

CHUNK_DIR = Path(os.environ.get("UKB_CHUNK_DIR", "data/ukb_features/chunks"))
DST_AF = Path(os.environ.get("ALIGNED_DIR", "data/aligned_features_ukb37775"))
DST_SP = Path(os.environ.get("SPLITS_DIR", "data/splits_ukb37775"))

RNG_SEED = 20260419


def main():
    DST_AF.mkdir(parents=True, exist_ok=True)
    (DST_AF / "meta").mkdir(parents=True, exist_ok=True)
    (DST_AF / "meta" / "feature_maps").mkdir(parents=True, exist_ok=True)
    DST_SP.mkdir(parents=True, exist_ok=True)

    chunks = sorted(CHUNK_DIR.glob("chunk_*.npz"))
    print(f"Found {len(chunks)} chunks at {CHUNK_DIR}")
    assert len(chunks) == 64, f"Expected 64 chunks, got {len(chunks)}"

    Xs, Ys, As, Ss, eids_all = [], [], [], [], []
    for c in chunks:
        d = np.load(c)
        status = d["status"].astype(str)
        keep = np.array([s.lower() == "ok" for s in status])
        Xs.append(d["X_gm"][keep])
        Ys.append(d["Y_fnc"][keep])
        As.append(d["ages"][keep])
        Ss.append(d["sexes"][keep])
        eids_all.append(d["eids"][keep].astype(str))

    X = np.concatenate(Xs).astype(np.float32)
    Y = np.concatenate(Ys).astype(np.float32)
    A = np.concatenate(As).astype(np.float32)
    Sx = np.concatenate(Ss).astype(np.float32)
    eids = np.concatenate(eids_all)
    N = X.shape[0]
    print(f"Aggregated: N={N}  X={X.shape}  Y={Y.shape}")

    _, uniq_idx = np.unique(eids, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    if len(uniq_idx) != N:
        print(f"  deduped {N - len(uniq_idx)} repeat eids → N={len(uniq_idx)}")
    X, Y, A, Sx, eids = X[uniq_idx], Y[uniq_idx], A[uniq_idx], Sx[uniq_idx], eids[uniq_idx]
    N = X.shape[0]

    rng = np.random.default_rng(RNG_SEED)
    perm = rng.permutation(N)
    ds1_n = int(N * 0.8)
    idx_ds1 = perm[:ds1_n]
    idx_ds2 = perm[ds1_n:]

    X1, Y1, A1, Sx1, sid1 = X[idx_ds1], Y[idx_ds1], A[idx_ds1], Sx[idx_ds1], eids[idx_ds1]
    X2, Y2, A2, Sx2, sid2 = X[idx_ds2], Y[idx_ds2], A[idx_ds2], Sx[idx_ds2], eids[idx_ds2]
    N1, N2 = len(sid1), len(sid2)
    print(f"DS1={N1}  DS2={N2}")

    rng_split = np.random.default_rng(RNG_SEED + 1)
    perm1 = rng_split.permutation(N1)
    n_tr = int(N1 * 0.70)
    n_va = int(N1 * 0.15)
    tr, va, te = perm1[:n_tr], perm1[n_tr:n_tr + n_va], perm1[n_tr + n_va:]
    print(f"CV: tr/va/te = {len(tr)}/{len(va)}/{len(te)}")

    cov1 = np.stack([A1, Sx1], axis=1)
    cov2 = np.stack([A2, Sx2], axis=1)

    muX = X1[tr].mean(axis=0, keepdims=True)
    sdX = X1[tr].std(axis=0, keepdims=True) + 1e-8
    X1_z = ((X1 - muX) / sdX).astype(np.float32)
    X2_z = ((X2 - muX) / sdX).astype(np.float32)
    muY = Y1[tr].mean(axis=0, keepdims=True)
    sdY = Y1[tr].std(axis=0, keepdims=True) + 1e-8
    Y1_z = ((Y1 - muY) / sdY).astype(np.float32)
    Y2_z = ((Y2 - muY) / sdY).astype(np.float32)
    D_tr = np.hstack([np.ones((len(tr), 1)), cov1[tr].astype(np.float64)])
    betaY, *_ = np.linalg.lstsq(D_tr, Y1[tr].astype(np.float64), rcond=None)
    D1 = np.hstack([np.ones((N1, 1)), cov1.astype(np.float64)])
    D2 = np.hstack([np.ones((N2, 1)), cov2.astype(np.float64)])
    Y1_resid = (Y1.astype(np.float64) - D1 @ betaY).astype(np.float32)
    Y2_resid = (Y2.astype(np.float64) - D2 @ betaY).astype(np.float32)

    def save(name, arr):
        np.save(DST_AF / name, arr)
        print(f"  wrote {name}  shape={arr.shape}")

    save("dataset1_X.npy", X1_z);     save("dataset1_X_raw.npy", X1.astype(np.float32));  save("dataset1_X_resid.npy", X1_z)
    save("dataset1_Y.npy", Y1_z);     save("dataset1_Y_raw.npy", Y1.astype(np.float32));  save("dataset1_Y_resid.npy", Y1_resid)
    save("dataset2_X.npy", X2_z);     save("dataset2_X_raw.npy", X2.astype(np.float32));  save("dataset2_X_resid.npy", X2_z)
    save("dataset2_Y.npy", Y2_z);     save("dataset2_Y_raw.npy", Y2.astype(np.float32));  save("dataset2_Y_resid.npy", Y2_resid)

    def write_meta(path, ids, ages, sexes):
        with open(path, "w") as f:
            f.write("SubjectID\tAge\tGender\tDiagnosis\n")
            for s, a, g in zip(ids, ages, sexes):
                f.write(f"{s}\t{float(a):.2f}\t{float(g):.1f}\t0\n")
        print(f"  wrote {path}")

    write_meta(DST_AF / "meta" / "dataset1_subjects.tsv", sid1, A1, Sx1)
    write_meta(DST_AF / "meta" / "dataset2_subjects.tsv", sid2, A2, Sx2)

    (DST_AF / "meta" / "feature_maps" / "gm_feature_names.txt").write_text(
        "\n".join([f"gm_{i}" for i in range(99)]) + "\n"
    )
    edges = [f"ic{i}-ic{j}" for i in range(53) for j in range(i + 1, 53)]
    assert len(edges) == 1378
    (DST_AF / "meta" / "feature_maps" / "fnc_edge_names.txt").write_text("\n".join(edges) + "\n")

    split1 = {
        "train": [sid1[i] for i in tr],
        "val":   [sid1[i] for i in va],
        "test":  [sid1[i] for i in te],
    }
    (DST_SP / "dataset1_split.json").write_text(json.dumps(split1))
    (DST_SP / "dataset2_split.json").write_text(json.dumps({"train": [], "val": [], "test": list(sid2)}))

    manifest = {
        "source": str(CHUNK_DIR), "rng_seed": RNG_SEED,
        "n_total": int(N), "ds1_n": int(N1), "ds2_n": int(N2),
        "split": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "dx": 99, "dy": 1378, "d_fnc_side": 53,
        "age_mean": float(A.mean()), "age_std": float(A.std()),
        "n_female": int((Sx == 0).sum()),
    }
    (DST_AF / "meta" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. Manifest: {manifest}")


if __name__ == "__main__":
    main()
