"""
Structured permutation null applied DIRECTLY to the subspace-overlap statistic O
(Reviewer 2.1 / Reviewer 1 cube-vs-sphere).

The manuscript's scrambled-GM null (Table S.B) is reported on PC-R^2. This script
reports the analogous null on O itself: we permute the GM<->FNC subject pairing in
the *training* partition (exactly the scrambled-GM procedure), refit the Ridge map,
form the GM-predicted FNC subspace on the held-out test set, and compute its overlap
with the observed test-FNC subspace. Breaking the pairing destroys the cross-modal
covariance, so the predicted subspace collapses to a random direction set and O falls
to its random-subspace floor (~ k/q). This is the number that distinguishes a genuine
shared subspace from two objects that merely share an ambient space.

NOTE: the `permute_rows` null in train/run_subspace_analysis.py is degenerate for O,
because permuting the *rows* of Y_pred does not change its right singular vectors
(null == observed, std 0). Permutation must happen BEFORE fitting, as done here.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path("/home/users/ybi3/sfcoupling")
sys.path.insert(0, str(ROOT))
from models.baselines import fit_ridge_grid          # noqa: E402
from models.utils import load_config, load_training_contracts, set_seed  # noqa: E402


def top_k(Y, k):
    _, _, Vt = np.linalg.svd(Y, full_matrices=False)
    return Vt[:k].T


def overlap(V1, V2):
    s = np.linalg.svd(V1.T @ V2, compute_uv=False)
    return float(np.mean(np.clip(s, 0.0, 1.0) ** 2))


def main():
    set_seed(42)
    cfg = load_config(str(ROOT / "train" / "config_baselines.yaml"))
    data = load_training_contracts(cfg)
    itr, iva, ite = data["idx1_train"], data["idx1_val"], data["idx1_test"]
    X1, Y1 = data["X1"].astype(np.float64), data["Y1"].astype(np.float64)
    Xtr, Ytr = X1[itr], Y1[itr]
    Xva, Yva = X1[iva], Y1[iva]
    Xte, Yte = X1[ite], Y1[ite]

    alphas = cfg.get("ridge", {}).get("alphas", [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    model, _ = fit_ridge_grid(Xtr, Ytr, Xva, Yva, alphas)
    alpha = float(getattr(model, "alpha", 1.0))
    B = model.coef_.T
    Ypred = Xte @ B

    q = Yte.shape[1]
    ks = [5, 10, 20]
    obs = {k: overlap(top_k(Yte, k), top_k(Ypred, k)) for k in ks}

    n_perm = 500
    rng = np.random.default_rng(42)
    null = {k: [] for k in ks}
    for _ in range(n_perm):
        perm = rng.permutation(Xtr.shape[0])            # break GM<->FNC pairing in TRAIN
        Bp = Ridge(alpha=alpha).fit(Xtr[perm], Ytr).coef_.T
        Yp = Xte @ Bp
        Vp = {k: top_k(Yp, k) for k in ks}
        for k in ks:
            null[k].append(overlap(top_k(Yte, k), Vp[k]))

    out = {"alpha": alpha, "q": int(q), "n_perm": n_perm, "ks": {}}
    for k in ks:
        a = np.array(null[k])
        out["ks"][str(k)] = {
            "observed_O": round(obs[k], 4),
            "null_mean": round(float(a.mean()), 4),
            "null_std": round(float(a.std()), 4),
            "kq_chance": round(k / q, 4),
            "sd_above_null": round(float((obs[k] - a.mean()) / (a.std() + 1e-12)), 1),
            "p_value": round(float((np.sum(a >= obs[k]) + 1) / (n_perm + 1)), 4),
        }
    outdir = ROOT / "results" / "subspace_perm_null"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "overlap_permutation_null.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
