"""
S.D (Appendix): Heteroscedastic spiked-matrix simulation.

Checks whether the O >> R^2 ordering predicted by the spiked-matrix analogy
(Eqs. 8 and 9 in the paper) survives a heteroscedastic noise structure matched
to empirical FNC residuals, rather than i.i.d. Gaussian noise.

Setup:
  - X sampled from the empirical DS1 training GM distribution (standard
    Gaussian since GM is already zero-mean unit-variance after preprocessing).
  - B_0 is a rank-6 matrix built from the observed singular values of a
    Ridge-fit B on the real data (matches BBP-detectable leading-block
    structure reported in the paper).
  - E drawn from N(0, diag(sigma^2_j)) with sigma^2_j matched to the per-edge
    variance of the Nuclear Norm residual on real DS1.

Reports:
  O_sim, R2_sim averaged over n_draws synthetic datasets.
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

SEED = 42
N_DRAWS = 20
N_TRAIN = 805
N_TEST = 173
P = 99
Q = 1378
R_SIGNAL = 6  # rank of planted signal
EVAL_K = 20


def _ids_to_idx(all_ids, target_ids):
    lookup = {sid: i for i, sid in enumerate(all_ids)}
    return np.array([lookup[s] for s in target_ids if s in lookup], dtype=np.int64)


def main():
    # ---- Estimate signal and noise from real data ----
    s1 = pd.read_csv(DATA / "meta" / "dataset1_subjects.tsv", sep="\t")
    ids1 = s1["SubjectID"].astype(str).tolist()
    X1 = np.load(DATA / "dataset1_X.npy").astype(np.float64)
    Y1 = np.load(DATA / "dataset1_Y.npy").astype(np.float64)
    split1 = json.loads((REPO / "splits" / "dataset1_split.json").read_text())
    idx_tr = _ids_to_idx(ids1, split1["train"])
    Xtr, Ytr = X1[idx_tr], Y1[idx_tr]

    # Fit Ridge to get plug-in signal coefficient and residual variance
    ridge = Ridge(alpha=1.0, random_state=0)
    ridge.fit(Xtr, Ytr)
    B_hat = ridge.coef_.T  # (p, q)
    resid = Ytr - Xtr @ B_hat
    sigma2_per_edge = resid.var(axis=0)  # (q,)
    print(f"Per-edge residual variance: mean={sigma2_per_edge.mean():.4f} "
          f"median={np.median(sigma2_per_edge):.4f} "
          f"range=[{sigma2_per_edge.min():.4f}, {sigma2_per_edge.max():.4f}]")

    # Construct a rank-R_SIGNAL B_0 matching the top singular subspace of B_hat
    U, S, Vt = np.linalg.svd(B_hat, full_matrices=False)
    print(f"Top {R_SIGNAL} singular values of B_hat: "
          f"{np.round(S[:R_SIGNAL], 3).tolist()}")
    B_0 = U[:, :R_SIGNAL] @ np.diag(S[:R_SIGNAL]) @ Vt[:R_SIGNAL, :]

    # ---- Simulate ----
    rng = np.random.default_rng(SEED)
    O_list, R2_list = [], []
    for draw in range(N_DRAWS):
        X_sim_tr = rng.standard_normal((N_TRAIN, P))
        X_sim_te = rng.standard_normal((N_TEST, P))
        # Heteroscedastic noise: each edge j has variance sigma2_per_edge[j]
        noise_scale = np.sqrt(sigma2_per_edge)[None, :]
        E_tr = rng.standard_normal((N_TRAIN, Q)) * noise_scale
        E_te = rng.standard_normal((N_TEST, Q)) * noise_scale
        Y_sim_tr = X_sim_tr @ B_0 + E_tr
        Y_sim_te = X_sim_te @ B_0 + E_te

        # Fit Ridge and compute O and R^2 in PC-space, same pipeline as paper
        r = Ridge(alpha=1.0, random_state=0)
        r.fit(X_sim_tr, Y_sim_tr)
        Y_pred_te = r.predict(X_sim_te)

        pca = PCA(n_components=EVAL_K, random_state=SEED)
        pca.fit(Y_sim_tr)
        yt = pca.transform(Y_sim_te)
        yp = pca.transform(Y_pred_te)
        per_pc = r2_score(yt, yp, multioutput="raw_values")
        per_pc = np.where(np.isfinite(per_pc), per_pc, 0.0)
        R2_list.append(float(per_pc.mean()))

        # Subspace overlap between predicted and observed FNC top-k right
        # singular vectors. We take the right-singular vectors of the predicted
        # vs observed Y_test matrices and compute mean squared cosines.
        Up, Sp, Vp = np.linalg.svd(Y_pred_te - Y_pred_te.mean(0),
                                    full_matrices=False)
        Uo, So, Vo = np.linalg.svd(Y_sim_te - Y_sim_te.mean(0),
                                    full_matrices=False)
        Vp_k = Vp[:EVAL_K]
        Vo_k = Vo[:EVAL_K]
        cos_angles = np.linalg.svd(Vp_k @ Vo_k.T, compute_uv=False)
        O_list.append(float(np.mean(cos_angles ** 2)))

    O_sim = np.mean(O_list)
    R2_sim = np.mean(R2_list)
    O_std = np.std(O_list)
    R2_std = np.std(R2_list)
    print(f"\nOver {N_DRAWS} draws:")
    print(f"  O_sim  = {O_sim:.4f} ± {O_std:.4f}")
    print(f"  R2_sim = {R2_sim:.4f} ± {R2_std:.4f}")
    print(f"  Ordering O > R2 holds in {sum(o>r for o,r in zip(O_list,R2_list))}/{N_DRAWS} draws")

    summary = {
        "n_draws": N_DRAWS,
        "rank_signal": R_SIGNAL,
        "O_sim_mean": float(O_sim),
        "O_sim_std": float(O_std),
        "R2_sim_mean": float(R2_sim),
        "R2_sim_std": float(R2_std),
        "O_values": [float(x) for x in O_list],
        "R2_values": [float(x) for x in R2_list],
        "per_edge_sigma2_mean": float(sigma2_per_edge.mean()),
    }
    (OUT / "M5_rmt_simulation.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {OUT / 'M5_rmt_simulation.json'}")


if __name__ == "__main__":
    main()
