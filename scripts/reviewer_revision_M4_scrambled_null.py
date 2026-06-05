"""
M4: Scrambled-GM null baseline for the NeuroImage revision.

Permutes rows of X (breaking the GM->FNC pairing) and retrains Ridge and a
PC-space Ridge (matching Table 3 pipeline) under the null. Reports the null
floor for DS1 and DS2 PC-R^2 across 200 permutations.

This directly addresses Major Comment M4: the current paper's PC-R^2 pivot
(Edge-R^2 negative on DS2 for 5/7 methods) could reflect overfitting rather
than real coupling. If the scrambled-GM null is near zero while the observed
PC-R^2 is far above it, the PC-R^2 advantage is genuine.

Run from repo root:
    python scripts/reviewer_revision_M4_scrambled_null.py

Outputs:
    results/reviewer_revision/M4_scrambled_null.json
    results/reviewer_revision/M4_scrambled_null.csv
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

N_PERM = 200
MODEL_K_LIST = [5, 10, 20]
EVAL_K = 20
ALPHAS = [1e-3, 1e-2, 1e-1, 1, 10, 100]
RNG_SEED = 42


def _ids_to_idx(all_ids: list[str], target_ids: list[str]) -> np.ndarray:
    lookup = {sid: i for i, sid in enumerate(all_ids)}
    return np.array([lookup[s] for s in target_ids if s in lookup], dtype=np.int64)


def pc_r2_mean(y_true: np.ndarray, y_pred: np.ndarray, pca: PCA) -> float:
    yt = pca.transform(y_true)
    yp = pca.transform(y_pred)
    per_pc = r2_score(yt, yp, multioutput="raw_values")
    per_pc = np.asarray(per_pc, dtype=np.float64)
    per_pc = np.where(np.isfinite(per_pc), per_pc, 0.0)
    return float(per_pc.mean())


def fit_ridge_pc(Xtr, Ytr_t, Xva, Yva_t, alphas):
    best_alpha, best_val = None, -np.inf
    for alpha in alphas:
        ridge = Ridge(alpha=alpha, random_state=0)
        ridge.fit(Xtr, Ytr_t)
        val_pred = ridge.predict(Xva)
        val_r2 = r2_score(Yva_t, val_pred, multioutput="uniform_average")
        if val_r2 > best_val:
            best_val = val_r2
            best_alpha = alpha
    ridge = Ridge(alpha=best_alpha, random_state=0)
    ridge.fit(Xtr, Ytr_t)
    return ridge


def eval_ridge_one_seed(Xtr, Ytr, Xva, Yva, Xte, Yte, X2, Y2,
                        seed: int, model_k: int, eval_pca: PCA):
    pca_m = PCA(n_components=model_k, svd_solver="randomized", random_state=seed)
    pca_m.fit(Ytr)
    Ytr_t = pca_m.transform(Ytr)
    Yva_t = pca_m.transform(Yva)
    ridge = fit_ridge_pc(Xtr, Ytr_t, Xva, Yva_t, ALPHAS)
    yte_full = pca_m.inverse_transform(ridge.predict(Xte))
    yext_full = pca_m.inverse_transform(ridge.predict(X2))
    return (pc_r2_mean(Yte, yte_full, eval_pca),
            pc_r2_mean(Y2, yext_full, eval_pca))


def main():
    s1 = pd.read_csv(DATA / "meta" / "dataset1_subjects.tsv", sep="\t")
    s2 = pd.read_csv(DATA / "meta" / "dataset2_subjects.tsv", sep="\t")
    ids1 = s1["SubjectID"].astype(str).tolist()

    # Use the residualized + standardized arrays that are used by the main pipeline
    X1 = np.load(DATA / "dataset1_X.npy")
    Y1 = np.load(DATA / "dataset1_Y.npy")
    X2 = np.load(DATA / "dataset2_X.npy")
    Y2 = np.load(DATA / "dataset2_Y.npy")

    split1 = json.loads((REPO / "splits" / "dataset1_split.json").read_text())
    idx_tr = _ids_to_idx(ids1, split1["train"])
    idx_va = _ids_to_idx(ids1, split1["val"])
    idx_te = _ids_to_idx(ids1, split1["test"])

    Xtr, Ytr = X1[idx_tr], Y1[idx_tr]
    Xva, Yva = X1[idx_va], Y1[idx_va]
    Xte, Yte = X1[idx_te], Y1[idx_te]

    print(f"DS1 train/val/test: {len(idx_tr)}/{len(idx_va)}/{len(idx_te)}  "
          f"DS2 ext: {X2.shape[0]}")
    print(f"Running {N_PERM} GM-row permutations x {len(MODEL_K_LIST)} k x 1 seed\n")

    # Reference (unpermuted) values
    eval_pca = PCA(n_components=EVAL_K, svd_solver="randomized", random_state=42)
    eval_pca.fit(Ytr)

    observed = {}
    for model_k in MODEL_K_LIST:
        ds1, ds2 = eval_ridge_one_seed(Xtr, Ytr, Xva, Yva, Xte, Yte, X2, Y2,
                                        seed=42, model_k=model_k, eval_pca=eval_pca)
        observed[f"k={model_k}"] = {"DS1": ds1, "DS2": ds2}
        print(f"  observed (no permute) k={model_k}: "
              f"DS1={ds1:.4f}  DS2={ds2:.4f}")

    # Scrambled-GM null: permute training X rows (breaks GM<->FNC pairing),
    # keep test/val/DS2 unchanged so the metric still measures "can Ridge
    # predict FNC from a randomly-relabeled GM mapping".
    rng = np.random.default_rng(RNG_SEED)
    null_results = {f"k={k}": {"DS1": [], "DS2": []} for k in MODEL_K_LIST}

    for p in range(N_PERM):
        perm = rng.permutation(Xtr.shape[0])
        Xtr_perm = Xtr[perm]
        # Validation and test use unpermuted X; the "mapping" was learned from
        # scrambled training only, so the null measures what Ridge can recover
        # after the structural signal is destroyed.
        for model_k in MODEL_K_LIST:
            ds1, ds2 = eval_ridge_one_seed(Xtr_perm, Ytr, Xva, Yva, Xte, Yte,
                                            X2, Y2, seed=42, model_k=model_k,
                                            eval_pca=eval_pca)
            null_results[f"k={model_k}"]["DS1"].append(ds1)
            null_results[f"k={model_k}"]["DS2"].append(ds2)
        if (p + 1) % 25 == 0:
            print(f"  permutation {p+1}/{N_PERM}")

    # Summaries
    summary = {"n_perm": N_PERM, "observed": observed, "null": {}}
    for model_k in MODEL_K_LIST:
        arr1 = np.array(null_results[f"k={model_k}"]["DS1"])
        arr2 = np.array(null_results[f"k={model_k}"]["DS2"])
        obs1 = observed[f"k={model_k}"]["DS1"]
        obs2 = observed[f"k={model_k}"]["DS2"]
        # One-sided p: fraction of permutations >= observed
        p1 = (np.sum(arr1 >= obs1) + 1) / (N_PERM + 1)
        p2 = (np.sum(arr2 >= obs2) + 1) / (N_PERM + 1)
        summary["null"][f"k={model_k}"] = {
            "DS1_null_mean": float(arr1.mean()),
            "DS1_null_std": float(arr1.std()),
            "DS1_null_min": float(arr1.min()),
            "DS1_null_max": float(arr1.max()),
            "DS1_null_p_onesided": float(p1),
            "DS2_null_mean": float(arr2.mean()),
            "DS2_null_std": float(arr2.std()),
            "DS2_null_min": float(arr2.min()),
            "DS2_null_max": float(arr2.max()),
            "DS2_null_p_onesided": float(p2),
        }
        print(f"\nk={model_k}:")
        print(f"  DS1 obs={obs1:.4f}   null={arr1.mean():.4f}±{arr1.std():.4f}   "
              f"range [{arr1.min():.4f}, {arr1.max():.4f}]   p={p1:.4f}")
        print(f"  DS2 obs={obs2:.4f}   null={arr2.mean():.4f}±{arr2.std():.4f}   "
              f"range [{arr2.min():.4f}, {arr2.max():.4f}]   p={p2:.4f}")

    (OUT / "M4_scrambled_null.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {OUT / 'M4_scrambled_null.json'}")


if __name__ == "__main__":
    main()
