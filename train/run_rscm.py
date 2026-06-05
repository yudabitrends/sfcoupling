#!/usr/bin/env python3
"""
Riemannian Spiked Coupling Model (RSCM): GM -> FNC on the SPD manifold.

This is the MIA-follow-up entry point. Maps structural features to FNC by
fitting a low-rank linear operator in the SPD tangent space at the Frechet
mean (RSFE), then evaluating predictions in both parent-metric vector form
(for direct comparison with the NeuroImage paper's Nuclear Norm baseline)
and in three manifold-native distances (AIRM, Log-Euclidean, Bures-Wasserstein).

At rank 1 with the flat (Euclidean) metric, the fit degenerates to the
vectorized Nuclear Norm solution, giving a narrative-continuity sanity check.

Usage:
    python train/run_rscm.py \
        --config train/config_rscm_smoke.yaml \
        --seeds 42 43 44 \
        --out_dir results/rscm_smoke
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.metrics import (
    fit_pca_on_train,
    pc_space_r2_from_pca,
    r2_summary,
    spd_manifold_summary,
)
from models.rsfe import RSFE, RSFEConfig
from models.utils import load_config, load_training_contracts, save_json, set_seed
from preprocess.features_fnc_spd import (
    batch_spd_to_vec,
    batch_vec_to_spd,
    make_spd_batch,
)
from preprocess.harmonize import load_covariates


def _load_variant(base: Path, dataset: str, variant: str) -> np.ndarray:
    """Load X/Y variants. For SPD reconstruction we need Y in valid
    Fisher-z range, so the 'resid' variant (residualized, not z-scored)
    is the default target. z-scored aligned_features{_X,_Y}.npy would
    saturate tanh() during inverse Fisher-z.
    """
    if variant == "default":
        suffix = ""
    elif variant in ("raw", "resid"):
        suffix = f"_{variant}"
    else:
        raise ValueError(f"Unknown variant {variant}; expected default|raw|resid")
    return np.load(base / f"{dataset}{suffix}.npy")

T_TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
           6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def _stats(values):
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    mean = float(np.mean(arr))
    if n <= 1:
        return {"mean": mean, "std": float("nan"), "ci95": float("nan")}
    std = float(np.std(arr, ddof=1))
    t = T_TABLE.get(n - 1, 2.0 if n - 1 < 30 else 1.96)
    return {"mean": mean, "std": std, "ci95": float(t * std / math.sqrt(n))}


def _y_to_spd(
    Y: np.ndarray, d: int, fisher_z: bool, min_eig: float = 1e-6
) -> np.ndarray:
    """Upper-triangle Fisher-z vector (N, d(d-1)/2) -> SPD (N, d, d).

    Raw Fisher-z (before any per-edge residualization) reconstructs to
    100% PD correlation matrices — they are Gram matrices of unit-norm
    ICA timecourses by construction. A small `make_spd_batch` floor
    (1e-6) handles float32 quantization, nothing more invasive.

    Per-edge residualization destroys the correlation structure (empirical:
    12% negative eigenvalues on Y_resid); the RSCM paper's contribution
    is to defer residualization to the tangent space, so this function
    expects PD-valid Y as input.
    """
    F = batch_vec_to_spd(Y, d=d, fisher_z=fisher_z)
    return make_spd_batch(F, min_eig=min_eig)


def evaluate_rscm(
    model: RSFE,
    X: np.ndarray,
    Y_vec: np.ndarray,
    Y_spd: np.ndarray,
    Y_train_vec: np.ndarray,
    pca_ks: List[int],
    seed: int,
    fisher_z: bool = True,
    calibration: Optional[Dict] = None,
    covariates: Optional[np.ndarray] = None,
    Y_resid_vec: Optional[np.ndarray] = None,
    Y_train_resid_vec: Optional[np.ndarray] = None,
) -> Dict:
    """Evaluate RSCM on a split using both vector- and SPD-form metrics.

    Vector form: predict_vec -> (N, q=d(d-1)/2) matches main-paper Nuclear
    Norm metric space (edge R^2, PC-space R^2 at several k).

    SPD form: predict_spd -> (N, d, d) evaluated with AIRM / LogE / BW.

    The AIRM exp-map strongly compresses predicted off-diagonal Fisher-z
    (empirical scale ratio ~0.2x on DS1). A per-edge linear calibration
    (a_e * y_pred_e + b_e) fit on training targets restores the scale
    without changing the subspace structure (affine, scale-equivariant).
    """
    Y_vec_pred = model.predict_vec(X, fisher_z=fisher_z, covariates=covariates)
    Y_spd_pred = model.predict_spd(X, covariates=covariates)

    # "Strict" predictions: X-only (drop covariate re-projection). This is
    # the cleanest head-to-head with Euclidean NN on Y_resid, because both
    # methods predict the X-explainable signal alone.
    if covariates is not None and model.harmonize_beta is not None:
        Y_vec_pred_strict = model.predict_vec(
            X, fisher_z=fisher_z,
            covariates=np.zeros_like(covariates),
        )
    else:
        Y_vec_pred_strict = Y_vec_pred

    if calibration is not None:
        Y_vec_pred = Y_vec_pred * calibration["scale"] + calibration["offset"]
        Y_vec_pred_strict = (
            Y_vec_pred_strict * calibration["scale"] + calibration["offset"]
        )

    edge_r2 = r2_summary(Y_vec, Y_vec_pred)
    pc_r2_by_k = {}
    for k in pca_ks:
        pca = fit_pca_on_train(Y_train_vec, k=k, seed=seed)
        pc_r2_by_k[f"k{k}"] = pc_space_r2_from_pca(Y_vec, Y_vec_pred, pca)

    # Strict / X-only: compare confound-free predictions to Y_resid, giving
    # a direct head-to-head with Euclidean NN on Y_resid (both now predicting
    # only the X-explainable variance). This is the fair MIA comparison.
    strict = None
    if Y_resid_vec is not None:
        strict_edge = r2_summary(Y_resid_vec, Y_vec_pred_strict)
        strict_pc = {}
        for k in pca_ks:
            pca_resid = fit_pca_on_train(Y_train_resid_vec, k=k, seed=seed)
            strict_pc[f"k{k}"] = pc_space_r2_from_pca(
                Y_resid_vec, Y_vec_pred_strict, pca_resid
            )
        strict = {"edge_r2": strict_edge, "pc_r2_by_k": strict_pc}

    spd_summary = spd_manifold_summary(Y_spd, Y_spd_pred)

    return {
        "edge_r2": edge_r2,
        "pc_r2_by_k": pc_r2_by_k,
        "spd_distances": spd_summary,
        "strict_vs_Yresid": strict,
    }


def run_single_seed(
    cfg: Dict,
    seed: int,
    d: int,
    pca_ks: List[int],
    rank_cap: int,
    lambda_grid: Optional[List[float]],
    save_decomposition: bool,
    out_dir: Path,
) -> Dict:
    set_seed(seed)
    data = load_training_contracts(cfg)

    idx_tr = data["idx1_train"]
    idx_val = data["idx1_val"]
    idx_te = data["idx1_test"]
    idx_ext = data["idx2_external"]

    rscm_opts = cfg.get("rscm", {})
    x_variant = rscm_opts.get("x_variant", "resid")
    y_variant = rscm_opts.get("y_variant", "resid")
    fisher_z = bool(rscm_opts.get("fisher_z", True))
    harmonize = bool(rscm_opts.get("harmonize", False))
    harmonize_cols = tuple(rscm_opts.get("harmonize_cols", ("Age", "Gender")))

    base = Path(cfg["paths"]["aligned_features_dir"]).resolve()
    X1 = _load_variant(base, "dataset1_X", x_variant).astype(np.float32)
    Y1 = _load_variant(base, "dataset1_Y", y_variant).astype(np.float32)
    X2 = _load_variant(base, "dataset2_X", x_variant).astype(np.float32)
    Y2 = _load_variant(base, "dataset2_Y", y_variant).astype(np.float32)
    # Always load residualized Y too (for strict head-to-head evaluation
    # against Euclidean NN on Y_resid).
    Y1_resid = np.load(base / "dataset1_Y_resid.npy").astype(np.float32)
    Y2_resid = np.load(base / "dataset2_Y_resid.npy").astype(np.float32)

    X_tr, Y_tr_vec = X1[idx_tr], Y1[idx_tr]
    X_val, Y_val_vec = X1[idx_val], Y1[idx_val]
    X_te, Y_te_vec = X1[idx_te], Y1[idx_te]
    X_ext, Y_ext_vec = X2[idx_ext], Y2[idx_ext]
    Y_tr_resid = Y1_resid[idx_tr]
    Y_te_resid = Y1_resid[idx_te]
    Y_ext_resid = Y2_resid[idx_ext]

    if harmonize:
        meta = base / "meta"
        ids1 = np.array(data["ids1"])
        ids2 = np.array(data["ids2"])
        C_all_d1 = load_covariates(
            meta / "dataset1_subjects.tsv", ids1, harmonize_cols
        )
        C_all_d2 = load_covariates(
            meta / "dataset2_subjects.tsv", ids2, harmonize_cols
        )
        cov_tr = C_all_d1[idx_tr]
        cov_val = C_all_d1[idx_val]
        cov_te = C_all_d1[idx_te]
        cov_ext = C_all_d2[idx_ext]
    else:
        cov_tr = cov_val = cov_te = cov_ext = None

    Y_tr_spd = _y_to_spd(Y_tr_vec, d=d, fisher_z=fisher_z)
    Y_val_spd = _y_to_spd(Y_val_vec, d=d, fisher_z=fisher_z)
    Y_te_spd = _y_to_spd(Y_te_vec, d=d, fisher_z=fisher_z)
    Y_ext_spd = _y_to_spd(Y_ext_vec, d=d, fisher_z=fisher_z)

    rscm_cfg = RSFEConfig(
        d=d,
        rank_cap=rank_cap,
        nn_lambda_grid=tuple(lambda_grid) if lambda_grid else None,
        metric=rscm_opts.get("metric", "airm"),
    )

    print(f"  [seed {seed}] Fitting RSCM (d={d}, "
          f"N_train={X_tr.shape[0]}, p={X_tr.shape[1]}, q={Y_tr_vec.shape[1]}) ...",
          flush=True)
    t0 = time.time()
    model = RSFE(rscm_cfg).fit(
        X_tr, Y_tr_spd, S_val=X_val, F_val_spd=Y_val_spd,
        covariates_train=cov_tr, covariates_val=cov_val,
    )
    fit_s = time.time() - t0

    eff_rank = int(model.effective_rank())
    U, S, Vt = model.coef_svd()

    # Per-edge linear calibration fit on the training split:
    # Y_tr_vec ≈ scale_e * Y_tr_pred_e + offset_e. Closed-form per-edge OLS.
    Y_tr_pred = model.predict_vec(
        X_tr, fisher_z=fisher_z, covariates=cov_tr
    ).astype(np.float64)
    Y_tr_true = Y_tr_vec.astype(np.float64)
    yp_mean = Y_tr_pred.mean(axis=0)
    yt_mean = Y_tr_true.mean(axis=0)
    yp_var = Y_tr_pred.var(axis=0) + 1e-12
    cov = np.mean((Y_tr_pred - yp_mean) * (Y_tr_true - yt_mean), axis=0)
    scale = cov / yp_var
    offset = yt_mean - scale * yp_mean
    calibration = {"scale": scale.astype(np.float32),
                   "offset": offset.astype(np.float32)}

    ds1 = evaluate_rscm(
        model, X_te, Y_te_vec, Y_te_spd, Y_tr_vec,
        pca_ks=pca_ks, seed=seed, fisher_z=fisher_z,
        calibration=calibration, covariates=cov_te,
        Y_resid_vec=Y_te_resid, Y_train_resid_vec=Y_tr_resid,
    )
    ds2 = evaluate_rscm(
        model, X_ext, Y_ext_vec, Y_ext_spd, Y_tr_vec,
        pca_ks=pca_ks, seed=seed, fisher_z=fisher_z,
        calibration=calibration, covariates=cov_ext,
        Y_resid_vec=Y_ext_resid, Y_train_resid_vec=Y_tr_resid,
    )

    result = {
        "seed": seed,
        "best_lambda": model.best_lambda,
        "val_mse": model.val_mse,
        "effective_rank": eff_rank,
        "singular_values": [float(s) for s in S[:min(20, len(S))]],
        "fit_time_s": round(fit_s, 2),
        "data_shapes": {
            "train": list(X_tr.shape),
            "val": list(X_val.shape),
            "test": list(X_te.shape),
            "external": list(X_ext.shape),
            "F_bar": list(model.F_bar.shape),
        },
        "dataset1_test": ds1,
        "dataset2_external": ds2,
    }

    if save_decomposition:
        dec_dir = out_dir / "decompositions"
        dec_dir.mkdir(parents=True, exist_ok=True)
        np.save(dec_dir / f"rscm_seed{seed}_B.npy", model.B)
        np.save(dec_dir / f"rscm_seed{seed}_F_bar.npy", model.F_bar)
        np.save(dec_dir / f"rscm_seed{seed}_U.npy", U)
        np.save(dec_dir / f"rscm_seed{seed}_S.npy", S)
        np.save(dec_dir / f"rscm_seed{seed}_Vt.npy", Vt)

    return result


def aggregate_results(all_results: List[Dict]) -> Dict:
    seeds = [r["seed"] for r in all_results]
    summary: Dict = {"n_seeds": len(seeds), "seeds": seeds}

    eval_pca_k = "k20"

    d1_pc = [r["dataset1_test"]["pc_r2_by_k"][eval_pca_k]["pc_r2_mean"]
             for r in all_results]
    d2_pc = [r["dataset2_external"]["pc_r2_by_k"][eval_pca_k]["pc_r2_mean"]
             for r in all_results]
    d1_edge = [r["dataset1_test"]["edge_r2"]["r2_edge_mean"] for r in all_results]
    d2_edge = [r["dataset2_external"]["edge_r2"]["r2_edge_mean"]
               for r in all_results]

    d1_airm = [r["dataset1_test"]["spd_distances"]["airm_mean"] for r in all_results]
    d2_airm = [r["dataset2_external"]["spd_distances"]["airm_mean"]
               for r in all_results]
    d1_loge = [r["dataset1_test"]["spd_distances"]["logE_mean"] for r in all_results]
    d2_loge = [r["dataset2_external"]["spd_distances"]["logE_mean"]
               for r in all_results]

    effective_ranks = [r["effective_rank"] for r in all_results]

    summary["pc_r2_mean_d1"] = _stats(d1_pc)
    summary["pc_r2_mean_d2"] = _stats(d2_pc)
    summary["edge_r2_mean_d1"] = _stats(d1_edge)
    summary["edge_r2_mean_d2"] = _stats(d2_edge)
    summary["airm_mean_d1"] = _stats(d1_airm)
    summary["airm_mean_d2"] = _stats(d2_airm)
    summary["logE_mean_d1"] = _stats(d1_loge)
    summary["logE_mean_d2"] = _stats(d2_loge)
    summary["effective_rank"] = _stats(effective_ranks)

    # Also expose per-PCA-k pc_r2 means, useful for the main-paper direct
    # comparison (k=20 is the published primary metric).
    per_k = {}
    for pk in all_results[0]["dataset1_test"]["pc_r2_by_k"].keys():
        per_k[pk] = {
            "d1": _stats([r["dataset1_test"]["pc_r2_by_k"][pk]["pc_r2_mean"]
                          for r in all_results]),
            "d2": _stats([r["dataset2_external"]["pc_r2_by_k"][pk]["pc_r2_mean"]
                          for r in all_results]),
        }
    summary["per_pca_k"] = per_k

    return summary


def main():
    parser = argparse.ArgumentParser(description="RSCM: Riemannian Spiked Coupling")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--d", type=int, default=53,
                        help="FNC side length (53-IC Neuromark default)")
    parser.add_argument("--rank_cap", type=int, default=50)
    parser.add_argument("--pca_ks", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--lambda_grid", type=float, nargs="*", default=None,
                        help="Optional custom lambda grid; else RSFE default.")
    parser.add_argument("--save_decomposition", action="store_true")
    parser.add_argument("--out_dir", type=str, default="results/rscm")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RSCM: Riemannian Spiked Coupling Model")
    print(f"  d={args.d}  rank_cap={args.rank_cap}")
    print(f"  Seeds: {args.seeds}")
    print(f"  PCA ks for evaluation: {args.pca_ks}")
    print(f"  Lambda grid: {args.lambda_grid or 'default'}")
    print(f"  Output: {out_dir}")
    print("=" * 70)

    all_results = []
    for i, seed in enumerate(args.seeds):
        t0 = time.time()
        print(f"\n[{i+1}/{len(args.seeds)}] seed={seed}", flush=True)
        result = run_single_seed(
            cfg, seed=seed, d=args.d, pca_ks=args.pca_ks,
            rank_cap=args.rank_cap, lambda_grid=args.lambda_grid,
            save_decomposition=args.save_decomposition, out_dir=out_dir,
        )
        elapsed = time.time() - t0
        pc_d1 = result["dataset1_test"]["pc_r2_by_k"]["k20"]["pc_r2_mean"]
        pc_d2 = result["dataset2_external"]["pc_r2_by_k"]["k20"]["pc_r2_mean"]
        airm_d1 = result["dataset1_test"]["spd_distances"]["airm_mean"]
        print(f"  pc_r2(k20) DS1={pc_d1:.4f}  DS2={pc_d2:.4f}  "
              f"AIRM DS1={airm_d1:.4f}  "
              f"eff_rank={result['effective_rank']}  "
              f"lam={result['best_lambda']}  [{elapsed:.1f}s]",
              flush=True)
        all_results.append(result)
        save_json(out_dir / f"seed_{seed}.json", result)

    summary = aggregate_results(all_results)
    save_json(out_dir / "summary.json", summary)

    print("\n" + "=" * 70)
    print("RSCM SUMMARY")
    print("=" * 70)
    pc1 = summary["pc_r2_mean_d1"]
    pc2 = summary["pc_r2_mean_d2"]
    airm1 = summary["airm_mean_d1"]
    er = summary["effective_rank"]
    print(f"  DS1 pc_r2(k20) = {pc1['mean']:.4f} +/- {pc1['ci95']:.4f}")
    print(f"  DS2 pc_r2(k20) = {pc2['mean']:.4f} +/- {pc2['ci95']:.4f}")
    print(f"  DS1 AIRM mean  = {airm1['mean']:.4f} +/- {airm1['ci95']:.4f}")
    print(f"  effective rank = {er['mean']:.1f} +/- {er['ci95']:.1f}")
    print("=" * 70)
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
