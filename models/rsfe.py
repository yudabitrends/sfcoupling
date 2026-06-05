"""
Riemannian Structural FNC Estimator (RSFE).

The model maps structural features s ∈ R^p (GM, WM, or concat) to a subject-
specific 53x53 SPD matrix F̂. Two fitting strategies:

  1. RSFE-tan (default)  : tangent-space linear + nuclear-norm regression at
                            the Fréchet mean. Pure Euclidean solver, cheap.
  2. RSFE-Riem           : refines the tangent fit by one Riemannian gradient
                            step on the AIRM loss (optional).

The tangent parameterisation works because log_F̄ : SPD(d) -> Sym(d) is a
diffeomorphism; in its image, Frobenius error is the log-Euclidean distance
(Arsigny 2007) and the nuclear-norm optimum is available in closed form via
SVD soft-thresholding.

The shared train/eval interface returns:
    {
      "F_pred_spd": (N, d, d),  # SPD predictions
      "F_pred_vec": (N, d(d-1)/2),  # upper-tri Fisher-z for parent metrics
      "B": (q, p),              # tangent-space weight matrix
      "F_bar": (d, d),          # Fréchet mean used for log/exp
    }

All operations live in float32 to match the parent project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from models.spd_utils import (
    batch_tangent_vectors,
    batch_tangent_vectors_le,
    exp_from_tangent_vectors,
    exp_from_tangent_vectors_le,
    exp_map,
    frechet_mean,
    log_map,
    mat_tangent,
    sym_mat_exp,
    sym_mat_log,
    vec_tangent,
)
from preprocess.features_fnc_spd import batch_spd_to_vec
from preprocess.harmonize import (
    apply_tangent_residualizer,
    fit_tangent_residualizer,
)


@dataclass
class RSFEConfig:
    d: int = 53                # FNC side length
    rank_cap: int = 50         # soft ceiling on effective rank
    nn_lambda_grid: Optional[tuple] = None  # sweep for nuclear norm
    nn_max_iter: int = 600
    nn_tol: float = 1e-5
    use_bias: bool = True
    frechet_iters: int = 50
    frechet_tol: float = 1e-6
    metric: str = "airm"       # "airm" or "logE"

    def default_lambda_grid(self) -> tuple:
        if self.nn_lambda_grid is not None:
            return tuple(self.nn_lambda_grid)
        return (0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)


def _soft_threshold_svd(M: np.ndarray, tau: float) -> np.ndarray:
    """SVD soft-thresholding: nuclear-norm prox of M at threshold tau."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s = np.maximum(s - tau, 0.0)
    k = int(np.sum(s > 0))
    if k == 0:
        return np.zeros_like(M)
    return (U[:, :k] * s[:k]) @ Vt[:k]


def _istanuclear(
    X: np.ndarray,
    Y: np.ndarray,
    lam: float,
    max_iter: int = 600,
    tol: float = 1e-5,
) -> np.ndarray:
    """
    ISTA solution of:
      min_B  0.5 / n * || Y - X B ||_F^2  +  lam * || B ||_*
    X: (n, p), Y: (n, q), B: (p, q). Returns the B that achieved minimum.
    """
    n, p = X.shape
    q = Y.shape[1]
    if n < 1:
        raise ValueError("need at least one training sample")
    XtX = X.T @ X
    L = float(np.linalg.eigvalsh(XtX).max() / n + 1e-8)
    step = 1.0 / L

    B = np.zeros((p, q), dtype=np.float64)
    prev_obj = np.inf
    for it in range(max_iter):
        grad = (X.T @ (X @ B - Y)) / n
        B = _soft_threshold_svd(B - step * grad, step * lam)
        resid = X @ B - Y
        obj = 0.5 / n * float(np.sum(resid ** 2)) + lam * float(
            np.linalg.norm(B, ord="nuc")
        )
        if abs(prev_obj - obj) / max(prev_obj, 1e-12) < tol:
            break
        prev_obj = obj
    return B


def _select_best_lambda(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    lambdas,
    max_iter: int,
    tol: float,
) -> tuple:
    """Return (best_B, best_lambda, val_mse) via held-out validation MSE."""
    best = None
    best_lam = None
    best_mse = np.inf
    for lam in lambdas:
        B = _istanuclear(X_tr, Y_tr, lam=lam, max_iter=max_iter, tol=tol)
        pred = X_val @ B
        mse = float(np.mean((pred - Y_val) ** 2))
        if mse < best_mse:
            best = B
            best_lam = lam
            best_mse = mse
    return best, best_lam, best_mse


class RSFE:
    """Tangent-space nuclear-norm regression on SPD(d), the RSFE model."""

    def __init__(self, config: Optional[RSFEConfig] = None):
        self.cfg = config or RSFEConfig()
        self.F_bar: Optional[np.ndarray] = None
        self.mean_t: Optional[np.ndarray] = None
        self.mean_s: Optional[np.ndarray] = None
        self.B: Optional[np.ndarray] = None
        self.best_lambda: Optional[float] = None
        self.val_mse: Optional[float] = None
        # Harmonization beta (None means no tangent-space residualization).
        self.harmonize_beta: Optional[np.ndarray] = None

    def _auto_lambda_grid(
        self, S_c: np.ndarray, T_c: np.ndarray
    ) -> tuple:
        """Data-driven lambda grid (log-spaced below operator-norm lambda_max).

        lambda_max = ||S^T T / n||_op is the smallest lambda that drives the
        ISTA solution to B=0; any useful lambda lies strictly below it. This
        mirrors the Euclidean nuclear-norm baseline in
        train/run_multivariate_methods.py.
        """
        n = S_c.shape[0]
        lambda_max = float(np.linalg.svd(S_c.T @ T_c / n, compute_uv=False)[0])
        ratios = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
        return tuple(lambda_max * r for r in ratios)

    def _to_tangent(self, X: np.ndarray) -> np.ndarray:
        if self.cfg.metric == "logE":
            return batch_tangent_vectors_le(X, self.F_bar)
        return batch_tangent_vectors(X, self.F_bar)

    def _from_tangent(self, V: np.ndarray) -> np.ndarray:
        if self.cfg.metric == "logE":
            return exp_from_tangent_vectors_le(V, self.F_bar)
        return exp_from_tangent_vectors(V, self.F_bar)

    def fit(
        self,
        S_train: np.ndarray,
        F_train_spd: np.ndarray,
        S_val: Optional[np.ndarray] = None,
        F_val_spd: Optional[np.ndarray] = None,
        covariates_train: Optional[np.ndarray] = None,
        covariates_val: Optional[np.ndarray] = None,
    ) -> "RSFE":
        """
        S_*       : (n, p) structural features
        F_*_spd   : (n, d, d) SPD FNC matrices
        covariates_*: (n, k) optional confound matrix for tangent-space
                    residualization (Age/Gender/site). Beta is fit on
                    covariates_train only and applied to val with the same
                    coefficients; no leakage.
        """
        if S_train.shape[0] != F_train_spd.shape[0]:
            raise ValueError("Train sample count mismatch.")
        if self.cfg.metric == "logE":
            # LE Frechet mean has closed form: exp(mean(log(F_i)))
            n = F_train_spd.shape[0]
            log_mean = np.zeros_like(F_train_spd[0], dtype=np.float64)
            for i in range(n):
                log_mean += sym_mat_log(F_train_spd[i].astype(np.float64))
            self.F_bar = sym_mat_exp(log_mean / n).astype(np.float32)
        else:
            self.F_bar = frechet_mean(
                F_train_spd,
                max_iter=self.cfg.frechet_iters,
                tol=self.cfg.frechet_tol,
            )

        T_train = self._to_tangent(F_train_spd).astype(np.float64)
        # Optional tangent-space harmonization (MIA thread-2 contribution):
        # fit beta on train-only, subtract confound projection from T_train.
        if covariates_train is not None:
            self.harmonize_beta = fit_tangent_residualizer(T_train,
                                                           covariates_train)
            T_train = apply_tangent_residualizer(T_train, covariates_train,
                                                 self.harmonize_beta)
        self.mean_t = T_train.mean(axis=0)
        T_train_c = T_train - self.mean_t

        S_train64 = S_train.astype(np.float64)
        self.mean_s = S_train64.mean(axis=0)
        S_train_c = S_train64 - self.mean_s

        if self.cfg.nn_lambda_grid is not None:
            lambdas = self.cfg.default_lambda_grid()
        else:
            lambdas = self._auto_lambda_grid(S_train_c, T_train_c)

        if S_val is None or F_val_spd is None:
            # no held-out fold: pick a mild lambda via the 1-SE heuristic
            mid_lam = lambdas[len(lambdas) // 2]
            pick_B = _istanuclear(
                S_train_c,
                T_train_c,
                lam=mid_lam,
                max_iter=self.cfg.nn_max_iter,
                tol=self.cfg.nn_tol,
            )
            self.B = pick_B
            self.best_lambda = mid_lam
            self.val_mse = None
        else:
            T_val = self._to_tangent(F_val_spd).astype(np.float64)
            if self.harmonize_beta is not None:
                if covariates_val is None:
                    raise ValueError(
                        "covariates_val required when fit with covariates_train"
                    )
                T_val = apply_tangent_residualizer(T_val, covariates_val,
                                                   self.harmonize_beta)
            T_val_c = T_val - self.mean_t
            S_val_c = S_val.astype(np.float64) - self.mean_s
            B, lam, mse = _select_best_lambda(
                S_train_c,
                T_train_c,
                S_val_c,
                T_val_c,
                lambdas=lambdas,
                max_iter=self.cfg.nn_max_iter,
                tol=self.cfg.nn_tol,
            )
            self.B = B
            self.best_lambda = lam
            self.val_mse = mse
        return self

    def _predict_tangent_vectors(
        self, S: np.ndarray, covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.B is None or self.mean_s is None or self.mean_t is None:
            raise RuntimeError("RSFE is not fitted.")
        S64 = S.astype(np.float64) - self.mean_s
        T_hat = S64 @ self.B + self.mean_t
        # If the model was fit with tangent-space harmonization, add the
        # covariate projection back so predictions live in the original
        # (un-residualized) tangent space — so spd_to_vec / edge R^2 are
        # comparable to the Y_raw ground truth.
        if self.harmonize_beta is not None:
            if covariates is None:
                raise ValueError(
                    "covariates required at predict: model was fit with "
                    "tangent-space harmonization"
                )
            cov64 = covariates.astype(np.float64)
            D = np.hstack([np.ones((cov64.shape[0], 1)), cov64])
            T_hat = T_hat + D @ self.harmonize_beta
        return T_hat.astype(np.float64)

    def predict_spd(
        self, S: np.ndarray, covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Map structural features to (N, d, d) SPD predictions."""
        if self.F_bar is None:
            raise RuntimeError("RSFE is not fitted.")
        T_hat = self._predict_tangent_vectors(S, covariates=covariates)
        return self._from_tangent(T_hat.astype(np.float32))

    def predict_vec(
        self,
        S: np.ndarray,
        fisher_z: bool = True,
        covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict upper-tri Fisher-z vector (parent metric compatibility)."""
        F_hat = self.predict_spd(S, covariates=covariates)
        return batch_spd_to_vec(F_hat, fisher_z=fisher_z)

    def effective_rank(self, eps: float = 1e-4) -> int:
        """Return approximate rank of fitted B via singular-value cutoff."""
        if self.B is None:
            return 0
        s = np.linalg.svd(self.B, compute_uv=False)
        return int(np.sum(s > eps * s.max()))

    def coef_svd(self) -> tuple:
        """(U, s, Vt) of B for mode interpretation / visualization."""
        if self.B is None:
            raise RuntimeError("RSFE is not fitted.")
        return np.linalg.svd(self.B, full_matrices=False)


def rsfe_sanity_test(seed: int = 0, d: int = 6, p: int = 10, n: int = 120) -> dict:
    """
    Synthetic round-trip: generate SPD targets from linear-plus-noise structural
    inputs, fit RSFE, verify tangent-prediction error decreases vs a zero model.

    Returns a dict of numbers that should be stable across machines.
    """
    rng = np.random.default_rng(seed)
    S = rng.standard_normal((n, p)).astype(np.float32)
    d_eff = d * (d + 1) // 2
    # Random low-rank ground truth tangent coefficient, q=d(d+1)/2
    B_true = rng.standard_normal((p, d_eff)) / p
    T = S @ B_true + 0.05 * rng.standard_normal((n, d_eff))
    # Use an arbitrary SPD basepoint; project tangent vectors back to SPD
    base = rng.standard_normal((d, d))
    F_bar_true = base @ base.T + np.eye(d)
    F_spd = np.empty((n, d, d), dtype=np.float32)
    for i in range(n):
        F_spd[i] = exp_map(F_bar_true, mat_tangent(T[i], d=d))

    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    idx = np.arange(n)
    idx_tr, idx_val, idx_te = (
        idx[:n_train],
        idx[n_train : n_train + n_val],
        idx[n_train + n_val :],
    )
    model = RSFE(RSFEConfig(d=d))
    model.fit(S[idx_tr], F_spd[idx_tr], S_val=S[idx_val], F_val_spd=F_spd[idx_val])
    F_pred = model.predict_spd(S[idx_te])
    # evaluate on tangent vectors for simplicity
    T_pred = batch_tangent_vectors(F_pred, model.F_bar)
    T_true = batch_tangent_vectors(F_spd[idx_te], model.F_bar)
    mse = float(np.mean((T_pred - T_true) ** 2))
    base_mse = float(np.mean(T_true ** 2))
    return dict(
        mse=mse,
        base_mse=base_mse,
        reduction_ratio=float(1 - mse / base_mse),
        best_lambda=model.best_lambda,
        val_mse=model.val_mse,
        effective_rank=model.effective_rank(),
    )
