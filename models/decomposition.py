"""
Coupled + residual decomposition of structural features with respect to a
functional-aligned subspace.

For each modality m ∈ {GM, WM} (and the concatenation GM+WM) we build an
orthogonal projector P_F: R^p -> R^p whose column space is the "FNC-aligned"
direction set. Then m = P_F(m) + R_F(m). Properties enforced numerically:

  1. Symmetric projector   : P_F = P_F.T
  2. Idempotent            : P_F @ P_F == P_F
  3. Orthogonal residual   : (I - P_F).T @ P_F == 0
  4. Info preservation     : R(m, F | P_F) close to R(m, F | full)

The projector is built from an SVD of a functional-regressed training weight
matrix B (e.g., the Nuclear-Norm or RSFE B) restricted to the top-k left
singular vectors.

This module implements the three axioms as unit-testable utilities and exposes
a decompose() function returning {coupled, residual, projector, rank, audit}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DecompositionReport:
    k: int
    projector_is_symmetric: bool
    projector_is_idempotent: bool
    residual_orthogonal: bool
    idempotence_err: float
    orthogonality_err: float
    info_retained_ratio: Optional[float]

    def axioms_pass(self, tol: float = 1e-4) -> bool:
        return (
            self.projector_is_symmetric
            and self.idempotence_err < tol
            and self.orthogonality_err < tol
        )

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "projector_is_symmetric": self.projector_is_symmetric,
            "projector_is_idempotent": self.projector_is_idempotent,
            "residual_orthogonal": self.residual_orthogonal,
            "idempotence_err": self.idempotence_err,
            "orthogonality_err": self.orthogonality_err,
            "info_retained_ratio": self.info_retained_ratio,
        }


def build_projector_from_weights(B: np.ndarray, k: int) -> np.ndarray:
    """
    Given a weight matrix B (p, q) mapping structure to a functional target,
    return the rank-k orthogonal projector P onto the top-k left singular
    vectors of B. P has shape (p, p).
    """
    if B.ndim != 2:
        raise ValueError(f"B must be 2D; got {B.shape}")
    if k < 1 or k > min(B.shape):
        raise ValueError(f"k={k} outside [1, {min(B.shape)}]")
    U, _, _ = np.linalg.svd(B, full_matrices=False)
    Uk = U[:, :k]
    return (Uk @ Uk.T).astype(np.float64)


def decompose(
    S: np.ndarray,
    B: np.ndarray,
    k: int,
    info_denominator_norm: Optional[float] = None,
) -> tuple:
    """
    Split S row-wise into (coupled, residual) using the projector derived from B.

    Returns (coupled, residual, projector, report) where coupled + residual = S
    up to float error and report captures the three axioms.
    """
    P = build_projector_from_weights(B, k=k)
    coupled = S.astype(np.float64) @ P
    residual = S.astype(np.float64) - coupled

    sym_err = float(np.linalg.norm(P - P.T, ord="fro"))
    idem_err = float(np.linalg.norm(P @ P - P, ord="fro"))
    ortho_err = float(np.linalg.norm((np.eye(P.shape[0]) - P) @ P, ord="fro"))

    if info_denominator_norm is not None and info_denominator_norm > 0:
        info_ratio = float(
            np.linalg.norm(coupled, ord="fro") / info_denominator_norm
        )
    else:
        info_ratio = None

    report = DecompositionReport(
        k=k,
        projector_is_symmetric=sym_err < 1e-6,
        projector_is_idempotent=idem_err < 1e-4,
        residual_orthogonal=ortho_err < 1e-4,
        idempotence_err=idem_err,
        orthogonality_err=ortho_err,
        info_retained_ratio=info_ratio,
    )
    return (
        coupled.astype(np.float32),
        residual.astype(np.float32),
        P.astype(np.float32),
        report,
    )


def information_retention_check(
    coupled: np.ndarray,
    residual: np.ndarray,
    F_target: np.ndarray,
    estimator_cls=None,
    estimator_kwargs: Optional[dict] = None,
) -> dict:
    """
    Empirically check whether coupled retains most of the mutual information
    with F, by regressing F on (coupled only) vs (coupled + residual) and on
    (residual only). Returns dict of R² values.

    estimator_cls: scikit-learn regressor with fit/predict (default Ridge).
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    estimator_cls = estimator_cls or Ridge
    estimator_kwargs = estimator_kwargs or dict(alpha=1.0)

    def fit_pred(X):
        m = estimator_cls(**estimator_kwargs).fit(X, F_target)
        return float(r2_score(F_target, m.predict(X)))

    full = np.concatenate([coupled, residual], axis=1)
    return dict(
        r2_from_coupled=fit_pred(coupled),
        r2_from_residual=fit_pred(residual),
        r2_from_full=fit_pred(full),
    )


def residual_biology(
    residual: np.ndarray,
    targets: dict,
    estimator_kwargs: Optional[dict] = None,
) -> dict:
    """
    Ridge-regress each continuous/binary target onto the residual. Returns a
    dict {target_name -> R² or AUC}.

    targets: {name -> (y, kind)} with kind in {"continuous", "binary"}.
    """
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import r2_score, roc_auc_score

    estimator_kwargs = estimator_kwargs or {}
    out = {}
    for name, (y, kind) in targets.items():
        if kind == "continuous":
            m = Ridge(alpha=1.0).fit(residual, y)
            yhat = m.predict(residual)
            out[name] = float(r2_score(y, yhat))
        elif kind == "binary":
            m = LogisticRegression(max_iter=500).fit(residual, y)
            proba = m.predict_proba(residual)[:, 1]
            out[name] = float(roc_auc_score(y, proba))
        else:
            raise ValueError(f"Unknown target kind {kind}")
    return out
