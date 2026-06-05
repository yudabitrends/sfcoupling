"""
Riemannian utilities on SPD(d), the manifold of d-by-d symmetric positive
definite matrices. Used for sfcoupling v2 (RSFE, SFIB, AIRM-based metrics).

Implements:
  - sym_mat_sqrt / sym_mat_inv_sqrt / sym_mat_log / sym_mat_exp: spectral operators.
  - log_map / exp_map: AIRM log/exp at an arbitrary basepoint.
  - airm_dist / log_euclidean_dist / bures_wasserstein_dist.
  - frechet_mean: Karcher iteration on SPD.
  - vec_tangent / mat_tangent: isomorphism T_I SPD(d) ~ Sym(d) ~ R^{d(d+1)/2}.

All operations are pure NumPy. A torch mirror lives near the RSFE loss
definition to stay out of the data pipeline.

References
----------
Pennec, Fillard, Ayache (IJCV 2006): A Riemannian Framework for Tensor Computing
Arsigny et al. (2007): Log-Euclidean metrics for fast simple calculus
Bhatia (2007): Positive Definite Matrices (Princeton)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _symmetrize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def sym_eig(A: np.ndarray, min_eig: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition of a symmetric matrix with eigenvalue clipping."""
    w, V = np.linalg.eigh(_symmetrize(A))
    w = np.maximum(w, min_eig)
    return w, V


def sym_mat_sqrt(A: np.ndarray) -> np.ndarray:
    w, V = sym_eig(A)
    return V @ np.diag(np.sqrt(w)) @ V.T


def sym_mat_inv_sqrt(A: np.ndarray) -> np.ndarray:
    w, V = sym_eig(A)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def sym_mat_log(A: np.ndarray) -> np.ndarray:
    w, V = sym_eig(A)
    return V @ np.diag(np.log(w)) @ V.T


_SYM_EXP_CLIP = 50.0  # eigenvalues above ~50 overflow float64 in np.exp


def sym_mat_exp(A: np.ndarray, clip: float = _SYM_EXP_CLIP) -> np.ndarray:
    """Matrix exponential with eigenvalue clipping.

    Tangent predictions from a noisy linear fit occasionally land far outside
    the log-spectral range of real SPD FNC matrices. Clipping the tangent
    eigenvalues to +/-50 keeps the output finite without meaningfully
    distorting the signal — a correlation-matrix eigenvalue of exp(50) is
    not physical and would imply the linear predictor is extrapolating.
    """
    w, V = np.linalg.eigh(_symmetrize(A))
    w = np.clip(w, -clip, clip)
    return V @ np.diag(np.exp(w)) @ V.T


def log_map(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    AIRM logarithm: map X in SPD(d) to the tangent space at basepoint P.
    log_P(X) = P^{1/2} log(P^{-1/2} X P^{-1/2}) P^{1/2}
    """
    P_inv_sqrt = sym_mat_inv_sqrt(P)
    P_sqrt = sym_mat_sqrt(P)
    M = P_inv_sqrt @ X @ P_inv_sqrt
    return P_sqrt @ sym_mat_log(M) @ P_sqrt


def exp_map(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    AIRM exponential: project a tangent vector V at P back to SPD(d).
    exp_P(V) = P^{1/2} exp(P^{-1/2} V P^{-1/2}) P^{1/2}
    """
    P_inv_sqrt = sym_mat_inv_sqrt(P)
    P_sqrt = sym_mat_sqrt(P)
    return P_sqrt @ sym_mat_exp(P_inv_sqrt @ V @ P_inv_sqrt) @ P_sqrt


def airm_dist(A: np.ndarray, B: np.ndarray) -> float:
    """
    Affine-Invariant Riemannian distance:
    d_AIRM(A, B) = || log(A^{-1/2} B A^{-1/2}) ||_F
    """
    A_inv_sqrt = sym_mat_inv_sqrt(A)
    M = A_inv_sqrt @ B @ A_inv_sqrt
    w, _ = sym_eig(M)
    return float(np.sqrt(np.sum(np.log(w) ** 2)))


def log_euclidean_dist(A: np.ndarray, B: np.ndarray) -> float:
    """||log(A) - log(B)||_F."""
    return float(np.linalg.norm(sym_mat_log(A) - sym_mat_log(B), ord="fro"))


def bures_wasserstein_dist(A: np.ndarray, B: np.ndarray) -> float:
    """sqrt(tr(A) + tr(B) - 2 tr((A^{1/2} B A^{1/2})^{1/2}))."""
    A_sqrt = sym_mat_sqrt(A)
    M = sym_mat_sqrt(A_sqrt @ B @ A_sqrt)
    val = float(np.trace(A) + np.trace(B) - 2.0 * np.trace(M))
    return float(np.sqrt(max(val, 0.0)))


def frechet_mean(
    X: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
    init: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Karcher mean on SPD(d) under AIRM.

    X: (N, d, d) stack of SPD matrices
    Returns the dxd Fréchet mean.
    """
    if X.ndim != 3 or X.shape[1] != X.shape[2]:
        raise ValueError(f"Expected (N, d, d), got {X.shape}")
    n, d, _ = X.shape
    M = init.copy() if init is not None else np.mean(X, axis=0)
    M = _symmetrize(M)
    for it in range(max_iter):
        M_inv_sqrt = sym_mat_inv_sqrt(M)
        M_sqrt = sym_mat_sqrt(M)
        # tangent mean at M
        acc = np.zeros((d, d), dtype=np.float64)
        for k in range(n):
            C = M_inv_sqrt @ X[k] @ M_inv_sqrt
            acc += sym_mat_log(C)
        V = M_sqrt @ (acc / n) @ M_sqrt  # tangent vector average
        M_new = exp_map(M, V)
        delta = np.linalg.norm(M_new - M, ord="fro") / max(
            np.linalg.norm(M, ord="fro"), 1e-12
        )
        M = M_new
        if verbose:
            print(f"[frechet_mean] iter={it} delta={delta:.3e}")
        if delta < tol:
            break
    return M.astype(np.float32)


def vec_tangent(S: np.ndarray) -> np.ndarray:
    """
    Isomorphism Sym(d) -> R^{d(d+1)/2}, diagonal kept as-is, off-diagonals
    scaled by sqrt(2) so that the Euclidean norm equals the Frobenius norm.
    """
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"Expected (d,d), got {S.shape}")
    d = S.shape[0]
    i_up, j_up = np.triu_indices(d, k=1)
    out = np.empty(d * (d + 1) // 2, dtype=np.float64)
    out[:d] = np.diagonal(S)
    out[d:] = np.sqrt(2.0) * S[i_up, j_up]
    return out.astype(np.float32)


def mat_tangent(v: np.ndarray, d: int) -> np.ndarray:
    """Inverse of vec_tangent."""
    expected = d * (d + 1) // 2
    if v.size != expected:
        raise ValueError(f"Vector length {v.size} != {expected} for d={d}.")
    S = np.zeros((d, d), dtype=np.float64)
    np.fill_diagonal(S, v[:d])
    i_up, j_up = np.triu_indices(d, k=1)
    off = v[d:] / np.sqrt(2.0)
    S[i_up, j_up] = off
    S[j_up, i_up] = off
    return S.astype(np.float32)


def log_map_at_frechet(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Batch tangent-space projection: T[k] = log_M(X[k]) for each subject.

    Returns (N, d, d) stack of tangent-space vectors.
    """
    if X.ndim != 3 or M.ndim != 2:
        raise ValueError("X must be (N,d,d), M must be (d,d).")
    n = X.shape[0]
    out = np.empty_like(X, dtype=np.float32)
    for k in range(n):
        out[k] = log_map(M, X[k])
    return out


def batch_tangent_vectors(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Convenience: log_map_at_frechet + vec_tangent, producing (N, d(d+1)/2).
    This is the standard "tangent vectorization" used for linear models.
    """
    T = log_map_at_frechet(X, M)
    n, d, _ = T.shape
    out = np.empty((n, d * (d + 1) // 2), dtype=np.float32)
    for k in range(n):
        out[k] = vec_tangent(T[k])
    return out


def exp_from_tangent_vectors(V: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Inverse of batch_tangent_vectors: project (N, d(d+1)/2) tangent vectors
    back to SPD(d) via exp_map at M.
    """
    if V.ndim != 2:
        raise ValueError(f"V must be (N, q); got {V.shape}")
    d = M.shape[0]
    n = V.shape[0]
    out = np.empty((n, d, d), dtype=np.float32)
    for k in range(n):
        S = mat_tangent(V[k], d=d)
        out[k] = exp_map(M, S)
    return out


# ---- Log-Euclidean variant (isometric to Euclidean on log-matrices) -----
#
# LE tangent: T_i = sym_mat_log(F_i) - sym_mat_log(F_bar). Linear regression
# in LE tangent space is just linear regression in log-matrix space, no
# non-linear exp-map compression at prediction time. LE matches AIRM to
# first order at the basepoint and is strictly more numerically stable at
# scale. The MIA plan picks LE as the production metric for this reason.

def batch_tangent_vectors_le(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """LE tangent vectorization at basepoint M: vec(log(F_i) - log(M))."""
    if X.ndim != 3 or M.ndim != 2:
        raise ValueError("X must be (N,d,d), M must be (d,d).")
    n, d, _ = X.shape
    logM = sym_mat_log(M)
    out = np.empty((n, d * (d + 1) // 2), dtype=np.float32)
    for k in range(n):
        out[k] = vec_tangent(sym_mat_log(X[k]) - logM)
    return out


def exp_from_tangent_vectors_le(V: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Inverse of batch_tangent_vectors_le: F_hat = exp(log(M) + mat(V))."""
    if V.ndim != 2:
        raise ValueError(f"V must be (N, q); got {V.shape}")
    d = M.shape[0]
    logM = sym_mat_log(M)
    n = V.shape[0]
    out = np.empty((n, d, d), dtype=np.float32)
    for k in range(n):
        out[k] = sym_mat_exp(logM + mat_tangent(V[k], d=d))
    return out
