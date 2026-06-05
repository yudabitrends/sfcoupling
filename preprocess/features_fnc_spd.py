"""
FNC SPD representation utilities for sfcoupling v2.

The parent project stores FNC as 1378-dim upper-triangle Fisher-z vectors (53
ICs). For Riemannian modeling we need the underlying 53x53 SPD matrix.

This module provides:
  - vec_to_spd(v, d): invert the 53-choose-2 upper-tri vector to full SPD, with
    Fisher-z inverted back to correlations and diagonal set to 1.0 (or 1+eps).
  - spd_to_vec(M): the forward map used by features_fnc.py.
  - batch_vec_to_spd / batch_spd_to_vec: ND helpers preserving subject axis.
  - make_spd(M): symmetrize + diagonal jitter to guarantee positive definiteness.

All routines are pure NumPy and side-effect free. Fréchet means and AIRM
distances live in models/spd_utils.py.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _triu_indices(d: int) -> Tuple[np.ndarray, np.ndarray]:
    i, j = np.triu_indices(d, k=1)
    return i, j


def spd_to_vec(M: np.ndarray, fisher_z: bool = True, clip: float = 0.999) -> np.ndarray:
    """
    Map a dxd symmetric matrix to its upper-tri vector of length d(d-1)/2.
    If fisher_z, apply arctanh after clipping to (-clip, +clip).
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Expected square 2D matrix, got {M.shape}")
    d = M.shape[0]
    i, j = _triu_indices(d)
    v = M[i, j].astype(np.float64)
    if fisher_z:
        v = np.arctanh(np.clip(v, -clip, clip))
    return v.astype(np.float32)


def vec_to_spd(
    v: np.ndarray,
    d: int = 53,
    fisher_z: bool = True,
    diag: float = 1.0,
    jitter: float = 1e-4,
) -> np.ndarray:
    """
    Invert the upper-triangle Fisher-z vector to a 53x53 correlation matrix.

    The FNC upper-tri convention (Fisher-z of Pearson correlations with 1 on
    diagonal) only yields an SPD matrix after inverse Fisher-z and adding a
    small jitter for numerical stability in log/exp maps.

    Parameters:
      v       : (d(d-1)/2,) vector
      d       : matrix side length (default 53)
      fisher_z: apply tanh() inverse before reconstructing
      diag    : value to place on the diagonal (default 1.0)
      jitter  : added to the diagonal to guarantee strict positive definiteness

    Returns a dxd float32 SPD matrix.
    """
    expected = d * (d - 1) // 2
    if v.size != expected:
        raise ValueError(f"Vector length {v.size} does not match d={d} (expected {expected}).")
    vv = v.astype(np.float64)
    if fisher_z:
        vv = np.tanh(vv)
    M = np.zeros((d, d), dtype=np.float64)
    i, j = _triu_indices(d)
    M[i, j] = vv
    M[j, i] = vv
    np.fill_diagonal(M, diag + jitter)
    return M.astype(np.float32)


def batch_vec_to_spd(
    V: np.ndarray,
    d: int = 53,
    fisher_z: bool = True,
    diag: float = 1.0,
    jitter: float = 1e-4,
) -> np.ndarray:
    """Stack version of vec_to_spd: V shape (N, d(d-1)/2) -> (N, d, d)."""
    if V.ndim != 2:
        raise ValueError(f"Expected (N, E) array, got {V.shape}")
    n = V.shape[0]
    out = np.empty((n, d, d), dtype=np.float32)
    for k in range(n):
        out[k] = vec_to_spd(V[k], d=d, fisher_z=fisher_z, diag=diag, jitter=jitter)
    return out


def batch_spd_to_vec(
    M: np.ndarray, fisher_z: bool = True, clip: float = 0.999
) -> np.ndarray:
    """Stack version of spd_to_vec: M shape (N, d, d) -> (N, d(d-1)/2)."""
    if M.ndim != 3 or M.shape[1] != M.shape[2]:
        raise ValueError(f"Expected (N, d, d), got {M.shape}")
    n, d, _ = M.shape
    e = d * (d - 1) // 2
    out = np.empty((n, e), dtype=np.float32)
    for k in range(n):
        out[k] = spd_to_vec(M[k], fisher_z=fisher_z, clip=clip)
    return out


def make_spd(
    M: np.ndarray, min_eig: float = 1e-6, symmetrize: bool = True
) -> np.ndarray:
    """
    Guarantee positive definiteness. Useful when a predicted matrix is only
    approximately valid.

    1. Symmetrize:  M <- (M + M.T) / 2
    2. Eigen clip:  eigenvalues below min_eig are raised to min_eig
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Expected square, got {M.shape}")
    X = M.astype(np.float64)
    if symmetrize:
        X = 0.5 * (X + X.T)
    w, Q = np.linalg.eigh(X)
    w = np.maximum(w, min_eig)
    return (Q @ np.diag(w) @ Q.T).astype(np.float32)


def make_spd_batch(M: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
    """Stack version of make_spd."""
    if M.ndim != 3 or M.shape[1] != M.shape[2]:
        raise ValueError(f"Expected (N, d, d), got {M.shape}")
    out = np.empty_like(M, dtype=np.float32)
    for k in range(M.shape[0]):
        out[k] = make_spd(M[k], min_eig=min_eig)
    return out


def check_spd(M: np.ndarray, tol: float = -1e-8) -> bool:
    """Return True if M is symmetric and its smallest eigenvalue is >= tol."""
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return False
    X = M.astype(np.float64)
    if not np.allclose(X, X.T, atol=1e-6):
        return False
    w = np.linalg.eigvalsh(X)
    return bool(w.min() >= tol)


def reconstruct_ukb_spd_from_vec_npy(
    vec_npy_path: str, d: int = 53, fisher_z: bool = True, jitter: float = 1e-4
) -> np.ndarray:
    """
    Convenience loader: read a (N, d(d-1)/2) npy of FNC vectors (Fisher-z
    upper-tri convention) and return (N, d, d) SPD stack.
    """
    V = np.load(vec_npy_path)
    return batch_vec_to_spd(V, d=d, fisher_z=fisher_z, jitter=jitter)
