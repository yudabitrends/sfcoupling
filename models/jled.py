"""
Joint Laplacian Eigenmode Decomposition (JLED).

Motivation: recent SFC literature (Pang 2023, Cai 2024 Nat Comms) reconstructs
functional activity as a linear combination of structural eigenmodes. JLED
extends that idea by letting GM and WM jointly define the structural graph and
then expressing FNC in the resulting harmonic basis.

Pipeline
--------
1. Compute a group-level structural similarity matrix A ∈ R^{d×d} over the d
   functional nodes (53 ICs). For each subject i, the feature-to-node mapping
   uses a fixed projector P ∈ R^{d×p} (e.g., group-average GM-to-IC loadings)
   so node-wise structural signatures live in R^d.
2. Build the symmetric normalized graph Laplacian L = I - D^{-1/2} A D^{-1/2},
   decompose L = U Λ U^T.
3. For each subject, project the FNC matrix to the Laplacian basis:
       F̃_i = U^T F_i U
4. Keep the top-k frequency modes (low-λ), i.e. F̂_i ≈ U[:, :k] F̃_i[:k, :k] U[:, :k]^T.

The free parameter is k. Group G-D OptShrink from the parent project gives a
principled cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class JLEDConfig:
    d: int = 53
    k: int = 20                # modes to retain
    knn: int = 10              # k-NN sparsification for the similarity graph
    sigma: Optional[float] = None  # Gaussian bandwidth; None -> median heuristic


def _pairwise_kernel(X: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    """Row-wise Gaussian affinity on rows of X. Returns a dense (d, d) matrix."""
    G = X @ X.T
    sq = np.diagonal(G)
    D = np.maximum(sq[:, None] + sq[None, :] - 2.0 * G, 0.0)
    if sigma is None:
        tri = D[np.triu_indices_from(D, k=1)]
        sigma = float(np.sqrt(np.median(tri) + 1e-12))
    return np.exp(-D / (2.0 * sigma * sigma + 1e-12))


def _knn_symmetrize(A: np.ndarray, knn: int) -> np.ndarray:
    d = A.shape[0]
    A = A.copy()
    np.fill_diagonal(A, 0.0)
    order = np.argsort(-A, axis=1)
    mask = np.zeros_like(A)
    for i in range(d):
        mask[i, order[i, :knn]] = 1.0
    mask = np.maximum(mask, mask.T)
    return A * mask


def _normalized_laplacian(A: np.ndarray) -> np.ndarray:
    d = A.sum(axis=1)
    d[d == 0] = 1.0
    Dinv_sqrt = 1.0 / np.sqrt(d)
    L = np.eye(A.shape[0]) - (Dinv_sqrt[:, None] * A * Dinv_sqrt[None, :])
    return 0.5 * (L + L.T)


class JLED:
    """Joint structural graph Laplacian projection of FNC."""

    def __init__(self, config: Optional[JLEDConfig] = None):
        self.cfg = config or JLEDConfig()
        self.U: Optional[np.ndarray] = None     # (d, d) eigenvectors
        self.eigvals: Optional[np.ndarray] = None
        self.loadings_mean: Optional[np.ndarray] = None

    def _node_signature_from_features(
        self, S: np.ndarray, P: np.ndarray
    ) -> np.ndarray:
        """Turn (N, p) features into (N, d) node-space signatures via P."""
        return S @ P.T

    def fit_graph(self, S_train: np.ndarray, P: np.ndarray) -> "JLED":
        """
        Build the joint structural graph from the training cohort.
        S_train : (n, p) subject-level features
        P       : (d, p) feature-to-node projector
        """
        Z = self._node_signature_from_features(S_train, P)  # (n, d)
        # d-by-d affinity built from node-space column signatures (per-node mean loading)
        # First z-score nodes across subjects, then Gaussian affinity over node signatures.
        mean = Z.mean(axis=0, keepdims=True)
        std = Z.std(axis=0, keepdims=True) + 1e-8
        Zc = (Z - mean) / std
        A = _pairwise_kernel(Zc.T, sigma=self.cfg.sigma)  # (d, d)
        if self.cfg.knn and self.cfg.knn < A.shape[0]:
            A = _knn_symmetrize(A, knn=self.cfg.knn)
        L = _normalized_laplacian(A)
        w, V = np.linalg.eigh(L)
        self.eigvals = w
        self.U = V
        return self

    def project_fnc(self, F_spd: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """
        Project (N, d, d) SPD FNC to low-freq truncation F̂_i = U_k F̃_i U_k^T.
        Returns (N, d, d) reconstructed stack.
        """
        if self.U is None:
            raise RuntimeError("JLED.fit_graph must be called first.")
        k = k or self.cfg.k
        if not (1 <= k <= self.U.shape[0]):
            raise ValueError(f"k={k} out of range [1, {self.U.shape[0]}].")
        Uk = self.U[:, :k]
        n = F_spd.shape[0]
        out = np.empty_like(F_spd, dtype=np.float32)
        for i in range(n):
            proj = Uk.T @ F_spd[i] @ Uk
            out[i] = (Uk @ proj @ Uk.T).astype(np.float32)
        return out

    def fit_predict(
        self,
        S_train: np.ndarray,
        F_train_spd: np.ndarray,
        S_test: np.ndarray,
        F_test_spd: np.ndarray,
        P: np.ndarray,
        k: Optional[int] = None,
    ) -> dict:
        """
        Convenience pipeline: fit graph on train, project train/test FNC to the
        retained subspace, also return loadings in the Laplacian basis for
        downstream regression.
        """
        self.fit_graph(S_train, P)
        F_train_rec = self.project_fnc(F_train_spd, k=k)
        F_test_rec = self.project_fnc(F_test_spd, k=k)
        return dict(
            F_train_rec=F_train_rec,
            F_test_rec=F_test_rec,
            eigvals=self.eigvals,
            U=self.U,
        )
