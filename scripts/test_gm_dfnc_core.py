#!/usr/bin/env python3
"""Small synthetic smoke test for gm_dfnc_core."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from gm_dfnc_core import (
    between_within_variance,
    build_prediction_targets,
    build_subject_dynamic_summary,
    compute_centroids_from_labels,
    compute_delta_retention,
    compute_subspace_retention,
    pairwise_distance_alignment,
    retention_vs_rank_curve,
)


def main() -> None:
    rng = np.random.default_rng(7)
    n_subjects = 10
    windows_per_subject = 8
    q = 12
    r = 3
    K = 4

    V = np.linalg.qr(rng.standard_normal((q, r)))[0]
    centroids = rng.standard_normal((K, q)) @ (V @ V.T) + 0.15 * rng.standard_normal((K, q))

    subject_ids = []
    window_idx = []
    labels = []
    windows = []
    for sid in range(n_subjects):
        labs = rng.integers(0, K, size=windows_per_subject)
        for t, lab in enumerate(labs):
            win = centroids[lab] + 0.2 * rng.standard_normal(q)
            subject_ids.append(f"S{sid:02d}")
            window_idx.append(t)
            labels.append(lab)
            windows.append(win)
    subject_ids = np.asarray(subject_ids)
    window_idx = np.asarray(window_idx)
    labels = np.asarray(labels)
    windows = np.asarray(windows)

    rho, proj, resid = compute_subspace_retention(V, centroids)
    assert rho.shape == (K,)
    assert proj.shape == centroids.shape
    assert resid.shape == centroids.shape
    assert np.all(rho >= 0.0)
    assert np.all(rho <= 1.0 + 1e-8)

    eval_centroids = compute_centroids_from_labels(windows, labels, K)
    delta = compute_delta_retention(V, eval_centroids)
    assert len(delta["pairs"]) == K * (K - 1) // 2

    manifold = pairwise_distance_alignment(eval_centroids, proj[:K], n_perm=50, seed=7)
    assert "pearson" in manifold and "mantel_p" in manifold

    variance = between_within_variance(windows, labels, eval_centroids, V)
    assert variance["coupled_between_frac"] >= variance["coupled_within_frac"] - 1e-6

    subject_df = build_subject_dynamic_summary(subject_ids, window_idx, labels, windows, V, K)
    assert len(subject_df) == n_subjects
    targets = build_prediction_targets(subject_df, K)
    assert "slow_bundle" in targets and "fast_bundle" in targets

    rank_curve = retention_vs_rank_curve(np.linalg.qr(rng.standard_normal((q, 6)))[0], eval_centroids, ranks=[1, 2, 3])
    assert rank_curve["rho_per_rank"].shape == (3, K)

    print("gm_dfnc_core smoke test passed")


if __name__ == "__main__":
    main()
