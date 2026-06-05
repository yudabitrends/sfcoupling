"""
Synthetic toy data tests for alignment and preprocessing pipeline.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from preprocess import align, covariates, diagnostics, features_fnc, features_gm, split
from preprocess.run_preprocess import load_config, run_pipeline


def _make_synthetic_data(n=20, dx=10, dy=15, seed=42):
    """Generate synthetic GM, FNC, and covariates."""
    rng = np.random.default_rng(seed)
    subject_ids = [f"sub_{i:03d}" for i in range(n)]
    X = rng.standard_normal((n, dx)).astype(np.float32)
    Y = rng.standard_normal((n, dy)).astype(np.float32)
    age = rng.integers(18, 70, n)
    sex = rng.integers(0, 2, n)
    diagnosis = rng.integers(0, 2, n)
    return subject_ids, X, Y, age, sex, diagnosis


def test_align_arrays():
    """Test align.align_arrays produces matching X, Y rows."""
    n, dx, dy = 20, 10, 15
    sub_ids, X_src, Y_src, _, _, _ = _make_synthetic_data(n, dx, dy)
    gm_dict = {s: X_src[i] for i, s in enumerate(sub_ids)}
    fnc_dict = {s: Y_src[i] for i, s in enumerate(sub_ids)}
    order = sorted(sub_ids)
    X, Y = align.align_arrays(gm_dict, fnc_dict, order)
    assert X.shape[0] == Y.shape[0] == len(order)
    assert X.shape[1] == dx
    assert Y.shape[1] == dy
    for i, s in enumerate(order):
        np.testing.assert_array_almost_equal(X[i], gm_dict[s])
        np.testing.assert_array_almost_equal(Y[i], fnc_dict[s])


def test_intersect_subjects():
    """Test subject intersection."""
    gm_ids = {"a", "b", "c", "d"}
    fnc_ids = {"b", "c", "d", "e"}
    cov_ids = {"b", "c"}
    inter = align.intersect_subjects(gm_ids, fnc_ids, cov_ids)
    assert inter == {"b", "c"}


def test_compute_alignment_hash():
    """Test hash is deterministic."""
    order = ["s1", "s2", "s3"]
    h1 = diagnostics.compute_alignment_hash(order)
    h2 = diagnostics.compute_alignment_hash(order)
    assert h1 == h2
    assert h1 != diagnostics.compute_alignment_hash(["s1", "s3", "s2"])


def test_row_integrity_checksums_deterministic():
    """Row checksum sampling must be deterministic."""
    subject_ids = ["s1", "s2", "s3", "s4", "s5"]
    X = np.arange(25, dtype=np.float32).reshape(5, 5)
    Y = (np.arange(15, dtype=np.float32).reshape(5, 3) / 10.0).astype(np.float32)
    c1 = diagnostics.compute_row_integrity_checksums(X, Y, subject_ids, sample_size=3, seed=1337)
    c2 = diagnostics.compute_row_integrity_checksums(X, Y, subject_ids, sample_size=3, seed=1337)
    assert c1 == c2
    assert len(c1) == 3


def test_split_stratified():
    """Test split preserves stratification."""
    sub_ids = [f"s{i}" for i in range(100)]
    stratify = np.array([i % 2 for i in range(100)])
    splits = split.create_splits(sub_ids, stratify, 0.7, 0.15, 42)
    for k in ["train", "val", "test"]:
        assert len(splits[k]) > 0
    all_ids = splits["train"] + splits["val"] + splits["test"]
    assert len(set(all_ids)) == len(all_ids)
    assert set(all_ids) == set(sub_ids)


def test_unseen_category_safe_design_matrix():
    """Unseen categorical levels in dataset2 should not crash and must be reported."""
    cov1 = pd.DataFrame(
        {
            "SubjectID": ["a", "b", "c", "d"],
            "Age": [20, 21, 22, 23],
            "Gender": ["M", "F", "M", "F"],
            "Site": ["S1", "S1", "S1", "S1"],
        }
    )
    cov2 = pd.DataFrame(
        {
            "SubjectID": ["e", "f"],
            "Age": [30, 31],
            "Gender": ["X", "F"],
            "Site": ["S2", "S3"],
        }
    )
    audit = {}
    d1, d2, cols = covariates.build_design_matrices_consistent(
        cov1,
        cov2,
        ["a", "b", "c", "d"],
        ["e", "f"],
        ["Age", "Gender", "Site"],
        "SubjectID",
        ["a", "b"],
        audit,
    )
    assert d1.shape[0] == 4 and d2.shape[0] == 2
    assert cols[0] == "intercept"
    assert "unseen_categories" in audit
    assert "Site" in audit["unseen_categories"]
    assert "S2" in audit["unseen_categories"]["Site"]["dataset2_unseen_values"]


def test_fisher_z_clipping_decision():
    """Near-boundary correlations should clip and remain finite after Fisher-z."""
    vals = np.array([-1.0, -0.9999999, 0.0, 0.9999999, 1.0], dtype=np.float64)
    decision = features_fnc.build_fisherz_decision(
        vals,
        apply_fisher_z_config=True,
        input_was_matrix=False,
        force_correlation=True,
        clip_eps=1e-6,
    )
    out, clip_applied = features_fnc.apply_fisher_z(vals, clip_eps=1e-6)
    assert decision["apply_fisher_z"] is True
    assert clip_applied is True
    assert np.all(np.isfinite(out))


def test_full_pipeline_synthetic():
    """Run full pipeline on synthetic CSV data."""
    n, dx, dy = 20, 10, 15
    sub_ids, X, Y, age, sex, diagnosis = _make_synthetic_data(n, dx, dy)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ds1_dir = tmp / "ds1"
        ds2_dir = tmp / "ds2"
        ds1_dir.mkdir()
        ds2_dir.mkdir()

        n1, n2 = 12, 8
        sub1, sub2 = sub_ids[:n1], sub_ids[n1:]
        X1, X2 = X[:n1], X[n1:]
        Y1, Y2 = Y[:n1], Y[n1:]
        age1, age2 = age[:n1], age[n1:]
        sex1, sex2 = sex[:n1], sex[n1:]
        diag1, diag2 = diagnosis[:n1], diagnosis[n1:]

        cov1 = pd.DataFrame({
            "SubjectID": sub1,
            "Age": age1,
            "Gender": sex1,
            "Diagnosis": diag1,
        })
        cov2 = pd.DataFrame({
            "SubjectID": sub2,
            "Age": age2,
            "Gender": sex2,
            "Diagnosis": diag2,
        })
        gm1 = pd.DataFrame({"SubjectID": sub1, **{f"gm_{j}": X1[:, j] for j in range(dx)}})
        gm2 = pd.DataFrame({"SubjectID": sub2, **{f"gm_{j}": X2[:, j] for j in range(dx)}})
        fnc1 = pd.DataFrame({"SubjectID": sub1, **{f"e{j}": Y1[:, j] for j in range(dy)}})
        fnc2 = pd.DataFrame({"SubjectID": sub2, **{f"e{j}": Y2[:, j] for j in range(dy)}})

        cov1.to_csv(ds1_dir / "covariates.csv", index=False)
        cov2.to_csv(ds2_dir / "covariates.csv", index=False)
        gm1.to_csv(ds1_dir / "gm.csv", index=False)
        gm2.to_csv(ds2_dir / "gm.csv", index=False)
        fnc1.to_csv(ds1_dir / "fnc.csv", index=False)
        fnc2.to_csv(ds2_dir / "fnc.csv", index=False)

        cfg = {
            "dataset1": {
                "gm_path": str(ds1_dir / "gm.csv"),
                "gm_format": "csv",
                "fnc_path": str(ds1_dir / "fnc.csv"),
                "fnc_format": "csv",
                "fnc_column": "SubjectID",
                "covariates_path": str(ds1_dir / "covariates.csv"),
                "covariates_format": "csv",
            },
            "dataset2": {
                "gm_path": str(ds2_dir / "gm.csv"),
                "gm_format": "csv",
                "fnc_path": str(ds2_dir / "fnc.csv"),
                "fnc_format": "csv",
                "fnc_column": "SubjectID",
                "covariates_path": str(ds2_dir / "covariates.csv"),
                "covariates_format": "csv",
            },
            "id_column": "SubjectID",
            "residualization": {
                "residualize_x": True,
                "residualize_y": True,
                "residualize_train_only": True,
                "covariate_cols": ["Age", "Gender"],
            },
            "scaling": {"zscore_train_only": True},
            "fnc": {"n_components": 6, "apply_fisher_z": False, "extract_upper_triangle": False},
            "gm": {"representation": "voxel"},
            "split": {"train_frac": 0.7, "val_frac": 0.15, "stratify_by": "Diagnosis", "seed": 42},
            "output_dir": str(tmp / "out"),
            "splits_dir": str(tmp / "out" / "splits"),
        }

        manifest = run_pipeline(cfg)
        out = Path(cfg["output_dir"])

        X1_loaded = np.load(out / "dataset1_X.npy")
        Y1_loaded = np.load(out / "dataset1_Y.npy")
        X2_loaded = np.load(out / "dataset2_X.npy")
        Y2_loaded = np.load(out / "dataset2_Y.npy")

        assert X1_loaded.shape[0] == Y1_loaded.shape[0] == n1
        assert X2_loaded.shape[0] == Y2_loaded.shape[0] == n2
        assert X1_loaded.shape[1] == X2_loaded.shape[1] == dx
        assert Y1_loaded.shape[1] == Y2_loaded.shape[1] == dy

        assert not np.any(np.isnan(X1_loaded))
        assert not np.any(np.isnan(Y1_loaded))
        assert not np.any(np.isinf(X1_loaded))
        assert not np.any(np.isinf(Y1_loaded))

        report1_path = out / "meta" / "alignment_report_dataset1.json"
        report2_path = out / "meta" / "alignment_report_dataset2.json"
        assert report1_path.exists()
        assert report2_path.exists()
        with open(report1_path) as f:
            report = json.load(f)
        assert report["sanity_passed"] is True
        assert report["alignment"]["x_rows_match_subjects"]
        assert report["alignment"]["y_rows_match_subjects"]
        assert "row_integrity_checksums" in report
        assert (out / "meta" / "audit_report_v2.md").exists()
        assert (out / "meta" / "verification_summary.json").exists()

        split_path = out.parent / "splits" / "dataset1_split.json"
        if split_path.exists():
            with open(split_path) as f:
                splits = json.load(f)
            assert "train" in splits and "val" in splits and "test" in splits
