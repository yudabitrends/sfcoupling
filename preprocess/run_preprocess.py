#!/usr/bin/env python3
"""
Single CLI entrypoint for multimodal GM-FNC preprocessing.
Usage: python preprocess/run_preprocess.py --config preprocess/config.yaml
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from preprocess import align, covariates, diagnostics, features_fnc, features_gm, io, split
else:
    from . import align, covariates, diagnostics, features_fnc, features_gm, io, split

logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict:
    """Load YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)


def _load_covariates(cfg: Dict, dataset_key: str) -> pd.DataFrame:
    """Load covariates DataFrame for a dataset."""
    ds = cfg[dataset_key]
    path = ds.get("covariates_path")
    fmt = ds.get("covariates_format", "h5")
    key = ds.get("covariates_key", "train")
    if fmt == "h5":
        return io.load_h5_table(path, key)
    return io.load_csv_table(path)


def extract_modality(
    cfg: Dict,
    dataset_key: str,
    modality: str,
) -> Tuple[Dict[str, np.ndarray], List[str], Dict]:
    """
    Extract GM or FNC for a dataset based on config.
    modality: "gm" or "fnc"
    Returns: (dict subject_id->array, feature_names)
    """
    ds = cfg[dataset_key]
    id_col = cfg.get("id_column", "SubjectID")

    if modality == "gm":
        path = ds.get("gm_path")
        fmt = ds.get("gm_format", "h5")
        key = ds.get("gm_key", "train")
        col = ds.get("gm_column", "sMRIPath")
        fail_missing = cfg.get("gm_fail_on_missing_nifti", False)
        gm_cfg = cfg.get("gm", {})
        gm_representation = gm_cfg.get("representation", "voxel")
        roi_atlas_path = gm_cfg.get("roi_atlas_path")
        roi_labels_path = gm_cfg.get("roi_labels_path")
        roi_threshold = float(gm_cfg.get("roi_threshold", 0.0))

        audit = {"modality": "gm", "format": fmt, "representation": gm_representation}
        if fmt == "h5":
            df = io.load_h5_table(path, key)
            gm_dict, feat_names = features_gm.extract_gm_from_h5(
                df,
                col,
                id_col,
                fail_on_missing=fail_missing,
                representation=gm_representation,
                roi_atlas_path=roi_atlas_path,
                roi_labels_path=roi_labels_path,
                roi_threshold=roi_threshold,
            )
        elif fmt == "csv":
            gm_dict, feat_names = features_gm.extract_gm_from_csv(path, id_col)
        elif fmt == "csv_paths":
            path_col = ds.get("gm_path_column", "sMRIPath")
            gm_dict, feat_names = features_gm.extract_gm_from_csv_paths(
                path,
                id_col,
                path_col,
                fail_missing,
                representation=gm_representation,
                roi_atlas_path=roi_atlas_path,
                roi_labels_path=roi_labels_path,
                roi_threshold=roi_threshold,
            )
        elif fmt == "npy":
            cov_df = _load_covariates(cfg, dataset_key)
            sub_ids = cov_df[id_col].astype(str).str.strip().tolist()
            gm_dict, feat_names = features_gm.extract_gm_from_npy(path, sub_ids)
        else:
            raise ValueError(f"Unsupported gm_format: {fmt}")
        return gm_dict, feat_names, audit

    if modality == "fnc":
        path = ds.get("fnc_path")
        fmt = ds.get("fnc_format", "h5")
        key = ds.get("fnc_key", "train")
        col = ds.get("fnc_column", "sFNC")
        fnc_cfg = cfg.get("fnc", {})
        n_comp = fnc_cfg.get("n_components", 53)
        fisher_z = fnc_cfg.get("apply_fisher_z", True)
        extract_tri = fnc_cfg.get("extract_upper_triangle", True)
        force_fz = fnc_cfg.get("force_fisher_z", False)

        audit = {"modality": "fnc", "format": fmt}
        if fmt == "h5":
            df = io.load_h5_table(path, key)
            fnc_dict, feat_names = features_fnc.extract_fnc_from_h5(
                df, col, id_col, n_comp, fisher_z, extract_tri,
                force_fisher_z=force_fz, audit_out=audit,
            )
        elif fmt == "csv":
            df = io.load_csv_table(path)
            sub_ids = (
                df[id_col].astype(str).str.strip().tolist()
                if id_col in df.columns
                else None
            )
            fnc_dict, feat_names = features_fnc.extract_fnc_from_csv(
                path, id_col, n_comp, fisher_z, sub_ids,
                force_fisher_z=force_fz, audit_out=audit,
            )
        else:
            raise ValueError(f"Unsupported fnc_format: {fmt}")
        return fnc_dict, feat_names, audit

    raise ValueError(f"Unknown modality: {modality}")


def process_dataset(
    cfg: Dict,
    dataset_key: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame, List[str], List[str], Dict]:
    """
    Full extraction + alignment for one dataset.
    Returns: (X, Y, subject_ids, cov_df, gm_names, fnc_names)
    """
    id_col = cfg.get("id_column", "SubjectID")

    gm_dict, gm_names, gm_audit = extract_modality(cfg, dataset_key, "gm")
    fnc_dict, fnc_names, fnc_audit = extract_modality(cfg, dataset_key, "fnc")
    cov_df = _load_covariates(cfg, dataset_key)

    gm_ids = set(str(k).strip() for k in gm_dict.keys())
    fnc_ids = set(str(k).strip() for k in fnc_dict.keys())
    cov_ids = set(cov_df[id_col].astype(str).str.strip()) if id_col in cov_df.columns else gm_ids

    inter = align.intersect_subjects(gm_ids, fnc_ids, cov_ids)
    subject_order = sorted(inter)
    alignment_proof = align.build_alignment_proof_payload(gm_ids, fnc_ids, cov_ids, inter)

    gm_dict_aligned = {s: gm_dict[s] for s in subject_order}
    fnc_dict_aligned = {s: fnc_dict[s] for s in subject_order}

    X, Y = align.align_arrays(gm_dict_aligned, fnc_dict_aligned, subject_order)

    cov_aligned = cov_df[cov_df[id_col].astype(str).str.strip().isin(subject_order)]
    cov_aligned = cov_aligned.drop_duplicates(subset=id_col)
    cov_aligned = cov_aligned.set_index(id_col)
    cov_aligned = cov_aligned.reindex(subject_order).reset_index()

    return X, Y, subject_order, cov_aligned, gm_names, fnc_names, {
        "alignment_proof": alignment_proof,
        "gm": gm_audit,
        "fnc": fnc_audit,
    }


def _build_subject_hash(subject_ids: List[str]) -> str:
    return diagnostics.compute_alignment_hash(subject_ids)


def _verify_required_files(base_dir: Path, meta_dir: Path) -> Dict:
    files = {
        "dataset1_X": base_dir / "dataset1_X.npy",
        "dataset1_Y": base_dir / "dataset1_Y.npy",
        "dataset2_X": base_dir / "dataset2_X.npy",
        "dataset2_Y": base_dir / "dataset2_Y.npy",
        "dataset1_subjects": meta_dir / "dataset1_subjects.tsv",
        "dataset2_subjects": meta_dir / "dataset2_subjects.tsv",
        "gm_feature_names": meta_dir / "feature_maps" / "gm_feature_names.txt",
        "fnc_edge_names": meta_dir / "feature_maps" / "fnc_edge_names.txt",
    }
    missing = [k for k, p in files.items() if not p.exists()]
    return {"files": {k: str(v) for k, v in files.items()}, "missing": missing}


def _verify_aligned_layout(cfg: Dict, base_dir: Path) -> Dict:
    meta_dir = base_dir / "meta"
    check = _verify_required_files(base_dir, meta_dir)
    if check["missing"]:
        raise FileNotFoundError(f"Missing required output files: {check['missing']}")

    X1 = np.load(base_dir / "dataset1_X.npy")
    Y1 = np.load(base_dir / "dataset1_Y.npy")
    X2 = np.load(base_dir / "dataset2_X.npy")
    Y2 = np.load(base_dir / "dataset2_Y.npy")
    s1 = pd.read_csv(meta_dir / "dataset1_subjects.tsv", sep="\t")
    s2 = pd.read_csv(meta_dir / "dataset2_subjects.tsv", sep="\t")
    id_col = cfg.get("id_column", "SubjectID")
    sub1 = s1[id_col].astype(str).str.strip().tolist()
    sub2 = s2[id_col].astype(str).str.strip().tolist()

    if X1.shape[0] != len(sub1) or Y1.shape[0] != len(sub1):
        raise ValueError("dataset1 dims mismatch with subject list.")
    if X2.shape[0] != len(sub2) or Y2.shape[0] != len(sub2):
        raise ValueError("dataset2 dims mismatch with subject list.")
    if np.any(~np.isfinite(X1)) or np.any(~np.isfinite(Y1)) or np.any(~np.isfinite(X2)) or np.any(~np.isfinite(Y2)):
        raise ValueError("NaN/Inf detected in output arrays.")

    gm_names = [x.strip() for x in (meta_dir / "feature_maps" / "gm_feature_names.txt").read_text().splitlines() if x.strip()]
    fnc_names = [x.strip() for x in (meta_dir / "feature_maps" / "fnc_edge_names.txt").read_text().splitlines() if x.strip()]
    if len(gm_names) != X1.shape[1]:
        raise ValueError(f"GM feature-name count mismatch: {len(gm_names)} != {X1.shape[1]}")
    if len(fnc_names) != Y1.shape[1]:
        raise ValueError(f"FNC feature-name count mismatch: {len(fnc_names)} != {Y1.shape[1]}")

    strict_roi = cfg.get("verify", {}).get("strict_roi", True)
    allow_voxelwise = cfg.get("verify", {}).get("allow_voxelwise", False)
    if strict_roi and not allow_voxelwise and X1.shape[1] > 5000:
        raise ValueError("Strict ROI verification failed: dx > 5000 and voxelwise not allowed.")

    fnc_cfg = cfg.get("fnc", {})
    if fnc_cfg.get("extract_upper_triangle", False):
        n = int(fnc_cfg.get("n_components", 53))
        expected = n * (n - 1) // 2
        if Y1.shape[1] != expected:
            raise ValueError(f"Upper-triangle check failed: dy={Y1.shape[1]} expected={expected}")

    edge_summary = {"first_5_edges": fnc_names[:5], "last_5_edges": fnc_names[-5:]}
    out = {
        "layout": "aligned_features",
        "verified": True,
        "dataset1": {"n": len(sub1), "dx": int(X1.shape[1]), "dy": int(Y1.shape[1]), "subject_hash": _build_subject_hash(sub1)},
        "dataset2": {"n": len(sub2), "dx": int(X2.shape[1]), "dy": int(Y2.shape[1]), "subject_hash": _build_subject_hash(sub2)},
        "edge_name_ordering": edge_summary,
    }
    rp1 = meta_dir / "alignment_report_dataset1.json"
    rp2 = meta_dir / "alignment_report_dataset2.json"
    if rp1.exists() and rp2.exists():
        with open(rp1) as f:
            rr1 = json.load(f)
        with open(rp2) as f:
            rr2 = json.load(f)
        if rr1.get("subject_order_sha256") and rr1.get("subject_order_sha256") != out["dataset1"]["subject_hash"]:
            raise ValueError("dataset1 subject hash mismatch with recorded report.")
        if rr2.get("subject_order_sha256") and rr2.get("subject_order_sha256") != out["dataset2"]["subject_hash"]:
            raise ValueError("dataset2 subject hash mismatch with recorded report.")
    return out


def _verify_final_pairs_layout(cfg: Dict, base_dir: Path) -> Dict:
    by_dataset = base_dir / "by_dataset"
    if not by_dataset.exists():
        raise FileNotFoundError(f"Missing by_dataset directory: {by_dataset}")
    datasets = []
    checks = []
    for xfile in sorted(by_dataset.glob("*_X.npy")):
        name = xfile.name[:-6]
        yfile = by_dataset / f"{name}_Y.npy"
        sfile = by_dataset / f"{name}_subjects.csv"
        gfile = by_dataset / f"{name}_gm_feature_names.txt"
        ffile = by_dataset / f"{name}_fnc_feature_names.txt"
        if not (yfile.exists() and sfile.exists() and gfile.exists() and ffile.exists()):
            raise FileNotFoundError(f"Missing paired files for dataset {name} in {by_dataset}")
        X = np.load(xfile)
        Y = np.load(yfile)
        subs = pd.read_csv(sfile)
        id_col = "SubjectID" if "SubjectID" in subs.columns else subs.columns[0]
        subject_ids = subs[id_col].astype(str).str.strip().tolist()
        if X.shape[0] != len(subject_ids) or Y.shape[0] != len(subject_ids):
            raise ValueError(f"{name}: row count mismatch among X/Y/subjects.")
        gm_names = [x.strip() for x in gfile.read_text().splitlines() if x.strip()]
        fnc_names = [x.strip() for x in ffile.read_text().splitlines() if x.strip()]
        if len(gm_names) != X.shape[1]:
            raise ValueError(f"{name}: gm_feature_names count mismatch.")
        if len(fnc_names) != Y.shape[1]:
            raise ValueError(f"{name}: fnc_feature_names count mismatch.")
        if np.any(~np.isfinite(X)) or np.any(~np.isfinite(Y)):
            raise ValueError(f"{name}: NaN/Inf found in arrays.")
        strict_roi = cfg.get("verify", {}).get("strict_roi", True)
        allow_voxelwise = cfg.get("verify", {}).get("allow_voxelwise", False)
        if strict_roi and not allow_voxelwise and X.shape[1] > 5000:
            raise ValueError(f"{name}: strict ROI check failed (dx > 5000).")
        datasets.append(
            {
                "name": name,
                "n": len(subject_ids),
                "dx": int(X.shape[1]),
                "dy": int(Y.shape[1]),
                "subject_hash": _build_subject_hash(subject_ids),
            }
        )
        checks.append({"name": name, "first_5_edges": fnc_names[:5], "last_5_edges": fnc_names[-5:]})
    return {"layout": "final_pairs", "verified": True, "datasets": datasets, "edge_name_ordering": checks}


def verify_outputs_only(cfg: Dict) -> Dict:
    """
    Verify already-produced outputs without re-running preprocessing.
    Supports aligned_features-style layout.
    """
    requested = cfg.get("verify", {}).get("target_path")
    default_base = Path(cfg.get("output_dir", "aligned_features")).resolve()
    base_dir = Path(requested).resolve() if requested else default_base

    aligned_like = (base_dir / "meta").exists() and (base_dir / "dataset1_X.npy").exists()
    final_pairs_like = (base_dir / "by_dataset").exists() or (base_dir.name == "final_pairs" and (base_dir / "by_dataset").exists())
    if aligned_like:
        summary = _verify_aligned_layout(cfg, base_dir)
        with open(base_dir / "meta" / "verification_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return summary
    if final_pairs_like:
        fp_base = base_dir
        summary = _verify_final_pairs_layout(cfg, fp_base)
        out_file = fp_base / "verification_summary.json"
        with open(out_file, "w") as f:
            json.dump(summary, f, indent=2)
        return summary
    raise FileNotFoundError(
        f"Could not auto-detect output layout at {base_dir}. "
        "Expected aligned_features layout or final_pairs/by_dataset layout."
    )


def _write_audit_report_v2(
    meta_dir: Path,
    report1: Dict,
    report2: Dict,
    verification_summary: Optional[Dict],
    cov_cols: List[str],
    design_cols: List[str],
) -> None:
    lines = []
    lines.append("# Preprocessing Audit Report v2")
    lines.append("")
    lines.append("## Dataset Shapes")
    lines.append(f"- dataset1: N={report1.get('n_subjects')} dx={report1.get('dx')} dy={report1.get('dy')}")
    lines.append(f"- dataset2: N={report2.get('n_subjects')} dx={report2.get('dx')} dy={report2.get('dy')}")
    lines.append("")
    lines.append("## Alignment And Drops")
    a1 = report1.get("alignment_proof", {})
    a2 = report2.get("alignment_proof", {})
    lines.append(f"- dataset1 pre={a1.get('counts_pre_alignment', {})} post={a1.get('counts_post_alignment', {})} drops={a1.get('dropped_reasons', {})}")
    lines.append(f"- dataset2 pre={a2.get('counts_pre_alignment', {})} post={a2.get('counts_post_alignment', {})} drops={a2.get('dropped_reasons', {})}")
    lines.append("")
    lines.append("## Covariates And Design")
    lines.append(f"- covariates_used: {cov_cols}")
    lines.append(f"- design_matrix_columns: {design_cols}")
    lines.append("")
    lines.append("## Fisher-z Decision")
    lines.append(f"- dataset1: {report1.get('fisher_z_decision', {})}")
    lines.append(f"- dataset2: {report2.get('fisher_z_decision', {})}")
    lines.append("")
    lines.append("## Hash Proofs")
    lines.append(f"- dataset1 subject_order_sha256: {report1.get('subject_order_sha256')}")
    lines.append(f"- dataset2 subject_order_sha256: {report2.get('subject_order_sha256')}")
    lines.append(f"- dataset1 sampled_row_checksums: {len(report1.get('row_integrity_checksums', []))}")
    lines.append(f"- dataset2 sampled_row_checksums: {len(report2.get('row_integrity_checksums', []))}")
    lines.append("")
    lines.append("## Leakage And Splits")
    lines.append(f"- dataset1 split_proof: {report1.get('split_proof', {})}")
    lines.append(f"- dataset2 split_proof: {report2.get('split_proof', {})}")
    lines.append(f"- dataset1 leakage_checks: {report1.get('leakage_checks', {})}")
    lines.append(f"- dataset2 leakage_checks: {report2.get('leakage_checks', {})}")
    if verification_summary:
        lines.append("")
        lines.append("## Verification Summary")
        lines.append(f"- {verification_summary}")
    (meta_dir / "audit_report_v2.md").write_text("\n".join(lines))


def run_pipeline(cfg: Dict, base_dir: Optional[Path] = None) -> Dict:
    """
    Run full preprocessing pipeline. Returns manifest dict.
    """
    base_dir = Path(cfg.get("output_dir", "aligned_features")).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = base_dir / "meta"
    meta_dir.mkdir(exist_ok=True)
    splits_dir = Path(cfg.get("splits_dir", "splits"))
    if not splits_dir.is_absolute():
        splits_dir = base_dir.parent / splits_dir
    splits_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config_path": str(cfg.get("_config_path", "")),
        "timestamp": datetime.now().isoformat(),
        "datasets": {},
        "decisions": {},
    }

    X1, Y1, sub1, cov1, gm_names1, fnc_names1, audit1 = process_dataset(cfg, "dataset1")
    X2, Y2, sub2, cov2, gm_names2, fnc_names2, audit2 = process_dataset(cfg, "dataset2")

    # --- Remove dataset2 subjects that overlap with dataset1 ---
    id_col = cfg.get("id_column", "SubjectID")
    set1 = set(sub1)
    set2 = set(sub2)
    overlap = set1 & set2
    if overlap:
        logger.warning(
            "Found %d subjects in BOTH dataset1 and dataset2. "
            "Removing them from dataset2 to prevent leakage.", len(overlap)
        )
        keep_mask = np.array([s not in overlap for s in sub2])
        X2 = X2[keep_mask]
        Y2 = Y2[keep_mask]
        sub2 = [s for s in sub2 if s not in overlap]
        cov2 = cov2[cov2[id_col].astype(str).str.strip().isin(sub2)].reset_index(drop=True)
        # Re-sort cov2 to match sub2 order after overlap removal
        cov2 = cov2.set_index(cov2[id_col].astype(str).str.strip()).reindex(sub2).reset_index(drop=True)
        logger.info("dataset2 after dedup: %d subjects (removed %d)", len(sub2), len(overlap))
    manifest["decisions"]["inter_dataset_overlap"] = {
        "n_overlap": len(overlap),
        "removed_from": "dataset2",
        "overlap_subjects": sorted(overlap),
    }

    if X1.shape[1] != X2.shape[1] or Y1.shape[1] != Y2.shape[1]:
        raise ValueError(
            f"Feature dims must match: dataset1 X={X1.shape[1]} Y={Y1.shape[1]}, "
            f"dataset2 X={X2.shape[1]} Y={Y2.shape[1]}"
        )

    dx, dy = X1.shape[1], Y1.shape[1]
    id_col = cfg.get("id_column", "SubjectID")
    res_cfg = cfg.get("residualization", {})
    scale_cfg = cfg.get("scaling", {})
    split_cfg = cfg.get("split", {})

    # ── Compute total GM (TIV proxy) and inject into covariates ──
    cov_cols_cfg = res_cfg.get("covariate_cols", ["Age", "Gender", "Site"])
    if "total_gm" in cov_cols_cfg:
        tgm1 = X1.sum(axis=1)
        tgm2 = X2.sum(axis=1)
        cov1["total_gm"] = tgm1
        cov2["total_gm"] = tgm2
        logger.info("Injected total_gm (TIV proxy): DS1 mean=%.2f std=%.2f, DS2 mean=%.2f std=%.2f",
                     tgm1.mean(), tgm1.std(), tgm2.mean(), tgm2.std())

    stratify_col = split_cfg.get("stratify_by")
    cov_cols = res_cfg.get("covariate_cols", ["Age", "Gender", "Site"])
    available_cov = [c for c in cov_cols if c in cov1.columns or c in cov2.columns]

    strat1 = None
    if stratify_col and stratify_col in cov1.columns:
        strat1 = cov1[stratify_col].values
        try:
            strat1 = pd.to_numeric(strat1, errors="coerce")
            strat1 = np.nan_to_num(strat1, nan=-1)
        except Exception:
            strat1 = np.array([hash(str(x)) % 1000 for x in strat1])
    strat2 = None
    if stratify_col and stratify_col in cov2.columns:
        strat2 = cov2[stratify_col].values
        try:
            strat2 = pd.to_numeric(strat2, errors="coerce")
            strat2 = np.nan_to_num(strat2, nan=-1)
        except Exception:
            strat2 = np.array([hash(str(x)) % 1000 for x in strat2])

    splits1 = split.create_splits(
        sub1,
        strat1,
        split_cfg.get("train_frac", 0.7),
        split_cfg.get("val_frac", 0.15),
        split_cfg.get("seed", 42),
    )
    splits2 = split.create_splits(
        sub2,
        strat2,
        split_cfg.get("train_frac", 0.7),
        split_cfg.get("val_frac", 0.15),
        split_cfg.get("seed", 42),
    )

    train_mask1 = np.array([s in splits1["train"] for s in sub1])
    design_audit: Dict = {}

    try:
        design1, design2, design_cols = covariates.build_design_matrices_consistent(
            cov1, cov2, sub1, sub2, available_cov, id_col, splits1["train"], design_audit
        )
    except (ValueError, KeyError) as e:
        logger.warning("Could not build consistent design matrices: %s. Using intercept only.", e)
        design1 = np.ones((len(sub1), 1))
        design2 = np.ones((len(sub2), 1))
        design_cols = ["intercept"]

    X1r, Y1r, res_params, lr_x, lr_y = covariates.residualize(
        X1,
        Y1,
        design1,
        train_mask1,
        res_cfg.get("residualize_x", True),
        res_cfg.get("residualize_y", True),
    )

    X2r, Y2r = covariates.apply_residualization(
        X2, Y2, design2, lr_x, lr_y,
        res_cfg.get("residualize_x", True),
        res_cfg.get("residualize_y", True),
    )

    scaler_x, scaler_y = covariates.fit_scalers_train_only(X1r, Y1r, train_mask1)
    X1s, Y1s = covariates.apply_scalers(X1r, Y1r, scaler_x, scaler_y)
    X2s, Y2s = covariates.apply_scalers(X2r, Y2r, scaler_x, scaler_y)

    if cfg.get("save_intermediate_arrays", True):
        np.save(base_dir / "dataset1_X_raw.npy", X1.astype(np.float32))
        np.save(base_dir / "dataset1_Y_raw.npy", Y1.astype(np.float32))
        np.save(base_dir / "dataset2_X_raw.npy", X2.astype(np.float32))
        np.save(base_dir / "dataset2_Y_raw.npy", Y2.astype(np.float32))
        np.save(base_dir / "dataset1_X_resid.npy", X1r.astype(np.float32))
        np.save(base_dir / "dataset1_Y_resid.npy", Y1r.astype(np.float32))
        np.save(base_dir / "dataset2_X_resid.npy", X2r.astype(np.float32))
        np.save(base_dir / "dataset2_Y_resid.npy", Y2r.astype(np.float32))

    np.save(base_dir / "dataset1_X.npy", X1s.astype(np.float32))
    np.save(base_dir / "dataset1_Y.npy", Y1s.astype(np.float32))
    np.save(base_dir / "dataset2_X.npy", X2s.astype(np.float32))
    np.save(base_dir / "dataset2_Y.npy", Y2s.astype(np.float32))

    cov1_out = cov1.copy()
    cov1_out.to_csv(meta_dir / "dataset1_subjects.tsv", sep="\t", index=False)
    cov2_out = cov2.copy()
    cov2_out.to_csv(meta_dir / "dataset2_subjects.tsv", sep="\t", index=False)

    (meta_dir / "feature_maps").mkdir(exist_ok=True)
    with open(meta_dir / "feature_maps" / "gm_feature_names.txt", "w") as f:
        f.write("\n".join(gm_names1))
    with open(meta_dir / "feature_maps" / "fnc_edge_names.txt", "w") as f:
        f.write("\n".join(fnc_names1))

    rp = {
        "covariate_cols_used": design_cols,
        "residualization": {k: v for k, v in res_params.items() if v is not None},
        "scaler_x_mean": scaler_x.mean_.tolist(),
        "scaler_x_scale": scaler_x.scale_.tolist(),
        "scaler_y_mean": scaler_y.mean_.tolist(),
        "scaler_y_scale": scaler_y.scale_.tolist(),
        "design_matrix_columns": design_audit.get("design_matrix_columns", design_cols),
        "categorical_mappings": design_audit.get("categorical_mappings", {}),
        "unseen_categories": design_audit.get("unseen_categories", {}),
        "fitted_on": {
            "dataset": "dataset1",
            "split": "train",
            "n_subjects": int(len(splits1["train"])),
            "subject_hash": _build_subject_hash(sorted(splits1["train"])),
        },
    }
    with open(meta_dir / "residualization_params.json", "w") as f:
        json.dump(rp, f, indent=2)
    with open(meta_dir / "covariates_used.json", "w") as f:
        json.dump({"covariate_cols": available_cov, "design_cols": design_cols}, f, indent=2)

    train_mask_arr = np.array([s in splits1["train"] for s in sub1])
    report1_base = diagnostics.run_diagnostics(
        X1s, Y1s, sub1, scalers_fit_on_train=True, train_mask=train_mask_arr
    )
    report2_base = diagnostics.run_diagnostics(
        X2s, Y2s, sub2, scalers_fit_on_train=True, train_mask=None
    )
    feature_map_checks = {
        "gm_feature_count_matches_dx": len(gm_names1) == X1s.shape[1],
        "fnc_feature_count_matches_dy": len(fnc_names1) == Y1s.shape[1],
        "gm_feature_count": len(gm_names1),
        "fnc_feature_count": len(fnc_names1),
    }
    report1 = diagnostics.augment_report(
        report1_base,
        "dataset1",
        audit1.get("alignment_proof", {}),
        split.build_split_proof(splits1, sanity_only=False),
        fisherz_decision=audit1.get("fnc", {}).get("decision", {}),
        feature_map_checks=feature_map_checks,
        design_matrix_proof=design_audit,
    )
    report2 = diagnostics.augment_report(
        report2_base,
        "dataset2",
        audit2.get("alignment_proof", {}),
        split.build_split_proof(splits2, sanity_only=True),
        fisherz_decision=audit2.get("fnc", {}).get("decision", {}),
        feature_map_checks=feature_map_checks,
        design_matrix_proof=design_audit,
    )
    diagnostics.write_alignment_report(report1, str(meta_dir / "alignment_report_dataset1.json"))
    diagnostics.write_alignment_report(report2, str(meta_dir / "alignment_report_dataset2.json"))
    diagnostics.write_alignment_report(
        {"dataset1_report": "alignment_report_dataset1.json", "dataset2_report": "alignment_report_dataset2.json"},
        str(meta_dir / "alignment_report.json"),
    )

    with open(splits_dir / "dataset1_split.json", "w") as f:
        json.dump(splits1, f, indent=2)
    with open(splits_dir / "dataset2_split.json", "w") as f:
        json.dump(splits2, f, indent=2)
    ext_map = split.create_external_test_mapping(splits1, splits2)
    with open(splits_dir / "external_test_mapping.json", "w") as f:
        json.dump(ext_map, f, indent=2)

    manifest["datasets"]["dataset1"] = {"n": len(sub1), "dx": dx, "dy": dy}
    manifest["datasets"]["dataset2"] = {"n": len(sub2), "dx": dx, "dy": dy}
    manifest["decisions"].update({
        "covariates_used": available_cov,
        "residualize_train_only": res_cfg.get("residualize_train_only", True),
        "zscore_train_only": scale_cfg.get("zscore_train_only", True),
        "seed": split_cfg.get("seed", 42),
    })
    verification_summary = verify_outputs_only(cfg)
    _write_audit_report_v2(meta_dir, report1, report2, verification_summary, available_cov, design_cols)
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal GM-FNC preprocessing for structure-function coupling"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="preprocess/config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output_dir from config",
    )
    parser.add_argument(
        "--verify_outputs_only",
        action="store_true",
        help="Only verify existing outputs without re-running pipeline",
    )
    args = parser.parse_args()

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    cfg = load_config(args.config)
    cfg["_config_path"] = args.config
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
        cfg["splits_dir"] = str(Path(args.output_dir).parent / "splits")

    try:
        if args.verify_outputs_only:
            verify = verify_outputs_only(cfg)
            print(json.dumps(verify, indent=2))
            return
        manifest = run_pipeline(cfg)
        manifest_path = Path(cfg.get("output_dir", "aligned_features")) / "meta" / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Preprocessing complete. Manifest: %s", manifest_path)
        print(f"\nPreprocessing complete. Outputs in {cfg.get('output_dir', 'aligned_features')}")
    except Exception as e:
        logger.exception("Preprocessing failed")
        print(f"\nERROR: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
