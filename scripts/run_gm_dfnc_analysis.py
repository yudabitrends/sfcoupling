#!/usr/bin/env python3
"""Run the full GM-dFNC analysis suite.

The pipeline is config-driven and supports:
  - rebuilding static GM->FNC maps from existing training configs
  - loading dynamic FNC windows / state labels / centroids
  - state geometry, manifold, hierarchy, and residual analyses
  - between-state vs within-state variance decomposition
  - subject-level dynamic phenotype prediction from GM
  - dynamic-feature clinical utility checks

Example
-------
python scripts/run_gm_dfnc_analysis.py --config train/config_gm_dfnc_template.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from models.baselines import fit_ridge_grid

from gm_dfnc_core import (
    analyze_residuals,
    between_within_variance,
    bootstrap_auc_ci,
    bootstrap_state_retention,
    build_prediction_targets,
    build_subject_dynamic_summary,
    case_control_state_effects,
    compute_centroids_from_labels,
    compute_delta_retention,
    compute_subspace_retention,
    fit_logistic_grid,
    fit_static_method,
    load_array,
    load_table,
    orthonormalize,
    pairwise_distance_alignment,
    per_mode_contribution,
    principal_angles,
    regression_summary,
    retention_vs_rank_curve,
    rotation_null_distribution,
    save_tsv,
    select_rank,
    split_subject_frame,
    subspace_overlap,
    tier_retention,
    top_k_subspace,
    vectorize_symmetric_block,
    write_summary_json,
    within_state_local_overlaps,
)


ID_CANDIDATES = ["subject_id", "SubjectID", "eid", "ID", "subject"]
WINIDX_CANDIDATES = ["window_idx", "window_index", "WindowIndex", "time_idx"]
DIAG_CANDIDATES = ["Diagnosis", "diagnosis", "DX", "dx"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GM-dFNC full analysis")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def infer_column(columns: Iterable[str], explicit: Optional[str], candidates: Sequence[str]) -> str:
    if explicit is not None:
        return explicit
    cols = list(columns)
    for cand in candidates:
        if cand in cols:
            return cand
    raise ValueError(f"Could not infer column from candidates {candidates}; available={cols}")


def normalize_state_labels(labels: np.ndarray, K_hint: Optional[int] = None) -> Tuple[np.ndarray, Dict[Any, int]]:
    labels = np.asarray(labels)
    uniq = sorted(pd.unique(labels).tolist())
    if K_hint is not None and len(uniq) != int(K_hint):
        # Preserve observed labels even if some states are absent in the current subset.
        if labels.min() >= 0 and labels.max() < int(K_hint):
            return labels.astype(np.int64), {int(i): int(i) for i in range(int(K_hint))}
    mapping = {raw: idx for idx, raw in enumerate(uniq)}
    out = np.asarray([mapping[val] for val in labels], dtype=np.int64)
    return out, mapping


def load_dynamic_base(dataset_spec: Dict[str, Any], dy: int) -> Dict[str, Any]:
    windows = load_array(dataset_spec["windows_path"], dataset_spec.get("windows_key"))
    windows = vectorize_symmetric_block(np.asarray(windows, dtype=np.float32))
    if windows.shape[1] != dy:
        raise ValueError(
            f"{dataset_spec['windows_path']} has dy={windows.shape[1]} but static maps expect {dy}"
        )
    meta = load_table(dataset_spec["metadata_path"], dataset_spec.get("metadata_sep"))
    subject_col = infer_column(meta.columns, dataset_spec.get("subject_id_col"), ID_CANDIDATES)
    window_col = infer_column(meta.columns, dataset_spec.get("window_idx_col"), WINIDX_CANDIDATES)
    if len(meta) != windows.shape[0]:
        raise ValueError(
            f"Dynamic metadata rows ({len(meta)}) != windows rows ({windows.shape[0]}) for {dataset_spec['metadata_path']}"
        )
    meta = meta.copy()
    meta["_dynamic_row_idx"] = np.arange(len(meta), dtype=np.int64)
    meta["subject_id"] = meta[subject_col].astype(str)
    meta["window_idx"] = meta[window_col].astype(int)
    return {
        "windows": windows,
        "meta": meta,
        "subject_col": subject_col,
        "window_col": window_col,
        "dataset_spec": dataset_spec,
    }


def attach_static_subject_info(
    meta: pd.DataFrame,
    reference_fit: Dict[str, Any],
    static_dataset: str,
) -> pd.DataFrame:
    if static_dataset not in {"dataset1", "dataset2"}:
        raise ValueError(f"Unsupported static_dataset: {static_dataset}")
    subjects_all = split_subject_frame(reference_fit, f"{static_dataset}_all")
    keep_cols = ["subject_id", "_X_index", "_X_matrix"]
    keep_cols += [
        c for c in subjects_all.columns
        if not c.startswith("_") and c not in meta.columns and c != "subject_id"
    ]
    merged = meta.merge(subjects_all[keep_cols], on="subject_id", how="inner")

    if static_dataset == "dataset1":
        train_ids = set(split_subject_frame(reference_fit, "dataset1_train")["subject_id"])
        val_ids = set(split_subject_frame(reference_fit, "dataset1_val")["subject_id"])
        test_ids = set(split_subject_frame(reference_fit, "dataset1_test")["subject_id"])
        split_name = []
        for sid in merged["subject_id"]:
            if sid in train_ids:
                split_name.append("train")
            elif sid in val_ids:
                split_name.append("val")
            elif sid in test_ids:
                split_name.append("test")
            else:
                split_name.append("unknown")
        merged["static_split"] = split_name
    else:
        merged["static_split"] = "external"
    return merged


def prepare_dynamic_solution(
    base: Dict[str, Any],
    solution_spec: Dict[str, Any],
    reference_fit: Dict[str, Any],
) -> Dict[str, Any]:
    meta = base["meta"].copy()
    static_dataset = solution_spec.get("static_dataset") or base["dataset_spec"]["static_dataset"]
    meta = attach_static_subject_info(meta, reference_fit, static_dataset)
    row_idx = meta["_dynamic_row_idx"].to_numpy(dtype=np.int64)
    windows = base["windows"][row_idx]

    if "state_label_col" in solution_spec:
        raw_labels = meta[solution_spec["state_label_col"]].to_numpy()
    elif "labels_path" in solution_spec:
        raw_labels = load_array(solution_spec["labels_path"], solution_spec.get("labels_key"))
        raw_labels = np.asarray(raw_labels)[row_idx]
    else:
        raise ValueError(f"Solution {solution_spec.get('label', '<unnamed>')} requires labels")

    K_hint = solution_spec.get("K")
    labels, _ = normalize_state_labels(raw_labels, K_hint=K_hint)
    K = int(K_hint or (labels.max() + 1))

    centroids_ref = None
    if solution_spec.get("centroids_path"):
        centroids_ref = load_array(solution_spec["centroids_path"], solution_spec.get("centroids_key"))
        centroids_ref = vectorize_symmetric_block(np.asarray(centroids_ref, dtype=np.float64))
    else:
        centroid_split = solution_spec.get("centroid_split")
        if centroid_split is None:
            centroids_ref = compute_centroids_from_labels(windows, labels, K)
        else:
            split_mask = meta["static_split"].astype(str).eq(str(centroid_split)).to_numpy()
            centroids_ref = compute_centroids_from_labels(windows[split_mask], labels[split_mask], K)

    return {
        "label": solution_spec["label"],
        "windowing_label": solution_spec.get("windowing_label", "default"),
        "report_name": solution_spec.get("report_name") or base["dataset_spec"].get("report_name"),
        "static_dataset": static_dataset,
        "windows": windows,
        "meta": meta.reset_index(drop=True),
        "labels": labels.astype(np.int64),
        "K": K,
        "reference_centroids": centroids_ref.astype(np.float64),
        "base_spec": base["dataset_spec"],
        "solution_spec": solution_spec,
    }


def solution_eval_mask(solution: Dict[str, Any]) -> np.ndarray:
    if solution["static_dataset"] == "dataset1":
        return solution["meta"]["static_split"].astype(str).eq("test").to_numpy()
    return np.ones(len(solution["meta"]), dtype=bool)


def eval_label_for_solution(solution: Dict[str, Any]) -> str:
    base = solution["report_name"] or solution["static_dataset"]
    if solution["static_dataset"] == "dataset1":
        return f"{base}"
    return f"{base}"


def save_detail_arrays(
    out_dir: Path,
    cohort: str,
    solution_label: str,
    method: str,
    seed: int,
    rank: int,
    payload: Dict[str, Any],
) -> str:
    arrays_dir = out_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    safe = f"{cohort}__{solution_label}__{method}__seed{seed}__r{rank}".replace("/", "_")
    path = arrays_dir / f"{safe}.npz"
    np.savez_compressed(path, **payload)
    return str(path)


def build_clinical_feature_sets(subject_df: pd.DataFrame, K: int) -> Dict[str, np.ndarray]:
    occ_cols = [f"occupancy_s{s}" for s in range(K)]
    dwell_cols = [f"dwell_s{s}" for s in range(K)]
    trans_cols = [f"trans_prob_s{i}_to_s{j}" for i in range(K) for j in range(K)]
    full_cols = [f"full_energy_s{s}" for s in range(K)]
    coup_cols = [f"coupled_energy_s{s}" for s in range(K)]
    unc_cols = [f"uncoupled_energy_s{s}" for s in range(K)]
    common = np.concatenate([
        subject_df[occ_cols].to_numpy(dtype=np.float64),
        np.log1p(subject_df[dwell_cols].to_numpy(dtype=np.float64)),
        np.log1p(subject_df[["switching_rate", "transition_entropy"]].to_numpy(dtype=np.float64)),
        subject_df[trans_cols].to_numpy(dtype=np.float64),
    ], axis=1)
    return {
        "full": np.concatenate([common, np.log1p(subject_df[full_cols].to_numpy(dtype=np.float64))], axis=1),
        "coupled": np.concatenate([common, np.log1p(subject_df[coup_cols].to_numpy(dtype=np.float64))], axis=1),
        "uncoupled": np.concatenate([common, np.log1p(subject_df[unc_cols].to_numpy(dtype=np.float64))], axis=1),
    }


def aggregate_table(df: pd.DataFrame, value_cols: Sequence[str], group_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    agg_spec = {}
    for col in value_cols:
        agg_spec[col] = ["mean", "std"]
    out = df.groupby(list(group_cols), dropna=False).agg(agg_spec)
    out.columns = ["_".join(col).strip("_") for col in out.columns.to_flat_index()]
    return out.reset_index()


def run_prediction_suite(
    out_rows: List[Dict[str, Any]],
    map_source: str,
    method: str,
    seed: int,
    rank: int,
    dataset1_solution: Dict[str, Any],
    dataset2_solution: Optional[Dict[str, Any]],
    static_fit: Dict[str, Any],
    V_r: np.ndarray,
    analysis_cfg: Dict[str, Any],
) -> None:
    alphas = analysis_cfg.get("prediction_alphas", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    K = dataset1_solution["K"]

    def subject_summary_for_solution(solution: Dict[str, Any]) -> pd.DataFrame:
        meta = solution["meta"]
        return build_subject_dynamic_summary(
            meta["subject_id"].to_numpy(dtype=str),
            meta["window_idx"].to_numpy(dtype=np.int64),
            solution["labels"],
            solution["windows"],
            V_r,
            K,
        )

    summary_d1 = subject_summary_for_solution(dataset1_solution).reset_index(drop=True)
    summary_d1["_summary_idx"] = np.arange(len(summary_d1), dtype=np.int64)
    targets_d1 = build_prediction_targets(summary_d1, K)

    train_df = split_subject_frame(static_fit, "dataset1_train")
    val_df = split_subject_frame(static_fit, "dataset1_val")
    test_df = split_subject_frame(static_fit, "dataset1_test")

    merged_train = train_df.merge(summary_d1, on="subject_id", how="inner")
    merged_val = val_df.merge(summary_d1, on="subject_id", how="inner")
    merged_test = test_df.merge(summary_d1, on="subject_id", how="inner")
    if merged_train.empty or merged_val.empty or merged_test.empty:
        return

    X1 = static_fit["contracts"]["X1"]
    X_train = X1[merged_train["_X_index"].to_numpy(dtype=np.int64)]
    X_val = X1[merged_val["_X_index"].to_numpy(dtype=np.int64)]
    X_test = X1[merged_test["_X_index"].to_numpy(dtype=np.int64)]

    external_payload = None
    if dataset2_solution is not None:
        summary_d2 = subject_summary_for_solution(dataset2_solution).reset_index(drop=True)
        summary_d2["_summary_idx"] = np.arange(len(summary_d2), dtype=np.int64)
        ext_df = split_subject_frame(static_fit, "dataset2_all")
        merged_ext = ext_df.merge(summary_d2, on="subject_id", how="inner")
        if not merged_ext.empty:
            X2 = static_fit["contracts"]["X2"]
            X_ext = X2[merged_ext["_X_index"].to_numpy(dtype=np.int64)]
            external_payload = (summary_d2, merged_ext, X_ext, dataset2_solution["report_name"])

    for target_name, target_info in targets_d1.items():
        Y_d1 = target_info["Y"]
        Y_train = Y_d1[merged_train["_summary_idx"].to_numpy(dtype=np.int64)]
        Y_val = Y_d1[merged_val["_summary_idx"].to_numpy(dtype=np.int64)]
        Y_test = Y_d1[merged_test["_summary_idx"].to_numpy(dtype=np.int64)]
        model, info = fit_ridge_grid(X_train, Y_train, X_val, Y_val, alphas)
        pred_test = model.predict(X_test)
        metrics_test = regression_summary(Y_test, pred_test)
        out_rows.append({
            "map_source": map_source,
            "cohort": dataset1_solution["report_name"],
            "target_group": target_name,
            "method": method,
            "seed": seed,
            "rank": rank,
            "best_alpha": info["best_alpha"],
            **metrics_test,
        })

        if external_payload is not None:
            summary_d2, merged_ext, X_ext, cohort_name = external_payload
            Y_ext_info = build_prediction_targets(summary_d2, K)[target_name]
            Y_ext = Y_ext_info["Y"][merged_ext["_summary_idx"].to_numpy(dtype=np.int64)]
            pred_ext = model.predict(X_ext)
            metrics_ext = regression_summary(Y_ext, pred_ext)
            out_rows.append({
                "map_source": map_source,
                "cohort": cohort_name,
                "target_group": target_name,
                "method": method,
                "seed": seed,
                "rank": rank,
                "best_alpha": info["best_alpha"],
                **metrics_ext,
            })


def run_clinical_suite(
    auc_rows: List[Dict[str, Any]],
    effect_rows: List[Dict[str, Any]],
    map_source: str,
    method: str,
    seed: int,
    rank: int,
    dataset1_solution: Dict[str, Any],
    dataset2_solution: Optional[Dict[str, Any]],
    static_fit: Dict[str, Any],
    V_r: np.ndarray,
    analysis_cfg: Dict[str, Any],
) -> None:
    K = dataset1_solution["K"]
    diag_col = None
    for cand in DIAG_CANDIDATES:
        if cand in static_fit["contracts"]["subjects1"].columns:
            diag_col = cand
            break
    if diag_col is None:
        return

    def summary_for(solution: Dict[str, Any]) -> pd.DataFrame:
        meta = solution["meta"]
        return build_subject_dynamic_summary(
            meta["subject_id"].to_numpy(dtype=str),
            meta["window_idx"].to_numpy(dtype=np.int64),
            solution["labels"],
            solution["windows"],
            V_r,
            K,
        )

    summary_d1 = summary_for(dataset1_solution)
    train_df = split_subject_frame(static_fit, "dataset1_train")
    val_df = split_subject_frame(static_fit, "dataset1_val")
    test_df = split_subject_frame(static_fit, "dataset1_test")
    merged_train = train_df.merge(summary_d1, on="subject_id", how="inner")
    merged_val = val_df.merge(summary_d1, on="subject_id", how="inner")
    merged_test = test_df.merge(summary_d1, on="subject_id", how="inner")
    if merged_train.empty or merged_val.empty or merged_test.empty:
        return
    y_train = merged_train[diag_col].to_numpy(dtype=np.int64)
    y_val = merged_val[diag_col].to_numpy(dtype=np.int64)
    y_test = merged_test[diag_col].to_numpy(dtype=np.int64)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
        return

    feature_sets_train = build_clinical_feature_sets(merged_train, K)
    feature_sets_val = build_clinical_feature_sets(merged_val, K)
    feature_sets_test = build_clinical_feature_sets(merged_test, K)
    Cs = analysis_cfg.get("clinical_Cs", [0.01, 0.1, 1.0, 10.0, 100.0])

    external_pack = None
    if dataset2_solution is not None and diag_col in static_fit["contracts"]["subjects2"].columns:
        summary_d2 = summary_for(dataset2_solution)
        ext_df = split_subject_frame(static_fit, "dataset2_all")
        merged_ext = ext_df.merge(summary_d2, on="subject_id", how="inner")
        if not merged_ext.empty and diag_col in merged_ext.columns:
            y_ext = merged_ext[diag_col].to_numpy(dtype=np.int64)
            if len(np.unique(y_ext)) >= 2:
                external_pack = (
                    merged_ext,
                    build_clinical_feature_sets(merged_ext, K),
                    y_ext,
                    dataset2_solution["report_name"],
                )

    rho_state = compute_centroids_from_labels(
        dataset1_solution["windows"][dataset1_solution["meta"]["static_split"].eq("test").to_numpy()],
        dataset1_solution["labels"][dataset1_solution["meta"]["static_split"].eq("test").to_numpy()],
        K,
    )
    rho_state, _, _ = compute_subspace_retention(V_r, rho_state)

    effects = case_control_state_effects(merged_test, diag_col, K, rho_state=rho_state)
    for row in effects["rows"]:
        row.update({
            "map_source": map_source,
            "cohort": dataset1_solution["report_name"],
            "method": method,
            "seed": seed,
            "rank": rank,
        })
        effect_rows.append(row)
    for row in effects.get("rho_association", []):
        row.update({
            "map_source": map_source,
            "cohort": dataset1_solution["report_name"],
            "method": method,
            "seed": seed,
            "rank": rank,
            "metric_kind": "rho_association",
        })
        effect_rows.append(row)

    for feat_name in ["full", "coupled", "uncoupled"]:
        clf, info = fit_logistic_grid(
            feature_sets_train[feat_name], y_train,
            feature_sets_val[feat_name], y_val,
            Cs,
        )
        y_score = clf.predict_proba(feature_sets_test[feat_name])[:, 1]
        auc = bootstrap_auc_ci(y_test, y_score, n_boot=int(analysis_cfg.get("clinical_boot", 500)), seed=seed)
        auc_rows.append({
            "map_source": map_source,
            "cohort": dataset1_solution["report_name"],
            "feature_set": feat_name,
            "method": method,
            "seed": seed,
            "rank": rank,
            "best_C": info["best_C"],
            **auc,
        })
        if external_pack is not None:
            merged_ext, feat_ext, y_ext, cohort_name = external_pack
            y_score_ext = clf.predict_proba(feat_ext[feat_name])[:, 1]
            auc_ext = bootstrap_auc_ci(y_ext, y_score_ext, n_boot=int(analysis_cfg.get("clinical_boot", 500)), seed=seed)
            auc_rows.append({
                "map_source": map_source,
                "cohort": cohort_name,
                "feature_set": feat_name,
                "method": method,
                "seed": seed,
                "rank": rank,
                "best_C": info["best_C"],
                **auc_ext,
            })


def main() -> None:
    args = parse_args()
    from models.utils import load_config  # local import to keep script startup symmetric with others

    cfg = load_config(args.config)
    analysis_cfg = cfg.get("analysis", {})
    out_dir = Path(analysis_cfg.get("out_dir", PROJECT_ROOT / "results" / "gm_dfnc"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("GM-dFNC full analysis")
    print("=" * 72)
    print(f"Config: {args.config}")
    print(f"Output: {out_dir}")

    map_sources = cfg["map_sources"]
    fit_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Step 1: static maps
    # ------------------------------------------------------------------
    print("\n[1/5] Rebuilding static GM->FNC maps ...")
    for source_name, source_cfg in map_sources.items():
        methods = source_cfg.get("methods", analysis_cfg.get("methods", ["nuclear_norm"]))
        seeds = source_cfg.get("seeds", analysis_cfg.get("seeds", [42]))
        for method in methods:
            for seed in seeds:
                print(f"  {source_name} | {method} | seed={seed}")
                fit_cache[(source_name, method, seed)] = fit_static_method(source_cfg["config"], method, seed)

    # ------------------------------------------------------------------
    # Step 2: dynamic data
    # ------------------------------------------------------------------
    print("\n[2/5] Loading dynamic datasets ...")
    dynamic_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for source_name, source_cfg in map_sources.items():
        reference_key = (source_name, source_cfg.get("methods", analysis_cfg.get("methods", ["nuclear_norm"]))[0],
                         source_cfg.get("seeds", analysis_cfg.get("seeds", [42]))[0])
        ref_fit = fit_cache[reference_key]
        dy = ref_fit["contracts"]["dy"]
        for dataset_name, dataset_spec in source_cfg.get("dynamic_datasets", {}).items():
            print(f"  {source_name} | {dataset_name}")
            base = load_dynamic_base(dataset_spec, dy)
            for solution_spec in dataset_spec.get("solutions", []):
                sol = prepare_dynamic_solution(base, solution_spec, ref_fit)
                dynamic_cache[(source_name, dataset_name, sol["label"])] = sol
                print(
                    f"    solution={sol['label']}  K={sol['K']}  matched_windows={sol['windows'].shape[0]}  "
                    f"subjects={sol['meta']['subject_id'].nunique()}"
                )

    # ------------------------------------------------------------------
    # Step 3: geometry / manifold / hierarchy / transition
    # ------------------------------------------------------------------
    print("\n[3/5] Running geometric and dynamic-state analyses ...")
    retention_rows: List[Dict[str, Any]] = []
    manifold_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    between_rows: List[Dict[str, Any]] = []
    local_rows: List[Dict[str, Any]] = []
    hierarchy_rows: List[Dict[str, Any]] = []
    transition_rows: List[Dict[str, Any]] = []
    detail_index: List[Dict[str, Any]] = []

    requested_ranks = analysis_cfg.get("ranks", ["eff", 5, 10, 20])

    for (source_name, method, seed), static_fit in fit_cache.items():
        V_full = static_fit["V_full"]
        Sigma = static_fit["Sigma"]
        rank_list = [select_rank(Sigma, item) for item in requested_ranks]
        rank_list = sorted(set([r for r in rank_list if r <= V_full.shape[1]]))
        fnc_names = static_fit["contracts"]["fnc_names"]

        for (src2, dataset_name, solution_label), solution in dynamic_cache.items():
            if src2 != source_name:
                continue

            eval_mask = solution_eval_mask(solution)
            eval_windows = solution["windows"][eval_mask]
            eval_labels = solution["labels"][eval_mask]
            eval_subjects = solution["meta"].loc[eval_mask, "subject_id"].to_numpy(dtype=str)
            eval_window_idx = solution["meta"].loc[eval_mask, "window_idx"].to_numpy(dtype=np.int64)
            eval_centroids = compute_centroids_from_labels(eval_windows, eval_labels, solution["K"])
            cohort = eval_label_for_solution(solution)

            for rank in rank_list:
                V_r = V_full[:, :rank]
                rho_obs, rho_null, p_vals, p_mean = rotation_null_distribution(
                    V_r,
                    eval_centroids,
                    n_perm=int(analysis_cfg.get("n_perm", 1000)),
                    seed=seed,
                )
                boot = bootstrap_state_retention(
                    V_r,
                    eval_windows,
                    eval_labels,
                    eval_subjects,
                    solution["K"],
                    n_boot=int(analysis_cfg.get("n_boot", 250)),
                    seed=seed,
                )
                chance = float(rank / eval_centroids.shape[1])
                cent_proj = eval_centroids @ orthonormalize(V_r[:, : min(3, rank)])
                delta = compute_delta_retention(V_r, eval_centroids)
                delta_contrib = per_mode_contribution(V_r, delta["proj"] if len(delta["pairs"]) else np.zeros((0, rank)))
                rho_rank = retention_vs_rank_curve(
                    V_full,
                    eval_centroids,
                    ranks=list(range(1, min(V_full.shape[1], int(analysis_cfg.get("rank_curve_max", 40))) + 1)),
                )
                _, centroid_proj, _ = compute_subspace_retention(V_r, eval_centroids)
                manifold = pairwise_distance_alignment(
                    eval_centroids,
                    centroid_proj,
                    n_perm=int(analysis_cfg.get("mantel_perm", 1000)),
                    seed=seed,
                )
                between = between_within_variance(eval_windows, eval_labels, eval_centroids, V_r)
                local = within_state_local_overlaps(
                    eval_windows,
                    eval_labels,
                    V_r,
                    max_k=min(rank, int(analysis_cfg.get("local_state_k", 10))),
                )
                hierarchy = tier_retention(V_r, eval_centroids, fnc_names)
                residuals = analyze_residuals(V_r, eval_centroids)

                trans_counts = np.zeros((solution["K"], solution["K"]), dtype=np.float64)
                for sid in pd.unique(eval_subjects):
                    smask = eval_subjects == sid
                    order = np.argsort(eval_window_idx[smask])
                    labs = eval_labels[smask][order]
                    for a, b in zip(labs[:-1], labs[1:]):
                        trans_counts[int(a), int(b)] += 1.0
                trans_total = trans_counts / max(np.sum(trans_counts), 1.0)
                pair_map = {pair: idx for idx, pair in enumerate(delta["pairs"])}
                for i in range(solution["K"]):
                    for j in range(solution["K"]):
                        if i == j:
                            continue
                        pair = (i, j) if i < j else (j, i)
                        pair_idx = pair_map.get(pair)
                        trans_rows = {
                            "map_source": source_name,
                            "cohort": cohort,
                            "solution": solution_label,
                            "method": method,
                            "seed": seed,
                            "rank": rank,
                            "from_state": i,
                            "to_state": j,
                            "transition_prob": float(trans_total[i, j]),
                            "delta_rho": float(delta["rho"][pair_idx]) if pair_idx is not None and len(delta["pairs"]) else float("nan"),
                            "state_coord_x": float(cent_proj[i, 0]) if cent_proj.shape[1] > 0 else 0.0,
                            "state_coord_y": float(cent_proj[i, 1]) if cent_proj.shape[1] > 1 else 0.0,
                            "next_coord_x": float(cent_proj[j, 0]) if cent_proj.shape[1] > 0 else 0.0,
                            "next_coord_y": float(cent_proj[j, 1]) if cent_proj.shape[1] > 1 else 0.0,
                        }
                        transition_rows.append(trans_rows)

                detail_path = save_detail_arrays(
                    out_dir,
                    cohort,
                    solution_label,
                    method,
                    seed,
                    rank,
                    {
                        "rho_obs": rho_obs,
                        "rho_null": rho_null,
                        "bootstrap_rho_ci_lo": boot["rho_ci_lo"],
                        "bootstrap_rho_ci_hi": boot["rho_ci_hi"],
                        "rank_curve_ranks": rho_rank["ranks"],
                        "rank_curve_rho": rho_rank["rho_per_rank"],
                        "rank_curve_chance": rho_rank["chance_curve"],
                        "state_coords": cent_proj[:, : min(3, cent_proj.shape[1])],
                        "resid_cosines": residuals["resid_cosines"],
                        "resid_explained": residuals["resid_explained_variance_ratio"],
                    },
                )
                detail_index.append({
                    "cohort": cohort,
                    "solution": solution_label,
                    "method": method,
                    "seed": seed,
                    "rank": rank,
                    "detail_path": detail_path,
                })

                rank_order = np.argsort(-rho_obs)
                for state, value in enumerate(rho_obs):
                    retention_rows.append({
                        "map_source": source_name,
                        "cohort": cohort,
                        "solution": solution_label,
                        "method": method,
                        "seed": seed,
                        "rank": rank,
                        "state": state,
                        "rho": float(value),
                        "chance": chance,
                        "lift": float(value / max(chance, 1e-30)),
                        "p_value": float(p_vals[state]),
                        "rho_ci_lo": float(boot["rho_ci_lo"][state]),
                        "rho_ci_hi": float(boot["rho_ci_hi"][state]),
                        "rank_order": int(np.where(rank_order == state)[0][0] + 1),
                        "mean_rho": float(np.mean(rho_obs)),
                        "mean_p_value": float(p_mean),
                        "mean_rho_ci_lo": float(boot["mean_rho_ci_lo"]),
                        "mean_rho_ci_hi": float(boot["mean_rho_ci_hi"]),
                    })

                manifold_rows.append({
                    "map_source": source_name,
                    "cohort": cohort,
                    "solution": solution_label,
                    "method": method,
                    "seed": seed,
                    "rank": rank,
                    "span_overlap": float(subspace_overlap(top_k_subspace(eval_centroids, min(rank, solution["K"])), V_r[:, : min(rank, solution["K"])])),
                    "distance_pearson": manifold["pearson"],
                    "distance_spearman": manifold["spearman"],
                    "mantel_p": manifold["mantel_p"],
                })

                for idx_pair, pair in enumerate(delta["pairs"]):
                    delta_rows.append({
                        "map_source": source_name,
                        "cohort": cohort,
                        "solution": solution_label,
                        "method": method,
                        "seed": seed,
                        "rank": rank,
                        "state_a": int(pair[0]),
                        "state_b": int(pair[1]),
                        "rho": float(delta["rho"][idx_pair]),
                        "mean_rho": float(delta["mean_rho"]),
                        "top_mode_contrib": float(np.max(delta_contrib[idx_pair])) if delta_contrib.size else float("nan"),
                    })

                between_rows.append({
                    "map_source": source_name,
                    "cohort": cohort,
                    "solution": solution_label,
                    "method": method,
                    "seed": seed,
                    "rank": rank,
                    **between,
                })

                for row in local:
                    row.update({
                        "map_source": source_name,
                        "cohort": cohort,
                        "solution": solution_label,
                        "method": method,
                        "seed": seed,
                        "rank": rank,
                    })
                    local_rows.append(row)

                for row in hierarchy:
                    row.update({
                        "map_source": source_name,
                        "cohort": cohort,
                        "solution": solution_label,
                        "method": method,
                        "seed": seed,
                        "rank": rank,
                    })
                    hierarchy_rows.append(row)

    # ------------------------------------------------------------------
    # Step 4: subject-level prediction and clinical utility
    # ------------------------------------------------------------------
    print("\n[4/5] Running subject-level dynamic prediction and clinical checks ...")
    prediction_rows: List[Dict[str, Any]] = []
    clinical_auc_rows: List[Dict[str, Any]] = []
    clinical_effect_rows: List[Dict[str, Any]] = []

    for source_name, source_cfg in map_sources.items():
        dataset_specs = source_cfg.get("dynamic_datasets", {})
        if "dataset1" not in dataset_specs:
            continue
        dataset1_solutions = [key for key in dynamic_cache if key[0] == source_name and key[1] == "dataset1"]
        dataset2_solutions = {
            key[2]: dynamic_cache[key]
            for key in dynamic_cache
            if key[0] == source_name and key[1] == "dataset2"
        }
        for method in source_cfg.get("methods", analysis_cfg.get("methods", ["nuclear_norm"])):
            for seed in source_cfg.get("seeds", analysis_cfg.get("seeds", [42])):
                static_fit = fit_cache[(source_name, method, seed)]
                rank_list = [select_rank(static_fit["Sigma"], item) for item in requested_ranks]
                rank_list = sorted(set([r for r in rank_list if r <= static_fit["V_full"].shape[1]]))
                for source_key in dataset1_solutions:
                    solution_d1 = dynamic_cache[source_key]
                    solution_d2 = dataset2_solutions.get(solution_d1["label"])
                    for rank in rank_list:
                        V_r = static_fit["V_full"][:, :rank]
                        run_prediction_suite(
                            prediction_rows,
                            source_name,
                            method,
                            seed,
                            rank,
                            solution_d1,
                            solution_d2,
                            static_fit,
                            V_r,
                            analysis_cfg,
                        )
                        run_clinical_suite(
                            clinical_auc_rows,
                            clinical_effect_rows,
                            source_name,
                            method,
                            seed,
                            rank,
                            solution_d1,
                            solution_d2,
                            static_fit,
                            V_r,
                            analysis_cfg,
                        )

    # ------------------------------------------------------------------
    # Step 5: write outputs
    # ------------------------------------------------------------------
    print("\n[5/5] Writing summary tables ...")
    retention_df = pd.DataFrame(retention_rows)
    manifold_df = pd.DataFrame(manifold_rows)
    delta_df = pd.DataFrame(delta_rows)
    between_df = pd.DataFrame(between_rows)
    local_df = pd.DataFrame(local_rows)
    hierarchy_df = pd.DataFrame(hierarchy_rows)
    transition_df = pd.DataFrame(transition_rows)
    prediction_df = pd.DataFrame(prediction_rows)
    clinical_auc_df = pd.DataFrame(clinical_auc_rows)
    clinical_effect_df = pd.DataFrame(clinical_effect_rows)
    detail_df = pd.DataFrame(detail_index)

    save_tsv(out_dir / "state_retention.tsv", retention_rows)
    save_tsv(out_dir / "manifold_alignment.tsv", manifold_rows)
    save_tsv(out_dir / "delta_retention.tsv", delta_rows)
    save_tsv(out_dir / "between_within.tsv", between_rows)
    save_tsv(out_dir / "local_state_overlap.tsv", local_rows)
    save_tsv(out_dir / "hierarchy_retention.tsv", hierarchy_rows)
    save_tsv(out_dir / "transition_graph.tsv", transition_rows)
    save_tsv(out_dir / "prediction.tsv", prediction_rows)
    save_tsv(out_dir / "clinical_auc.tsv", clinical_auc_rows)
    save_tsv(out_dir / "case_control_effects.tsv", clinical_effect_rows)
    save_tsv(out_dir / "detail_arrays.tsv", detail_index)

    agg_outputs = {
        "state_retention_summary.tsv": aggregate_table(
            retention_df,
            ["rho", "lift", "p_value", "mean_rho", "mean_p_value", "rho_ci_lo", "rho_ci_hi"],
            ["cohort", "solution", "method", "rank", "state"],
        ),
        "manifold_alignment_summary.tsv": aggregate_table(
            manifold_df,
            ["span_overlap", "distance_pearson", "distance_spearman", "mantel_p"],
            ["cohort", "solution", "method", "rank"],
        ),
        "delta_retention_summary.tsv": aggregate_table(
            delta_df,
            ["rho", "mean_rho", "top_mode_contrib"],
            ["cohort", "solution", "method", "rank", "state_a", "state_b"],
        ),
        "between_within_summary.tsv": aggregate_table(
            between_df,
            ["between_var_frac", "within_var_frac", "coupled_between_frac", "coupled_within_frac", "mean_between_rho", "mean_within_rho"],
            ["cohort", "solution", "method", "rank"],
        ),
        "prediction_summary.tsv": aggregate_table(
            prediction_df,
            ["r2_mean", "corr_mean", "calibration_slope"],
            ["cohort", "target_group", "method", "rank"],
        ),
        "clinical_auc_summary.tsv": aggregate_table(
            clinical_auc_df,
            ["auc", "auc_ci_lo", "auc_ci_hi"],
            ["cohort", "feature_set", "method", "rank"],
        ),
        "hierarchy_retention_summary.tsv": aggregate_table(
            hierarchy_df,
            ["rho", "lift", "chance"],
            ["cohort", "solution", "method", "rank", "tier", "state"],
        ),
    }
    for filename, df in agg_outputs.items():
        if not df.empty:
            df.to_csv(out_dir / filename, sep="\t", index=False)

    table_aliases = {
        "table1_state_retention.tsv": agg_outputs["state_retention_summary.tsv"],
        "table2_prediction_hierarchy.tsv": agg_outputs["prediction_summary.tsv"],
        "table3_hierarchy_retention.tsv": agg_outputs["hierarchy_retention_summary.tsv"],
        "table4_clinical_auc.tsv": agg_outputs["clinical_auc_summary.tsv"],
    }
    for filename, df in table_aliases.items():
        if not df.empty:
            df.to_csv(out_dir / filename, sep="\t", index=False)

    primary_method = analysis_cfg.get("primary_method", "nuclear_norm")
    primary_rank_selector = analysis_cfg.get("primary_rank", "eff")
    primary_solution = analysis_cfg.get("primary_solution")
    primary_rows = []
    if not retention_df.empty:
        for cohort in sorted(retention_df["cohort"].unique()):
            sub = retention_df[
                (retention_df["cohort"] == cohort)
                & (retention_df["method"] == primary_method)
            ]
            if primary_solution is not None:
                sub = sub[sub["solution"] == primary_solution]
            if sub.empty:
                continue
            ranks_here = sorted(sub["rank"].unique())
            rank_target = None
            if isinstance(primary_rank_selector, str) and primary_rank_selector == "eff":
                rank_target = max(ranks_here)
            else:
                rank_target = int(primary_rank_selector)
            best = sub[sub["rank"] == rank_target]
            if best.empty:
                best = sub[sub["rank"] == max(ranks_here)]
            primary_rows.append(best.iloc[0].to_dict())

    summary = {
        "config_path": args.config,
        "out_dir": str(out_dir),
        "map_sources": sorted(map_sources.keys()),
        "n_static_fits": len(fit_cache),
        "n_dynamic_solutions": len(dynamic_cache),
        "primary_results": primary_rows,
        "files": {
            "state_retention": str(out_dir / "state_retention.tsv"),
            "state_retention_summary": str(out_dir / "state_retention_summary.tsv"),
            "prediction_summary": str(out_dir / "prediction_summary.tsv"),
            "clinical_auc_summary": str(out_dir / "clinical_auc_summary.tsv"),
            "table1_state_retention": str(out_dir / "table1_state_retention.tsv"),
            "table2_prediction_hierarchy": str(out_dir / "table2_prediction_hierarchy.tsv"),
            "table3_hierarchy_retention": str(out_dir / "table3_hierarchy_retention.tsv"),
            "table4_clinical_auc": str(out_dir / "table4_clinical_auc.tsv"),
            "detail_arrays": str(out_dir / "detail_arrays.tsv"),
        },
    }
    write_summary_json(out_dir / "summary.json", summary)

    print("\nSaved:")
    for rel in [
        "summary.json",
        "state_retention.tsv",
        "state_retention_summary.tsv",
        "between_within_summary.tsv",
        "prediction_summary.tsv",
        "clinical_auc_summary.tsv",
    ]:
        print(f"  {out_dir / rel}")


if __name__ == "__main__":
    main()
