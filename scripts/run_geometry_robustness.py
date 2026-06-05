#!/usr/bin/env python3
"""
Geometry-robustness analysis for the GM->FNC dissociation story.

This script reuses saved coefficient matrices and evaluates the overlap
between observed test FNC geometry and three alternative predicted-geometry
definitions:

1. top PCs of predicted test FNC Y_hat
2. top right singular vectors of the learned map B
3. top right singular vectors of the train cross-covariance X_train^T Y_train

It also recomputes a correct random-subspace null, which should be close to
k / d_y rather than the stale values in the old subspace_stats artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.metrics import r2_summary
from models.utils import load_config, load_training_contracts
from train.run_subspace_analysis import principal_angles


DS1_METHOD_PATTERNS = {
    "Nuclear_Norm": PROJECT_ROOT / "results" / "multivariate_methods" / "decompositions" / "nuclear_norm_seed{seed}_B.npy",
    "Linear_OptShrink": PROJECT_ROOT / "results" / "kernel_spectral_regression" / "decompositions" / "linear_optshrink_seed{seed}_B.npy",
    "Rrr": PROJECT_ROOT / "results" / "multivariate_methods" / "decompositions" / "rrr_seed{seed}_B.npy",
    "Pls": PROJECT_ROOT / "results" / "multivariate_methods" / "decompositions" / "pls_seed{seed}_B.npy",
}

UKB_METHOD_PATTERNS = {
    "Nuclear_Norm": PROJECT_ROOT / "results" / "ukb" / "multivariate_methods" / "decompositions" / "nuclear_norm_seed{seed}_B.npy",
    "Linear_OptShrink": PROJECT_ROOT / "results" / "ukb" / "kernel_spectral_regression" / "decompositions" / "linear_optshrink_seed{seed}_B.npy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geometry robustness analysis")
    parser.add_argument(
        "--config-ds1",
        default=str(PROJECT_ROOT / "train" / "config_baselines.yaml"),
    )
    parser.add_argument(
        "--config-ukb",
        default=str(PROJECT_ROOT / "train" / "config_baselines_ukb.yaml"),
    )
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46, 47, 48],
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "results" / "geometry_robustness"),
    )
    parser.add_argument(
        "--ukb-motion-csv",
        default="/home/users/ybi3/PNAS/ukb_motion_covariates.csv",
        help="Optional UKB motion covariate table with eid and mean_fd columns.",
    )
    parser.add_argument(
        "--motion-fd-threshold",
        type=float,
        default=0.2,
        help="Mean-FD threshold for a strict low-motion UKB subset.",
    )
    return parser.parse_args()


def right_subspace(mat: np.ndarray, k: int) -> np.ndarray:
    gram = np.asarray(mat.T @ mat, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    keep = eigvals > 1e-12
    rank = int(np.sum(keep))
    k_eff = min(k, rank)
    if k_eff == 0:
        return np.zeros((mat.shape[1], 0), dtype=np.float64)
    basis = eigvecs[:, keep][:, :k_eff]
    q, _ = np.linalg.qr(basis)
    return q[:, :k_eff]


def overlap(v_true: np.ndarray, v_pred: np.ndarray) -> float:
    k_eff = min(v_true.shape[1], v_pred.shape[1])
    if k_eff == 0:
        return 0.0
    cos_angles = principal_angles(v_true[:, :k_eff], v_pred[:, :k_eff])
    return float(np.mean(cos_angles ** 2))


def random_null(v_true: np.ndarray, n_perm: int, rng: np.random.Generator) -> Dict[str, float]:
    d, k = v_true.shape
    vals = np.empty(n_perm, dtype=np.float64)
    for idx in range(n_perm):
        q, _ = np.linalg.qr(rng.standard_normal((d, k)))
        vals[idx] = overlap(v_true, q[:, :k])
    return {
        "null_mean": float(np.mean(vals)),
        "null_std": float(np.std(vals)),
        "chance_k_over_d": float(k / d),
    }


def discover_method_paths(
    patterns: Dict[str, Path],
    seeds: List[int],
) -> Dict[str, List[Tuple[int, Path]]]:
    out: Dict[str, List[Tuple[int, Path]]] = {}
    for method, pattern in patterns.items():
        found: List[Tuple[int, Path]] = []
        for seed in seeds:
            path = Path(str(pattern).format(seed=seed))
            if path.exists():
                found.append((seed, path))
        if found:
            out[method] = found
    return out


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def regress_out(mat: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    cov = np.asarray(covariate, dtype=np.float64).reshape(-1)
    design = np.column_stack([np.ones(cov.shape[0], dtype=np.float64), cov])
    beta, *_ = np.linalg.lstsq(design, mat, rcond=None)
    return mat - design @ beta


def ds1_analysis(
    cfg_path: str,
    seeds: List[int],
    ks: List[int],
    n_perm: int,
    rng_seed: int,
) -> Dict[str, object]:
    cfg = load_config(cfg_path)
    data = load_training_contracts(cfg)
    x_train = data["X1"][data["idx1_train"]].astype(np.float64)
    y_train = data["Y1"][data["idx1_train"]].astype(np.float64)
    x_test = data["X1"][data["idx1_test"]].astype(np.float64)
    y_test = data["Y1"][data["idx1_test"]].astype(np.float64)

    v_true = {k: right_subspace(y_test, k) for k in ks}
    v_crosscov = {k: right_subspace(x_train.T @ y_train, k) for k in ks}
    nulls = {
        k: random_null(v_true[k], n_perm=n_perm, rng=np.random.default_rng(rng_seed + k))
        for k in ks
    }

    discovered = discover_method_paths(DS1_METHOD_PATTERNS, seeds)
    methods: Dict[str, object] = {}

    for method, seed_paths in discovered.items():
        by_k: Dict[str, object] = {}
        r2_vals: List[float] = []
        for k in ks:
            pred_vals: List[float] = []
            b_vals: List[float] = []
            for _, path in seed_paths:
                b = np.load(path)
                y_hat = x_test @ b
                r2_vals.append(r2_summary(y_test.astype(np.float32), y_hat.astype(np.float32))["r2_global"])
                pred_vals.append(overlap(v_true[k], right_subspace(y_hat, k)))
                b_vals.append(overlap(v_true[k], right_subspace(b, k)))
            by_k[f"k={k}"] = {
                "predicted_test_subspace": summarize(pred_vals),
                "map_right_singular_subspace": summarize(b_vals),
                "train_crosscov_subspace": {
                    "mean": float(overlap(v_true[k], v_crosscov[k])),
                    "std": 0.0,
                },
                "null": nulls[k],
            }
        methods[method] = {
            "n_seeds": len(seed_paths),
            "seeds": [seed for seed, _ in seed_paths],
            "r2_global": summarize(r2_vals),
            "by_k": by_k,
        }

    return {
        "dataset": "DS1_test",
        "n_test": int(y_test.shape[0]),
        "d_y": int(y_test.shape[1]),
        "k_values": ks,
        "methods": methods,
    }


def ukb_analysis(
    cfg_path: str,
    seeds: List[int],
    ks: List[int],
) -> Dict[str, object]:
    cfg = load_config(cfg_path)
    data = load_training_contracts(cfg)
    x_ext = data["X2"][data["idx2_external"]].astype(np.float64)
    y_ext = data["Y2"][data["idx2_external"]].astype(np.float64)
    v_true = {k: right_subspace(y_ext, k) for k in ks}

    discovered = discover_method_paths(UKB_METHOD_PATTERNS, seeds)
    methods: Dict[str, object] = {}
    for method, seed_paths in discovered.items():
        by_k: Dict[str, object] = {}
        r2_vals: List[float] = []
        for k in ks:
            pred_vals: List[float] = []
            b_vals: List[float] = []
            for _, path in seed_paths:
                b = np.load(path)
                y_hat = x_ext @ b
                r2_vals.append(r2_summary(y_ext.astype(np.float32), y_hat.astype(np.float32))["r2_global"])
                pred_vals.append(overlap(v_true[k], right_subspace(y_hat, k)))
                b_vals.append(overlap(v_true[k], right_subspace(b, k)))
            by_k[f"k={k}"] = {
                "predicted_test_subspace": summarize(pred_vals),
                "map_right_singular_subspace": summarize(b_vals),
                "chance_k_over_d": float(k / y_ext.shape[1]),
            }
        methods[method] = {
            "n_seeds": len(seed_paths),
            "seeds": [seed for seed, _ in seed_paths],
            "r2_global": summarize(r2_vals),
            "by_k": by_k,
        }

    return {
        "dataset": "UKB_external",
        "n_external": int(y_ext.shape[0]),
        "d_y": int(y_ext.shape[1]),
        "k_values": ks,
        "methods": methods,
    }


def ukb_motion_analysis(
    cfg_path: str,
    seeds: List[int],
    k: int,
    motion_csv: str,
    fd_threshold: float,
) -> Dict[str, object]:
    motion_path = Path(motion_csv)
    if not motion_path.exists():
        return {
            "available": False,
            "status": f"Motion file not found: {motion_path}",
        }

    cfg = load_config(cfg_path)
    data = load_training_contracts(cfg)
    x_ext = data["X2"][data["idx2_external"]].astype(np.float64)
    y_ext = data["Y2"][data["idx2_external"]].astype(np.float64)
    ext_ids = [str(data["ids2"][i]) for i in data["idx2_external"]]

    motion = pd.read_csv(motion_path, usecols=["eid", "mean_fd"])
    motion["eid"] = motion["eid"].astype(str)
    aligned = pd.DataFrame({"eid": ext_ids}).merge(motion, on="eid", how="left")
    if aligned["mean_fd"].isna().any():
        n_missing = int(aligned["mean_fd"].isna().sum())
        return {
            "available": False,
            "status": f"Motion file matched only {len(ext_ids) - n_missing}/{len(ext_ids)} UKB external subjects.",
        }

    fd = aligned["mean_fd"].to_numpy(dtype=np.float64)
    fd_median = float(np.median(fd))
    conditions = {
        "all": np.ones(fd.shape[0], dtype=bool),
        "low_motion_median": fd <= fd_median,
        "low_motion_fd020": fd <= float(fd_threshold),
    }

    discovered = discover_method_paths(UKB_METHOD_PATTERNS, seeds)
    methods: Dict[str, object] = {}
    for method, seed_paths in discovered.items():
        by_condition: Dict[str, object] = {}
        for condition_name, mask in conditions.items():
            y_true = y_ext[mask]
            v_true = right_subspace(y_true, k)
            pred_vals: List[float] = []
            b_vals: List[float] = []
            r2_vals: List[float] = []
            for _, path in seed_paths:
                b = np.load(path)
                y_hat = x_ext @ b
                y_hat_sub = y_hat[mask]
                r2_vals.append(r2_summary(y_true.astype(np.float32), y_hat_sub.astype(np.float32))["r2_global"])
                pred_vals.append(overlap(v_true, right_subspace(y_hat_sub, k)))
                b_vals.append(overlap(v_true, right_subspace(b, k)))
            by_condition[condition_name] = {
                "n_subjects": int(mask.sum()),
                "chance_k_over_d": float(k / y_true.shape[1]),
                "O_predY": summarize(pred_vals),
                "O_B": summarize(b_vals),
                "r2_global": summarize(r2_vals),
                "fd_summary": {
                    "mean_fd_mean": float(np.mean(fd[mask])),
                    "mean_fd_median": float(np.median(fd[mask])),
                },
            }

        y_true_resid = regress_out(y_ext, fd)
        v_true_resid = right_subspace(y_true_resid, k)
        pred_vals = []
        b_vals = []
        r2_vals = []
        for _, path in seed_paths:
            b = np.load(path)
            y_hat = x_ext @ b
            y_hat_resid = regress_out(y_hat, fd)
            r2_vals.append(r2_summary(y_true_resid.astype(np.float32), y_hat_resid.astype(np.float32))["r2_global"])
            pred_vals.append(overlap(v_true_resid, right_subspace(y_hat_resid, k)))
            b_vals.append(overlap(v_true_resid, right_subspace(b, k)))
        by_condition["motion_residualized"] = {
            "n_subjects": int(fd.shape[0]),
            "chance_k_over_d": float(k / y_true_resid.shape[1]),
            "O_predY": summarize(pred_vals),
            "O_B": summarize(b_vals),
            "r2_global": summarize(r2_vals),
            "fd_summary": {
                "mean_fd_mean": float(np.mean(fd)),
                "mean_fd_median": fd_median,
            },
        }

        methods[method] = {
            "n_seeds": len(seed_paths),
            "seeds": [seed for seed, _ in seed_paths],
            "by_condition": by_condition,
        }

    return {
        "available": True,
        "status": "ok",
        "dataset": "UKB_external",
        "k": int(k),
        "fd_median": fd_median,
        "fd_threshold": float(fd_threshold),
        "n_external": int(fd.shape[0]),
        "methods": methods,
    }


def build_tsv_rows(ds1: Dict[str, object], ukb: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset_name, payload in [("DS1_test", ds1), ("UKB_external", ukb)]:
        methods = payload["methods"]
        for method, method_info in methods.items():
            for k_label, item in method_info["by_k"].items():
                row = {
                    "dataset": dataset_name,
                    "method": method,
                    "n_seeds": method_info["n_seeds"],
                    "k": int(k_label.split("=")[1]),
                    "r2_global_mean": method_info["r2_global"]["mean"],
                    "r2_global_std": method_info["r2_global"]["std"],
                    "O_predY_mean": item["predicted_test_subspace"]["mean"],
                    "O_predY_std": item["predicted_test_subspace"]["std"],
                    "O_B_mean": item["map_right_singular_subspace"]["mean"],
                    "O_B_std": item["map_right_singular_subspace"]["std"],
                }
                if dataset_name == "DS1_test":
                    row["O_crosscov"] = item["train_crosscov_subspace"]["mean"]
                    row["chance_k_over_d"] = item["null"]["chance_k_over_d"]
                    row["null_mean"] = item["null"]["null_mean"]
                    row["null_std"] = item["null"]["null_std"]
                else:
                    row["O_crosscov"] = ""
                    row["chance_k_over_d"] = item["chance_k_over_d"]
                    row["null_mean"] = ""
                    row["null_std"] = ""
                rows.append(row)
    return rows


def build_motion_rows(payload: Dict[str, object]) -> List[Dict[str, object]]:
    if not payload.get("available", False):
        return []
    rows: List[Dict[str, object]] = []
    for method, info in payload["methods"].items():
        for condition, item in info["by_condition"].items():
            rows.append({
                "dataset": payload["dataset"],
                "method": method,
                "n_seeds": info["n_seeds"],
                "condition": condition,
                "n_subjects": item["n_subjects"],
                "k": payload["k"],
                "chance_k_over_d": item["chance_k_over_d"],
                "O_predY_mean": item["O_predY"]["mean"],
                "O_predY_std": item["O_predY"]["std"],
                "O_B_mean": item["O_B"]["mean"],
                "O_B_std": item["O_B"]["std"],
                "r2_global_mean": item["r2_global"]["mean"],
                "r2_global_std": item["r2_global"]["std"],
                "ratio_predY_over_r2": (
                    item["O_predY"]["mean"] / item["r2_global"]["mean"]
                    if abs(item["r2_global"]["mean"]) > 1e-12
                    else float("inf")
                ),
                "mean_fd_mean": item["fd_summary"]["mean_fd_mean"],
                "mean_fd_median": item["fd_summary"]["mean_fd_median"],
            })
    return rows


def save_tsv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds1 = ds1_analysis(
        cfg_path=args.config_ds1,
        seeds=args.seeds,
        ks=args.ks,
        n_perm=args.n_perm,
        rng_seed=args.rng_seed,
    )
    ukb = ukb_analysis(
        cfg_path=args.config_ukb,
        seeds=args.seeds,
        ks=args.ks,
    )
    ukb_motion = ukb_motion_analysis(
        cfg_path=args.config_ukb,
        seeds=args.seeds,
        k=max(args.ks),
        motion_csv=args.ukb_motion_csv,
        fd_threshold=args.motion_fd_threshold,
    )

    payload = {
        "ds1": ds1,
        "ukb": ukb,
        "ukb_motion": ukb_motion,
        "notes": {
            "subspace_stats_warning": (
                "Older results/subspace_analysis/subspace_stats.json contains stale "
                "null summaries and should not be used for reviewer-facing null statistics."
            )
        },
    }
    json_path = out_dir / "geometry_robustness_summary.json"
    json_path.write_text(json.dumps(payload, indent=2))

    rows = build_tsv_rows(ds1, ukb)
    tsv_path = out_dir / "geometry_robustness_summary.tsv"
    save_tsv(rows, tsv_path)

    motion_rows = build_motion_rows(ukb_motion)
    if motion_rows:
        motion_tsv_path = out_dir / "ukb_motion_robustness.tsv"
        save_tsv(motion_rows, motion_tsv_path)
    else:
        motion_tsv_path = None

    print(f"Saved JSON to {json_path}")
    print(f"Saved TSV to {tsv_path}")
    if motion_tsv_path is not None:
        print(f"Saved motion TSV to {motion_tsv_path}")

    for method in ["Nuclear_Norm", "Linear_OptShrink"]:
        if method in ds1["methods"]:
            k20 = ds1["methods"][method]["by_k"]["k=20"]
            print(
                f"DS1 {method}: R2={ds1['methods'][method]['r2_global']['mean']:.4f} "
                f"O_predY={k20['predicted_test_subspace']['mean']:.4f} "
                f"O_B={k20['map_right_singular_subspace']['mean']:.4f} "
                f"O_XY={k20['train_crosscov_subspace']['mean']:.4f}"
            )
        if method in ukb["methods"]:
            k20 = ukb["methods"][method]["by_k"]["k=20"]
            print(
                f"UKB {method}: R2={ukb['methods'][method]['r2_global']['mean']:.4f} "
                f"O_predY={k20['predicted_test_subspace']['mean']:.4f} "
                f"O_B={k20['map_right_singular_subspace']['mean']:.4f}"
            )
        if ukb_motion.get("available", False) and method in ukb_motion["methods"]:
            sens = ukb_motion["methods"][method]["by_condition"]
            low = sens["low_motion_median"]
            resid = sens["motion_residualized"]
            print(
                f"UKB motion {method}: low-motion median O={low['O_predY']['mean']:.4f} "
                f"R2={low['r2_global']['mean']:.4f}; residualized O={resid['O_predY']['mean']:.4f} "
                f"R2={resid['r2_global']['mean']:.4f}"
            )


if __name__ == "__main__":
    main()
