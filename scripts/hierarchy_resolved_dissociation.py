#!/usr/bin/env python3
"""
Hierarchy-resolved dissociation analysis for the Nuclear Norm model.

This script reuses the trained GM->FNC coefficient matrices and evaluates
direction-amplitude dissociation on tier-specific edge subsets without
retraining any model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.metrics import r2_edgewise
from models.utils import load_config, load_training_contracts
from train.run_subspace_analysis import principal_angles


FNC_DOMAIN_RANGES = {
    "SC": (0, 5),
    "AUD": (5, 7),
    "SM": (7, 16),
    "VS": (16, 25),
    "CC": (25, 42),
    "DM": (42, 49),
    "CB": (49, 53),
}

PRIMARY_TIER_MAP = {
    "SM": "sensorimotor",
    "VS": "sensorimotor",
    "AUD": "sensorimotor",
    "CC": "heteromodal",
    "CB": "heteromodal",
    "DM": "transmodal",
    "SC": "transmodal",
}

CB_TO_SENSORIMOTOR_MAP = {
    "SM": "sensorimotor",
    "VS": "sensorimotor",
    "AUD": "sensorimotor",
    "CB": "sensorimotor",
    "CC": "heteromodal",
    "DM": "transmodal",
    "SC": "transmodal",
}

GROUP_ORDER = [
    "sensorimotor",
    "heteromodal",
    "transmodal",
    "within_all",
    "between_all",
    "SM-HM",
    "SM-TM",
    "HM-TM",
]

MAIN_TEXT_GROUPS = [
    "sensorimotor",
    "heteromodal",
    "transmodal",
    "within_all",
    "between_all",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hierarchy-resolved dissociation")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "train" / "config_baselines.yaml"),
    )
    parser.add_argument(
        "--decomp-dir",
        default=str(PROJECT_ROOT / "results" / "multivariate_methods" / "decompositions"),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46, 47, 48],
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "results" / "hierarchy_analysis"),
    )
    return parser.parse_args()


def get_fnc_domain(ic_idx: int) -> str:
    for domain, (lo, hi) in FNC_DOMAIN_RANGES.items():
        if lo <= ic_idx < hi:
            return domain
    return "Other"


def parse_fnc_edges(fnc_names: Iterable[str]) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for name in fnc_names:
        left, right = name.split("--")
        edges.append((int(left.replace("IC_", "")), int(right.replace("IC_", ""))))
    return edges


def build_masks(
    edges: List[Tuple[int, int]],
    tier_map: Dict[str, str],
) -> Dict[str, np.ndarray]:
    masks = {name: [] for name in GROUP_ORDER}
    for edge_idx, (ic_i, ic_j) in enumerate(edges):
        tier_i = tier_map[get_fnc_domain(ic_i)]
        tier_j = tier_map[get_fnc_domain(ic_j)]
        if tier_i == tier_j:
            masks[tier_i].append(edge_idx)
            masks["within_all"].append(edge_idx)
        else:
            masks["between_all"].append(edge_idx)
            pair = tuple(sorted((tier_i, tier_j)))
            if pair == ("heteromodal", "sensorimotor"):
                masks["SM-HM"].append(edge_idx)
            elif pair == ("sensorimotor", "transmodal"):
                masks["SM-TM"].append(edge_idx)
            elif pair == ("heteromodal", "transmodal"):
                masks["HM-TM"].append(edge_idx)

    return {name: np.asarray(idx, dtype=np.int64) for name, idx in masks.items()}


def mean_edge_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(r2_edgewise(y_true.astype(np.float32), y_pred.astype(np.float32))))


def top_k_subspace_fast(y: np.ndarray, k: int) -> np.ndarray:
    """
    Algebraically equivalent to taking the top-k right singular vectors of y,
    but computed through YY^T to make repeated bootstrap evaluations feasible.
    """
    gram = y @ y.T
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    positive = eigvals > 1e-12
    rank = int(np.sum(positive))
    k_eff = min(k, rank)
    if k_eff == 0:
        return np.zeros((y.shape[1], 0), dtype=np.float64)
    u = eigvecs[:, :k_eff]
    s = np.sqrt(np.clip(eigvals[:k_eff], 1e-12, None))
    v = (y.T @ u) / s
    q, _ = np.linalg.qr(v)
    return q[:, :k_eff]


def subset_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    edge_idx: np.ndarray,
    k: int,
) -> Dict[str, float]:
    y_true_sub = y_true[:, edge_idx]
    y_pred_sub = y_pred[:, edge_idx]
    v_true = top_k_subspace_fast(y_true_sub, k)
    v_pred = top_k_subspace_fast(y_pred_sub, k)
    k_use = min(v_true.shape[1], v_pred.shape[1])
    if k_use == 0:
        overlap = 0.0
    else:
        overlap = float(np.mean(principal_angles(v_true[:, :k_use], v_pred[:, :k_use]) ** 2))
    r2 = mean_edge_r2(y_true_sub, y_pred_sub)
    ratio = float(overlap / r2) if abs(r2) > 1e-12 else float("inf")
    return {
        "n_edges": int(y_true_sub.shape[1]),
        "k": int(k_use),
        "chance": float(k_use / y_true_sub.shape[1]),
        "O": overlap,
        "R2": r2,
        "ratio": ratio,
    }


def summarize_seed_metrics(seed_metrics: Dict[int, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for group in GROUP_ORDER:
        metrics = [seed_metrics[seed][group] for seed in sorted(seed_metrics)]
        o_arr = np.array([m["O"] for m in metrics], dtype=np.float64)
        r2_arr = np.array([m["R2"] for m in metrics], dtype=np.float64)
        ratio_arr = np.array([m["ratio"] for m in metrics], dtype=np.float64)
        chance_arr = np.array([m["chance"] for m in metrics], dtype=np.float64)
        summary[group] = {
            "n_edges": int(metrics[0]["n_edges"]),
            "k": int(metrics[0]["k"]),
            "chance_mean": float(np.mean(chance_arr)),
            "chance_std": float(np.std(chance_arr)),
            "O_mean": float(np.mean(o_arr)),
            "O_std": float(np.std(o_arr)),
            "R2_mean": float(np.mean(r2_arr)),
            "R2_std": float(np.std(r2_arr)),
            "ratio_mean": float(np.mean(ratio_arr)),
            "ratio_std": float(np.std(ratio_arr)),
        }
    return summary


def basic_interval(theta_hat: float, boot_samples: np.ndarray) -> List[float]:
    lo = float(np.percentile(boot_samples, 2.5))
    hi = float(np.percentile(boot_samples, 97.5))
    return [float(2 * theta_hat - hi), float(2 * theta_hat - lo)]


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    masks: Dict[str, np.ndarray],
    k: int,
    n_boot: int,
    rng: np.random.Generator,
    theta_hat: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {group: {} for group in MAIN_TEXT_GROUPS}
    n_subjects = y_true.shape[0]
    boot_store = {
        group: {"O": [], "R2": [], "ratio": []}
        for group in MAIN_TEXT_GROUPS
    }

    for _ in range(n_boot):
        rows = rng.integers(0, n_subjects, size=n_subjects)
        y_true_b = y_true[rows]
        y_pred_b = y_pred_mean[rows]
        for group in MAIN_TEXT_GROUPS:
            metrics = subset_metrics(y_true_b, y_pred_b, masks[group], k)
            boot_store[group]["O"].append(metrics["O"])
            boot_store[group]["R2"].append(metrics["R2"])
            boot_store[group]["ratio"].append(metrics["ratio"])

    for group in MAIN_TEXT_GROUPS:
        for metric_key, theta_key in [("O", "O_mean"), ("R2", "R2_mean"), ("ratio", "ratio_mean")]:
            arr = np.asarray(boot_store[group][metric_key], dtype=np.float64)
            out[group][metric_key] = basic_interval(theta_hat[group][theta_key], arr)
    return out


def tsv_rows(
    mapping_name: str,
    mapping_summary: Dict[str, Dict[str, float]],
    bootstrap_summary: Dict[str, Dict[str, List[float]]] | None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for group in GROUP_ORDER:
        item = mapping_summary[group]
        row = {
            "mapping": mapping_name,
            "group": group,
            "n_edges": item["n_edges"],
            "k": item["k"],
            "chance_mean": item["chance_mean"],
            "O_mean": item["O_mean"],
            "O_std": item["O_std"],
            "R2_mean": item["R2_mean"],
            "R2_std": item["R2_std"],
            "ratio_mean": item["ratio_mean"],
            "ratio_std": item["ratio_std"],
        }
        if bootstrap_summary and group in bootstrap_summary:
            row.update({
                "O_ci_lo": bootstrap_summary[group]["O"][0],
                "O_ci_hi": bootstrap_summary[group]["O"][1],
                "R2_ci_lo": bootstrap_summary[group]["R2"][0],
                "R2_ci_hi": bootstrap_summary[group]["R2"][1],
                "ratio_ci_lo": bootstrap_summary[group]["ratio"][0],
                "ratio_ci_hi": bootstrap_summary[group]["ratio"][1],
            })
        rows.append(row)
    return rows


def write_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    data = load_training_contracts(cfg)
    idx_test = data["idx1_test"]
    x_test = data["X1"][idx_test].astype(np.float64)
    y_test = data["Y1"][idx_test].astype(np.float64)
    edges = parse_fnc_edges(data["fnc_names"])

    decomp_dir = Path(args.decomp_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_masks = build_masks(edges, PRIMARY_TIER_MAP)
    sensitivity_masks = build_masks(edges, CB_TO_SENSORIMOTOR_MAP)

    y_pred_by_seed: Dict[int, np.ndarray] = {}
    seed_summary_primary: Dict[int, Dict[str, Dict[str, float]]] = {}
    seed_summary_sensitivity: Dict[int, Dict[str, Dict[str, float]]] = {}

    for seed in args.seeds:
        b_path = decomp_dir / f"nuclear_norm_seed{seed}_B.npy"
        if not b_path.exists():
            raise FileNotFoundError(f"Missing decomposition matrix: {b_path}")
        b = np.load(b_path)
        y_pred = x_test @ b
        y_pred_by_seed[seed] = y_pred
        seed_summary_primary[seed] = {
            group: subset_metrics(y_test, y_pred, primary_masks[group], args.k)
            for group in GROUP_ORDER
        }
        seed_summary_sensitivity[seed] = {
            group: subset_metrics(y_test, y_pred, sensitivity_masks[group], args.k)
            for group in GROUP_ORDER
        }

    primary_summary = summarize_seed_metrics(seed_summary_primary)
    sensitivity_summary = summarize_seed_metrics(seed_summary_sensitivity)
    rng = np.random.default_rng(args.bootstrap_seed)
    y_pred_mean = np.mean(
        np.stack([y_pred_by_seed[seed] for seed in args.seeds], axis=0),
        axis=0,
    )
    primary_boot = bootstrap_ci(
        y_true=y_test,
        y_pred_mean=y_pred_mean,
        masks=primary_masks,
        k=args.k,
        n_boot=args.n_boot,
        rng=rng,
        theta_hat=primary_summary,
    )

    payload = {
        "mapping_version": "fnc_7domain_tier_v1",
        "tier_definitions": {
            "primary": PRIMARY_TIER_MAP,
            "cb_to_sensorimotor_sensitivity": CB_TO_SENSORIMOTOR_MAP,
        },
        "k": args.k,
        "bootstrap": {
            "n_boot": args.n_boot,
            "interval": "basic",
            "seed": args.bootstrap_seed,
        },
        "edge_counts": {
            "primary": {group: int(len(primary_masks[group])) for group in GROUP_ORDER},
            "cb_to_sensorimotor_sensitivity": {
                group: int(len(sensitivity_masks[group])) for group in GROUP_ORDER
            },
        },
        "seed_summary": {
            "primary": {str(seed): seed_summary_primary[seed] for seed in args.seeds},
            "cb_to_sensorimotor_sensitivity": {
                str(seed): seed_summary_sensitivity[seed] for seed in args.seeds
            },
        },
        "summary": {
            "primary": primary_summary,
            "cb_to_sensorimotor_sensitivity": sensitivity_summary,
        },
        "bootstrap_ci": {
            "primary": primary_boot,
        },
    }

    json_path = out_dir / "hierarchy_resolved_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2))

    rows = []
    rows.extend(tsv_rows("primary", primary_summary, primary_boot))
    rows.extend(tsv_rows("cb_to_sensorimotor_sensitivity", sensitivity_summary, None))
    write_tsv(out_dir / "hierarchy_resolved_metrics.tsv", rows)

    print(json.dumps({
        "json": str(json_path),
        "tsv": str(out_dir / "hierarchy_resolved_metrics.tsv"),
        "primary_main_text": {group: primary_summary[group] for group in MAIN_TEXT_GROUPS},
        "primary_bootstrap_ci": primary_boot,
    }, indent=2))


if __name__ == "__main__":
    main()
