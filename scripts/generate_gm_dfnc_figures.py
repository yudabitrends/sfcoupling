#!/usr/bin/env python3
"""Generate manuscript-style figures from GM-dFNC analysis outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GM-dFNC figures")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--solution", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None)
    return parser.parse_args()


def load_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def primary_selection(results_dir: Path, method: Optional[str], solution: Optional[str], rank: Optional[int]) -> Dict[str, object]:
    summary_path = results_dir / "summary.json"
    selected = {"method": method, "solution": solution, "rank": rank}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        primary = summary.get("primary_results", [])
        if primary and not method:
            selected["method"] = primary[0].get("method")
        if primary and not solution:
            selected["solution"] = primary[0].get("solution")
        if primary and rank is None:
            selected["rank"] = int(primary[0].get("rank"))
    return selected


def first_detail_path(results_dir: Path, cohort: str, method: str, solution: str, rank: int) -> Optional[Path]:
    detail = load_tsv(results_dir / "detail_arrays.tsv")
    if detail.empty:
        return None
    sub = detail[
        (detail["cohort"] == cohort)
        & (detail["method"] == method)
        & (detail["solution"] == solution)
        & (detail["rank"] == rank)
    ]
    if sub.empty:
        return None
    return Path(sub.iloc[0]["detail_path"])


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1_state_manifold(results_dir: Path, fig_dir: Path, method: str, solution: str, rank: int) -> None:
    transition = load_tsv(results_dir / "transition_graph.tsv")
    if transition.empty:
        return
    cohorts = [c for c in ["DS1", "DS2", "UKB"] if c in set(transition["cohort"])]
    if not cohorts:
        cohorts = sorted(transition["cohort"].unique().tolist())[:3]

    fig, axes = plt.subplots(1, len(cohorts), figsize=(4.8 * len(cohorts), 4.0), squeeze=False)
    for ax, cohort in zip(axes[0], cohorts):
        sub = transition[
            (transition["cohort"] == cohort)
            & (transition["method"] == method)
            & (transition["solution"] == solution)
            & (transition["rank"] == rank)
        ]
        if sub.empty:
            ax.set_axis_off()
            continue
        nodes = sub[["from_state", "state_coord_x", "state_coord_y"]].drop_duplicates().sort_values("from_state")
        prob_thr = np.quantile(sub["transition_prob"], 0.8) if len(sub) >= 2 else 0.0
        edges = sub[sub["transition_prob"] >= prob_thr]
        for _, row in edges.iterrows():
            ax.plot(
                [row["state_coord_x"], row["next_coord_x"]],
                [row["state_coord_y"], row["next_coord_y"]],
                color="lightsteelblue",
                alpha=min(0.9, 0.2 + 3.0 * row["transition_prob"]),
                linewidth=1.5,
            )
        ax.scatter(nodes["state_coord_x"], nodes["state_coord_y"], s=120, c=np.arange(len(nodes)), cmap="Set2", edgecolors="k")
        for _, row in nodes.iterrows():
            ax.text(row["state_coord_x"], row["state_coord_y"], f"S{int(row['from_state'])}", ha="center", va="center", fontsize=9)
        ax.set_title(cohort)
        ax.set_xlabel("GM mode 1")
        ax.set_ylabel("GM mode 2")
        ax.axhline(0, color="0.85", linewidth=0.8)
        ax.axvline(0, color="0.85", linewidth=0.8)
    fig.suptitle("Figure 1. Pipeline-adapted state manifold overview", fontsize=14, fontweight="bold")
    save_figure(fig, fig_dir / "figure1_state_manifold.pdf")


def fig2_retention(results_dir: Path, fig_dir: Path, method: str, solution: str, rank: int) -> None:
    retention = load_tsv(results_dir / "state_retention.tsv")
    if retention.empty:
        return
    ds1 = retention[
        (retention["cohort"] == "DS1")
        & (retention["method"] == method)
        & (retention["solution"] == solution)
        & (retention["rank"] == rank)
    ]
    if ds1.empty:
        ds1 = retention[
            (retention["method"] == method)
            & (retention["solution"] == solution)
            & (retention["rank"] == rank)
        ].copy()
    if ds1.empty:
        return
    ds1 = ds1.sort_values("state")

    detail_path = first_detail_path(results_dir, ds1.iloc[0]["cohort"], method, solution, rank)
    detail = np.load(detail_path) if detail_path is not None and detail_path.exists() else None

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    ax = axes[0]
    ax.bar(ds1["state"].to_numpy(), ds1["rho"].to_numpy(), color="steelblue", edgecolor="k", linewidth=0.5)
    if "rho_ci_lo" in ds1 and "rho_ci_hi" in ds1:
        y = ds1["rho"].to_numpy()
        yerr = np.vstack([
            y - ds1["rho_ci_lo"].to_numpy(),
            ds1["rho_ci_hi"].to_numpy() - y,
        ])
        ax.errorbar(ds1["state"].to_numpy(), y, yerr=yerr, fmt="none", ecolor="0.2", capsize=3)
    ax.axhline(ds1["chance"].iloc[0], color="crimson", linestyle="--", linewidth=1.5)
    ax.set_title("State retention")
    ax.set_xlabel("State")
    ax.set_ylabel(r"$\rho_k$")

    ax = axes[1]
    if detail is not None and "rho_null" in detail:
        mean_null = np.mean(detail["rho_null"], axis=1)
        ax.hist(mean_null, bins=40, color="0.75", edgecolor="white")
        ax.axvline(float(np.mean(ds1["rho"])), color="crimson", linewidth=2.0)
        ax.set_title("Rotation null")
        ax.set_xlabel("Mean retention")
    else:
        ax.set_axis_off()

    ax = axes[2]
    if detail is not None and "rank_curve_ranks" in detail:
        ranks = detail["rank_curve_ranks"]
        rho_rank = detail["rank_curve_rho"]
        ax.plot(ranks, np.mean(rho_rank, axis=1), "o-", color="navy", linewidth=2)
        ax.plot(ranks, detail["rank_curve_chance"], "--", color="crimson", linewidth=1.5)
        ax.set_title("Retention vs rank")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Mean retention")
    else:
        ax.set_axis_off()
    fig.suptitle("Figure 2. State centroid retention + null + rank curve", fontsize=14, fontweight="bold")
    save_figure(fig, fig_dir / "figure2_retention.pdf")


def fig3_between_within(results_dir: Path, fig_dir: Path, method: str, solution: str, rank: int) -> None:
    between = load_tsv(results_dir / "between_within_summary.tsv")
    local = load_tsv(results_dir / "local_state_overlap.tsv")
    if between.empty:
        return
    sub = between[
        (between["method"] == method)
        & (between["solution"] == solution)
        & (between["rank"] == rank)
    ]
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    x = np.arange(len(sub))
    ax.bar(x - 0.15, sub["coupled_between_frac_mean"], width=0.3, label="Between-state", color="teal")
    ax.bar(x + 0.15, sub["coupled_within_frac_mean"], width=0.3, label="Within-state", color="goldenrod")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["cohort"].tolist())
    ax.set_ylabel("Coupled variance fraction")
    ax.set_title("Between > within")
    ax.legend()

    ax = axes[1]
    local_sub = local[
        (local["method"] == method)
        & (local["solution"] == solution)
        & (local["rank"] == rank)
        & (local["cohort"] == "DS1")
    ]
    if local_sub.empty:
        local_sub = local[
            (local["method"] == method)
            & (local["solution"] == solution)
            & (local["rank"] == rank)
        ]
    if not local_sub.empty:
        ax.bar(local_sub["state"].to_numpy(), local_sub["overlap"].to_numpy(), color="mediumpurple", edgecolor="k")
        ax.set_xlabel("State")
        ax.set_ylabel("Local overlap")
        ax.set_title("Within-state local subspaces")
    else:
        ax.set_axis_off()
    fig.suptitle("Figure 3. State geometry vs within-state fluctuation", fontsize=14, fontweight="bold")
    save_figure(fig, fig_dir / "figure3_between_within.pdf")


def fig4_prediction(results_dir: Path, fig_dir: Path, method: str, rank: int) -> None:
    pred = load_tsv(results_dir / "prediction_summary.tsv")
    if pred.empty:
        return
    pred = pred[(pred["method"] == method) & (pred["rank"] == rank)]
    pred = pred[pred["target_group"].isin(["slow_bundle", "fast_bundle", "occupancy", "transition_matrix"])]
    if pred.empty:
        return

    order = ["occupancy", "slow_bundle", "transition_matrix", "fast_bundle"]
    pred["target_group"] = pd.Categorical(pred["target_group"], categories=order, ordered=True)
    pred = pred.sort_values(["cohort", "target_group"])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    cohorts = list(dict.fromkeys(pred["cohort"].tolist()))
    width = 0.18
    x = np.arange(len(cohorts))
    for i, group in enumerate(order):
        sub = pred[pred["target_group"] == group]
        vals = [sub[sub["cohort"] == cohort]["r2_mean_mean"].iloc[0] if not sub[sub["cohort"] == cohort].empty else np.nan for cohort in cohorts]
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=group)
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts)
    ax.set_ylabel("Mean predictive $R^2$")
    ax.set_title("Subject-level dynamic phenotype prediction")
    ax.legend(ncol=2, fontsize=9)
    save_figure(fig, fig_dir / "figure4_prediction_hierarchy.pdf")


def fig5_hierarchy(results_dir: Path, fig_dir: Path, method: str, solution: str, rank: int) -> None:
    hierarchy = load_tsv(results_dir / "hierarchy_retention_summary.tsv")
    if hierarchy.empty:
        return
    sub = hierarchy[
        (hierarchy["method"] == method)
        & (hierarchy["solution"] == solution)
        & (hierarchy["rank"] == rank)
    ]
    if sub.empty:
        return
    cohorts = list(dict.fromkeys(sub["cohort"].tolist()))
    fig, axes = plt.subplots(1, len(cohorts), figsize=(4.6 * len(cohorts), 4.0), squeeze=False)
    tiers = ["sensorimotor", "heteromodal", "transmodal"]
    for ax, cohort in zip(axes[0], cohorts):
        cur = sub[sub["cohort"] == cohort]
        if cur.empty:
            ax.set_axis_off()
            continue
        states = sorted(cur["state"].unique().tolist())
        mat = np.zeros((len(states), len(tiers)), dtype=np.float64)
        for i, state in enumerate(states):
            for j, tier in enumerate(tiers):
                row = cur[(cur["state"] == state) & (cur["tier"] == tier)]
                mat[i, j] = row["rho_mean"].iloc[0] if not row.empty else np.nan
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels(tiers, rotation=25, ha="right")
        ax.set_yticks(range(len(states)))
        ax.set_yticklabels([f"S{s}" for s in states])
        ax.set_title(cohort)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Figure 5. Hierarchy-resolved state retention", fontsize=14, fontweight="bold")
    save_figure(fig, fig_dir / "figure5_hierarchy.pdf")


def fig6_clinical(results_dir: Path, fig_dir: Path, method: str, rank: int) -> None:
    auc = load_tsv(results_dir / "clinical_auc_summary.tsv")
    effects = load_tsv(results_dir / "case_control_effects.tsv")
    if auc.empty:
        return
    auc = auc[(auc["method"] == method) & (auc["rank"] == rank)]
    cohorts = list(dict.fromkeys(auc["cohort"].tolist()))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    feature_sets = ["full", "coupled", "uncoupled"]
    x = np.arange(len(cohorts))
    width = 0.22
    for i, feat in enumerate(feature_sets):
        sub = auc[auc["feature_set"] == feat]
        vals = [sub[sub["cohort"] == cohort]["auc_mean"].iloc[0] if not sub[sub["cohort"] == cohort].empty else np.nan for cohort in cohorts]
        ax.bar(x + (i - 1) * width, vals, width=width, label=feat)
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts)
    ax.set_ylabel("AUC")
    ax.set_title("Clinical utility")
    ax.legend()

    ax = axes[1]
    if not effects.empty and {"metric", "cohen_d", "rho_state"}.issubset(effects.columns):
        sub = effects[(effects["metric"] == "coupled_energy") & (effects["cohort"] == "DS1")]
        if sub.empty:
            sub = effects[effects["metric"] == "coupled_energy"]
        if not sub.empty:
            ax.scatter(sub["rho_state"], np.abs(sub["cohen_d"]), c=sub["state"], cmap="Set2", s=80, edgecolors="k")
            for _, row in sub.iterrows():
                ax.text(row["rho_state"], abs(row["cohen_d"]), f"S{int(row['state'])}", fontsize=8)
            ax.set_xlabel(r"State retention $\rho_k$")
            ax.set_ylabel(r"$|$Cohen's $d|$")
            ax.set_title("Effect size concentrates in high-$\\rho$ states")
        else:
            ax.set_axis_off()
    else:
        ax.set_axis_off()
    fig.suptitle("Figure 6. Clinical utility + case-control concentration", fontsize=14, fontweight="bold")
    save_figure(fig, fig_dir / "figure6_clinical.pdf")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    fig_dir = results_dir / "figures"
    primary = primary_selection(results_dir, args.method, args.solution, args.rank)
    method = primary["method"] or "nuclear_norm"
    solution = primary["solution"]
    rank = int(primary["rank"] or 20)

    if solution is None:
        detail = load_tsv(results_dir / "detail_arrays.tsv")
        if detail.empty:
            raise SystemExit("No detail_arrays.tsv found and no --solution provided")
        solution = str(detail.iloc[0]["solution"])

    print(f"Generating figures from {results_dir}")
    print(f"Primary method={method} solution={solution} rank={rank}")

    fig1_state_manifold(results_dir, fig_dir, method, solution, rank)
    fig2_retention(results_dir, fig_dir, method, solution, rank)
    fig3_between_within(results_dir, fig_dir, method, solution, rank)
    fig4_prediction(results_dir, fig_dir, method, rank)
    fig5_hierarchy(results_dir, fig_dir, method, solution, rank)
    fig6_clinical(results_dir, fig_dir, method, rank)


if __name__ == "__main__":
    main()
