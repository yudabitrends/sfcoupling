#!/usr/bin/env python3
"""Fig — Spectral-rank descent (finer N-grid) + two controls.

Panel a: effective rank vs N (log-x), primary tiers + finer tiers (N=2k/4k/8k);
         a monotonic descent (19 -> 7) spanning Helmer's N>=1000 onset.
Panel b: largest singular-value gap (orders of magnitude) by estimator on the
         canonical Tier-3 training set — nuclear-norm (hard gap) vs ridge / OLS
         (graded) vs the gray-matter--FNC label-permutation null (no structure);
         effective rank annotated beneath each bar.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from plot_style import apply_nature_strict, nature_panel_label
from utils import style_axes

RR = PROJECT_ROOT / "results/reviewer_revision"
OUT_PDF = Path(__file__).parent / "fig_rank_descent.pdf"
C_LINE = "#0072B2"
C_FINER = "#D55E00"


def main() -> None:
    apply_nature_strict()
    finer = json.loads((RR / "finer_ngrid_crossover.json").read_text())
    est = json.loads((RR / "estimator_rank_and_null.json").read_text())

    pts = [(1079, 19, False), (11820, 7, False), (37775, 7, False)]
    for r in finer["finer_tiers"]:
        pts.append((r["N_total"], r["eff_rank"], True))
    pts.sort()

    e = est["A3_estimator_comparison"]
    bars = [
        ("Nuclear\nnorm", e["nuclear_lam0.3"]["max_log10_gap"],
         e["nuclear_lam0.3"]["eff_rank_eps1e-4"], "#003D6B"),
        ("Ridge", e["ridge_alpha1"]["max_log10_gap"],
         e["ridge_alpha1"]["eff_rank_eps1e-4"], "#88B0D6"),
        ("OLS", e["ols"]["max_log10_gap"], e["ols"]["eff_rank_eps1e-4"], "#88B0D6"),
        ("Permutation\nnull", 0.0, round(est["A2_null"]["null_eff_rank_mean"]), "#BBBBBB"),
    ]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    plt.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.17, wspace=0.32)

    # ── Panel a ──
    Ns = [p[0] for p in pts]
    ranks = [p[1] for p in pts]
    axA.plot(Ns, ranks, "-", color=C_LINE, lw=1.3, zorder=2)
    for nN, rk, finer_pt in pts:
        axA.scatter([nN], [rk], s=34, zorder=3,
                    color=C_FINER if finer_pt else C_LINE,
                    edgecolor="white", linewidth=0.5)
        axA.annotate(str(rk), (nN, rk), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=6.5,
                     color=C_FINER if finer_pt else "#333333")
    axA.axvline(1000, ls="--", color="#888888", lw=0.8)
    axA.text(930, 11, "Helmer $N{\\gtrsim}1{,}000$", fontsize=6,
             color="#888888", rotation=90, va="center", ha="center")
    axA.scatter([], [], s=34, color=C_FINER, edgecolor="white", linewidth=0.5,
                label="finer tiers (new)")
    axA.scatter([], [], s=34, color=C_LINE, edgecolor="white", linewidth=0.5,
                label="primary tiers")
    axA.legend(frameon=False, fontsize=6, loc="upper right")
    axA.set_xscale("log")
    axA.set_xlabel("Sample size $N$")
    axA.set_ylabel("Effective rank of $\\hat{B}$")
    axA.set_ylim(0, 23)
    nature_panel_label(axA, "a", x=-0.16, y=1.07)
    axA.set_title("Monotonic rank descent", fontsize=7.5, loc="center")
    style_axes(axA, ygrid=True)

    # ── Panel b ──
    xs = list(range(len(bars)))
    axB.bar(xs, [b[1] for b in bars], color=[b[3] for b in bars], width=0.66,
            edgecolor="white", linewidth=0.4)
    axB.set_xticks(xs)
    axB.set_xticklabels([b[0] for b in bars], fontsize=6.5)
    axB.set_ylabel("Largest singular-value gap\n(orders of magnitude)")
    axB.set_ylim(0, 15.5)
    for x, b in zip(xs, bars):
        axB.annotate(f"{b[1]:.1f}", (x, b[1]), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=6.5)
        axB.annotate(f"rank {b[2]}", (x, b[1]), textcoords="offset points",
                     xytext=(0, 2), ha="center", fontsize=5.8, color="#555")
    nature_panel_label(axB, "b", x=-0.16, y=1.07)
    axB.set_title("Hard gap is estimator-specific and signal-driven",
                  fontsize=7.5, loc="center")
    style_axes(axB, ygrid=True)

    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_PDF.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("Wrote", OUT_PDF)


if __name__ == "__main__":
    main()
