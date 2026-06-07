#!/usr/bin/env python3
"""Figure 2 — Spectral structure and low dimensionality of the GM->FNC coupling.

a  Singular-value spectra of the coefficient matrix across the three tiers (log-y).
b  Effective rank at fixed lambda=0.3 vs N (primary + finer tiers) -- the operating-point descent.
c  Largest singular-value gap by estimator (nuclear-norm vs ridge/OLS vs permutation null).
d  Model-free dimensionality of the GM<->FNC cross-covariance vs N (participation ratio,
   Gavish-Donoho, Roy-Vetterli erank) -- the sample-size-invariant low-dimensionality.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from nature_style import apply_nature_strict, nature_panel_label
from utils import style_axes

OUT = Path(__file__).parent / "fig_spectral.pdf"
RR = PROJECT_ROOT / "results/reviewer_revision"
DEC = PROJECT_ROOT / "results"
SCREE = [
    ("Tier 1: $N$=1,079", DEC / "rscm_smoke_ukb1079_le_harmon/decompositions/rscm_seed42_S.npy", "#92C5DE"),
    ("Tier 2: $N$=11,820", DEC / "rscm_ukb11820_le_harmon_lam03/decompositions/rscm_seed42_S.npy", "#2166AC"),
    ("Tier 3: $N$=37,775", DEC / "rscm_ukb37775_le_harmon_lam03/decompositions/rscm_seed42_S.npy", "#053061"),
]
EPS = 1e-4


def panel_scree(ax):
    s_t3 = None
    for label, path, color in SCREE:
        if not path.exists():
            continue
        s = np.load(path)
        ax.semilogy(np.arange(1, len(s) + 1), np.maximum(s, 1e-18), "-o",
                    color=color, markersize=2.3, linewidth=1.0, label=label)
        if "37" in label:
            s_t3 = s
    if s_t3 is not None:
        ax.axhline(EPS * s_t3.max(), ls="--", color="#888", lw=0.6,
                   label="$\\varepsilon\\cdot\\sigma_{\\max}$")
    ax.set_xlabel("Singular value index $i$"); ax.set_ylabel("$\\sigma_i$")
    ax.set_xlim(0, 30); ax.set_ylim(1e-18, 5)
    ax.legend(frameon=False, fontsize=5.5, loc="lower left")
    style_axes(ax, ygrid=True, xgrid=False)


def panel_rankdescent(ax):
    finer = json.loads((RR / "finer_ngrid_crossover.json").read_text())
    # Tier-1 rung uses the epsilon-threshold rank (21), consistent with the
    # finer-tier (13/9/8) and Tier-2/3 (7) epsilon-threshold ladder; the Kaiser
    # value (19) is reported separately in the text where the criterion is named.
    pts = [(1079, 21, False), (11820, 7, False), (37775, 7, False)]
    for r in finer["finer_tiers"]:
        pts.append((r["N_total"], r["eff_rank"], True))
    pts.sort()
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color="#2166AC", lw=1.3, zorder=2)
    for nN, rk, fp in pts:
        ax.scatter([nN], [rk], s=28, zorder=3, color="#B2182B" if fp else "#2166AC",
                   edgecolor="white", linewidth=0.5)
        ax.annotate(str(rk), (nN, rk), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=6, color="#B2182B" if fp else "#333")
    ax.axvline(1000, ls="--", color="#888", lw=0.8)
    ax.text(930, 11, "Helmer $N{\\gtrsim}1{,}000$", fontsize=5.5, color="#888",
            rotation=90, va="center", ha="center")
    ax.set_xscale("log"); ax.set_ylim(0, 23)
    ax.set_xlabel("Sample size $N$"); ax.set_ylabel("Effective rank (fixed $\\lambda{=}0.3$)")
    style_axes(ax, ygrid=True)


def panel_estimatorgap(ax):
    est = json.loads((RR / "estimator_rank_and_null.json").read_text())
    e = est["A3_estimator_comparison"]
    bars = [("Nuclear\nnorm", e["nuclear_lam0.3"]["max_log10_gap"], e["nuclear_lam0.3"]["eff_rank_eps1e-4"], "#053061"),
            ("Ridge", e["ridge_alpha1"]["max_log10_gap"], e["ridge_alpha1"]["eff_rank_eps1e-4"], "#92C5DE"),
            ("OLS", e["ols"]["max_log10_gap"], e["ols"]["eff_rank_eps1e-4"], "#92C5DE"),
            ("Perm.\nnull", 0.0, round(est["A2_null"]["null_eff_rank_mean"]), "#BBBBBB")]
    xs = range(len(bars))
    ax.bar(list(xs), [b[1] for b in bars], color=[b[3] for b in bars], width=0.66,
           edgecolor="white", linewidth=0.4)
    ax.set_xticks(list(xs)); ax.set_xticklabels([b[0] for b in bars], fontsize=6)
    ax.set_ylim(0, 15.5)
    ax.set_ylabel("Largest s.v. gap (orders of mag.)")
    for x, b in zip(xs, bars):
        ax.annotate(f"{b[1]:.1f}", (x, b[1]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=6)
        ax.annotate(f"rank {b[2]}", (x, b[1]), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=5.5, color="#555")
    style_axes(ax, ygrid=True)


def panel_modelfree(ax):
    mf = json.loads((RR / "model_free_dim.json").read_text())
    rows = mf["rows"]; Ns = [r["N_total"] for r in rows]
    ax.plot(Ns, [r["participation_ratio"] for r in rows], "o-", color="#5AAE61", ms=4, lw=1.2, label="participation ratio")
    ax.plot(Ns, [r["gavish_donoho_rank"] for r in rows], "^-", color="#762A83", ms=4, lw=1.2, label="Gavish-Donoho")
    ax.plot(Ns, [r["roy_vetterli_erank"] for r in rows], "v-", color="#888", ms=3.5, lw=1.0, label="Roy-Vetterli erank")
    ax.axhline(mf["nuclear_norm_rank_reference"], ls=":", color="#2166AC", lw=0.8)
    ax.text(Ns[0], mf["nuclear_norm_rank_reference"] + 2.5, "nuclear-norm rank 7", fontsize=5.5, color="#2166AC")
    ax.set_xscale("log"); ax.set_xlabel("Sample size $N$"); ax.set_ylabel("Model-free dimensionality")
    ax.legend(frameon=False, fontsize=5.5, loc="upper right")
    style_axes(ax, ygrid=True)


def main():
    apply_nature_strict()
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30, left=0.085, right=0.985, top=0.95, bottom=0.085)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    panel_scree(axes[0]); panel_rankdescent(axes[1]); panel_estimatorgap(axes[2]); panel_modelfree(axes[3])
    for ax, lab in zip(axes, "abcd"):
        nature_panel_label(ax, lab, x=-0.16, y=1.06)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
