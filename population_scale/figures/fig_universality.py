#!/usr/bin/env python3
"""Fig — Cross-cohort conservation of the GM->FNC coupling subspace.

Model-free cross-covariance coupling subspace (GM-side singular vectors,
99-dim shared NeuroMark basis), compared by rotation-invariant overlap and
principal angles. Lead main figure for the reframed manuscript.

(a) Leading-axis cosine with UKB per clinical cohort (bootstrap 95% CI), with
    within-UKB references (cross-tier) and a Haar-random null band.
(b) Principal-angle spectra (cos of angles 1..6) per cohort -> ~3-5 shared dims.
(c) Subspace overlap (residualized) vs each cohort's matched-N ceiling, with
    %-of-ceiling labels and the random baseline.
(d) Metric contrast: the naive Mode-1 *vector* alignment of the regularized
    estimator (r=0.03, a near-degeneracy/basepoint artifact) vs the model-free
    *subspace* leading cosine (~0.93) -- the wrong-metric -> right-metric lesson.

Reads:
  results/discovery/universality_persite_replication.json
  results/discovery/universality_modefree_ds1_ukb.json
  results/discovery/universality_within_ukb_tiers.json
  results/reviewer_revision/cross_cohort_mode_procrustes.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from plot_style import apply_neuroimage_style, nature_panel_label  # noqa: E402

DISC = ROOT / "results/discovery"
OUT = Path(__file__).parent / "fig_universality.pdf"

# cohort order + colors (US = blues, China = vermillion)
SITES = ["COBRE", "FBIRN", "PK_MPRC", "ChineseSZ"]
LABEL = {"COBRE": "COBRE", "FBIRN": "FBIRN", "PK_MPRC": "MPRC", "ChineseSZ": "ChineseSZ"}
REGION = {"COBRE": "US", "FBIRN": "US", "PK_MPRC": "US", "ChineseSZ": "China"}
CC = {"COBRE": "#0072B2", "FBIRN": "#56B4E9", "PK_MPRC": "#3C7DB0",
      "ChineseSZ": "#D55E00", "pooled": "#444444"}
C_REF = "#009E73"     # within-UKB reference
C_RAND = "#999999"    # random null
C_CEIL = "#BBBBBB"


def load():
    ps = json.loads((DISC / "universality_persite_replication.json").read_text())
    pooled = json.loads((DISC / "universality_modefree_ds1_ukb.json").read_text())
    tiers = json.loads((DISC / "universality_within_ukb_tiers.json").read_text())
    proc = json.loads((ROOT / "results/reviewer_revision/cross_cohort_mode_procrustes.json").read_text())
    return ps, pooled, tiers, proc


def random_lead_cos_null(k=6, reps=2000, seed=3):
    """95th pct of the leading principal cosine between two random k-subspaces in R^99."""
    g = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        A = np.linalg.qr(g.standard_normal((99, k)))[0]
        B = np.linalg.qr(g.standard_normal((99, k)))[0]
        vals.append(np.linalg.svd(A.T @ B, compute_uv=False)[0])
    return float(np.mean(vals)), float(np.quantile(vals, 0.95))


def main():
    apply_neuroimage_style()
    ps, pooled, tiers, proc = load()
    rand_mean, rand_p95 = random_lead_cos_null()

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.3))
    plt.subplots_adjust(left=0.085, right=0.975, top=0.92, bottom=0.10,
                        hspace=0.48, wspace=0.30)
    (axA, axB), (axC, axD) = axes

    # ── (a) leading-axis cosine forest ──
    rows = []
    for S in SITES:
        ci = ps["sites"][S]["lead_cos_resid_ci"]  # [mean, lo, hi]
        rows.append((f"{LABEL[S]}  ($N{{=}}{ps['sites'][S]['n']}$)", ci, CC[S], REGION[S]))
    pooled_lead = pooled["ds1_vs_ukb_resid"]["k6"]["principal_cos"][0]
    rows.append(("pooled DS1  ($N{=}1{,}151$)", [pooled_lead, pooled_lead, pooled_lead], CC["pooled"], None))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, ci, col, reg) in zip(ys, rows):
        m, lo, hi = ci
        axA.plot([lo, hi], [y, y], color=col, lw=1.4, zorder=2)
        axA.scatter([m], [y], s=34, color=col, edgecolor="white", lw=0.5, zorder=3)
        axA.text(-0.012, y, lab, transform=axA.get_yaxis_transform(),
                 ha="right", va="center", fontsize=6.3, color="#222")
        if reg == "China":
            axA.text(hi + 0.012, y, "China", va="center", fontsize=5.6, color=col)
    # within-UKB references
    t13 = tiers["T1_vs_T3"]["k6"]["pcos"][0]
    t23 = tiers["T2_vs_T3"]["k6"]["pcos"][0]
    axA.axvline(t23, color=C_REF, ls="--", lw=0.8, zorder=1)
    axA.text(t23, len(rows) - 0.35, "within-UKB\n(T2$\\leftrightarrow$T3)", color=C_REF,
             fontsize=5.4, ha="center", va="bottom")
    # random null band
    axA.axvspan(0, rand_p95, color=C_RAND, alpha=0.13, zorder=0)
    axA.text(rand_p95 / 2, 0.0, "random", color="#666", fontsize=5.4, ha="center", va="center")
    axA.set_xlim(0, 1.02)
    axA.set_ylim(-0.6, len(rows) - 0.4)
    axA.set_yticks([])
    axA.set_xlabel("leading-axis cosine with UKB")
    for sp in ("left", "right", "top"):
        axA.spines[sp].set_visible(False)
    axA.set_title("Conserved leading coupling axis", fontsize=8)
    nature_panel_label(axA, "a", x=-0.04)

    # ── (b) principal-angle spectra ──
    idx = np.arange(1, 7)
    for S in SITES:
        pc = ps["sites"][S]["vs_ukb_resid"]["k6"]["pcos"]
        axB.plot(idx, pc, "-o", color=CC[S], ms=3, lw=1.1, label=LABEL[S])
    axB.axhline(rand_p95, color=C_RAND, ls=":", lw=0.8)
    axB.text(6, rand_p95 + 0.02, "random 95%", color="#666", fontsize=5.4, ha="right")
    axB.set_xlabel("principal angle index")
    axB.set_ylabel("cosine (resid)")
    axB.set_ylim(0, 1.0)
    axB.set_xticks(idx)
    axB.legend(fontsize=5.6, loc="upper right", ncol=2, handlelength=1.0)
    axB.set_title("Shared coupling dimensions", fontsize=8)
    nature_panel_label(axB, "b")

    # ── (c) overlap vs ceiling bars ──
    x = np.arange(len(SITES))
    w = 0.36
    ov = [ps["sites"][S]["vs_ukb_resid"]["k6"]["overlap"] for S in SITES]
    ce = [ps["sites"][S]["matched_N_ceiling"]["6"][0] for S in SITES]
    axC.bar(x - w / 2, ce, w, color=C_CEIL, label="matched-$N$ ceiling")
    axC.bar(x + w / 2, ov, w, color=[CC[S] for S in SITES], label="cross-cohort (resid)")
    for xi, o, c in zip(x, ov, ce):
        axC.text(xi + w / 2, o + 0.012, f"{100*o/c:.0f}%", ha="center", fontsize=5.6, color="#222")
    rb = ps["random_baseline"]["k6"][0]
    axC.axhline(rb, color=C_RAND, ls=":", lw=0.8)
    axC.text(len(SITES) - 0.5, rb + 0.012, "random", color="#666", fontsize=5.4, ha="right")
    axC.set_xticks(x)
    axC.set_xticklabels([LABEL[S] for S in SITES], rotation=20, ha="right", fontsize=6)
    axC.set_ylabel("subspace overlap ($k{=}6$)")
    axC.set_ylim(0, max(ce) * 1.15)
    axC.legend(fontsize=5.6, loc="upper left")
    axC.set_title("Overlap vs achievable ceiling", fontsize=8)
    nature_panel_label(axC, "c")

    # ── (d) metric contrast ──
    naive = proc["comparisons"]["same_method_DS1RSCM_vs_UKBRSCM"]["mode1_abs_r_mean"]
    correct = pooled["ds1_vs_ukb_resid"]["k6"]["principal_cos"][0]
    bars = [("Naive Mode-1\nvector (RSCM)", naive, "#C0C0C0"),
            ("Model-free\nsubspace", correct, "#0072B2")]
    bx = np.arange(len(bars))
    axD.bar(bx, [b[1] for b in bars], 0.55, color=[b[2] for b in bars],
            edgecolor="#333", lw=0.5)
    for xi, b in zip(bx, bars):
        axD.text(xi, b[1] + 0.02, f"{b[1]:.2f}", ha="center", fontsize=7, color="#222")
    axD.set_xticks(bx)
    axD.set_xticklabels([b[0] for b in bars], fontsize=6.2)
    axD.set_ylabel("DS1$\\leftrightarrow$UKB alignment")
    axD.set_ylim(0, 1.05)
    axD.set_title("Why the metric matters", fontsize=8)
    axD.text(0.27, 0.55, "near-degenerate\nspectrum + per-cohort\nbasepoint $\\Rightarrow$ vector\n$r$ is artifactual",
             transform=axD.transAxes, fontsize=5.3, ha="center", color="#666")
    nature_panel_label(axD, "d")

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    print(f"Wrote {OUT}  (random lead-cos 95%={rand_p95:.3f})")


if __name__ == "__main__":
    main()
