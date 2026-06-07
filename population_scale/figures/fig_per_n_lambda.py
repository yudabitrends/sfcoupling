#!/usr/bin/env python3
"""Supp Fig — Robustness of the rank to per-N lambda re-tuning + model-free dimensionality.

Panel a: effective rank vs N under fixed lambda=0.3 (published) vs lambda re-selected by
         validation-MSE at each N (R1) -- does the descent survive per-N tuning?
Panel b: model-free dimensionality of the GM<->FNC cross-covariance vs N (R2/R4):
         participation ratio, Roy-Vetterli erank, and Gavish-Donoho signal rank -- no
         nuclear-norm fit. The participation ratio sits at the nuclear-norm rank-7 line.
Data: results/reviewer_revision/{per_n_lambda_ladder,model_free_dim}.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from plot_style import apply_nature_strict, nature_panel_label
from utils import style_axes

RR = PROJECT_ROOT / "results/reviewer_revision"
OUT = Path(__file__).parent / "fig_per_n_lambda.pdf"


def main():
    apply_nature_strict()
    ladder = json.loads((RR / "per_n_lambda_ladder.json").read_text())
    mf = json.loads((RR / "model_free_dim.json").read_text())

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    plt.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.17, wspace=0.30)

    # Panel a: fixed vs per-N-tuned rank
    fx = ladder["fixed_lambda_reference"]
    axA.plot(fx["N"], fx["eff_rank_ladder"], "o-", color="#0072B2", ms=4.5, lw=1.2,
             label="fixed $\\lambda=0.3$ (published)")
    pn = ladder["per_n_tuned"]
    axA.plot([r["N_total"] for r in pn], [r["eff_rank_eps1e-4"] for r in pn], "s--",
             color="#D55E00", ms=4.5, lw=1.2, label="$\\lambda$ re-tuned per $N$")
    axA.set_xscale("log")
    axA.set_xlabel("Sample size $N$")
    axA.set_ylabel("Effective rank of $\\hat{B}$")
    axA.legend(frameon=False, fontsize=6, loc="upper right")
    nature_panel_label(axA, "a", x=-0.16, y=1.07)
    axA.set_title("Rank under per-$N$ $\\lambda$ re-tuning", fontsize=7.5)
    style_axes(axA, ygrid=True)

    # Panel b: model-free dimensionality
    rows = mf["rows"]
    Ns = [r["N_total"] for r in rows]
    axB.plot(Ns, [r["participation_ratio"] for r in rows], "o-", color="#009E73", ms=4.5,
             lw=1.2, label="participation ratio")
    axB.plot(Ns, [r["gavish_donoho_rank"] for r in rows], "^-", color="#CC79A7", ms=4.5,
             lw=1.2, label="Gavish-Donoho rank")
    axB.plot(Ns, [r["roy_vetterli_erank"] for r in rows], "v-", color="#888888", ms=4,
             lw=1.0, label="Roy-Vetterli erank")
    axB.axhline(mf["nuclear_norm_rank_reference"], ls=":", color="#0072B2", lw=0.8)
    axB.text(Ns[0], mf["nuclear_norm_rank_reference"] + 2, "nuclear-norm rank 7",
             fontsize=5.8, color="#0072B2")
    axB.set_xscale("log")
    axB.set_xlabel("Sample size $N$")
    axB.set_ylabel("Model-free dimensionality")
    axB.legend(frameon=False, fontsize=6, loc="upper left")
    nature_panel_label(axB, "b", x=-0.16, y=1.07)
    axB.set_title("Model-free dimensionality (no fit)", fontsize=7.5)
    style_axes(axB, ygrid=True)

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
