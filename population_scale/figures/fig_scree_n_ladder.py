#!/usr/bin/env python3
"""Fig 2 — Scree plot across 3-tier UKB N-ladder. Nature single column."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from plot_style import NATURE_SINGLE, apply_nature_strict
from utils import style_axes

TIERS = [
    ("Tier 1: N=1,079", 1_079,
     PROJECT_ROOT / "results/rscm_smoke_ukb1079_le_harmon/decompositions/rscm_seed42_S.npy",
     "#88B0D6"),
    ("Tier 2: N=11,820", 11_820,
     PROJECT_ROOT / "results/rscm_ukb11820_le_harmon_lam03/decompositions/rscm_seed42_S.npy",
     "#0072B2"),
    ("Tier 3: N=37,775", 37_775,
     PROJECT_ROOT / "results/rscm_ukb37775_le_harmon_lam03/decompositions/rscm_seed42_S.npy",
     "#003D6B"),
]

EPS_THRESHOLD = 1e-4
OUT_PDF = Path(__file__).parent / "fig_scree_n_ladder.pdf"


def main() -> None:
    apply_nature_strict()
    fig, ax = plt.subplots(figsize=NATURE_SINGLE)

    s_t3 = None
    for label, n, path, color in TIERS:
        if not path.exists():
            continue
        s = np.load(path)
        idx = np.arange(1, len(s) + 1)
        ax.semilogy(idx, np.maximum(s, 1e-18), "-o", color=color,
                    markersize=2.5, linewidth=1.0, label=label)
        if "37" in label:
            s_t3 = s

    # ε-threshold line
    if s_t3 is not None:
        thresh = EPS_THRESHOLD * s_t3.max()
        ax.axhline(thresh, linestyle="--", color="#888888", linewidth=0.6,
                   label=f"$\\varepsilon\\cdot\\sigma_{{\\max}}$ ($\\varepsilon={EPS_THRESHOLD:.0e}$)")

    ax.set_xlabel("Singular value index $i$")
    ax.set_ylabel("$\\sigma_i$")
    ax.set_xlim(0, 30)
    ax.set_ylim(1e-18, 5)
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    style_axes(ax, ygrid=True, xgrid=False)

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_PDF.with_suffix(".png"), dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
