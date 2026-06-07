#!/usr/bin/env python3
"""Supp Fig — Bootstrap stability of Mode-1/2/3 (raincloud on a shared axis).

100 bootstrap resamples of the Tier-3 training partition; for each, RSCM is
refit at lambda=0.3 and the leading three GM-side singular vectors are aligned
to the canonical Tier-3 reference. A single shared |r| axis makes the three
mode distributions directly comparable (Mode-3 has the widest tail). Each mode:
half-violin (density) + jittered bootstrap points + mean with 95% percentile CI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from plot_style import apply_nature_strict
from utils import style_axes

BOOT_DIR = PROJECT_ROOT / "results/rscm_ukb37775_bootstrap100"
OUT_PDF = Path(__file__).parent / "fig_bootstrap_dist.pdf"
COLORS = ["#0072B2", "#009E73", "#D55E00"]


def main() -> None:
    apply_nature_strict()
    boots = [json.loads(p.read_text()) for p in sorted(BOOT_DIR.glob("per_boot_*.json"))]
    data = [np.array([b[f"mode{m}_abs_r_vs_main"] for b in boots]) for m in (1, 2, 3)]
    n = len(boots)

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    plt.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.13)
    rng = np.random.default_rng(0)

    positions = [1, 2, 3]
    for i, (vals, color, pos) in enumerate(zip(data, COLORS, positions)):
        # half-violin (right side)
        vp = ax.violinplot(vals, positions=[pos], widths=0.7, showextrema=False,
                           points=200)
        for body in vp["bodies"]:
            verts = body.get_paths()[0].vertices
            verts[:, 0] = np.clip(verts[:, 0], pos, np.inf)  # keep right half
            body.set_facecolor(color)
            body.set_alpha(0.25)
            body.set_edgecolor(color)
            body.set_linewidth(0.6)
        # jittered points (left side)
        jit = pos - 0.10 - rng.uniform(0, 0.18, size=len(vals))
        ax.scatter(jit, vals, s=7, color=color, alpha=0.55, edgecolor="none", zorder=3)
        # mean + 95% CI
        mean = vals.mean()
        lo, hi = np.percentile(vals, [2.5, 97.5])
        ax.errorbar(pos - 0.02, mean, yerr=[[mean - lo], [hi - mean]], fmt="o",
                    color=color, ms=5, capsize=2.5, elinewidth=1.0, mec="white",
                    mew=0.5, zorder=4)
        ax.text(pos + 0.30, mean, f"{mean:.4f}\n[{lo:.3f}, {hi:.3f}]",
                va="center", ha="left", fontsize=5.8, color=color)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Mode 1", "Mode 2", "Mode 3"])
    ax.set_xlim(0.5, 3.7)
    ax.set_ylim(0.94, 1.002)
    ax.set_ylabel("Bootstrap alignment $|r|$ with Tier-3 reference")
    ax.set_title(f"Tier-3 leading-mode stability across {n} bootstraps "
                 f"(effective rank $=7$ in all {n})", fontsize=7, loc="left")
    style_axes(ax, ygrid=True)

    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_PDF.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
