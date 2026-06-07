#!/usr/bin/env python3
"""Fig 3 — Mode-1 N-trajectory + within-UKB matrix. Nature double column."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from plot_style import apply_nature_strict, nature_panel_label
from utils import style_axes

OUT_PDF = Path(__file__).parent / "fig_mode1_trajectory.pdf"


def load_within_ukb():
    return json.loads((PROJECT_ROOT / "results/reviewer_revision/within_ukb_tier_mode1_alignment.json").read_text())


def load_r1():
    return json.loads((PROJECT_ROOT / "results/reviewer_revision/ukb_downsample_N805_mode1_control_disjoint.json").read_text())


def main() -> None:
    apply_nature_strict()
    within = load_within_ukb()
    r1 = load_r1()

    fig = plt.figure(figsize=(7.2, 2.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0],
                          wspace=0.40, left=0.07, right=0.98,
                          top=0.86, bottom=0.18)

    # ── Panel a: Mode-1 r vs N ──
    ax_a = fig.add_subplot(gs[0, 0])
    Ns = np.array([805, 1079, 11820, 37775])
    rs = np.array([
        r1["mode1_r_mean"],
        next(p for p in within["pairs"] if p["pair"] == "tier1_N1079__tier3_N37775")["mode1_abs_r"],
        next(p for p in within["pairs"] if p["pair"] == "tier2_N11820__tier3_N37775")["mode1_abs_r"],
        1.0,
    ])
    rs_err = np.array([r1["mode1_r_std"], 0.0, 0.0, 0.0])

    ax_a.errorbar(Ns, rs, yerr=rs_err, fmt="o-", color="#0072B2",
                  markersize=4, linewidth=1.0, capsize=2,
                  label="UKB Mode-1 vs Tier 3")
    ax_a.axhline(0.8, linestyle=":", color="#999", linewidth=0.5)
    ax_a.scatter([805], [0.030], color="#D55E00", s=30, marker="^", zorder=5,
                 label="DS1 (clinical, 52% SZ)")
    ax_a.axvline(1000, linestyle="--", color="#888", linewidth=0.5)
    ax_a.text(1080, 0.08, "Helmer\n$N{\\approx}1{,}000$",
              fontsize=5.5, color="#666")
    ax_a.set_xscale("log")
    ax_a.set_xlabel("Sample size $N$")
    ax_a.set_ylabel("Mode-1 $|r|$ vs Tier 3")
    ax_a.set_ylim(-0.05, 1.06)
    ax_a.set_xlim(500, 60000)
    ax_a.legend(frameon=False, fontsize=5.5, loc="lower right")
    nature_panel_label(ax_a, "a", x=-0.22, y=1.10)
    style_axes(ax_a, ygrid=True)

    # ── Panel b: within-UKB Mode-1 matrix ──
    ax_b = fig.add_subplot(gs[0, 1])
    tiers = ["Tier 1\n1,079", "Tier 2\n11,820", "Tier 3\n37,775"]
    M = np.eye(3)
    pair_map = {
        ("tier1_N1079", "tier2_N11820"): (0, 1),
        ("tier1_N1079", "tier3_N37775"): (0, 2),
        ("tier2_N11820", "tier3_N37775"): (1, 2),
    }
    for p in within["pairs"]:
        a, b = p["pair"].split("__")
        i, j = pair_map[(a, b)]
        M[i, j] = M[j, i] = p["mode1_abs_r"]
    im_b = ax_b.imshow(M, vmin=0.85, vmax=1.0, cmap="viridis", aspect="equal")
    for i in range(3):
        for j in range(3):
            ax_b.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                      color="white" if M[i, j] < 0.95 else "black",
                      fontsize=6.5)
    ax_b.set_xticks([0, 1, 2]); ax_b.set_yticks([0, 1, 2])
    ax_b.set_xticklabels(tiers, fontsize=6)
    ax_b.set_yticklabels(tiers, fontsize=6)
    nature_panel_label(ax_b, "b", x=-0.18, y=1.10)
    cb = plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=5.5)

    # ── Panel c: per-mode heatmap ──
    ax_c = fig.add_subplot(gs[0, 2])
    pair_labels = ["T1↔T2", "T1↔T3", "T2↔T3"]
    mode_labels = ["Mode 1", "Mode 2", "Mode 3"]
    pair_order = ["tier1_N1079__tier2_N11820",
                  "tier1_N1079__tier3_N37775",
                  "tier2_N11820__tier3_N37775"]
    HM = np.zeros((3, 3))
    for j, pair in enumerate(pair_order):
        d = next(p for p in within["pairs"] if p["pair"] == pair)
        HM[0, j] = d["mode1_abs_r"]
        HM[1, j] = d["mode2_abs_r"]
        HM[2, j] = d["mode3_abs_r"]
    im_c = ax_c.imshow(HM, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    for i in range(3):
        for j in range(3):
            ax_c.text(j, i, f"{HM[i, j]:.2f}", ha="center", va="center",
                      color="white" if HM[i, j] < 0.5 else "black", fontsize=6.5)
    ax_c.set_xticks([0, 1, 2]); ax_c.set_yticks([0, 1, 2])
    ax_c.set_xticklabels(pair_labels, fontsize=6.5)
    ax_c.set_yticklabels(mode_labels, fontsize=6.5)
    nature_panel_label(ax_c, "c", x=-0.18, y=1.10)
    cb = plt.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=5.5)

    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_PDF.with_suffix(".png"), dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
