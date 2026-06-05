"""
Per-tier break-pairing (structured) null on the subspace-overlap statistic O.

Goal: localize the GENUINELY GM-specific component of the GM->FNC directional
overlap along the cortical hierarchy. The manuscript's per-tier overlap result
uses only the weak random-subspace null (~k/n_edges). Here we apply the SAME
structured break-pairing null used globally (scripts/overlap_permutation_null.py)
SEPARATELY within each hierarchy tier: permute the GM<->FNC subject pairing in the
training partition, refit Ridge, and recompute the tier-restricted overlap. The
margin (observed O minus break-pairing null) is the part of each tier's overlap
that requires the correct cross-modal correspondence, i.e. the genuinely
GM-specific coupling, after removing generic shared low-dimensional structure.

Reuses tier masks + subset_metrics from hierarchy_resolved_dissociation.py and the
data/fit path from overlap_permutation_null.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path("/home/users/ybi3/sfcoupling")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from models.baselines import fit_ridge_grid                       # noqa: E402
from models.utils import load_config, load_training_contracts, set_seed  # noqa: E402
from hierarchy_resolved_dissociation import (                     # noqa: E402
    build_masks, subset_metrics, parse_fnc_edges,
    PRIMARY_TIER_MAP, CB_TO_SENSORIMOTOR_MAP,
)

GROUPS = ["sensorimotor", "heteromodal", "transmodal",
          "within_all", "between_all", "SM-HM", "SM-TM", "HM-TM"]
KS = [5, 10, 20]
N_PERM = 300


def run(masks, Xtr, Ytr, Xte, Yte, alpha, tag):
    B = Ridge(alpha=alpha).fit(Xtr, Ytr).coef_.T
    Ypred = Xte @ B
    obs = {g: {k: subset_metrics(Yte, Ypred, masks[g], k)["O"]
               for k in KS} for g in GROUPS if masks[g].size}

    rng = np.random.default_rng(42)
    null = {g: {k: [] for k in KS} for g in obs}
    for _ in range(N_PERM):
        perm = rng.permutation(Xtr.shape[0])
        Bp = Ridge(alpha=alpha).fit(Xtr[perm], Ytr).coef_.T
        Yp = Xte @ Bp
        for g in obs:
            for k in KS:
                null[g][k].append(subset_metrics(Yte, Yp, masks[g], k)["O"])

    out = {}
    for g in obs:
        out[g] = {"n_edges": int(masks[g].size), "k": {}}
        for k in KS:
            a = np.array(null[g][k])
            o = obs[g][k]
            out[g]["k"][str(k)] = {
                "observed_O": round(float(o), 4),
                "null_mean": round(float(a.mean()), 4),
                "null_std": round(float(a.std()), 4),
                "margin": round(float(o - a.mean()), 4),
                "sd_above": round(float((o - a.mean()) / (a.std() + 1e-12)), 2),
                "p_value": round(float((np.sum(a >= o) + 1) / (N_PERM + 1)), 4),
            }
    return out


def main():
    set_seed(42)
    cfg = load_config(str(ROOT / "train" / "config_baselines.yaml"))
    data = load_training_contracts(cfg)
    X1, Y1 = data["X1"].astype(np.float64), data["Y1"].astype(np.float64)
    itr, iva, ite = data["idx1_train"], data["idx1_val"], data["idx1_test"]
    Xtr, Ytr, Xva, Yva, Xte, Yte = (X1[itr], Y1[itr], X1[iva], Y1[iva],
                                    X1[ite], Y1[ite])
    edges = parse_fnc_edges(data["fnc_names"])

    alphas = cfg.get("ridge", {}).get("alphas", [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    model, _ = fit_ridge_grid(Xtr, Ytr, Xva, Yva, alphas)
    alpha = float(getattr(model, "alpha", 1.0))

    result = {
        "alpha": alpha, "n_perm": N_PERM,
        "primary": run(build_masks(edges, PRIMARY_TIER_MAP),
                       Xtr, Ytr, Xte, Yte, alpha, "primary"),
        "cb_to_sensorimotor": run(build_masks(edges, CB_TO_SENSORIMOTOR_MAP),
                                  Xtr, Ytr, Xte, Yte, alpha, "cb_sm"),
    }
    outdir = ROOT / "results" / "hierarchy_perm_null"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "hierarchy_break_pairing_null.json").write_text(json.dumps(result, indent=2))

    # compact console summary (primary mapping)
    print(f"alpha={alpha}  n_perm={N_PERM}")
    print(f"{'group':14s} {'n':>4s}  " + "  ".join(f"k={k}: O/null/marg/sd/p" for k in KS))
    for g in result["primary"]:
        row = result["primary"][g]
        cells = []
        for k in KS:
            d = row["k"][str(k)]
            cells.append(f"{d['observed_O']:.3f}/{d['null_mean']:.3f}/"
                         f"{d['margin']:+.3f}/{d['sd_above']:+.1f}/{d['p_value']:.3f}")
        print(f"{g:14s} {row['n_edges']:4d}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
