"""R6 — covariate-stratified permutation null (strengthens A2).

The plain permutation null (A2) destroys ALL GM<->FNC association, including any residual
Age/Gender structure that survives tangent residualization. This control instead permutes the
GM<->FNC subject correspondence ONLY WITHIN Age-decile x Gender strata, preserving residual
covariate structure while destroying the within-stratum GM-FNC coupling. If the fitted rank
still collapses to ~0, the rank-7 is driven by genuine GM-FNC coupling, not a residual confound.

Output: results/reviewer_revision/stratified_null.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.rsfe import RSFE, RSFEConfig, _istanuclear  # noqa: E402
from models.utils import load_config, load_training_contracts  # noqa: E402
from preprocess.harmonize import load_covariates, apply_tangent_residualizer  # noqa: E402
from train.run_rscm import _load_variant, _y_to_spd  # noqa: E402

CONFIG = PROJECT_ROOT / "natcomm/configs/config_rscm_ukb37775_le_harmon_lam03.yaml"
OUT = PROJECT_ROOT / "results/reviewer_revision/stratified_null.json"
D = 53; LAM = 0.3; EPS = 1e-4; NPERM = 25


def eff_rank(s): return int(np.sum(s > EPS * s.max()))


def main():
    cfg = load_config(str(CONFIG)); data = load_training_contracts(cfg)
    base = Path(cfg["paths"]["aligned_features_dir"]).resolve(); r = cfg["rscm"]
    X1 = _load_variant(base, "dataset1_X", r.get("x_variant", "default")).astype(np.float32)
    Y1 = _load_variant(base, "dataset1_Y", r.get("y_variant", "raw")).astype(np.float32)
    cols = tuple(r.get("harmonize_cols", ("Age", "Gender")))
    C1 = load_covariates(base / "meta" / "dataset1_subjects.tsv", np.array(data["ids1"]), cols)
    meta = pd.read_csv(base / "meta" / "dataset1_subjects.tsv", sep="\t")
    itr = data["idx1_train"]; iva = data["idx1_val"]
    Ytr = _y_to_spd(Y1[itr], d=D, fisher_z=True); Yva = _y_to_spd(Y1[iva], d=D, fisher_z=True)
    m = RSFE(RSFEConfig(d=D, nn_lambda_grid=(LAM,), metric="logE")).fit(
        X1[itr], Ytr, S_val=X1[iva], F_val_spd=Yva, covariates_train=C1[itr], covariates_val=C1[iva])
    T = apply_tangent_residualizer(m._to_tangent(Ytr).astype(np.float64), C1[itr], m.harmonize_beta)
    T_c = T - m.mean_t; X_c = X1[itr].astype(np.float64) - m.mean_s
    n = X_c.shape[0]
    obs = eff_rank(np.linalg.svd(m.B, compute_uv=False))
    print(f"Tier-3 train N={n}, observed eff_rank={obs}", flush=True)

    # Build Age-decile x Gender strata over the training subjects
    mtr = meta.iloc[itr].reset_index(drop=True)
    ad = pd.qcut(mtr["Age"], 10, labels=False, duplicates="drop").astype(int).values
    gv = mtr["Gender"].astype(int).values
    strata = ad * 10 + gv
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]

    rng = np.random.default_rng(20260604); ranks = []
    for p in range(NPERM):
        perm = np.arange(n)
        for g in groups:           # permute T_c rows only within each stratum
            perm[g] = g[rng.permutation(len(g))]
        Bp = _istanuclear(X_c, T_c[perm], lam=LAM, max_iter=600, tol=1e-5)
        er = eff_rank(np.linalg.svd(Bp, compute_uv=False)); ranks.append(er)
        if p < 3 or p == NPERM - 1:
            print(f"  strat-null perm {p}: eff_rank={er}", flush=True)
    ranks = np.array(ranks)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "design": "Permute GM<->FNC within Age-decile x Gender strata (residual covariate "
                  "structure preserved); refit nuclear-norm lambda=0.3; %d perms." % NPERM,
        "observed_eff_rank": obs, "n_strata": len(groups),
        "strat_null_eff_rank_mean": float(ranks.mean()), "strat_null_eff_rank_max": int(ranks.max()),
        "strat_null_values": [int(x) for x in ranks]}, indent=2))
    print("HEADLINE R6: stratified-null eff_rank mean=%.2f max=%d vs observed %d" % (
        ranks.mean(), ranks.max(), obs))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
