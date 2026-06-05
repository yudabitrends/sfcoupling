#!/usr/bin/env python3
"""
TIV-proxy validation (Reviewer 4): does the total-GM proxy track true TIV?

For a deterministic sample of UK Biobank subjects we compute, from the SPM
tissue segmentations in each subject's anat folder:
  GM  = sum(c1) * voxel_volume_mL      (native gray-matter volume)
  WM  = sum(c2) * voxel_volume_mL
  CSF = sum(c3) * voxel_volume_mL
  TIV = GM + WM + CSF                    (standard total intracranial volume)
  proxy_modulated = sum(wc1) * voxel_volume_mL   (matches the paper's GM feature source)

and report the Pearson correlation between the total-GM proxy and TIV.
A high correlation confirms the proxy captures the global brain-size variance
that TIV is meant to absorb.

Output: results/tiv_proxy_validation.json
"""
from __future__ import annotations
import csv, json, os
from pathlib import Path
import numpy as np
import nibabel as nib

MANIFEST = "/home/users/ybi3/PNAS/ukb_paths_with_age_sex_timeseries.csv"
ATLAS = "/data/qneuromark/Network_templates/NeuroMark3/T1.nii"  # 4D, 100 ROI maps
OUT = Path("/home/users/ybi3/sfcoupling/results/tiv_proxy_validation.json")
N_TARGET = 200          # subjects to sample
STRIDE = 180            # deterministic spread across the 37,775-row manifest

# Atlas-weighted ROI proxy: x_r = sum_v A_r(v) GM(v) / sum_v A_r(v); proxy = sum_r x_r.
_ATLAS = nib.load(ATLAS)
_A = np.asarray(_ATLAS.dataobj, dtype=np.float32).reshape(-1, _ATLAS.shape[-1])  # (V, 100)
_A_DEN = _A.sum(0)                                                                # (100,)
_A_DEN[_A_DEN == 0] = np.nan


def roi_proxy(wc1_path):
    """Paper's total-GM proxy: sum over ROIs of the atlas-weighted mean GM."""
    gm = np.asarray(nib.load(str(wc1_path)).dataobj, dtype=np.float32).reshape(-1)
    roi = (_A.T @ gm) / _A_DEN          # (100,) weighted mean GM per ROI map
    return float(np.nansum(roi))


def vox_ml(img):
    """Voxel volume in mL (cm^3) from the affine."""
    return float(abs(np.linalg.det(img.affine[:3, :3])) / 1000.0)


def tissue_sum(path):
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    return float(data.sum()), vox_ml(img)


def main():
    rows = list(csv.DictReader(open(MANIFEST)))
    sample = rows[::STRIDE][:N_TARGET]
    gm, wm, csf, tiv, proxy_mod, eids = [], [], [], [], [], []
    used = 0
    for r in sample:
        wc1 = r["nii_path_y"]
        d = os.path.dirname(wc1)
        c1, c2, c3 = (os.path.join(d, f"{t}pT1.nii.nii") for t in ("c1", "c2", "c3"))
        if not all(os.path.exists(p) for p in (c1, c2, c3, wc1)):
            continue
        try:
            s1, v = tissue_sum(c1)
            s2, _ = tissue_sum(c2)
            s3, _ = tissue_sum(c3)
            px = roi_proxy(wc1)          # paper's atlas-weighted ROI-sum proxy
        except Exception:
            continue
        gm.append(s1 * v); wm.append(s2 * v); csf.append(s3 * v)
        tiv.append((s1 + s2 + s3) * v); proxy_mod.append(px)
        eids.append(r.get("eid"))
        used += 1
        if used % 25 == 0:
            print(f"  processed {used} subjects...", flush=True)

    gm, wm, csf, tiv, proxy_mod = map(np.array, (gm, wm, csf, tiv, proxy_mod))

    def pearson(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    res = {
        "n_subjects": int(used),
        "stride": STRIDE,
        "r_totalGMproxy_vs_TIV": pearson(proxy_mod, tiv),
        "r_nativeGM_vs_TIV": pearson(gm, tiv),
        "r_totalGMproxy_vs_nativeGM": pearson(proxy_mod, gm),
        "GM_frac_of_TIV_mean": float((gm / tiv).mean()),
        "TIV_mean_mL": float(tiv.mean()), "TIV_sd_mL": float(tiv.std()),
        "GM_mean_mL": float(gm.mean()),
        "note": "Native TIV = sum(c1+c2+c3)*voxel_mL; total-GM proxy = paper's atlas-weighted ROI-sum of modulated GM (Eq. 1).",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
