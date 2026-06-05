#!/usr/bin/env python3
"""
Definitive TIV-proxy validation against the measured UK Biobank intracranial
volume (FreeSurfer eTIV, field 26521), now that the phenotype basket is available.

For a deterministic sample of UKB subjects we compute, from the SPM tissue maps:
  proxy   = paper's total-GM proxy = sum_r [ atlas-weighted mean GM in ROI r ]  (Eq. 1, modulated wc1)
  GMvol   = sum(c1) * voxel_mL        (native total gray-matter volume)
  TIVseg  = sum(c1+c2+c3) * voxel_mL  (segmentation-based TIV)
and look up, by eid, from ukb674036_2025.csv:
  ICV     = 26521-2.0  (FreeSurfer EstimatedTotalIntraCranialVolume, mm^3)
  GMukb   = 25008-2.0  (UKB grey-matter volume)
  hscale  = 25000-2.0  (T1->MNI volumetric scaling; inverse ~ head size)
Then report Pearson correlations. This tells us, with UKB's own ICV, what the
proxy actually measures.

Output: results/tiv_icv_validation.json
"""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path
import numpy as np
import nibabel as nib

csv.field_size_limit(sys.maxsize)
# 0-based column indices in ukb674036_2025.csv (verified against header)
IDX_EID, IDX_HSCALE, IDX_GMUKB, IDX_ICV = 0, 17707, 17723, 20471

MANIFEST = "/home/users/ybi3/PNAS/ukb_paths_with_age_sex_timeseries.csv"
PHENO = "/data/qneuromark/Data/UKBiobank/Data_info/Basket/B4033904/ukb674036_2025.csv"
ATLAS = "/data/qneuromark/Network_templates/NeuroMark3/T1.nii"
OUT = Path("/home/users/ybi3/sfcoupling/results/tiv_icv_validation.json")
N_TARGET, STRIDE = 250, 140

# --- atlas-weighted ROI proxy (paper Eq. 1) ---
_AT = nib.load(ATLAS)
_A = np.asarray(_AT.dataobj, dtype=np.float32).reshape(-1, _AT.shape[-1])
_DEN = _A.sum(0); _DEN[_DEN == 0] = np.nan

def vox_ml(img): return float(abs(np.linalg.det(img.affine[:3, :3])) / 1000.0)
def vsum(p):
    img = nib.load(str(p)); return float(np.asarray(img.dataobj, np.float32).sum()), vox_ml(img)
def roi_proxy(wc1):
    gm = np.asarray(nib.load(str(wc1)).dataobj, np.float32).reshape(-1)
    return float(np.nansum((_A.T @ gm) / _DEN))

def main():
    print("streaming UKB phenotype (eid, 25000, 25008, 26521) by index...", flush=True)
    def num(x):
        try: return float(x)
        except (TypeError, ValueError): return np.nan
    icv_map = {}
    with open(PHENO, newline="") as f:
        rdr = csv.reader(f)
        hdr = next(rdr)
        assert hdr[IDX_ICV] == "26521-2.0" and hdr[IDX_EID] == "eid", "column index mismatch"
        n = 0
        for row in rdr:
            icv = num(row[IDX_ICV])
            if not np.isfinite(icv):
                continue
            icv_map[row[IDX_EID]] = {"hscale": num(row[IDX_HSCALE]),
                                      "gmukb": num(row[IDX_GMUKB]), "icv": icv}
            n += 1
    print("subjects with ICV:", n, flush=True)

    rows = list(csv.DictReader(open(MANIFEST)))
    sample = rows[::STRIDE][:N_TARGET]
    proxy, gmvol, tivseg, icv, gmukb, hscale = [], [], [], [], [], []
    used = 0
    for r in sample:
        eid = str(r.get("eid")); wc1 = r["nii_path_y"]; d = os.path.dirname(wc1)
        c1, c2, c3 = (os.path.join(d, f"{t}pT1.nii.nii") for t in ("c1", "c2", "c3"))
        ph_row = icv_map.get(eid)
        if ph_row is None:
            continue
        if not all(os.path.exists(p) for p in (c1, c2, c3, wc1)):
            continue
        try:
            s1, v = vsum(c1); s2, _ = vsum(c2); s3, _ = vsum(c3); px = roi_proxy(wc1)
        except Exception:
            continue
        proxy.append(px); gmvol.append(s1 * v); tivseg.append((s1 + s2 + s3) * v)
        icv.append(ph_row["icv"]); gmukb.append(ph_row["gmukb"]); hscale.append(ph_row["hscale"])
        used += 1
        if used % 25 == 0: print(f"  matched {used} subjects...", flush=True)

    A = {k: np.array(v, float) for k, v in
         dict(proxy=proxy, gmvol=gmvol, tivseg=tivseg, icv=icv, gmukb=gmukb, hscale=hscale).items()}
    def r(a, b):
        m = np.isfinite(A[a]) & np.isfinite(A[b])
        return float(np.corrcoef(A[a][m], A[b][m])[0, 1]) if m.sum() > 3 else None
    res = {
        "n_matched": int(used),
        "r_proxy_vs_ICV": r("proxy", "icv"),
        "r_nativeGMvol_vs_ICV": r("gmvol", "icv"),
        "r_segTIV_vs_ICV": r("tivseg", "icv"),
        "r_proxy_vs_UKBgm": r("proxy", "gmukb"),
        "r_proxy_vs_headscale": r("proxy", "hscale"),
        "ICV_median_mm3": float(np.nanmedian(A["icv"])),
        "note": "proxy = paper Eq.1 atlas-weighted ROI-mean-sum of modulated GM; ICV = UKB field 26521 (FreeSurfer eTIV).",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
