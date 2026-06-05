"""
Generate K-fold cross-validation splits for dataset1.

Each fold produces a dataset1_split.json with the same format as the original,
so existing training scripts run unmodified — just point to a different splits_dir.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocess.split import build_split_proof


def main():
    parser = argparse.ArgumentParser(description="Generate K-fold CV splits")
    parser.add_argument("--K", type=int, default=5, help="Number of folds")
    parser.add_argument("--cv_seed", type=int, default=0, help="Random seed for KFold")
    parser.add_argument("--val_frac", type=float, default=0.15,
                        help="Fraction of non-test data to use as validation")
    parser.add_argument("--splits_dir", type=str,
                        default="splits",
                        help="Original splits directory")
    parser.add_argument("--config_template", type=str,
                        default="train/config_baselines.yaml",
                        help="Config template to copy for each fold")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    splits_dir = project_dir / args.splits_dir
    cv_dir = splits_dir / "cv"

    # Load original dataset1 split to get all subject IDs
    orig_split = json.loads((splits_dir / "dataset1_split.json").read_text())
    all_ids = orig_split["train"] + orig_split["val"] + orig_split["test"]
    all_ids = sorted(set(all_ids))  # deduplicate and sort for reproducibility
    n_total = len(all_ids)
    print(f"Total dataset1 subjects: {n_total}")

    all_ids = np.array(all_ids)
    kf = KFold(n_splits=args.K, shuffle=True, random_state=args.cv_seed)

    manifest = {"K": args.K, "cv_seed": args.cv_seed, "n_total": n_total, "folds": []}

    for k, (trainval_idx, test_idx) in enumerate(kf.split(all_ids)):
        fold_dir = cv_dir / f"fold_{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        test_ids = all_ids[test_idx].tolist()
        trainval_ids = all_ids[trainval_idx]

        # Split trainval into train/val
        train_ids, val_ids = train_test_split(
            trainval_ids, test_size=args.val_frac, random_state=args.cv_seed + k
        )
        train_ids = train_ids.tolist()
        val_ids = val_ids.tolist()

        split = {"train": train_ids, "val": val_ids, "test": test_ids}

        # Validate no overlap
        proof = build_split_proof(split)
        assert proof["no_overlap"], f"Fold {k}: overlap detected!"
        assert proof["counts"]["union"] == n_total, \
            f"Fold {k}: union={proof['counts']['union']} != {n_total}"

        # Write dataset1_split.json
        (fold_dir / "dataset1_split.json").write_text(json.dumps(split, indent=2))

        # Symlink dataset2_split.json from original
        ds2_link = fold_dir / "dataset2_split.json"
        if not ds2_link.exists():
            ds2_orig = splits_dir / "dataset2_split.json"
            if ds2_orig.exists():
                os.symlink(ds2_orig.resolve(), ds2_link)

        # Symlink external_test_mapping.json
        ext_link = fold_dir / "external_test_mapping.json"
        if not ext_link.exists():
            ext_orig = splits_dir / "external_test_mapping.json"
            if ext_orig.exists():
                os.symlink(ext_orig.resolve(), ext_link)

        fold_info = {
            "fold": k,
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
        }
        manifest["folds"].append(fold_info)
        print(f"  Fold {k}: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # Write manifest
    (cv_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to {cv_dir / 'manifest.json'}")

    # Verify: no test overlap across folds, and union of all test sets = all subjects
    all_test = []
    for k in range(args.K):
        fold_split = json.loads((cv_dir / f"fold_{k}" / "dataset1_split.json").read_text())
        all_test.extend(fold_split["test"])
    assert len(all_test) == n_total, f"Test union size {len(all_test)} != {n_total}"
    assert len(set(all_test)) == n_total, "Duplicate subjects across test folds!"
    print(f"Verification passed: {n_total} unique subjects across {args.K} test folds")

    # Generate per-fold config YAML files
    import yaml
    config_template = yaml.safe_load((project_dir / args.config_template).read_text())
    config_dir = project_dir / "train"
    for k in range(args.K):
        cfg = json.loads(json.dumps(config_template))  # deep copy
        cfg["paths"]["splits_dir"] = str((cv_dir / f"fold_{k}").resolve())
        out_path = config_dir / f"config_cv_fold_{k}.yaml"
        out_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        print(f"  Config: {out_path}")

    print("\nDone. Ready for CV training.")


if __name__ == "__main__":
    main()
