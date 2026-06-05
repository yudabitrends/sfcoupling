#!/usr/bin/env python3
"""Assemble all figure outputs into one `bigfigure` directory.

This script keeps your existing `figures/` structure untouched and creates:

bigfigure/
  figurexa/
  figurexb/
  ...

Each item is copied as:
- existing panel folders: all files under the corresponding `figurex*` folder.
- single-panel files: saved as `figurex*/panel_a.*`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from string import ascii_lowercase
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "figures"
DEFAULT_DEST = ROOT / "bigfigure"
DEFAULT_PREFIX = "figurex"


def _index_to_figure_name(idx: int, *, prefix: str = DEFAULT_PREFIX) -> str:
    """Return figurexa, figurexb, ..., figurexz, figurexaa, figurexab..."""
    if idx < 26:
        return f"{prefix}{ascii_lowercase[idx]}"
    idx -= 26
    major = idx // 26
    minor = idx % 26
    return f"{prefix}{ascii_lowercase[major]}{ascii_lowercase[minor]}"


PREFERRED_ORDER: Sequence[str] = [
    "fig1",
    "fig2",
    "fig3",
    "fig4",
    "fig5",
    "fig6",
    "fig7",
    "fig10",
    "fig11",
    "fig12",
    "fig13",
    "fig14",
    "fig15",
    "figS1",
    "figS2",
    "figS5",
]

ROOT_FILE_HINTS: Dict[str, int] = {
    "fig_biological_composite": 16,
    "fig_clinical": 17,
    "fig_linearity_test": 18,
    "fig_lowrank_geometric": 19,
    "fig_mode_detail": 20,
    "fig_robustness": 21,
    "fig_clinical_prediction": 22,
    "fig_nonlinear_residual": 23,
    "fig_rank_sensitivity": 24,
    "fig_roi_coupled_fraction": 25,
    "fig_variance_decomposition": 26,
}

VALID_EXTS = {".pdf", ".png"}


def _build_legacy_map(prefix: str) -> Dict[str, str]:
    legacy: Dict[str, str] = {
        name: _index_to_figure_name(i, prefix=prefix)
        for i, name in enumerate(PREFERRED_ORDER)
    }
    for name, idx in ROOT_FILE_HINTS.items():
        legacy[name] = _index_to_figure_name(idx, prefix=prefix)
    return legacy


def _resolve_target(
    name: str,
    used: Dict[str, None],
    counter: int,
    *,
    prefix: str,
) -> tuple[str, int]:
    legacy_map = _build_legacy_map(prefix)
    if name in legacy_map:
        target = legacy_map[name]
        while target in used:
            counter += 1
            target = _index_to_figure_name(counter, prefix=prefix)
    else:
        target = _index_to_figure_name(counter, prefix=prefix)
        while target in used:
            counter += 1
            target = _index_to_figure_name(counter, prefix=prefix)
    return target, max(counter, len(used))


def _iter_source_figures(src_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    for path in src_dir.iterdir():
        if path.name.startswith(".") or path.name == "bigfigure":
            continue
        if path.is_dir() and path.name.startswith("fig"):
            candidates.append(path)
        elif path.is_file() and path.suffix.lower() in VALID_EXTS and path.stem.startswith("fig"):
            candidates.append(path)
    return candidates


def _ordered_sources(src_dir: Path) -> List[Path]:
    all_paths = _iter_source_figures(src_dir)
    order_map = {p.name: p for p in all_paths}
    ordered: List[Path] = []
    consumed = set()
    for name in PREFERRED_ORDER:
        if name in order_map:
            ordered.append(order_map[name])
            consumed.add(name)
    for name in sorted(order_map):
        if name not in consumed:
            ordered.append(order_map[name])
    return ordered


def _copy_file(src: Path, dst: Path, overwrite: bool = True):
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)


def _copy_directory(src_dir: Path, dst_dir: Path, overwrite: bool = True):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_file in sorted(src_dir.iterdir()):
        if src_file.suffix.lower() not in VALID_EXTS:
            continue
        dst = dst_dir / src_file.name
        _copy_file(src_file, dst, overwrite=overwrite)


def _copy_root_file(src_file: Path, dst_dir: Path, overwrite: bool = True):
    for ext in VALID_EXTS:
        if src_file.suffix.lower() == ext:
            _copy_file(src_file, dst_dir / f"panel_a{ext}", overwrite=overwrite)


def build_bundle(
    source: Path,
    dest: Path,
    overwrite: bool = True,
    *,
    prefix: str = DEFAULT_PREFIX,
) -> List[tuple[str, str]]:
    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")
    if dest.exists():
        if overwrite:
            for child in dest.glob("*"):
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        else:
            raise FileExistsError(f"Destination exists: {dest}")

    dest.mkdir(parents=True, exist_ok=True)
    ordered = _ordered_sources(source)
    used: Dict[str, None] = {}
    counter = 0
    mapping: List[tuple[str, str]] = []

    for item in ordered:
        target, counter = _resolve_target(item.name, used, counter, prefix=prefix)
        used[target] = None
        target_dir = dest / target
        target_dir.mkdir(parents=True, exist_ok=True)
        if item.is_dir():
            _copy_directory(item, target_dir, overwrite=overwrite)
        else:
            _copy_root_file(item, target_dir, overwrite=overwrite)
        mapping.append((item.name, target))

        if isinstance(used[target], type(None)):
            counter += 1

    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Directory containing figure outputs.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Output bigfigure directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in destination.")
    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help="Output figure-folder prefix, default=figurex.",
    )
    args = parser.parse_args()

    prefix = args.prefix.strip() or DEFAULT_PREFIX
    mapping = build_bundle(args.source, args.dest, overwrite=args.overwrite, prefix=prefix)
    print(f"[bigfigure] source: {args.source}")
    print(f"[bigfigure] output: {args.dest}")
    print(f"[bigfigure] prefix: {prefix}")
    for legacy, target in mapping:
        print(f"  {legacy:36s} -> {target}")


if __name__ == "__main__":
    main()
