import json
import os
import random
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    if p.suffix.lower() == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported config extension: {p.suffix}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(tmp))
    tmp.replace(path)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    return torch.load(str(path), map_location="cpu")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_expected_files(exp_dir: Path, expected: List[str]) -> None:
    missing = []
    for rel in expected:
        p = exp_dir / rel
        if not p.exists():
            missing.append(rel)
    if missing:
        raise RuntimeError(f"Missing expected artifacts in {exp_dir}: {missing}")


def _index_from_ids(all_ids: List[str], wanted: List[str], name: str) -> np.ndarray:
    mp = {s: i for i, s in enumerate(all_ids)}
    missing = [s for s in wanted if s not in mp]
    if missing:
        raise ValueError(f"{name}: {len(missing)} split IDs not found, first={missing[:5]}")
    return np.array([mp[s] for s in wanted], dtype=np.int64)


def load_training_contracts(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load aligned arrays, subjects, feature maps and split indices.
    Enforces basic no-leakage data contracts.
    """
    base = Path(cfg["paths"]["aligned_features_dir"]).resolve()
    splits_dir = Path(cfg["paths"]["splits_dir"]).resolve()

    X1 = np.load(base / "dataset1_X.npy")
    Y1 = np.load(base / "dataset1_Y.npy")
    X2 = np.load(base / "dataset2_X.npy")
    Y2 = np.load(base / "dataset2_Y.npy")

    s1 = pd.read_csv(base / "meta" / "dataset1_subjects.tsv", sep="\t")
    s2 = pd.read_csv(base / "meta" / "dataset2_subjects.tsv", sep="\t")
    id_col = cfg.get("id_column", "SubjectID")
    ids1 = s1[id_col].astype(str).tolist()
    ids2 = s2[id_col].astype(str).tolist()

    overlap = set(ids1) & set(ids2)
    if overlap:
        raise ValueError(
            f"FATAL: {len(overlap)} subjects overlap between dataset1 and dataset2. "
            f"Re-run preprocessing to fix. First 5: {sorted(overlap)[:5]}"
        )

    if X1.shape[0] != len(ids1) or Y1.shape[0] != len(ids1):
        raise ValueError("dataset1 rows mismatch between arrays and subjects.tsv")
    if X2.shape[0] != len(ids2) or Y2.shape[0] != len(ids2):
        raise ValueError("dataset2 rows mismatch between arrays and subjects.tsv")
    if X1.shape[1] != X2.shape[1] or Y1.shape[1] != Y2.shape[1]:
        raise ValueError("dataset1 and dataset2 feature dimensions mismatch")

    gm_names = [x.strip() for x in (base / "meta" / "feature_maps" / "gm_feature_names.txt").read_text().splitlines() if x.strip()]
    fnc_names = [x.strip() for x in (base / "meta" / "feature_maps" / "fnc_edge_names.txt").read_text().splitlines() if x.strip()]
    if len(gm_names) != X1.shape[1]:
        raise ValueError(f"gm_feature_names count {len(gm_names)} != dx {X1.shape[1]}")
    if len(fnc_names) != Y1.shape[1]:
        raise ValueError(f"fnc_edge_names count {len(fnc_names)} != dy {Y1.shape[1]}")

    split1 = json.loads((splits_dir / "dataset1_split.json").read_text())
    split2 = json.loads((splits_dir / "dataset2_split.json").read_text()) if (splits_dir / "dataset2_split.json").exists() else None

    idx1_train = _index_from_ids(ids1, split1["train"], "dataset1_train")
    idx1_val = _index_from_ids(ids1, split1["val"], "dataset1_val")
    idx1_test = _index_from_ids(ids1, split1["test"], "dataset1_test")
    # Dataset2 is external-only by default to prevent leakage.
    if cfg.get("dataset2_external_use_all", True):
        idx2_external = np.arange(len(ids2), dtype=np.int64)
    else:
        if split2 is None:
            raise ValueError("dataset2 split requested but splits/dataset2_split.json missing")
        idx2_external = _index_from_ids(ids2, split2["test"], "dataset2_external_test")

    return {
        "X1": X1.astype(np.float32),
        "Y1": Y1.astype(np.float32),
        "X2": X2.astype(np.float32),
        "Y2": Y2.astype(np.float32),
        "ids1": ids1,
        "ids2": ids2,
        "subjects1": s1,
        "subjects2": s2,
        "gm_names": gm_names,
        "fnc_names": fnc_names,
        "dx": int(X1.shape[1]),
        "dy": int(Y1.shape[1]),
        "idx1_train": idx1_train,
        "idx1_val": idx1_val,
        "idx1_test": idx1_test,
        "idx2_external": idx2_external,
    }


def append_log_row(csv_path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(csv_path.parent)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)


def build_results_dirs(exp_dir: Path) -> Dict[str, Path]:
    ckpt = exp_dir / "checkpoints"
    figs = exp_dir / "figs"
    ensure_dir(exp_dir)
    ensure_dir(ckpt)
    ensure_dir(figs)
    return {"exp": exp_dir, "ckpt": ckpt, "figs": figs}


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def batched_indices(indices: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    idx = indices.copy()
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield idx[i : i + batch_size]


def get_ddp_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ddp = cfg.get("ddp", {})
    return {
        "enabled": bool(ddp.get("enabled", False)),
        "world_size": int(ddp.get("world_size", 1)),
        "backend": str(ddp.get("backend", "nccl")),
        "find_unused_parameters": bool(ddp.get("find_unused_parameters", False)),
    }


def ddp_setup(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ddp_cfg = get_ddp_config(cfg)
    use_cuda = bool(cfg.get("use_cuda", False)) and torch.cuda.is_available()
    if not ddp_cfg["enabled"]:
        device = torch.device("cuda" if use_cuda else "cpu")
        return {
            "is_distributed": False,
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "is_main": True,
            "device": device,
            "find_unused_parameters": ddp_cfg["find_unused_parameters"],
        }

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", str(ddp_cfg["world_size"])))
    backend = ddp_cfg["backend"]
    if backend == "nccl" and not use_cuda:
        backend = "gloo"

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return {
        "is_distributed": True,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "is_main": rank == 0,
        "device": device,
        "find_unused_parameters": ddp_cfg["find_unused_parameters"],
    }


def ddp_cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def dist_avg_scalar(value: float, device: torch.device, is_distributed: bool) -> float:
    if not is_distributed:
        return float(value)
    t = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t = t / dist.get_world_size()
    return float(t.item())

