#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.ae import YAutoencoder
from models.metrics import r2_summary
from models.utils import (
    append_log_row,
    build_results_dirs,
    ddp_cleanup,
    ddp_setup,
    dist_avg_scalar,
    load_config,
    load_training_contracts,
    save_checkpoint,
    save_json,
    set_seed,
    to_tensor,
)


def run(cfg: Dict, exp_name: str) -> None:
    ddp_state = ddp_setup(cfg)
    is_dist = ddp_state["is_distributed"]
    rank = ddp_state["rank"]
    world_size = ddp_state["world_size"]
    is_main = ddp_state["is_main"]
    device = ddp_state["device"]

    try:
        seed = int(cfg.get("seed", 42))
        set_seed(seed + rank)
        data = load_training_contracts(cfg)
        ytr = data["Y1"][data["idx1_train"]]
        yva = data["Y1"][data["idx1_val"]]
        yte = data["Y1"][data["idx1_test"]]
        yext = data["Y2"][data["idx2_external"]]

        if np.max(data["idx1_train"]) >= data["Y1"].shape[0]:
            raise RuntimeError("dataset1_train indices out of bounds for dataset1_Y.")

        result_root = Path(cfg.get("results_dir", "results")) / exp_name
        dirs = build_results_dirs(result_root)
        if is_main:
            save_json(dirs["exp"] / "config.json", cfg)

        mcfg = cfg.get("ae", {})
        model = YAutoencoder(
            dy=data["dy"],
            latent_dim=int(mcfg.get("latent_dim", 64)),
            hidden=int(mcfg.get("hidden", 256)),
            depth=int(mcfg.get("depth", 2)),
            dropout=float(mcfg.get("dropout", 0.1)),
        ).to(device)
        if is_dist:
            model = DDP(
                model,
                device_ids=[ddp_state["local_rank"]] if device.type == "cuda" else None,
                find_unused_parameters=ddp_state["find_unused_parameters"],
            )

        train_cfg = cfg.get("train", {})
        epochs = int(train_cfg.get("epochs", 50))
        batch_size = int(train_cfg.get("batch_size", 128))
        lr = float(train_cfg.get("lr", 1e-3))
        weight_decay = float(train_cfg.get("weight_decay", 1e-5))
        patience = int(train_cfg.get("patience", 10))
        grad_clip = float(train_cfg.get("grad_clip", 5.0))

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        yva_t = to_tensor(yva, device)
        best_val = float("inf")
        wait = 0
        best_ckpt = dirs["ckpt"] / "best_y_ae.pt"

        for ep in range(epochs):
            idx = np.arange(len(ytr))
            rng = np.random.default_rng(seed + ep)
            rng.shuffle(idx)
            idx = idx[rank::world_size]
            model.train()
            losses = []
            for i in range(0, len(idx), batch_size):
                b = idx[i : i + batch_size]
                yb = to_tensor(ytr[b], device)
                opt.zero_grad()
                y_hat, _ = model(yb)
                loss = F.mse_loss(y_hat, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                losses.append(float(loss.detach().cpu()))

            train_loss = float(np.mean(losses)) if losses else 0.0
            train_loss = dist_avg_scalar(train_loss, device, is_dist)

            model.eval()
            with torch.no_grad():
                yv_hat, _ = model(yva_t)
                val_loss_local = float(F.mse_loss(yv_hat, yva_t).cpu())
            val_loss = dist_avg_scalar(val_loss_local, device, is_dist)

            if is_main:
                append_log_row(
                    dirs["exp"] / "logs.csv",
                    {
                        "epoch": ep + 1,
                        "model": "y_autoencoder",
                        "train_split_role": "dataset1_train",
                        "val_split_role": "dataset1_val",
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                )

            if val_loss < best_val:
                best_val = val_loss
                wait = 0
                if is_main:
                    raw = model.module if isinstance(model, DDP) else model
                    save_checkpoint(
                        best_ckpt,
                        {
                            "model_state": raw.state_dict(),
                            "cfg": cfg,
                            "dy": data["dy"],
                            "best_epoch": ep + 1,
                            "best_val_loss": best_val,
                        },
                    )
            else:
                wait += 1
                if wait >= patience:
                    break

        if is_dist:
            torch.distributed.barrier()
        if not best_ckpt.exists():
            raise RuntimeError("Best checkpoint was not written by rank0.")

        ckpt = torch.load(best_ckpt, map_location=device)
        raw = model.module if isinstance(model, DDP) else model
        raw.load_state_dict(ckpt["model_state"])
        raw.eval()
        with torch.no_grad():
            yte_hat = raw(to_tensor(yte, device))[0].cpu().numpy()
            yext_hat = raw(to_tensor(yext, device))[0].cpu().numpy()
        out1 = {"dataset": "dataset1_test", "model": "y_autoencoder", "metrics": r2_summary(yte, yte_hat)}
        out2 = {"dataset": "dataset2_external", "model": "y_autoencoder", "metrics": r2_summary(yext, yext_hat)}
        if is_main:
            save_json(dirs["exp"] / "metrics_dataset1.json", out1)
            save_json(dirs["exp"] / "metrics_dataset2.json", out2)
            print(json.dumps({"exp_dir": str(dirs["exp"]), "dataset1_test": out1, "dataset2_external": out2}, indent=2))
    finally:
        ddp_cleanup()


def main():
    parser = argparse.ArgumentParser(description="Train Y-only autoencoder sanity model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg, args.exp_name)


if __name__ == "__main__":
    main()
