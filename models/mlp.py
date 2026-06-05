from typing import List, Optional

import torch
from torch import nn


def build_mlp(
    in_dim: int,
    hidden_dims: List[int],
    out_dim: int,
    dropout: float = 0.0,
    activation: str = "relu",
    final_activation: Optional[str] = None,
) -> nn.Sequential:
    """Build a simple MLP with configurable hidden layers."""
    if in_dim <= 0 or out_dim <= 0:
        raise ValueError(f"Invalid dimensions in_dim={in_dim}, out_dim={out_dim}")

    act_map = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
    }
    if activation not in act_map:
        raise ValueError(f"Unsupported activation: {activation}")
    if final_activation is not None and final_activation not in act_map:
        raise ValueError(f"Unsupported final_activation: {final_activation}")

    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(act_map[activation]())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    if final_activation is not None:
        layers.append(act_map[final_activation]())
    return nn.Sequential(*layers)

