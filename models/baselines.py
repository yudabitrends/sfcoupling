import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import Ridge

from .metrics import r2_summary
from .mlp import build_mlp


def fit_ridge_grid(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    alphas: List[float],
) -> Tuple[Ridge, Dict]:
    best = None
    best_score = -1e18
    trials = []
    for a in alphas:
        model = Ridge(alpha=float(a), random_state=0)
        model.fit(X_train, Y_train)
        pred = model.predict(X_val)
        score = r2_summary(Y_val, pred)["r2_global"]
        trials.append({"alpha": float(a), "val_r2_global": float(score)})
        if score > best_score:
            best_score = score
            best = model
    assert best is not None
    return best, {"best_alpha": float(best.alpha), "val_best_r2_global": float(best_score), "trials": trials}


class MLPRegressorTorch(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.net = build_mlp(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp_regressor(
    model: MLPRegressorTorch,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int,
    device: torch.device,
) -> Dict:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(Y_train, dtype=torch.float32, device=device)
    xval = torch.tensor(X_val, dtype=torch.float32, device=device)
    yval = torch.tensor(Y_val, dtype=torch.float32, device=device)

    best_state = None
    best_val = float("inf")
    wait = 0
    logs = []
    rng = np.random.default_rng(seed)

    for ep in range(epochs):
        model.train()
        idx = np.arange(len(X_train))
        rng.shuffle(idx)
        batch_losses = []
        for i in range(0, len(idx), batch_size):
            b = idx[i : i + batch_size]
            xb = xtr[b]
            yb = ytr[b]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            batch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_pred = model(xval)
            val_loss = float(loss_fn(val_pred, yval).cpu())
        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        logs.append({"epoch": ep + 1, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_loss": best_val, "logs": logs}

