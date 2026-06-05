import math

import torch
import torch.nn.functional as F


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, kind: str = "mse") -> torch.Tensor:
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    raise ValueError(f"Unsupported reconstruction loss: {kind}")


def smooth_warmup(
    epoch: int,
    total_epochs: int,
    target: float,
    warmup_frac: float = 0.3,
    floor: float = 1e-4,
) -> float:
    """From a tiny positive floor, linearly ramp to target.

    Unlike delayed_warmup, the gradient signal is present (but tiny)
    from epoch 0, which prevents the catastrophic jump that caused
    the epoch-42 collapse observed with delayed_warmup + beta=1.0.
    """
    warm_epochs = max(1, int(total_epochs * warmup_frac))
    if epoch >= warm_epochs:
        return float(target)
    t = float(epoch) / float(warm_epochs)
    return float(floor + (target - floor) * t)


def linear_warmup(epoch: int, total_epochs: int, target: float, frac: float = 0.3) -> float:
    warm_epochs = max(1, int(total_epochs * frac))
    if epoch >= warm_epochs:
        return float(target)
    return float(target) * float(epoch + 1) / float(warm_epochs)


def delayed_warmup(
    epoch: int,
    total_epochs: int,
    target: float,
    delay_frac: float = 0.2,
    warmup_frac: float = 0.2,
) -> float:
    """Delayed linear warmup: zero for first delay_frac epochs, then ramp up."""
    delay_end = int(total_epochs * delay_frac)
    warmup_end = delay_end + int(total_epochs * warmup_frac)
    if epoch < delay_end:
        return 0.0
    if epoch < warmup_end:
        return float(target) * float(epoch - delay_end) / float(max(warmup_end - delay_end, 1))
    return float(target)


def schedule_value(
    epoch: int,
    total_epochs: int,
    target: float,
    schedule: str = "warmup",
    warmup_frac: float = 0.3,
    delay_frac: float = 0.2,
) -> float:
    if schedule == "constant":
        return float(target)
    if schedule == "warmup":
        return linear_warmup(epoch, total_epochs, target, frac=warmup_frac)
    if schedule == "delayed_warmup":
        return delayed_warmup(epoch, total_epochs, target, delay_frac=delay_frac, warmup_frac=warmup_frac)
    if schedule == "smooth_warmup":
        return smooth_warmup(epoch, total_epochs, target, warmup_frac=warmup_frac)
    if schedule == "cosine":
        t = min(max(epoch, 0), max(total_epochs - 1, 1))
        frac = 0.5 * (1.0 - math.cos(math.pi * t / max(total_epochs - 1, 1)))
        return float(target) * float(frac)
    raise ValueError(f"Unsupported schedule: {schedule}")
