from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    *,
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    metrics_history: list[dict[str, Any]],
    best_val_loss: float | None,
    best_val_dice: float | None,
    scaler: Any | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "metrics_history": metrics_history,
        "best_val_loss": best_val_loss,
        "best_val_dice": best_val_dice,
    }

    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()

    torch.save(payload, path)


def load_checkpoint(
    *,
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {path}")

    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scaler is not None and "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])

    return checkpoint
