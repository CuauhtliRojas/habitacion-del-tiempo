from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

"""
Este archivo guarda y carga checkpoints del entrenamiento.

En el código antiguo se guardaba principalmente un archivo de pesos del modelo.
Eso servía para probar, pero no era suficiente para continuar entrenamientos,
comparar épocas o explicar exactamente de dónde salió un resultado.

Aquí el checkpoint guarda más información:
- pesos del modelo;
- estado del optimizador;
- configuración usada;
- historial de métricas;
- mejor pérdida de validación;
- mejor Dice de validación;
- estado del scaler si se usó mixed precision.

Así podemos reanudar un entrenamiento real o cargar solo pesos para hacer
fine-tuning.
"""

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
    """
    Guarda un checkpoint completo.

    No se guarda solo el modelo. También se guarda el estado necesario para
    saber cómo iba el entrenamiento y, si hace falta, continuarlo después.

    Esto permite tres usos importantes:
    - last.pth: recuperar la última época.
    - epoch_XXX.pth: comparar una época específica.
    - best_val_dice.pth o best_val_loss.pth: seleccionar el mejor modelo según
      el criterio de validación.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    """
    El payload es la "foto completa" del entrenamiento en esa época.

    model_state guarda lo aprendido por la red.
    optimizer_state guarda cómo venía actualizando los pesos.
    config guarda los hiperparámetros.
    metrics_history guarda la historia numérica del experimento.
    """
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
        """
        Si se entrenó con mixed precision, también se guarda el scaler.

        Esto ayuda a reanudar el entrenamiento sin perder el estado numérico
        que PyTorch estaba usando para estabilizar gradientes.
        """
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
    """
    Carga un checkpoint previamente guardado.

    Esta función se usa principalmente para resume, es decir, continuar el mismo
    entrenamiento con modelo, optimizador y scaler.

    Para fine-tuning normalmente no queremos restaurar el optimizador completo;
    en ese caso se cargan solo los pesos desde train_orchestrator.py usando
    `weights`.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {path}")

    checkpoint = torch.load(path, map_location=map_location)

    """
    Siempre se restauran los pesos del modelo. Sin esto, no hay continuidad real
    del aprendizaje.
    """
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        """
        Restaurar el optimizador sirve para resume.

        El optimizador no solo guarda learning rate; también guarda acumulados
        internos de Adam. Por eso, si continuamos el mismo entrenamiento, conviene
        cargarlo. Para fine-tuning normalmente se prefiere empezar con optimizer
        nuevo.
        """
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scaler is not None and "scaler_state" in checkpoint:
        """
        Si el entrenamiento usó mixed precision, el scaler también forma parte
        del estado de entrenamiento. Restaurarlo ayuda a continuar sin saltos
        raros en la escala de gradientes.
        """
        scaler.load_state_dict(checkpoint["scaler_state"])

    return checkpoint
