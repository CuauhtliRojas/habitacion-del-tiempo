"""
PRUEBA 2: BCE + Dice + IoU como pérdida configurable.

Objetivo:
Dejar evidencia de cómo se migró la pérdida del código  Deepshield hacia
src/metrics.py.

En el código legacy, la pérdida se construía directamente dentro de train.py:

    BCEWithLogitsLoss + dice_loss + iou_loss

Ese diseño tenía dos problemas:
- estaba fijo dentro del script de entrenamiento;
- no dejaba controlar fácilmente cuánto pesaba cada término.

En src/metrics.py se conserva la idea de combinar BCE, Dice e IoU, pero ahora se
controla con lambdas:

- lambda_bce controla BCE;
- lambda_dice controla Dice;
- lambda_iou permite activar o apagar IoU loss.

Así se puede comparar:
- BCE + Dice;
- BCE + Dice + IoU;
sin cambiar el modelo ni reescribir el entrenamiento.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import torch

from src.metrics import dual_segmentation_loss


def gradient_norm(loss: torch.Tensor, logits: torch.Tensor) -> float:
    """
    Calcula la norma del gradiente que una pérdida produce sobre los logits.
    """
    logits.grad = None
    loss.backward(retain_graph=True)
    return float(logits.grad.detach().norm().cpu())


def main() -> None:
    torch.manual_seed(117)

    """
    Simulamos una máscara pequeña, como suele ocurrir en segmentación de regiones
    manipuladas: hay muchos píxeles de fondo y pocos píxeles positivos.
    """
    target_fake = (torch.rand(1, 1, 64, 64) > 0.93).float()
    target_authentic = 1.0 - target_fake

    pred_fake_base = torch.randn(1, 1, 64, 64)
    pred_authentic_base = torch.randn(1, 1, 64, 64)

    pos_weight = 33.4754694047

    """
    Caso A:
    Variante SRC base. Usa BCE + Dice y deja IoU solo como métrica/reportable.
    """
    pred_fake_bce_dice = pred_fake_base.clone().detach().requires_grad_(True)
    pred_authentic_bce_dice = pred_authentic_base.clone().detach().requires_grad_(True)

    loss_bce_dice, parts_bce_dice = dual_segmentation_loss(
        pred_fake=pred_fake_bce_dice,
        target_fake=target_fake,
        pred_authentic=pred_authentic_bce_dice,
        target_authentic=target_authentic,
        pos_weight=pos_weight,
        lambda_bce=1.0,
        lambda_dice=2.0,
        lambda_iou=0.0,
    )

    grad_bce_dice = gradient_norm(loss_bce_dice, pred_fake_bce_dice)

    """
    Caso B:
    Variante SRC con presión geométrica adicional. Activa IoU loss de forma
    explícita mediante lambda_iou.
    """
    pred_fake_iou = pred_fake_base.clone().detach().requires_grad_(True)
    pred_authentic_iou = pred_authentic_base.clone().detach().requires_grad_(True)

    loss_with_iou, parts_with_iou = dual_segmentation_loss(
        pred_fake=pred_fake_iou,
        target_fake=target_fake,
        pred_authentic=pred_authentic_iou,
        target_authentic=target_authentic,
        pos_weight=pos_weight,
        lambda_bce=1.0,
        lambda_dice=2.0,
        lambda_iou=1.0,
    )

    grad_with_iou = gradient_norm(loss_with_iou, pred_fake_iou)

    """
    La diferencia entre ambos casos muestra el costo añadido por IoU loss.
    """
    added_loss = loss_with_iou.detach() - loss_bce_dice.detach()
    added_grad = grad_with_iou - grad_bce_dice

    print("\n=== PRUEBA 2: BCE + DICE + IOU CONFIGURABLE ===")
    print(f"Píxeles positivos fake: {int(target_fake.sum().item())}")
    print(f"Píxeles totales: {target_fake.numel()}")
    print(f"pos_weight usado: {pos_weight:.6f}")
    print("")

    print("--- CASO A: SRC BASE, BCE + DICE ---")
    print("Configuración: lambda_bce=1.0, lambda_dice=2.0, lambda_iou=0.0")
    print(f"loss_total: {parts_bce_dice['loss_total']:.6f}")
    print(f"loss_fake_bce: {parts_bce_dice['loss_fake_bce']:.6f}")
    print(f"loss_fake_dice: {parts_bce_dice['loss_fake_dice']:.6f}")
    print(f"loss_fake_iou: {parts_bce_dice['loss_fake_iou']:.6f}")
    print(f"gradiente fake: {grad_bce_dice:.6f}")
    print("")

    print("--- CASO B: SRC CON IOU LOSS, BCE + DICE + IOU ---")
    print("Configuración: lambda_bce=1.0, lambda_dice=2.0, lambda_iou=1.0")
    print(f"loss_total: {parts_with_iou['loss_total']:.6f}")
    print(f"loss_fake_bce: {parts_with_iou['loss_fake_bce']:.6f}")
    print(f"loss_fake_dice: {parts_with_iou['loss_fake_dice']:.6f}")
    print(f"loss_fake_iou: {parts_with_iou['loss_fake_iou']:.6f}")
    print(f"gradiente fake: {grad_with_iou:.6f}")
    print("")

    print("--- DIFERENCIA PRODUCIDA POR IOU LOSS ---")
    print(f"loss añadida por lambda_iou=1.0: {float(added_loss.cpu()):.6f}")
    print(f"cambio en norma de gradiente fake: {added_grad:.6f}")
    print("")

    print("=== CONCLUSIÓN ===")
    print("El orquestador SRC no elimina la idea geométrica de Deepshield.")
    print("La vuelve configurable mediante lambda_iou.")
    print("Con lambda_iou=0.0 se entrena BCE + Dice.")
    print("Con lambda_iou=1.0 se añade IoU loss como restricción geométrica.")
    print("Esto permite comparar variantes de entrenamiento sin modificar la red ni")
    print("mezclar la lógica de pérdida dentro del script principal.")


if __name__ == "__main__":
    main()