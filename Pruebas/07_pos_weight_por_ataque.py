from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.metrics import dual_segmentation_loss, weighted_bce_from_logits


"""
Prueba testimonial 07.

Objetivo:
    Demostrar por qué el orquestador moderno necesita pos_weight por ataque
    cuando se mezclan faceswap y local_inpainting.

Contexto:
    En codigo Deepshield/train.py existía un único pos_weight fijo. Eso era
    aceptable para un entrenamiento homogéneo, pero no describe bien un dataset
    mixto donde faceswap tiene una región fake grande y local_inpainting una
    región fake mucho más pequeña.

Conclusión esperada:
    Un tensor de pos_weight por muestra permite que cada ataque contribuya a la
    BCE con su propio desbalance, sin cambiar arquitectura ni máscaras.
"""


def main() -> None:
    torch.manual_seed(117)

    logits = torch.zeros((2, 1, 2, 2), dtype=torch.float32)
    target = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 0.0]]],
            [[[1.0, 0.0], [0.0, 0.0]]],
        ],
        dtype=torch.float32,
    )

    pos_weight_faceswap = 2.1
    pos_weight_local_inpainting = 33.4754694047

    pos_weight_by_attack = torch.tensor(
        [pos_weight_faceswap, pos_weight_local_inpainting],
        dtype=torch.float32,
    ).view(-1, 1, 1, 1)

    loss_global_faceswap = weighted_bce_from_logits(
        logits,
        target,
        pos_weight=pos_weight_faceswap,
    )
    loss_global_local = weighted_bce_from_logits(
        logits,
        target,
        pos_weight=pos_weight_local_inpainting,
    )
    loss_by_attack = weighted_bce_from_logits(
        logits,
        target,
        pos_weight=pos_weight_by_attack,
    )

    pred_fake = logits.clone()
    pred_authentic = logits.clone()
    target_fake = target.clone()
    target_authentic = 1.0 - target

    total_loss, parts = dual_segmentation_loss(
        pred_fake=pred_fake,
        target_fake=target_fake,
        pred_authentic=pred_authentic,
        target_authentic=target_authentic,
        pos_weight_fake=pos_weight_by_attack,
        pos_weight_authentic=1.0,
        lambda_bce=1.0,
        lambda_dice=2.0,
        lambda_iou=1.0,
    )

    print("=== PRUEBA 7: POS_WEIGHT POR ATAQUE ===")
    print(f"pos_weight faceswap: {pos_weight_faceswap}")
    print(f"pos_weight local_inpainting: {pos_weight_local_inpainting:.10f}")
    print("")
    print(f"BCE global faceswap: {float(loss_global_faceswap):.6f}")
    print(f"BCE global local_inpainting: {float(loss_global_local):.6f}")
    print(f"BCE por ataque: {float(loss_by_attack):.6f}")
    print("")
    print(f"dual loss total: {float(total_loss):.6f}")
    print(f"dual loss fake BCE: {parts['loss_fake_bce']:.6f}")
    print(f"dual loss fake Dice: {parts['loss_fake_dice']:.6f}")
    print(f"dual loss fake IoU: {parts['loss_fake_iou']:.6f}")
    print("")

    assert loss_global_faceswap < loss_by_attack < loss_global_local
    assert torch.isfinite(total_loss).all()
    print("OK: pos_weight por ataque queda activo y finito.")


if __name__ == "__main__":
    main()
