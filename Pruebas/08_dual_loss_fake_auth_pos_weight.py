from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.metrics import dual_segmentation_loss, weighted_bce_from_logits


def main() -> None:
    pred_fake = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
    pred_authentic = torch.zeros((2, 1, 4, 4), dtype=torch.float32)

    target_fake = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
    target_fake[0, :, :3, :3] = 1.0
    target_fake[1, :, 1:2, 1:2] = 1.0
    target_authentic = 1.0 - target_fake

    pos_weight_fake = torch.tensor(
        [2.1, 33.4754694047],
        dtype=torch.float32,
    ).view(-1, 1, 1, 1)

    pos_weight_authentic = 1.0

    expected_fake_bce = weighted_bce_from_logits(
        pred_fake,
        target_fake,
        pos_weight=pos_weight_fake,
    )

    expected_authentic_bce = weighted_bce_from_logits(
        pred_authentic,
        target_authentic,
        pos_weight=pos_weight_authentic,
    )

    total_loss, parts = dual_segmentation_loss(
        pred_fake=pred_fake,
        target_fake=target_fake,
        pred_authentic=pred_authentic,
        target_authentic=target_authentic,
        pos_weight_fake=pos_weight_fake,
        pos_weight_authentic=pos_weight_authentic,
        lambda_bce=1.0,
        lambda_dice=2.0,
        lambda_iou=1.0,
    )

    print("=== PRUEBA 8: BCE FAKE/AUTHENTIC SEPARADA ===")
    print(f"fake BCE esperado: {float(expected_fake_bce):.6f}")
    print(f"fake BCE dual: {parts['loss_fake_bce']:.6f}")
    print(f"authentic BCE esperado: {float(expected_authentic_bce):.6f}")
    print(f"authentic BCE dual: {parts['loss_authentic_bce']:.6f}")
    print(f"loss total: {float(total_loss):.6f}")

    assert torch.isfinite(total_loss).all()
    assert abs(parts["loss_fake_bce"] - float(expected_fake_bce)) < 1e-6
    assert abs(parts["loss_authentic_bce"] - float(expected_authentic_bce)) < 1e-6

    print("OK: la perdida dual separa pos_weight fake y authentic.")


if __name__ == "__main__":
    main()
