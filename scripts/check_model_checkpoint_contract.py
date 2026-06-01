from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.checkpoints import load_checkpoint, save_checkpoint
from src.metrics import dual_segmentation_loss, segmentation_metric_row
from src.model import DualSegmentationModel


def main() -> None:
    device = torch.device("cpu")

    print("=== STATIC CHECK: modelo/loss/checkpoint ===")
    print("Project root:", PROJECT_ROOT)
    print("Device usado:", device)

    model = DualSegmentationModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    batch_size = 1
    image_size = 64

    images = torch.rand(batch_size, 3, image_size, image_size, device=device)
    fake_masks = torch.randint(0, 2, (batch_size, 1, image_size, image_size), device=device).float()
    authentic_masks = torch.randint(0, 2, (batch_size, 1, image_size, image_size), device=device).float()

    with torch.no_grad():
        pred_fake, pred_authentic = model(images)

    print("pred_fake shape:", tuple(pred_fake.shape))
    print("pred_authentic shape:", tuple(pred_authentic.shape))
    print("pred_fake min/max:", float(pred_fake.min()), float(pred_fake.max()))
    print("pred_authentic min/max:", float(pred_authentic.min()), float(pred_authentic.max()))

    if pred_fake.shape != fake_masks.shape:
        raise RuntimeError(f"Shape incompatible pred_fake={pred_fake.shape}, target={fake_masks.shape}")

    if pred_authentic.shape != authentic_masks.shape:
        raise RuntimeError(
            f"Shape incompatible pred_authentic={pred_authentic.shape}, target={authentic_masks.shape}"
        )

    if float(pred_fake.min()) < 0.0 or float(pred_fake.max()) > 1.0:
        raise RuntimeError("pred_fake no esta en rango [0, 1].")

    if float(pred_authentic.min()) < 0.0 or float(pred_authentic.max()) > 1.0:
        raise RuntimeError("pred_authentic no esta en rango [0, 1].")

    loss, loss_parts = dual_segmentation_loss(
        pred_fake=pred_fake,
        target_fake=fake_masks,
        pred_authentic=pred_authentic,
        target_authentic=authentic_masks,
        pos_weight=65.0,
    )

    print("loss_total:", float(loss))
    print("loss_parts:", loss_parts)

    metric_row = segmentation_metric_row(
        pred_fake=pred_fake,
        target_fake=fake_masks,
        pred_authentic=pred_authentic,
        target_authentic=authentic_masks,
        threshold=0.5,
    )

    print("metric_keys:", sorted(metric_row.keys()))

    output_dir = PROJECT_ROOT / "outputs" / "static_checks"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint_roundtrip_cpu.pth"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        config={
            "check": "static_cpu_roundtrip",
            "image_size": image_size,
            "batch_size": batch_size,
            "note": "No entrenamiento real; solo validacion de contrato.",
        },
        metrics_history=[
            {
                "epoch": 0,
                "loss_total": float(loss),
                **loss_parts,
                **metric_row,
            }
        ],
        best_val_loss=float(loss),
        best_val_dice=metric_row["dice_fake"],
        scaler=None,
    )

    restored_model = DualSegmentationModel().to(device)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=5e-4)

    checkpoint = load_checkpoint(
        path=checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scaler=None,
        map_location=device,
    )

    print("checkpoint epoch:", checkpoint["epoch"])
    print("checkpoint keys:", sorted(checkpoint.keys()))
    print("checkpoint path:", checkpoint_path)

    print("STATIC CHECK OK: modelo, loss, metricas y checkpoint/resume son compatibles en CPU.")


if __name__ == "__main__":
    main()
