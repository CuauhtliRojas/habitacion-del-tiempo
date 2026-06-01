from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def tensor_image_to_uint8(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().float().clamp(0.0, 1.0)

    if image.ndim == 3:
        array = image.numpy()
        array = np.transpose(array, (1, 2, 0))
    else:
        raise ValueError("La imagen debe tener forma [C, H, W].")

    return (array * 255.0).astype(np.uint8)


def tensor_mask_to_uint8(mask: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    mask = mask.detach().cpu().float()

    if mask.ndim == 3:
        mask = mask[0]

    array = (mask.numpy() >= threshold).astype(np.uint8) * 255
    return array


def make_overlay(
    *,
    image_uint8: np.ndarray,
    mask_uint8: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    if image_uint8.ndim != 3 or image_uint8.shape[2] != 3:
        raise ValueError("image_uint8 debe tener forma [H, W, 3].")

    if mask_uint8.ndim != 2:
        raise ValueError("mask_uint8 debe tener forma [H, W].")

    overlay = image_uint8.copy().astype(np.float32)
    color = np.zeros_like(overlay)
    color[..., 0] = 255.0

    mask_bool = mask_uint8 > 0
    overlay[mask_bool] = (
        (1.0 - alpha) * overlay[mask_bool] + alpha * color[mask_bool]
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_prediction_sample(
    *,
    output_dir: str | Path,
    stem: str,
    image: torch.Tensor,
    gt_fake: torch.Tensor,
    pred_fake: torch.Tensor,
    gt_authentic: torch.Tensor,
    pred_authentic: torch.Tensor,
    threshold: float = 0.5,
) -> None:
    output_dir = Path(output_dir) / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    image_uint8 = tensor_image_to_uint8(image)
    gt_fake_uint8 = tensor_mask_to_uint8(gt_fake, threshold=0.5)
    pred_fake_uint8 = tensor_mask_to_uint8(pred_fake, threshold=threshold)
    gt_authentic_uint8 = tensor_mask_to_uint8(gt_authentic, threshold=0.5)
    pred_authentic_uint8 = tensor_mask_to_uint8(pred_authentic, threshold=threshold)

    overlay_fake = make_overlay(
        image_uint8=image_uint8,
        mask_uint8=pred_fake_uint8,
        alpha=0.45,
    )

    Image.fromarray(image_uint8).save(output_dir / "image.png")
    Image.fromarray(gt_fake_uint8).save(output_dir / "gt_fake.png")
    Image.fromarray(pred_fake_uint8).save(output_dir / "pred_fake.png")
    Image.fromarray(gt_authentic_uint8).save(output_dir / "gt_authentic.png")
    Image.fromarray(pred_authentic_uint8).save(output_dir / "pred_authentic.png")
    Image.fromarray(overlay_fake).save(output_dir / "overlay_fake.png")


@torch.no_grad()
def save_epoch_samples(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    output_dir: str | Path,
    epoch: int,
    threshold: float,
    max_samples: int = 8,
) -> None:
    model.eval()

    epoch_dir = Path(output_dir) / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    saved = 0

    for batch in loader:
        images = batch["image"].to(device)
        fake_masks = batch["fake_mask"].to(device)
        authentic_masks = batch["authentic_mask"].to(device)
        stems = batch["stem"]

        pred_fake, pred_authentic = model(images)

        batch_size = images.shape[0]

        for index in range(batch_size):
            save_prediction_sample(
                output_dir=epoch_dir,
                stem=str(stems[index]),
                image=images[index],
                gt_fake=fake_masks[index],
                pred_fake=pred_fake[index],
                gt_authentic=authentic_masks[index],
                pred_authentic=pred_authentic[index],
                threshold=threshold,
            )

            saved += 1

            if saved >= max_samples:
                return
