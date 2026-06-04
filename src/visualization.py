from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

"""
Este archivo se encarga de convertir resultados del modelo en imágenes fáciles
de revisar.

Durante el entrenamiento, las métricas numéricas dicen si el modelo mejora, pero
las imágenes permiten ver errores que una tabla no siempre explica: si la máscara
se infló, si quedó muy pequeña, si se fue fuera del rostro o si detectó bordes
razonables.

Aquí también se respeta una regla importante del proyecto: el modelo entrega
logits, y solo para visualizar se convierten a probabilidades con sigmoid.

Referencia:
https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
"""

def tensor_image_to_uint8(image: torch.Tensor) -> np.ndarray:
    """
    Convierte una imagen tensor a formato uint8.

    El entrenamiento trabaja con tensores en rango 0 a 1. Para guardar una imagen
    como PNG necesitamos convertirla a valores normales de imagen: 0 a 255.

    También se mueve a CPU y se separa del grafo de PyTorch porque aquí solo se
    va a guardar una imagen, no a seguir entrenando.
    """
    image = image.detach().cpu().float().clamp(0.0, 1.0)

    if image.ndim == 3:
        array = image.numpy()
        array = np.transpose(array, (1, 2, 0))
    else:
        raise ValueError("La imagen debe tener forma [C, H, W].")

    return (array * 255.0).astype(np.uint8)


def tensor_mask_to_uint8(mask: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """
    Convierte una máscara tensor a imagen blanco/negro.

    La máscara se binariza con un umbral:
    - blanco significa región detectada;
    - negro significa fondo.

    Esto facilita revisar visualmente si el modelo pinta de más, pinta de menos
    o deja vacía la máscara.
    """
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
    """
    Crea una superposición visual entre imagen y máscara.

    La zona detectada se pinta en rojo semitransparente sobre la imagen original.
    Esto ayuda a revisar rápido si la predicción cae sobre la zona manipulada o
    si se extendió a regiones incorrectas.
    """
    if image_uint8.ndim != 3 or image_uint8.shape[2] != 3:
        raise ValueError("image_uint8 debe tener forma [H, W, 3].")

    if mask_uint8.ndim != 2:
        raise ValueError("mask_uint8 debe tener forma [H, W].")

    overlay = image_uint8.copy().astype(np.float32)
    color = np.zeros_like(overlay)

    """
    Se usa rojo porque visualmente destaca sobre la mayoría de tonos de piel y
    fondos. No cambia la predicción; solo es una ayuda para inspección humana.
    """
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
    """
    Guarda una muestra completa de evaluación visual.

    Por cada imagen se guardan seis archivos:
    - image.png: imagen original.
    - gt_fake.png: máscara fake real.
    - pred_fake.png: máscara fake predicha.
    - gt_authentic.png: máscara auténtica real.
    - pred_authentic.png: máscara auténtica predicha.
    - overlay_fake.png: predicción fake sobre la imagen original.

    Esto reemplaza el flujo más limitado del código antiguo, donde se guardaban
    máscaras predichas, pero no quedaba una comparación visual tan directa contra
    el ground truth.
    """
    output_dir = Path(output_dir) / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    """
    La imagen se guarda como imagen normal. Las máscaras reales usan umbral 0.5
    porque ya deberían venir binarias, pero se fuerza el formato blanco/negro
    para evitar grises accidentales.
    """
    image_uint8 = tensor_image_to_uint8(image)
    gt_fake_uint8 = tensor_mask_to_uint8(gt_fake, threshold=0.5)
    pred_fake_uint8 = tensor_mask_to_uint8(pred_fake, threshold=threshold)
    gt_authentic_uint8 = tensor_mask_to_uint8(gt_authentic, threshold=0.5)
    pred_authentic_uint8 = tensor_mask_to_uint8(pred_authentic, threshold=threshold)

    """
    La superposición fake es la imagen más útil para revisión rápida: permite ver
    la predicción directamente encima del rostro.
    """
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
    """
    Guarda un pequeño conjunto de muestras visuales al final de una época.

    Se usa `torch.no_grad()` porque aquí no estamos entrenando. Solo queremos
    revisar cómo se ven las predicciones del modelo en validación.

    Esto ayuda a elegir entre checkpoints. Por ejemplo, una época puede tener
    buen Dice, pero visualmente sobrepintar; las muestras permiten detectar eso.
    """
    model.eval()

    epoch_dir = Path(output_dir) / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    saved = 0

    for batch in loader:
        images = batch["image"].to(device)
        fake_masks = batch["fake_mask"].to(device)
        authentic_masks = batch["authentic_mask"].to(device)
        stems = batch["stem"]

        pred_fake_logits, pred_authentic_logits = model(images)

        """
        El modelo devuelve logits. Para guardar máscaras visibles necesitamos
        probabilidades entre 0 y 1, por eso aquí sí se aplica sigmoid.

        Esta conversión vive aquí, no dentro del modelo, para mantener estable
        la pérdida BCE con logits durante el entrenamiento.
        """
        pred_fake = torch.sigmoid(pred_fake_logits.float())
        pred_authentic = torch.sigmoid(pred_authentic_logits.float())

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

            """
            No se guardan todas las imágenes de validación porque serían miles
            de archivos por época. Con unas cuantas muestras basta para revisar
            visualmente si el modelo va bien o mal.
            """
            if saved >= max_samples:
                return
