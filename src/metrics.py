from __future__ import annotations

import torch
import torch.nn.functional as F


"""
Este archivo concentra las métricas y pérdidas del entrenamiento.

Estas ideas estaban mezcladas dentro del codigo Deepshield\train.py.
Aquí se separan para que sea más fácil entender qué mide el
modelo, qué se usa para entrenar y qué solo se usa para evaluar.

Idea general:
- El modelo devuelve logits.
- La pérdida BCE trabaja con logits.
- Las métricas convierten esos logits a probabilidades con sigmoid.
- Dice se usa como parte de la pérdida porque ayuda a cuidar la forma de la máscara.
- IoU se conserva como métrica porque es útil para evaluar, pero no se suma a la
  pérdida principal para no duplicar la presión geométrica.

Referencias:
https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
"""

EPS = 1e-7


def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """
    Convierte logits a probabilidades.

    El modelo entrega valores crudos, llamados logits. Para calcular métricas
    o guardar máscaras visuales necesitamos valores entre 0 y 1, por eso aquí
    se aplica sigmoid.

    Se dejó fuera del modelo para que la pérdida BCEWithLogits pueda trabajar
    de forma más estable.
    """
    return torch.sigmoid(logits.float())


def threshold_mask(pred: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Convierte una probabilidad en máscara binaria.

    Si el valor es mayor o igual al umbral, se considera región detectada.
    Si está por debajo, se considera fondo.

    - 1 = el modelo cree que ahí está la región manipulada.
    - 0 = el modelo cree que ahí no está.
    """
    return (pred >= threshold).float()


def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calcula Dice Score.

    Dice mide qué tanto se empalman la máscara predicha y la máscara real.
    Es una métrica muy útil en segmentación 
    
    Dice = (2 · Verdaderos Positivos) / (2 · Verdaderos Positivos + Falsos Positivos + Falsos Negativos)

    - 1.0 = empalme perfecto.
    - 0.0 = no hay empalme útil.
    """
    pred_bin = threshold_mask(pred, threshold)
    target = target.float()

    intersection = (pred_bin * target).sum()
    denominator = pred_bin.sum() + target.sum()

    return float(((2.0 * intersection + EPS) / (denominator + EPS)).detach().cpu())


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calcula IoU, también llamado Intersection over Union.

    IoU compara la intersección contra la unión entre predicción y ground truth.
    Es más estricto que Dice, por eso normalmente da un valor menor.

    IoU = Verdaderos Positivos / (Verdaderos Positivos + Falsos Positivos + Falsos Negativos)

    - 1.0 = empalme perfecto.
    - 0.0 = no hay empalme útil.
    """
    pred_bin = threshold_mask(pred, threshold)
    target = target.float()

    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection

    return float(((intersection + EPS) / (union + EPS)).detach().cpu())


def precision_score_mask(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calcula precisión sobre la máscara.

    La precisión responde: de todo lo que el modelo pintó como fake, cuánto era
    realmente fake.

    Si la precisión baja, normalmente significa que el modelo está pintando de
    más y genera muchos falsos positivos.
    """
    pred_bin = threshold_mask(pred, threshold)
    target = target.float()

    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1.0 - target)).sum()

    return float(((tp + EPS) / (tp + fp + EPS)).detach().cpu())


def recall_score_mask(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calcula recall sobre la máscara.

    El recall responde: de toda la región fake real, cuánto logró encontrar el
    modelo.

    Si el recall baja, significa que el modelo está dejando zonas manipuladas
    sin detectar.
    """
    pred_bin = threshold_mask(pred, threshold)
    target = target.float()

    tp = (pred_bin * target).sum()
    fn = ((1.0 - pred_bin) * target).sum()

    return float(((tp + EPS) / (tp + fn + EPS)).detach().cpu())


def f1_score_mask(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calcula F1 Score.

    F1 combina precisión y recall en una sola métrica. Es útil cuando queremos
    saber si el modelo está equilibrado: que no pinte de más, pero que tampoco
    deje sin detectar la región manipulada.
    """
    precision = precision_score_mask(pred, target, threshold)
    recall = recall_score_mask(pred, target, threshold)

    return (2.0 * precision * recall) / (precision + recall + EPS)


def accuracy_score_mask(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calcula accuracy pixel a pixel.

    Esta métrica se reporta, pero no debe ser la principal en segmentación.
    Si hay mucho fondo negro, un modelo puede tener accuracy alta aunque no
    segmente bien la región fake.
    """
    pred_bin = threshold_mask(pred, threshold)
    target = target.float()

    correct = (pred_bin == target).float().mean()

    return float(correct.detach().cpu())


def segmentation_metric_row(
    *,
    pred_fake: torch.Tensor,
    target_fake: torch.Tensor,
    pred_authentic: torch.Tensor,
    target_authentic: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    """
    Calcula una fila de métricas para una batch.

    Aquí se convierten los logits a probabilidades porque las métricas se
    interpretan mejor en escala 0 a 1. Esto mantiene separado el contrato:

    - entrenamiento: logits;
    - evaluación: probabilidades;
    - reporte final: valores numéricos comparables.
    """
    pred_fake_prob = logits_to_probabilities(pred_fake)
    pred_authentic_prob = logits_to_probabilities(pred_authentic)

    return {
        "dice_fake": dice_score(pred_fake_prob, target_fake, threshold),
        "iou_fake": iou_score(pred_fake_prob, target_fake, threshold),
        "precision_fake": precision_score_mask(pred_fake_prob, target_fake, threshold),
        "recall_fake": recall_score_mask(pred_fake_prob, target_fake, threshold),
        "f1_fake": f1_score_mask(pred_fake_prob, target_fake, threshold),
        "accuracy_fake": accuracy_score_mask(pred_fake_prob, target_fake, threshold),
        "dice_authentic": dice_score(pred_authentic_prob, target_authentic, threshold),
        "iou_authentic": iou_score(pred_authentic_prob, target_authentic, threshold),
    }


def weighted_bce_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """
    Calcula BCE estable usando logits.

    Esta es una de las correcciones más importantes frente al código antiguo.
    En vez de aplicar sigmoide dentro del modelo y después calcular BCE, dejamos
    que PyTorch haga el cálculo estable con binary_cross_entropy_with_logits.

    `pos_weight` sirve para darle más peso a los píxeles positivos cuando la
    máscara fake ocupa poca área. Por eso conviene calcularlo con la proporción 
    real de píxeles negativos/positivos del dataset.

    Referencia:
    https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
    """
    logits = logits.float()
    target = target.float()
    pos_weight_tensor = torch.as_tensor(
        pos_weight,
        dtype=logits.dtype,
        device=logits.device,
    )

    return F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight_tensor,
    )


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Calcula la pérdida Dice suave.

    Se llama "suave" porque no convierte la predicción en blanco/negro antes de
    calcular la pérdida. Usa probabilidades continuas para que la red todavía pueda
    aprender durante el retroceso del error.

    Forma parte de la pérdida total junto con BCE.
    """
    pred = logits_to_probabilities(logits)
    target = target.float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    denominator = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    return 1.0 - dice.mean()


def dual_segmentation_loss(
    *,
    pred_fake: torch.Tensor,
    target_fake: torch.Tensor,
    pred_authentic: torch.Tensor,
    target_authentic: torch.Tensor,
    pos_weight: float,
    lambda_bce: float = 1.0,
    lambda_dice: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Calcula la pérdida total de las dos salidas del modelo.

    La primera salida aprende la máscara manipulada.
    La segunda salida aprende la máscara auténtica/original.

    La pérdida actual combina:
    - BCE con logits: ayuda a corregir errores pixel por pixel y permite usar
    pos_weight cuando hay desbalance.
    - Dice suave: ayuda a que la forma completa de la máscara se parezca más a la
    máscara real.

    Esta función es la pérdida principal.
    """
    loss_fake_bce = weighted_bce_from_logits(
        pred_fake,
        target_fake,
        pos_weight=pos_weight,
    )
    loss_fake_dice = soft_dice_loss(pred_fake, target_fake)
    loss_fake = (lambda_bce * loss_fake_bce) + (lambda_dice * loss_fake_dice)

    loss_authentic_bce = weighted_bce_from_logits(
        pred_authentic,
        target_authentic,
        pos_weight=pos_weight,
    )
    loss_authentic_dice = soft_dice_loss(pred_authentic, target_authentic)
    loss_authentic = (lambda_bce * loss_authentic_bce) + (lambda_dice * loss_authentic_dice)

    total_loss = loss_fake + loss_authentic

    return total_loss, {
        "loss_total": float(total_loss.detach().cpu()),
        "loss_fake": float(loss_fake.detach().cpu()),
        "loss_fake_bce": float(loss_fake_bce.detach().cpu()),
        "loss_fake_dice": float(loss_fake_dice.detach().cpu()),
        "loss_authentic": float(loss_authentic.detach().cpu()),
        "loss_authentic_bce": float(loss_authentic_bce.detach().cpu()),
        "loss_authentic_dice": float(loss_authentic_dice.detach().cpu()),
    }