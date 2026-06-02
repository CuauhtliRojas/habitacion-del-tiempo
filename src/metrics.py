from __future__ import annotations

import torch
import torch.nn.functional as F


EPS = 1e-7


def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.float())


def threshold_mask(pred: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (pred >= threshold).float()


def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = threshold_mask(pred, threshold)
    target = target.float()

    intersection = (pred_bin * target).sum()
    denominator = pred_bin.sum() + target.sum()

    return float(((2.0 * intersection + EPS) / (denominator + EPS)).detach().cpu())


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = threshold_mask(pred, threshold)
    target = target.float()

    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection

    return float(((intersection + EPS) / (union + EPS)).detach().cpu())


def precision_score_mask(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = threshold_mask(pred, threshold)
    target = target.float()

    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1.0 - target)).sum()

    return float(((tp + EPS) / (tp + fp + EPS)).detach().cpu())


def recall_score_mask(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = threshold_mask(pred, threshold)
    target = target.float()

    tp = (pred_bin * target).sum()
    fn = ((1.0 - pred_bin) * target).sum()

    return float(((tp + EPS) / (tp + fn + EPS)).detach().cpu())


def f1_score_mask(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    precision = precision_score_mask(pred, target, threshold)
    recall = recall_score_mask(pred, target, threshold)

    return (2.0 * precision * recall) / (precision + recall + EPS)


def accuracy_score_mask(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
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
    pred = logits_to_probabilities(logits)
    target = target.float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    denominator = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    return 1.0 - dice.mean()


def soft_iou_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    pred = logits_to_probabilities(logits)
    target = target.float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + smooth) / (union + smooth)

    return 1.0 - iou.mean()


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