from __future__ import annotations

import torch


EPS = 1e-7


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
    return {
        "dice_fake": dice_score(pred_fake, target_fake, threshold),
        "iou_fake": iou_score(pred_fake, target_fake, threshold),
        "precision_fake": precision_score_mask(pred_fake, target_fake, threshold),
        "recall_fake": recall_score_mask(pred_fake, target_fake, threshold),
        "f1_fake": f1_score_mask(pred_fake, target_fake, threshold),
        "accuracy_fake": accuracy_score_mask(pred_fake, target_fake, threshold),
        "dice_authentic": dice_score(pred_authentic, target_authentic, threshold),
        "iou_authentic": iou_score(pred_authentic, target_authentic, threshold),
    }


def weighted_bce_from_probabilities(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    pred = pred.clamp(min=EPS, max=1.0 - EPS)
    target = target.float()

    positive_loss = -pos_weight * target * torch.log(pred)
    negative_loss = -(1.0 - target) * torch.log(1.0 - pred)

    return (positive_loss + negative_loss).mean()


def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred = pred.float()
    target = target.float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    denominator = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    return 1.0 - dice.mean()


def soft_iou_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    pred = pred.float()
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
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_fake = (
        weighted_bce_from_probabilities(pred_fake, target_fake, pos_weight=pos_weight)
        + soft_dice_loss(pred_fake, target_fake)
        + soft_iou_loss(pred_fake, target_fake)
    )

    loss_authentic = (
        weighted_bce_from_probabilities(pred_authentic, target_authentic, pos_weight=pos_weight)
        + soft_dice_loss(pred_authentic, target_authentic)
        + soft_iou_loss(pred_authentic, target_authentic)
    )

    total_loss = loss_fake + loss_authentic

    return total_loss, {
        "loss_total": float(total_loss.detach().cpu()),
        "loss_fake": float(loss_fake.detach().cpu()),
        "loss_authentic": float(loss_authentic.detach().cpu()),
    }
