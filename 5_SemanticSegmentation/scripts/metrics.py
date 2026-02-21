"""
Метрики для оценки качества семантической сегментации.

Реализованы:
- Dice coefficient (F1 score для сегментации)
- IoU (Intersection over Union, Jaccard index)
- Pixel Accuracy
- Sensitivity (Recall) — обнаружение сосудов
- Specificity — правильное определение фона
"""

import torch
import numpy as np


def dice_coefficient(pred, target, threshold=0.5, smooth=1e-6):
    """
    Dice coefficient = 2*|A∩B| / (|A| + |B|).

    Args:
        pred: тензор предсказаний (logits или вероятности)
        target: тензор масок {0, 1}
        threshold: порог бинаризации
        smooth: сглаживание для числовой стабильности

    Returns:
        float: Dice coefficient [0, 1]
    """
    if pred.requires_grad:
        pred = pred.detach()
    if target.requires_grad:
        target = target.detach()

    # Применяем sigmoid если logits
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)

    pred_binary = (pred > threshold).float()
    target = target.float()

    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    IoU (Intersection over Union) = |A∩B| / |A∪B|.

    Args:
        pred: тензор предсказаний
        target: тензор масок {0, 1}
        threshold: порог бинаризации
        smooth: сглаживание

    Returns:
        float: IoU score [0, 1]
    """
    if pred.requires_grad:
        pred = pred.detach()
    if target.requires_grad:
        target = target.detach()

    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)

    pred_binary = (pred > threshold).float()
    target = target.float()

    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection

    return (intersection + smooth) / (union + smooth)


def pixel_accuracy(pred, target, threshold=0.5):
    """
    Pixel Accuracy = правильно классифицированные пиксели / все пиксели.

    Args:
        pred: тензор предсказаний
        target: тензор масок {0, 1}
        threshold: порог бинаризации

    Returns:
        float: accuracy [0, 1]
    """
    if pred.requires_grad:
        pred = pred.detach()
    if target.requires_grad:
        target = target.detach()

    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)

    pred_binary = (pred > threshold).float()
    target = target.float()

    correct = (pred_binary == target).float().sum()
    total = target.numel()

    return correct / total


def sensitivity(pred, target, threshold=0.5, smooth=1e-6):
    """
    Sensitivity (Recall, True Positive Rate).

    TP / (TP + FN) — способность обнаруживать сосуды.

    Args:
        pred: тензор предсказаний
        target: тензор масок {0, 1}
        threshold: порог бинаризации
        smooth: сглаживание

    Returns:
        float: sensitivity [0, 1]
    """
    if pred.requires_grad:
        pred = pred.detach()
    if target.requires_grad:
        target = target.detach()

    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)

    pred_binary = (pred > threshold).float()
    target = target.float()

    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)

    tp = (pred_flat * target_flat).sum()
    fn = ((1 - pred_flat) * target_flat).sum()

    return (tp + smooth) / (tp + fn + smooth)


def specificity(pred, target, threshold=0.5, smooth=1e-6):
    """
    Specificity (True Negative Rate).

    TN / (TN + FP) — правильное определение фона.

    Args:
        pred: тензор предсказаний
        target: тензор масок {0, 1}
        threshold: порог бинаризации
        smooth: сглаживание

    Returns:
        float: specificity [0, 1]
    """
    if pred.requires_grad:
        pred = pred.detach()
    if target.requires_grad:
        target = target.detach()

    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)

    pred_binary = (pred > threshold).float()
    target = target.float()

    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)

    tn = ((1 - pred_flat) * (1 - target_flat)).sum()
    fp = (pred_flat * (1 - target_flat)).sum()

    return (tn + smooth) / (tn + fp + smooth)


def compute_all_metrics(pred, target, threshold=0.5):
    """
    Вычисляет все метрики.

    Args:
        pred: тензор предсказаний (logits или вероятности)
        target: тензор масок {0, 1}
        threshold: порог бинаризации

    Returns:
        dict: словарь со всеми метриками
    """
    return {
        'dice': dice_coefficient(pred, target, threshold).item(),
        'iou': iou_score(pred, target, threshold).item(),
        'accuracy': pixel_accuracy(pred, target, threshold).item(),
        'sensitivity': sensitivity(pred, target, threshold).item(),
        'specificity': specificity(pred, target, threshold).item(),
    }


if __name__ == '__main__':
    # Тест метрик
    print("Тестирование метрик")
    print("=" * 40)

    # Случайные предсказания
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()

    metrics = compute_all_metrics(pred, target)
    print("Случайные предсказания:")
    for name, value in metrics.items():
        print(f"  {name:>15s}: {value:.4f}")

    # Идеальные предсказания
    perfect_pred = target * 10 - 5
    metrics_perfect = compute_all_metrics(perfect_pred, target)
    print("\nИдеальные предсказания:")
    for name, value in metrics_perfect.items():
        print(f"  {name:>15s}: {value:.4f}")
