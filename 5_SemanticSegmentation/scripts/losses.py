"""
Loss-функции для задачи семантической сегментации.

Реализованы:
- BCEWithLogitsLoss: стандартная бинарная кросс-энтропия
- DiceLoss: 1 - Dice coefficient (дифференцируемая)
- CombinedLoss: BCE + Dice с настраиваемыми весами
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss = 1 - Dice Coefficient.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Используется smooth для числовой стабильности.
    Принимает logits (без sigmoid).
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) — выход модели (без sigmoid)
            targets: (B, 1, H, W) — бинарные маски {0, 1}

        Returns:
            loss: скаляр
        """
        probs = torch.sigmoid(logits)

        # Расплющиваем до (B, N)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Комбинированная loss: BCE + Dice.

    loss = bce_weight * BCE + dice_weight * Dice

    По умолчанию: 0.5 * BCE + 0.5 * Dice
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) — выход модели (без sigmoid)
            targets: (B, 1, H, W) — бинарные маски {0, 1}

        Returns:
            loss: скаляр
        """
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def create_criterion(loss_type='bce'):
    """
    Фабричная функция для создания loss-функции.

    Args:
        loss_type: 'bce', 'dice' или 'combined'

    Returns:
        nn.Module — loss-функция
    """
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'combined':
        return CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    else:
        raise ValueError(
            f"Неизвестная loss-функция: {loss_type}. "
            "Доступны: bce, dice, combined"
        )


if __name__ == '__main__':
    # Тест loss-функций
    print("Тестирование loss-функций")
    print("=" * 40)

    logits = torch.randn(2, 1, 64, 64)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    for name in ['bce', 'dice', 'combined']:
        criterion = create_criterion(name)
        loss = criterion(logits, targets)
        print(f"  {name:>10s}: {loss.item():.4f}")

    # Тест: идеальное предсказание → Dice Loss ≈ 0
    print("\nТест с идеальным предсказанием:")
    perfect_logits = targets * 10 - 5  # sigmoid(10)≈1, sigmoid(-5)≈0
    for name in ['bce', 'dice', 'combined']:
        criterion = create_criterion(name)
        loss = criterion(perfect_logits, targets)
        print(f"  {name:>10s}: {loss.item():.6f}")
