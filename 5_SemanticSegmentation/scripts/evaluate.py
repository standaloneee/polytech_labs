"""
Скрипт оценки обученной модели сегментации.

Функции:
- Загрузка обученной модели из checkpoint
- Вычисление метрик на test set
- Визуализация результатов: оригинал | GT | предсказание | overlay
- Сохранение результатов в JSON и PNG
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from dataset import download_drive, create_dataloaders, DATA_DIR
from model import create_model
from losses import create_criterion
from metrics import compute_all_metrics


def load_model(model_path, device):
    """
    Загружает модель из checkpoint.

    Args:
        model_path: путь к .pth файлу
        device: устройство

    Returns:
        model: загруженная модель
        config: конфигурация обучения
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = create_model(arch=config['arch'], pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Модель загружена: {model_path}")
    print(f"  Эпоха: {checkpoint.get('epoch', '?')}, "
          f"Val Dice: {checkpoint.get('val_dice', 0):.4f}")

    return model, config


def evaluate_model(model, dataloader, device):
    """
    Оценивает модель на наборе данных.

    Args:
        model: обученная модель
        dataloader: DataLoader с test данными
        device: устройство

    Returns:
        avg_metrics: словарь со средними метриками
        all_predictions: список (image, mask, prediction) для визуализации
    """
    model.eval()
    all_metrics = {'dice': 0, 'iou': 0, 'accuracy': 0, 'sensitivity': 0, 'specificity': 0}
    num_samples = 0
    all_predictions = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            # Метрики для батча
            for i in range(images.size(0)):
                metrics = compute_all_metrics(
                    outputs[i:i+1], masks[i:i+1]
                )
                for k, v in metrics.items():
                    all_metrics[k] += v
                num_samples += 1

                # Сохраняем для визуализации
                all_predictions.append((
                    images[i].cpu().numpy().transpose(1, 2, 0),  # (H, W, 3)
                    masks[i, 0].cpu().numpy(),                    # (H, W)
                    probs[i, 0].cpu().numpy(),                    # (H, W)
                ))

    avg_metrics = {k: v / num_samples for k, v in all_metrics.items()}
    return avg_metrics, all_predictions


def visualize_predictions(predictions, experiment_name, save_path=None, num_samples=5):
    """
    Визуализация результатов сегментации.

    Показывает: оригинал | Ground Truth | Предсказание | Overlay

    Args:
        predictions: список (image, mask, pred) от evaluate_model
        experiment_name: название эксперимента
        save_path: путь для сохранения PNG
        num_samples: количество примеров
    """
    n = min(num_samples, len(predictions))
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))

    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        image, gt_mask, pred_prob = predictions[i]
        pred_mask = (pred_prob > 0.5).astype(np.float32)

        # Оригинал
        axes[i, 0].imshow(image)
        axes[i, 0].set_title('Оригинал' if i == 0 else '')
        axes[i, 0].axis('off')

        # Ground Truth
        axes[i, 1].imshow(gt_mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth' if i == 0 else '')
        axes[i, 1].axis('off')

        # Предсказание
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title('Предсказание' if i == 0 else '')
        axes[i, 2].axis('off')

        # Overlay: зелёный = верно, красный = ошибка
        overlay = image.copy()
        # True Positive — зелёный
        tp = (pred_mask > 0.5) & (gt_mask > 0.5)
        overlay[tp] = [0, 1, 0]
        # False Positive — красный
        fp = (pred_mask > 0.5) & (gt_mask < 0.5)
        overlay[fp] = [1, 0, 0]
        # False Negative — синий
        fn = (pred_mask < 0.5) & (gt_mask > 0.5)
        overlay[fn] = [0, 0, 1]

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay (TP/FP/FN)' if i == 0 else '')
        axes[i, 3].axis('off')

    plt.suptitle(f'Результаты: {experiment_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Визуализация сохранена: {save_path}")

    plt.close()


def run_evaluation(model_path, data_dir=None, device=None):
    """
    Полный цикл оценки модели.

    Args:
        model_path: путь к checkpoint (.pth)
        data_dir: путь к данным DRIVE
        device: устройство

    Returns:
        dict: результаты с метриками
    """
    if data_dir is None:
        data_dir = DATA_DIR

    model_path = Path(model_path)
    experiment_name = model_path.stem.replace('_best', '')

    # Загрузка модели
    model, config = load_model(model_path, device)

    # Скачивание данных (если нужно)
    download_drive(data_dir)

    # Создание DataLoader'ов
    _, _, test_loader = create_dataloaders(
        data_dir=data_dir, batch_size=4, augment=False, num_workers=2
    )

    # Оценка
    print(f"\nОценка на test set...")
    metrics, predictions = evaluate_model(model, test_loader, device)

    print(f"\nРезультаты ({experiment_name}):")
    for name, value in metrics.items():
        print(f"  {name:>15s}: {value:.4f}")

    # Сохранение результатов
    project_dir = Path(__file__).parent.parent
    results_dir = project_dir / 'results' / 'metrics'
    results_dir.mkdir(parents=True, exist_ok=True)

    # JSON с метриками
    results = {
        'experiment_name': experiment_name,
        'config': config,
        'test_metrics': metrics,
    }

    json_path = results_dir / f"{experiment_name}_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nМетрики сохранены: {json_path}")

    # Визуализация
    vis_path = results_dir / f"{experiment_name}_examples.png"
    visualize_predictions(
        predictions, experiment_name,
        save_path=vis_path, num_samples=5
    )

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Оценка модели сегментации сосудов сетчатки'
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Путь к checkpoint (.pth)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Путь к данным DRIVE')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Используется устройство: {device}")

    run_evaluation(
        model_path=args.model,
        data_dir=args.data_dir,
        device=device,
    )
