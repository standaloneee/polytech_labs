"""
Скрипт обучения модели сегментации сосудов сетчатки.

Поддерживает:
- Разные архитектуры: базовый U-Net, U-Net + ResNet34
- Разные loss-функции: BCE, Dice, Combined (BCE + Dice)
- Аугментации данных
- Early stopping, ReduceLROnPlateau
- Логирование в TensorBoard
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Добавляем путь к скриптам
sys.path.insert(0, str(Path(__file__).parent))

from dataset import download_drive, create_dataloaders, DATA_DIR
from model import create_model
from losses import create_criterion
from metrics import compute_all_metrics, dice_coefficient


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Одна эпоха обучения.

    Args:
        model: модель
        dataloader: DataLoader с train данными
        criterion: loss-функция
        optimizer: оптимизатор
        device: устройство (cuda/mps/cpu)

    Returns:
        avg_loss: средний loss за эпоху
        avg_dice: средний Dice за эпоху
    """
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0

    for images, masks in tqdm(dataloader, desc='  Train', leave=False):
        images = images.to(device)
        masks = masks.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Метрики
        total_loss += loss.item()
        total_dice += dice_coefficient(outputs, masks).item()
        num_batches += 1

    return total_loss / num_batches, total_dice / num_batches


def validate(model, dataloader, criterion, device):
    """
    Валидация модели.

    Args:
        model: модель
        dataloader: DataLoader с val данными
        criterion: loss-функция
        device: устройство

    Returns:
        avg_loss: средний loss
        avg_metrics: словарь с метриками
    """
    model.eval()
    total_loss = 0.0
    all_metrics = {'dice': 0, 'iou': 0, 'accuracy': 0, 'sensitivity': 0, 'specificity': 0}
    num_batches = 0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='  Val', leave=False):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()

            metrics = compute_all_metrics(outputs, masks)
            for k, v in metrics.items():
                all_metrics[k] += v

            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in all_metrics.items()}

    return avg_loss, avg_metrics


def train(arch='unet', loss_type='bce', augment=False,
          num_epochs=50, batch_size=4, lr=1e-4,
          patience=10, data_dir=None, device=None,
          experiment_name=None):
    """
    Полный цикл обучения модели.

    Args:
        arch: архитектура ('unet' или 'resnet34')
        loss_type: тип loss ('bce', 'dice', 'combined')
        augment: использовать аугментации
        num_epochs: количество эпох
        batch_size: размер батча
        lr: learning rate
        patience: терпение для early stopping
        data_dir: путь к данным
        device: устройство
        experiment_name: имя эксперимента

    Returns:
        dict: история обучения и лучшие метрики
    """
    if data_dir is None:
        data_dir = DATA_DIR

    # Имя эксперимента
    if experiment_name is None:
        aug_str = 'aug' if augment else 'noaug'
        experiment_name = f"{arch}_{loss_type}_{aug_str}"

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{experiment_name}_{timestamp}"

    print("=" * 60)
    print(f"Эксперимент: {experiment_name}")
    print(f"  Архитектура: {arch}")
    print(f"  Loss: {loss_type}")
    print(f"  Аугментации: {'да' if augment else 'нет'}")
    print(f"  Эпохи: {num_epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"  Устройство: {device}")
    print("=" * 60)

    # Пути для сохранения
    project_dir = Path(__file__).parent.parent
    models_dir = project_dir / 'models'
    results_dir = project_dir / 'results' / 'metrics'
    runs_dir = project_dir / 'runs'
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Скачивание данных
    download_drive(data_dir)

    # Создание DataLoader'ов
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        augment=augment,
        num_workers=2
    )

    # Создание модели
    model = create_model(arch=arch)
    model = model.to(device)

    # Loss и оптимизатор
    criterion = create_criterion(loss_type)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=str(runs_dir / run_name))

    # История обучения
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'val_iou': [], 'val_accuracy': [],
        'val_sensitivity': [], 'val_specificity': [],
        'lr': []
    }

    # Early stopping
    best_val_dice = 0.0
    epochs_without_improvement = 0
    best_model_path = models_dir / f"{experiment_name}_best.pth"

    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nЭпоха {epoch}/{num_epochs} (lr={current_lr:.2e})")

        # Обучение
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Валидация
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        # Планировщик
        scheduler.step(val_metrics['dice'])

        # Логирование
        print(f"  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Dice: {val_metrics['dice']:.4f}, "
              f"Val IoU: {val_metrics['iou']:.4f}")

        # TensorBoard
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalar('Dice/train', train_dice, epoch)
        writer.add_scalar('Dice/val', val_metrics['dice'], epoch)
        writer.add_scalar('IoU/val', val_metrics['iou'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('LR', current_lr, epoch)

        # История
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_metrics['dice'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_sensitivity'].append(val_metrics['sensitivity'])
        history['val_specificity'].append(val_metrics['specificity'])
        history['lr'].append(current_lr)

        # Сохранение лучшей модели
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            epochs_without_improvement = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_val_dice,
                'val_metrics': val_metrics,
                'config': {
                    'arch': arch,
                    'loss_type': loss_type,
                    'augment': augment,
                    'num_epochs': num_epochs,
                    'batch_size': batch_size,
                    'lr': lr,
                }
            }
            torch.save(checkpoint, best_model_path)
            print(f"  >>> Лучшая модель сохранена (Dice: {best_val_dice:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  Без улучшения: {epochs_without_improvement}/{patience}")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping на эпохе {epoch}")
            break

    writer.close()

    # Сохранение истории
    history_path = results_dir / f"{experiment_name}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nОбучение завершено. Лучший Val Dice: {best_val_dice:.4f}")
    print(f"Модель: {best_model_path}")
    print(f"История: {history_path}")

    return {
        'experiment_name': experiment_name,
        'best_val_dice': best_val_dice,
        'best_model_path': str(best_model_path),
        'history': history,
    }


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description='Обучение модели сегментации сосудов сетчатки'
    )
    parser.add_argument('--arch', type=str, default='unet',
                        choices=['unet', 'resnet34'],
                        help='Архитектура модели (default: unet)')
    parser.add_argument('--loss', type=str, default='bce',
                        choices=['bce', 'dice', 'combined'],
                        help='Loss-функция (default: bce)')
    parser.add_argument('--augment', action='store_true',
                        help='Использовать аугментации')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Количество эпох (default: 50)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Размер батча (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience для early stopping (default: 10)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Путь к данным DRIVE')
    parser.add_argument('--name', type=str, default=None,
                        help='Имя эксперимента')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Определяем устройство
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Используется устройство: {device}")

    train(
        arch=args.arch,
        loss_type=args.loss,
        augment=args.augment,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        data_dir=args.data_dir,
        device=device,
        experiment_name=args.name,
    )
