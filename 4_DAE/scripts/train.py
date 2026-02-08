"""
Скрипт обучения Denoising Autoencoder
Поддерживает MSE и MAE функции потерь, TensorBoard логирование
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from model import create_model
from noise import apply_noise, create_noise_levels


class BloodCellDataset(Dataset):
    """
    Датасет для изображений клеток крови

    Возвращает пары (чистое_изображение, зашумленное_изображение)
    """

    def __init__(self, image_dir, noise_config, transform=None):
        """
        Args:
            image_dir: Путь к директории с изображениями
            noise_config: Конфигурация шума (из noise.create_noise_levels())
            transform: Torchvision transforms для предобработки
        """
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.glob('*.jpg'))
        self.noise_config = noise_config
        self.transform = transform or transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Загрузить изображение
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Применить transforms
        clean_image = self.transform(image)

        # Добавить шум
        noisy_image = apply_noise(clean_image, self.noise_config)

        return noisy_image, clean_image


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Обучение на одной эпохе

    Args:
        model: Модель автокодировщика
        dataloader: DataLoader с данными
        criterion: Функция потерь (MSE или MAE)
        optimizer: Оптимизатор
        device: Устройство для вычислений
        epoch: Номер эпохи

    Returns:
        Средний loss на эпохе
    """
    model.train()
    total_loss = 0.0

    with tqdm(dataloader, desc=f'Epoch {epoch+1}') as pbar:
        for noisy_images, clean_images in pbar:
            # Переместить на устройство
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            # Forward pass
            reconstructed = model(noisy_images)
            loss = criterion(reconstructed, clean_images)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Статистика
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device):
    """
    Валидация модели

    Args:
        model: Модель автокодировщика
        dataloader: DataLoader с валидационными данными
        criterion: Функция потерь
        device: Устройство для вычислений

    Returns:
        Средний loss на валидации
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for noisy_images, clean_images in dataloader:
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            reconstructed = model(noisy_images)
            loss = criterion(reconstructed, clean_images)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train(
    data_dir,
    noise_type='gaussian_medium',
    loss_fn='mse',
    base_channels=32,
    batch_size=8,
    num_epochs=50,
    learning_rate=0.001,
    val_split=0.2,
    device='cpu',
    save_dir='models',
    log_dir='runs'
):
    """
    Полный цикл обучения модели

    Args:
        data_dir: Путь к данным
        noise_type: Тип шума из noise.create_noise_levels()
        loss_fn: Функция потерь ('mse' или 'mae')
        base_channels: Базовое количество каналов в модели
        batch_size: Размер батча
        num_epochs: Количество эпох
        learning_rate: Learning rate
        val_split: Доля данных для валидации
        device: Устройство ('cpu', 'cuda', 'mps')
        save_dir: Куда сохранять модели
        log_dir: Куда сохранять TensorBoard логи

    Returns:
        История обучения
    """
    # Создать директории
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp для уникальных имен
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'{noise_type}_{loss_fn}_{timestamp}'

    # Создать датасет
    noise_configs = create_noise_levels()
    noise_config = noise_configs[noise_type]

    full_dataset = BloodCellDataset(data_dir, noise_config)

    # Разделить на train и val
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Создать DataLoader'ы
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(f"\nДатасет: {len(full_dataset)} изображений")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Создать модель
    model = create_model(base_channels=base_channels, device=device)

    # Функция потерь
    if loss_fn == 'mse':
        criterion = nn.MSELoss()
    elif loss_fn == 'mae':
        criterion = nn.L1Loss()  # MAE = L1 Loss
    else:
        raise ValueError(f"Неизвестная функция потерь: {loss_fn}")

    # Оптимизатор
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler для уменьшения learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # TensorBoard writer
    writer = SummaryWriter(log_dir / run_name)

    # История обучения
    history = {
        'train_loss': [],
        'val_loss': [],
        'config': {
            'noise_type': noise_type,
            'noise_config': noise_config,
            'loss_fn': loss_fn,
            'base_channels': base_channels,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs
        }
    }

    best_val_loss = float('inf')

    print(f"\nНачало обучения: {run_name}")
    print(f"Функция потерь: {loss_fn.upper()}")
    print(f"Тип шума: {noise_type}")
    print("=" * 70)

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)

        # Логирование
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f}")

        # Сохранить лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = save_dir / f'{run_name}_best.pth'

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'config': history['config']
            }, best_model_path)

            print(f"✓ Лучшая модель сохранена: {best_model_path.name}")

    # Закрыть writer
    writer.close()

    # Сохранить историю
    history['best_val_loss'] = best_val_loss
    history_path = save_dir / f'{run_name}_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Обучение завершено!")
    print(f"Лучший val loss: {best_val_loss:.6f}")
    print(f"Модель сохранена: {best_model_path}")
    print(f"История: {history_path}")

    return history


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Обучение Denoising Autoencoder')

    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Путь к данным')
    parser.add_argument('--noise_type', type=str, default='gaussian_medium',
                       choices=['gaussian_light', 'gaussian_medium', 'gaussian_heavy',
                               'salt_pepper_light', 'salt_pepper_medium', 'salt_pepper_heavy',
                               'mixed_light', 'mixed_medium', 'mixed_heavy'],
                       help='Тип шума')
    parser.add_argument('--loss_fn', type=str, default='mse',
                       choices=['mse', 'mae'],
                       help='Функция потерь')
    parser.add_argument('--base_channels', type=int, default=32,
                       help='Базовое количество каналов')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Размер батча')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Количество эпох')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Доля данных для валидации')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Определить устройство
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Используется устройство: {device}")

    # Обучение
    history = train(
        data_dir=args.data_dir,
        noise_type=args.noise_type,
        loss_fn=args.loss_fn,
        base_channels=args.base_channels,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        val_split=args.val_split,
        device=device
    )
