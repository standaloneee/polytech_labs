"""
Оценка Denoising Autoencoder с метрикой SSIM
SSIM (Structural Similarity Index) - мера структурной схожести изображений
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

from model import DenoisingAutoencoder
from noise import apply_noise, create_noise_levels


def calculate_ssim(image1, image2):
    """
    Вычислить SSIM между двумя изображениями

    Args:
        image1: Первое изображение (numpy array или torch tensor)
        image2: Второе изображение

    Returns:
        SSIM значение в диапазоне [-1, 1] (1 = идентичные изображения)
    """
    # Конвертировать tensor в numpy если нужно
    if isinstance(image1, torch.Tensor):
        image1 = image1.cpu().numpy()
    if isinstance(image2, torch.Tensor):
        image2 = image2.cpu().numpy()

    # SSIM требует формат (H, W, C)
    if image1.ndim == 3 and image1.shape[0] == 3:  # (C, H, W)
        image1 = np.transpose(image1, (1, 2, 0))
    if image2.ndim == 3 and image2.shape[0] == 3:
        image2 = np.transpose(image2, (1, 2, 0))

    # Вычислить SSIM для RGB изображения
    return ssim(image1, image2, channel_axis=2, data_range=1.0)


def calculate_psnr(image1, image2):
    """
    Вычислить PSNR (Peak Signal-to-Noise Ratio)

    Args:
        image1: Первое изображение
        image2: Второе изображение

    Returns:
        PSNR в dB (чем выше, тем лучше)
    """
    if isinstance(image1, torch.Tensor):
        image1 = image1.cpu().numpy()
    if isinstance(image2, torch.Tensor):
        image2 = image2.cpu().numpy()

    if image1.ndim == 3 and image1.shape[0] == 3:
        image1 = np.transpose(image1, (1, 2, 0))
    if image2.ndim == 3 and image2.shape[0] == 3:
        image2 = np.transpose(image2, (1, 2, 0))

    return psnr(image1, image2, data_range=1.0)


def calculate_mse(image1, image2):
    """
    Вычислить MSE (Mean Squared Error)

    Args:
        image1: Первое изображение
        image2: Второе изображение

    Returns:
        MSE значение (чем ниже, тем лучше)
    """
    if isinstance(image1, torch.Tensor):
        image1 = image1.cpu().numpy()
    if isinstance(image2, torch.Tensor):
        image2 = image2.cpu().numpy()

    return mse(image1, image2)


def evaluate_model(model, test_loader, device):
    """
    Оценить модель на тестовых данных

    Args:
        model: Обученная модель
        test_loader: DataLoader с тестовыми данными
        device: Устройство для вычислений

    Returns:
        Словарь с метриками
    """
    model.eval()

    ssim_scores = []
    psnr_scores = []
    mse_scores = []

    # Метрики для сравнения: зашумленное vs чистое
    ssim_noisy_scores = []
    psnr_noisy_scores = []

    with torch.no_grad():
        for noisy_images, clean_images in tqdm(test_loader, desc='Оценка'):
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            # Получить восстановленные изображения
            reconstructed = model(noisy_images)

            # Вычислить метрики для каждого изображения в батче
            for i in range(noisy_images.size(0)):
                noisy = noisy_images[i]
                clean = clean_images[i]
                recon = reconstructed[i]

                # Метрики: восстановленное vs чистое
                ssim_scores.append(calculate_ssim(recon, clean))
                psnr_scores.append(calculate_psnr(recon, clean))
                mse_scores.append(calculate_mse(recon, clean))

                # Метрики: зашумленное vs чистое (для сравнения)
                ssim_noisy_scores.append(calculate_ssim(noisy, clean))
                psnr_noisy_scores.append(calculate_psnr(noisy, clean))

    # Средние значения
    results = {
        'ssim_reconstructed': float(np.mean(ssim_scores)),
        'psnr_reconstructed': float(np.mean(psnr_scores)),
        'mse_reconstructed': float(np.mean(mse_scores)),
        'ssim_noisy': float(np.mean(ssim_noisy_scores)),
        'psnr_noisy': float(np.mean(psnr_noisy_scores)),
        'ssim_improvement': float(np.mean(ssim_scores) - np.mean(ssim_noisy_scores)),
        'psnr_improvement': float(np.mean(psnr_scores) - np.mean(psnr_noisy_scores))
    }

    return results


def visualize_results(model, test_dataset, device, num_samples=5, save_path=None):
    """
    Визуализировать результаты работы модели

    Args:
        model: Обученная модель
        test_dataset: Тестовый датасет
        device: Устройство
        num_samples: Количество примеров для визуализации
        save_path: Путь для сохранения графика
    """
    model.eval()

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    with torch.no_grad():
        for i in range(num_samples):
            # Получить случайное изображение
            idx = np.random.randint(0, len(test_dataset))
            noisy_image, clean_image = test_dataset[idx]

            # Добавить batch dimension
            noisy_batch = noisy_image.unsqueeze(0).to(device)

            # Получить восстановленное изображение
            reconstructed = model(noisy_batch)[0].cpu()

            # Вычислить метрики
            ssim_noisy = calculate_ssim(noisy_image, clean_image)
            ssim_recon = calculate_ssim(reconstructed, clean_image)
            psnr_recon = calculate_psnr(reconstructed, clean_image)

            # Конвертировать для отображения
            clean_np = clean_image.permute(1, 2, 0).numpy()
            noisy_np = noisy_image.permute(1, 2, 0).numpy()
            recon_np = reconstructed.permute(1, 2, 0).numpy()
            diff_np = np.abs(recon_np - clean_np)

            # Отобразить
            axes[i, 0].imshow(clean_np)
            axes[i, 0].set_title('Оригинал (чистый)', fontsize=12)
            axes[i, 0].axis('off')

            axes[i, 1].imshow(noisy_np)
            axes[i, 1].set_title(f'Зашумленный\nSSIM: {ssim_noisy:.3f}', fontsize=12)
            axes[i, 1].axis('off')

            axes[i, 2].imshow(recon_np)
            axes[i, 2].set_title(f'Восстановленный\nSSIM: {ssim_recon:.3f} | PSNR: {psnr_recon:.1f}',
                               fontsize=12)
            axes[i, 2].axis('off')

            axes[i, 3].imshow(diff_np)
            axes[i, 3].set_title('Разница (error map)', fontsize=12)
            axes[i, 3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Визуализация сохранена: {save_path}")
    else:
        plt.show()

    plt.close()


def load_model(model_path, device):
    """
    Загрузить обученную модель из checkpoint

    Args:
        model_path: Путь к сохраненной модели
        device: Устройство

    Returns:
        Загруженная модель и конфигурация
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Получить конфигурацию
    config = checkpoint.get('config', {})
    base_channels = config.get('base_channels', 32)

    # Создать модель
    model = DenoisingAutoencoder(base_channels=base_channels)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Модель загружена: {model_path}")
    print(f"Эпоха: {checkpoint.get('epoch', 'N/A')}")
    print(f"Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")

    return model, config


def run_evaluation(model_path, data_dir, output_dir, device):
    """
    Запустить полную оценку модели

    Args:
        model_path: Путь к модели
        data_dir: Путь к данным
        output_dir: Куда сохранять результаты
        device: Устройство
    """
    from train import BloodCellDataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Загрузить модель
    model, config = load_model(model_path, device)

    # Создать тестовый датасет с тем же шумом
    noise_configs = create_noise_levels()
    noise_type = config.get('noise_type', 'gaussian_medium')
    noise_config = noise_configs[noise_type]

    test_dataset = BloodCellDataset(data_dir, noise_config)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print(f"\nТестовый датасет: {len(test_dataset)} изображений")
    print(f"Тип шума: {noise_type}")
    print("=" * 70)

    # Оценка
    results = evaluate_model(model, test_loader, device)

    # Вывести результаты
    print("\nРезультаты оценки:")
    print("=" * 70)
    print(f"SSIM (восстановленное): {results['ssim_reconstructed']:.4f}")
    print(f"SSIM (зашумленное):     {results['ssim_noisy']:.4f}")
    print(f"Улучшение SSIM:         {results['ssim_improvement']:.4f}")
    print()
    print(f"PSNR (восстановленное): {results['psnr_reconstructed']:.2f} dB")
    print(f"PSNR (зашумленное):     {results['psnr_noisy']:.2f} dB")
    print(f"Улучшение PSNR:         {results['psnr_improvement']:.2f} dB")
    print()
    print(f"MSE:                    {results['mse_reconstructed']:.6f}")
    print("=" * 70)

    # Сохранить результаты
    model_name = Path(model_path).stem
    results_path = output_dir / f'{model_name}_results.json'
    results['config'] = config
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nРезультаты сохранены: {results_path}")

    # Визуализация
    viz_path = output_dir / f'{model_name}_examples.png'
    visualize_results(model, test_dataset, device, num_samples=5, save_path=viz_path)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Оценка Denoising Autoencoder')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Путь к модели')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Путь к данным')
    parser.add_argument('--output_dir', type=str, default='results/metrics',
                       help='Куда сохранять результаты')

    args = parser.parse_args()

    # Устройство
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Используется устройство: {device}")

    # Оценка
    run_evaluation(args.model_path, args.data_dir, args.output_dir, device)
