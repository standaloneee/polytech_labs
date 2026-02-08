"""
Функции для добавления шумов к изображениям
Два типа шума: Gaussian (стохастический) и Salt & Pepper (импульсный)
"""

import torch
import numpy as np
from PIL import Image


def add_gaussian_noise(image, mean=0.0, std=0.1):
    """
    Добавить гауссовский (стохастический) шум к изображению

    Гауссовский шум моделирует шум от датчиков камеры,
    температурный шум электроники и т.д.

    Args:
        image: Тензор изображения [C, H, W] или [B, C, H, W] в диапазоне [0, 1]
        mean: Среднее значение распределения шума
        std: Стандартное отклонение (сила шума)

    Returns:
        Зашумленное изображение с теми же размерами
    """
    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    # Обрезать значения чтобы остаться в диапазоне [0, 1]
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
    return noisy_image


def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """
    Добавить импульсный шум (Salt & Pepper) к изображению

    Salt & Pepper шум моделирует ошибки передачи данных,
    мертвые пиксели, пыль на объективе и т.д.

    Args:
        image: Тензор изображения [C, H, W] или [B, C, H, W] в диапазоне [0, 1]
        salt_prob: Вероятность появления белого пикселя (salt)
        pepper_prob: Вероятность появления черного пикселя (pepper)

    Returns:
        Зашумленное изображение с теми же размерами
    """
    noisy_image = image.clone()

    # Генерировать случайные маски для salt и pepper
    salt_mask = torch.rand_like(image) < salt_prob
    pepper_mask = torch.rand_like(image) < pepper_prob

    # Применить шум
    noisy_image[salt_mask] = 1.0  # Белые пиксели
    noisy_image[pepper_mask] = 0.0  # Черные пиксели

    return noisy_image


def add_mixed_noise(image, gaussian_std=0.05, salt_prob=0.005, pepper_prob=0.005):
    """
    Добавить комбинацию гауссовского и импульсного шума

    Моделирует реальные условия, где присутствуют оба типа шума

    Args:
        image: Тензор изображения [C, H, W] или [B, C, H, W] в диапазоне [0, 1]
        gaussian_std: Сила гауссовского шума
        salt_prob: Вероятность salt шума
        pepper_prob: Вероятность pepper шума

    Returns:
        Зашумленное изображение с теми же размерами
    """
    # Сначала добавить гауссовский шум
    noisy_image = add_gaussian_noise(image, std=gaussian_std)
    # Затем добавить импульсный шум
    noisy_image = add_salt_pepper_noise(noisy_image, salt_prob, pepper_prob)
    return noisy_image


def create_noise_levels():
    """
    Создать словарь с различными уровнями шума для экспериментов

    Returns:
        Словарь {название: параметры_шума}
    """
    noise_configs = {
        # Только гауссовский шум
        'gaussian_light': {
            'type': 'gaussian',
            'mean': 0.0,
            'std': 0.05
        },
        'gaussian_medium': {
            'type': 'gaussian',
            'mean': 0.0,
            'std': 0.1
        },
        'gaussian_heavy': {
            'type': 'gaussian',
            'mean': 0.0,
            'std': 0.2
        },

        # Только импульсный шум
        'salt_pepper_light': {
            'type': 'salt_pepper',
            'salt_prob': 0.005,
            'pepper_prob': 0.005
        },
        'salt_pepper_medium': {
            'type': 'salt_pepper',
            'salt_prob': 0.01,
            'pepper_prob': 0.01
        },
        'salt_pepper_heavy': {
            'type': 'salt_pepper',
            'salt_prob': 0.02,
            'pepper_prob': 0.02
        },

        # Смешанный шум
        'mixed_light': {
            'type': 'mixed',
            'gaussian_std': 0.03,
            'salt_prob': 0.003,
            'pepper_prob': 0.003
        },
        'mixed_medium': {
            'type': 'mixed',
            'gaussian_std': 0.05,
            'salt_prob': 0.005,
            'pepper_prob': 0.005
        },
        'mixed_heavy': {
            'type': 'mixed',
            'gaussian_std': 0.1,
            'salt_prob': 0.01,
            'pepper_prob': 0.01
        }
    }

    return noise_configs


def apply_noise(image, noise_config):
    """
    Применить шум к изображению согласно конфигурации

    Args:
        image: Тензор изображения [C, H, W] или [B, C, H, W]
        noise_config: Словарь с параметрами шума

    Returns:
        Зашумленное изображение
    """
    noise_type = noise_config['type']

    if noise_type == 'gaussian':
        return add_gaussian_noise(
            image,
            mean=noise_config.get('mean', 0.0),
            std=noise_config['std']
        )

    elif noise_type == 'salt_pepper':
        return add_salt_pepper_noise(
            image,
            salt_prob=noise_config['salt_prob'],
            pepper_prob=noise_config['pepper_prob']
        )

    elif noise_type == 'mixed':
        return add_mixed_noise(
            image,
            gaussian_std=noise_config['gaussian_std'],
            salt_prob=noise_config['salt_prob'],
            pepper_prob=noise_config['pepper_prob']
        )

    else:
        raise ValueError(f"Неизвестный тип шума: {noise_type}")


# Пример использования
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from pathlib import Path

    # Загрузить тестовое изображение
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    img_path = list(data_dir.glob('*.jpg'))[0]

    # Преобразовать в тензор
    transform = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor()
    ])

    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image)

    # Применить различные типы шума
    noise_configs = create_noise_levels()

    # Показать примеры
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    # Оригинал
    axes[0].imshow(image_tensor.permute(1, 2, 0))
    axes[0].set_title('Оригинал')
    axes[0].axis('off')

    # Примеры с шумом
    for i, (name, config) in enumerate(list(noise_configs.items())[:11], 1):
        noisy = apply_noise(image_tensor, config)
        axes[i].imshow(noisy.permute(1, 2, 0))
        axes[i].set_title(name.replace('_', ' ').title())
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('../results/plots/noise_examples.png', dpi=150, bbox_inches='tight')
    print("Примеры шума сохранены в results/plots/noise_examples.png")
