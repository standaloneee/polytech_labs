"""
Скрипт для предобработки данных:
- Разделение на train/val/test (70/15/15)
- Проверка качества изображений
- Статистика по классам
- Визуализация примеров
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import random
from collections import Counter

from PIL import Image
import numpy as np
from tqdm import tqdm


# Константы
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'

TRAIN_DIR = PROCESSED_DIR / 'train'
VAL_DIR = PROCESSED_DIR / 'val'
TEST_DIR = PROCESSED_DIR / 'test'

# Разделение датасета
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed для воспроизводимости
RANDOM_SEED = 42

# Минимальный размер изображения
MIN_IMAGE_SIZE = 64


def set_seed(seed: int = RANDOM_SEED):
    """Установить seed для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)


def check_image(image_path: Path) -> bool:
    """
    Проверить что изображение можно открыть и оно больше минимального размера

    Returns:
        True если изображение валидно, False иначе
    """
    try:
        img = Image.open(image_path)
        img.verify()  # Проверка на corrupted

        # Переоткрыть после verify (verify закрывает файл)
        img = Image.open(image_path)
        width, height = img.size

        if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
            return False

        return True

    except Exception as e:
        print(f"  ✗ Corrupted image: {image_path.name} ({e})")
        return False


def collect_class_images(raw_dir: Path) -> Dict[str, List[Path]]:
    """
    Собрать все валидные изображения по классам

    Returns:
        Словарь {class_name: [image_paths]}
    """
    print("\n[1/4] Collecting and validating images...")

    class_images = {}

    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        images = []

        print(f"\nProcessing class: {class_name}")

        for img_path in tqdm(list(class_dir.glob('*.jpg')), desc="  Checking images"):
            if check_image(img_path):
                images.append(img_path)

        class_images[class_name] = images
        print(f"  ✓ Valid images: {len(images)}")

    return class_images


def split_dataset(class_images: Dict[str, List[Path]]) -> Tuple[Dict, Dict, Dict]:
    """
    Разделить датасет на train/val/test с сохранением баланса классов

    Returns:
        (train_dict, val_dict, test_dict)
    """
    print("\n[2/4] Splitting dataset (70/15/15)...")

    train_data = {}
    val_data = {}
    test_data = {}

    for class_name, images in class_images.items():
        # Перемешать изображения
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)

        # Разделить
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]

        train_data[class_name] = train_images
        val_data[class_name] = val_images
        test_data[class_name] = test_images

        print(f"  {class_name}: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")

    return train_data, val_data, test_data


def copy_images(data_dict: Dict[str, List[Path]], output_dir: Path, split_name: str):
    """
    Скопировать изображения в соответствующие папки

    Args:
        data_dict: Словарь {class_name: [image_paths]}
        output_dir: Директория назначения (train/val/test)
        split_name: Название split для логов
    """
    print(f"\n[3/4] Copying {split_name} images...")

    for class_name, images in data_dict.items():
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(images, desc=f"  {class_name}"):
            destination = class_dir / img_path.name
            shutil.copy2(img_path, destination)

    print(f"  ✓ {split_name} images saved to: {output_dir}")


def print_statistics(train_data: Dict, val_data: Dict, test_data: Dict):
    """Вывести статистику по датасету"""
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    # Подсчет по классам
    print("\nPer-class statistics:")
    print(f"{'Class':<20} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 70)

    total_train = 0
    total_val = 0
    total_test = 0

    for class_name in sorted(train_data.keys()):
        n_train = len(train_data[class_name])
        n_val = len(val_data[class_name])
        n_test = len(test_data[class_name])
        n_total = n_train + n_val + n_test

        print(f"{class_name:<20} {n_train:<10} {n_val:<10} {n_test:<10} {n_total:<10}")

        total_train += n_train
        total_val += n_val
        total_test += n_test

    print("-" * 70)
    print(f"{'TOTAL':<20} {total_train:<10} {total_val:<10} {total_test:<10} {total_train + total_val + total_test:<10}")

    # Общая статистика
    print("\n" + "=" * 70)
    print(f"Number of classes: {len(train_data)}")
    print(f"Total images: {total_train + total_val + total_test}")
    print(f"Train/Val/Test split: {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%}")
    print("=" * 70)


def main():
    """Основная функция предобработки"""

    print("=" * 70)
    print("Data Preprocessing Pipeline")
    print("=" * 70)

    # Установить seed
    set_seed(RANDOM_SEED)

    # Проверить что raw данные существуют
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        print(f"\n✗ Error: Raw data not found in {RAW_DIR}")
        print("Please run: python scripts/download_data.py")
        return

    # Создать директории для processed данных
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Собрать и валидировать изображения
    class_images = collect_class_images(RAW_DIR)

    # 2. Разделить на train/val/test
    train_data, val_data, test_data = split_dataset(class_images)

    # 3. Скопировать изображения
    copy_images(train_data, TRAIN_DIR, "train")
    copy_images(val_data, VAL_DIR, "val")
    copy_images(test_data, TEST_DIR, "test")

    # 4. Вывести статистику
    print_statistics(train_data, val_data, test_data)

    print("\n✓ Data preprocessing complete!")
    print("\nNext steps:")
    print("  1. Open notebook: jupyter notebook notebooks/finetuning_dogs.ipynb")
    print("  2. Or train directly: python scripts/train.py")


if __name__ == '__main__':
    main()
