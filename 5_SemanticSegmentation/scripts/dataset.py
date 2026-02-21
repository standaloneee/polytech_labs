"""
Модуль для работы с датасетом DRIVE (Digital Retinal Images for Vessel Extraction).

Обеспечивает автоматическое скачивание, загрузку изображений сетчатки глаза
и бинарных масок сосудов, а также аугментации для обучения.
"""

import os
import sys
import zipfile
import shutil
import requests
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

import cv2


# Пути по умолчанию
DATA_DIR = Path(__file__).parent.parent / 'data' / 'DRIVE'

# Размер изображений для модели (кратный 32 для U-Net)
IMAGE_SIZE = 512

# URL для скачивания датасета DRIVE
# Репозиторий содержит полный DRIVE: 20 train + 20 test (.tif + .gif маски)
DRIVE_URL = 'https://github.com/IstiaqAnsari/Retinal-Vessel-Segmentation/archive/refs/heads/master.zip'


def _reorganize_drive(data_dir):
    """
    Приводит структуру из репозитория к стандартной DRIVE:

    Исходная (в репо):
        data/train/images/  data/train/vein/   data/train/mask/
        data/test/images/   data/test/1st_manual/  data/test/mask/

    Целевая (стандартная DRIVE):
        training/images/    training/1st_manual/   training/mask/
        test/images/        test/1st_manual/       test/mask/
    """
    src_data = data_dir / 'data'
    if not src_data.exists():
        return

    # Training: data/train → training, vein → 1st_manual
    src_train = src_data / 'train'
    dst_train = data_dir / 'training'
    if src_train.exists():
        dst_train.mkdir(parents=True, exist_ok=True)

        # images
        src_imgs = src_train / 'images'
        dst_imgs = dst_train / 'images'
        if src_imgs.exists():
            if dst_imgs.exists():
                shutil.rmtree(dst_imgs)
            shutil.move(str(src_imgs), str(dst_imgs))

        # vein → 1st_manual (маски сосудов)
        src_vein = src_train / 'vein'
        dst_manual = dst_train / '1st_manual'
        if src_vein.exists():
            if dst_manual.exists():
                shutil.rmtree(dst_manual)
            shutil.move(str(src_vein), str(dst_manual))

        # mask (маски FOV)
        src_mask = src_train / 'mask'
        dst_mask = dst_train / 'mask'
        if src_mask.exists():
            if dst_mask.exists():
                shutil.rmtree(dst_mask)
            shutil.move(str(src_mask), str(dst_mask))

    # Test: data/test → test
    src_test = src_data / 'test'
    dst_test = data_dir / 'test'
    if src_test.exists():
        if dst_test.exists():
            shutil.rmtree(dst_test)
        shutil.move(str(src_test), str(dst_test))

    # Удаляем исходную data/
    if src_data.exists():
        shutil.rmtree(src_data)


def download_drive(data_dir=None):
    """
    Скачивает датасет DRIVE.

    Args:
        data_dir: путь для сохранения данных
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)

    # Проверяем, есть ли уже данные
    train_dir = data_dir / 'training' / 'images'
    test_dir = data_dir / 'test' / 'images'
    if train_dir.exists() and test_dir.exists():
        train_count = len(list(train_dir.glob('*.*')))
        test_count = len(list(test_dir.glob('*.*')))
        if train_count >= 20 and test_count >= 20:
            print(f"Датасет DRIVE уже скачан: {train_count} train, {test_count} test")
            return

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / 'drive.zip'

    # Скачиваем архив
    print(f"Скачивание DRIVE датасета...")
    try:
        response = requests.get(DRIVE_URL, stream=True, timeout=180)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = downloaded / total_size * 100
                    print(f"\r  Прогресс: {pct:.1f}%", end='', flush=True)

        print(f"\n  Скачано: {zip_path} ({downloaded / 1024 / 1024:.1f} MB)")
    except Exception as e:
        if zip_path.exists():
            zip_path.unlink()
        raise RuntimeError(
            f"Не удалось скачать датасет DRIVE: {e}\n"
            "Скачайте вручную и поместите в: " + str(data_dir)
        )

    # Распаковываем
    print("Распаковка архива...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)

    # Находим распакованную папку (например, Retinal-Vessel-Segmentation-master/)
    # и переносим содержимое на уровень data_dir
    extracted_dirs = [
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name not in ('training', 'test')
    ]
    if extracted_dirs:
        src = extracted_dirs[0]
        # Внутри может быть подпапка data/ с данными
        for item in src.iterdir():
            dst = data_dir / item.name
            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            shutil.move(str(item), str(dst))
        shutil.rmtree(src)

    # Приводим к стандартной структуре DRIVE
    _reorganize_drive(data_dir)

    # Удаляем архив и лишние файлы
    if zip_path.exists():
        zip_path.unlink()
    for f in data_dir.glob('*.py'):
        f.unlink()
    for f in data_dir.glob('*.ipynb'):
        f.unlink()
    for f in data_dir.glob('*.md'):
        f.unlink()
    for f in data_dir.glob('.git*'):
        if f.is_dir():
            shutil.rmtree(f)
        else:
            f.unlink()

    # Проверяем результат
    train_count = len(list((data_dir / 'training' / 'images').glob('*.*'))) if (data_dir / 'training' / 'images').exists() else 0
    test_count = len(list((data_dir / 'test' / 'images').glob('*.*'))) if (data_dir / 'test' / 'images').exists() else 0
    print(f"Датасет DRIVE готов: {train_count} train, {test_count} test")


def load_image(path, is_mask=False):
    """
    Загружает изображение или маску.

    Args:
        path: путь к файлу
        is_mask: если True, загружает как бинарную маску

    Returns:
        numpy array (H, W, C) для изображений или (H, W) для масок
    """
    img = Image.open(path)

    if is_mask:
        # Маски могут быть в формате .gif или .tif, конвертируем в grayscale
        img = img.convert('L')
        arr = np.array(img)
        # Бинаризация: > 128 = 1, остальное = 0
        arr = (arr > 128).astype(np.float32)
        return arr
    else:
        img = img.convert('RGB')
        return np.array(img).astype(np.float32) / 255.0


class DRIVEDataset(Dataset):
    """
    Dataset для загрузки изображений DRIVE.

    Args:
        data_dir: путь к корню DRIVE
        split: 'train', 'val' или 'test'
        image_size: размер выходного изображения
        augment: применять аугментации
        val_indices: индексы изображений для валидации (из training set)
    """

    def __init__(self, data_dir=None, split='train', image_size=IMAGE_SIZE,
                 augment=False, val_indices=None):
        super().__init__()

        if data_dir is None:
            data_dir = DATA_DIR
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')

        self.images = []
        self.masks = []

        if split in ('train', 'val'):
            self._load_training_split(val_indices)
        else:
            self._load_test_split()

    def _load_training_split(self, val_indices=None):
        """Загрузка training данных с разделением на train/val."""
        img_dir = self.data_dir / 'training' / 'images'
        mask_dir = self.data_dir / 'training' / '1st_manual'

        if not img_dir.exists():
            raise FileNotFoundError(
                f"Директория не найдена: {img_dir}\n"
                "Запустите download_drive() для скачивания датасета."
            )

        # Собираем пары (изображение, маска)
        image_files = sorted(img_dir.glob('*.tif'))
        if not image_files:
            image_files = sorted(img_dir.glob('*.*'))

        all_pairs = []
        for img_path in image_files:
            # Извлекаем номер изображения (например, 21 из 21_training.tif)
            img_num = img_path.stem.split('_')[0]

            # Ищем соответствующую маску
            mask_candidates = list(mask_dir.glob(f'{img_num}_manual1.*'))
            if not mask_candidates:
                mask_candidates = list(mask_dir.glob(f'{img_num}*.*'))

            if mask_candidates:
                all_pairs.append((img_path, mask_candidates[0]))

        if not all_pairs:
            raise FileNotFoundError(
                f"Не найдены пары изображение-маска в {img_dir}"
            )

        # Разделяем на train/val
        if val_indices is None:
            # По умолчанию: последние 4 из 20 → val
            val_indices = list(range(16, len(all_pairs)))

        if self.split == 'val':
            pairs = [all_pairs[i] for i in val_indices if i < len(all_pairs)]
        else:
            pairs = [all_pairs[i] for i in range(len(all_pairs)) if i not in val_indices]

        for img_path, mask_path in pairs:
            self.images.append(img_path)
            self.masks.append(mask_path)

        print(f"  {self.split}: {len(self.images)} изображений")

    def _load_test_split(self):
        """Загрузка test данных."""
        img_dir = self.data_dir / 'test' / 'images'
        mask_dir = self.data_dir / 'test' / '1st_manual'

        if not img_dir.exists():
            raise FileNotFoundError(
                f"Директория не найдена: {img_dir}\n"
                "Запустите download_drive() для скачивания датасета."
            )

        image_files = sorted(img_dir.glob('*.tif'))
        if not image_files:
            image_files = sorted(img_dir.glob('*.*'))

        for img_path in image_files:
            img_num = img_path.stem.split('_')[0]
            mask_candidates = list(mask_dir.glob(f'{img_num}_manual1.*'))
            if not mask_candidates:
                mask_candidates = list(mask_dir.glob(f'{img_num}*.*'))

            if mask_candidates:
                self.images.append(img_path)
                self.masks.append(mask_candidates[0])

        print(f"  {self.split}: {len(self.images)} изображений")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Возвращает пару (изображение, маска).

        Returns:
            image: тензор (3, H, W) float32 [0, 1]
            mask: тензор (1, H, W) float32 {0, 1}
        """
        image = load_image(self.images[idx], is_mask=False)
        mask = load_image(self.masks[idx], is_mask=True)

        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size),
                          interpolation=cv2.INTER_NEAREST)

        # Аугментации
        if self.augment:
            image, mask = self._apply_augmentations(image, mask)

        # Конвертация в тензоры
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()  # (3, H, W)
        mask = torch.from_numpy(mask[np.newaxis, :, :]).float()      # (1, H, W)

        return image, mask

    def _apply_augmentations(self, image, mask):
        """
        Применяет аугментации к паре изображение-маска.

        Используемые аугментации:
        - Горизонтальное отражение (p=0.5)
        - Вертикальное отражение (p=0.5)
        - Случайный поворот на 0/90/180/270 градусов
        - Яркость и контрастность (только изображение)
        - Эластическая деформация
        """
        # Горизонтальное отражение
        if np.random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # Вертикальное отражение
        if np.random.random() < 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        # Случайный поворот на 90 градусов
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()

        # Яркость и контрастность (только для изображения)
        if np.random.random() < 0.3:
            # Яркость
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 1)

        if np.random.random() < 0.3:
            # Контрастность
            mean = image.mean()
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip((image - mean) * factor + mean, 0, 1)

        # Гауссов шум (лёгкий)
        if np.random.random() < 0.2:
            noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0, 1)

        return image, mask


def create_dataloaders(data_dir=None, batch_size=4, image_size=IMAGE_SIZE,
                       augment=False, num_workers=2, val_indices=None):
    """
    Создаёт DataLoader'ы для train, val и test.

    Args:
        data_dir: путь к датасету DRIVE
        batch_size: размер батча
        image_size: размер изображений
        augment: использовать аугментации для train
        num_workers: количество процессов загрузки
        val_indices: индексы для валидации

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if data_dir is None:
        data_dir = DATA_DIR

    print("Создание датасетов...")

    train_dataset = DRIVEDataset(
        data_dir=data_dir, split='train', image_size=image_size,
        augment=augment, val_indices=val_indices
    )
    val_dataset = DRIVEDataset(
        data_dir=data_dir, split='val', image_size=image_size,
        augment=False, val_indices=val_indices
    )
    test_dataset = DRIVEDataset(
        data_dir=data_dir, split='test', image_size=image_size,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Тест загрузки данных
    print("Тестирование модуля dataset...")
    download_drive()

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=2, augment=True
    )

    for images, masks in train_loader:
        print(f"Батч изображений: {images.shape}, мин={images.min():.3f}, макс={images.max():.3f}")
        print(f"Батч масок: {masks.shape}, уникальные значения: {torch.unique(masks).tolist()}")
        break
