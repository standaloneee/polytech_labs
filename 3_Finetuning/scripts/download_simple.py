"""
Простой скрипт для скачивания Stanford Dogs через прямую ссылку
Использует только стандартную библиотеку Python
"""

import os
import urllib.request
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm


# Выбранные породы (ImageNet синтаксис)
TARGET_BREEDS = {
    'golden_retriever': 'n02099601',
    'german_shepherd': 'n02106662',
    'beagle': 'n02088364',
    'boxer': 'n02108089',
    'poodle': ['n02113624', 'n02113712', 'n02113799']  # toy, miniature, standard
}

# URL Stanford Dogs
IMAGES_URL = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'

# Пути
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
TEMP_DIR = DATA_DIR / 'temp'


def download_file(url, destination):
    """Скачать файл с прогресс-баром"""

    class ProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with ProgressBar(unit='B', unit_scale=True, desc=destination.name) as t:
        urllib.request.urlretrieve(url, destination, reporthook=t.update_to)

    print(f"✓ Downloaded: {destination}")


def verify_archive(tar_path):
    """Проверить целостность архива"""
    try:
        print(f"Verifying {tar_path.name}...")
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()
            print(f"✓ Valid archive ({len(members)} files)")
            return True
    except Exception as e:
        print(f"✗ Corrupted: {e}")
        return False


def extract_and_organize():
    """Распаковать и выбрать нужные породы"""

    tar_path = TEMP_DIR / 'images.tar'

    # Скачать если нет
    if not tar_path.exists():
        print(f"\nDownloading {tar_path.name} (~750MB)...")
        download_file(IMAGES_URL, tar_path)

    # Проверить
    if not verify_archive(tar_path):
        print("Removing corrupted archive...")
        tar_path.unlink()
        return False

    # Распаковать
    print(f"\nExtracting to {TEMP_DIR}...")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(TEMP_DIR, filter='data')
    print("✓ Extracted")

    # Найти папку Images
    images_dir = TEMP_DIR / 'Images'
    if not images_dir.exists():
        # Поиск альтернативных путей
        for path in TEMP_DIR.rglob('Images'):
            if path.is_dir():
                images_dir = path
                break

    if not images_dir.exists():
        print("✗ Images directory not found")
        return False

    # Выбрать породы
    print(f"\nSelecting breeds from {len(list(images_dir.iterdir()))} available...")

    selected = 0
    for breed_name, breed_ids in TARGET_BREEDS.items():
        if isinstance(breed_ids, str):
            breed_ids = [breed_ids]

        for breed_id in breed_ids:
            # Найти папку породы
            breed_dirs = list(images_dir.glob(f'{breed_id}*'))

            if breed_dirs:
                source = breed_dirs[0]
                dest = RAW_DIR / breed_name

                if dest.exists():
                    shutil.rmtree(dest)

                shutil.copytree(source, dest)
                num_images = len(list(dest.glob('*.jpg')))
                print(f"✓ {breed_name}: {num_images} images")
                selected += 1
                break

    if selected == 0:
        print("\n✗ No breeds found. Available:")
        for d in sorted(list(images_dir.iterdir())[:10]):
            print(f"  {d.name}")

    return selected > 0


def main():
    print("=" * 70)
    print("Stanford Dogs Dataset - Simple Downloader")
    print("=" * 70)

    # Создать директории
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Скачать и организовать
    if not extract_and_organize():
        print("\n✗ Failed to prepare dataset")
        return

    # Очистка (оставляем архив, удаляем только распакованные файлы)
    images_dir = TEMP_DIR / 'Images'
    if images_dir.exists():
        print(f"\nCleaning up extracted files...")
        shutil.rmtree(images_dir)
        print("✓ Cleanup complete (archive preserved)")

    print("\n" + "=" * 70)
    print("✓ Dataset ready!")
    print("=" * 70)
    print(f"Breeds saved to: {RAW_DIR}")
    print("\nNext step: python scripts/preprocess.py")


if __name__ == '__main__':
    main()
