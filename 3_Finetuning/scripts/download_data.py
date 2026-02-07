"""
Скрипт для скачивания датасета Stanford Dogs Dataset

Stanford Dogs Dataset содержит 120 пород собак, ~150 изображений на породу.
Мы выберем 5-7 наиболее различимых пород для классификации.

Источник: http://vision.stanford.edu/aditya86/ImageNetDogs/
"""

import os
import urllib.request
import tarfile
from pathlib import Path
import shutil
from tqdm import tqdm


# Выбранные породы (ImageNet ID -> Readable name)
TARGET_BREEDS = {
    'golden_retriever': 'n02099601',
    'german_shepherd': 'n02106662',
    'beagle': 'n02088364',
    'boxer': 'n02108089',
    'poodle': ['n02113624', 'n02113712', 'n02113799']  # toy, miniature, standard
}

# URLs для Stanford Dogs Dataset
IMAGES_URL = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'

# Пути
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
TEMP_DIR = DATA_DIR / 'temp'


def download_file(url: str, destination: Path, desc: str = "Downloading"):
    """Скачать файл с прогресс-баром"""

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, destination, reporthook=t.update_to)

    print(f"✓ Downloaded: {destination}")


def verify_tar(tar_path: Path) -> bool:
    """
    Проверить целостность tar архива

    Returns:
        True если файл валидный, False если поврежден
    """
    try:
        print(f"Verifying archive: {tar_path.name}")
        with tarfile.open(tar_path, 'r') as tar:
            # Попытка прочитать список файлов
            members = tar.getmembers()
            print(f"✓ Archive is valid ({len(members)} files)")
            return True
    except (tarfile.ReadError, EOFError, OSError) as e:
        print(f"✗ Archive is corrupted: {e}")
        return False


def extract_tar(tar_path: Path, extract_to: Path, desc: str = "Extracting"):
    """Распаковать tar архив"""
    print(f"{desc}: {tar_path.name}")

    try:
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(extract_to, filter='data')
        print(f"✓ Extracted to: {extract_to}")
        return True
    except (tarfile.ReadError, EOFError, OSError) as e:
        print(f"✗ Extraction failed: {e}")
        return False


def organize_breeds():
    """Организовать выбранные породы в структуру папок"""
    print("\nOrganizing selected breeds...")

    images_dir = TEMP_DIR / 'Images'

    if not images_dir.exists():
        print(f"✗ Images directory not found: {images_dir}")
        return False

    selected_count = 0

    for breed_name, breed_ids in TARGET_BREEDS.items():
        if isinstance(breed_ids, str):
            breed_ids = [breed_ids]

        for breed_id in breed_ids:
            # Найти папки с этим ID
            breed_dirs = list(images_dir.glob(f'{breed_id}*'))

            if breed_dirs:
                source = breed_dirs[0]
                destination = RAW_DIR / breed_name

                if destination.exists():
                    shutil.rmtree(destination)

                shutil.copytree(source, destination)
                num_images = len(list(destination.glob('*.jpg')))
                print(f"✓ {breed_name}: {num_images} images")
                selected_count += 1
                break

    if selected_count == 0:
        print("\n⚠️ No breeds found. Available breeds:")
        for d in sorted(list(images_dir.iterdir())[:20]):
            print(f"  {d.name}")
        return False

    print(f"\n✓ Selected {selected_count} breeds saved to: {RAW_DIR}")
    return True


def cleanup_temp():
    """Удалить только распакованные файлы, архив оставить"""
    images_dir = TEMP_DIR / 'Images'

    if images_dir.exists():
        print(f"\nCleaning up extracted files (preserving archive)...")
        shutil.rmtree(images_dir)
        print("✓ Cleanup complete")
        print(f"✓ Archive preserved at: {TEMP_DIR / 'images.tar'}")
    else:
        print("\n✓ No cleanup needed")


def main():
    """Основная функция скачивания датасета"""

    print("=" * 70)
    print("Stanford Dogs Dataset Downloader")
    print("=" * 70)

    # Создать директории
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Пути для архивов
    images_tar = TEMP_DIR / 'images.tar'

    # 1. Скачать images.tar (~750MB) с проверкой целостности
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        print(f"\n[1/3] Checking/Downloading images (~750MB) - Attempt {attempt}/{max_retries}")

        # Если файл существует - проверить его
        if images_tar.exists():
            print(f"Archive exists, verifying...")
            if verify_tar(images_tar):
                print("✓ Using existing valid archive")
                break
            else:
                print("✗ Archive is corrupted, removing and re-downloading...")
                images_tar.unlink()

        # Скачать файл
        try:
            download_file(IMAGES_URL, images_tar, desc="Images")

            # Проверить скачанный файл
            if verify_tar(images_tar):
                print("✓ Download successful and verified")
                break
            else:
                print("✗ Downloaded file is corrupted")
                images_tar.unlink()
                if attempt == max_retries:
                    print("\n✗ Failed to download valid archive after 3 attempts")
                    print("Please check your internet connection and try again")
                    return
        except Exception as e:
            print(f"✗ Download error: {e}")
            if images_tar.exists():
                images_tar.unlink()
            if attempt == max_retries:
                print("\n✗ Failed to download after 3 attempts")
                return

    # 2. Распаковать
    print("\n[2/3] Extracting archives...")
    if not (TEMP_DIR / 'Images').exists():
        if not extract_tar(images_tar, TEMP_DIR, desc="Extracting images"):
            print("\n✗ Failed to extract archive")
            print("The archive may be corrupted. Please delete it and try again:")
            print(f"  rm {images_tar}")
            return
    else:
        print("✓ Images already extracted")

    # 3. Организовать выбранные породы
    print("\n[3/3] Organizing selected breeds...")
    if not organize_breeds():
        return

    # 4. Очистка (сохраняем архив!)
    cleanup_temp()

    print("\n" + "=" * 70)
    print("✓ Dataset download complete!")
    print("=" * 70)
    print(f"\nSelected breeds saved to: {RAW_DIR}")
    print("\nNext steps:")
    print("  1. Run: python scripts/preprocess.py")
    print("  2. Train models: python scripts/train.py")


if __name__ == '__main__':
    main()
