"""
Вспомогательный скрипт для повторной распаковки архива БЕЗ перезакачки
"""

import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TEMP_DIR = PROJECT_ROOT / 'data' / 'temp'
IMAGES_TAR = TEMP_DIR / 'images.tar'

# Создать temp
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Проверить нужно ли скачивать
if not IMAGES_TAR.exists():
    print("Archive not found. Need to download again:")
    print("  python scripts/download_data.py")
else:
    # Распаковать
    print(f"Extracting {IMAGES_TAR}...")
    with tarfile.open(IMAGES_TAR, 'r') as tar:
        tar.extractall(TEMP_DIR, filter='data')

    # Показать доступные породы
    images_dir = TEMP_DIR / 'Images'
    if images_dir.exists():
        breeds = sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
        print(f"\nFound {len(breeds)} breeds:")
        for i, breed in enumerate(breeds[:20], 1):
            num_images = len(list((images_dir / breed).glob('*.jpg')))
            print(f"  {i}. {breed} ({num_images} images)")

        if len(breeds) > 20:
            print(f"  ... and {len(breeds) - 20} more")
    else:
        print("Images directory not found after extraction")
