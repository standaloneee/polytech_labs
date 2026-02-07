"""
БЫСТРЫЙ вариант - скачивание только выбранных пород

Использует API https://dog.ceo/dog-api/ для получения реальных фото собак
Скачает по 100-150 фото каждой породы
"""

import requests
import urllib.request
from pathlib import Path
from tqdm import tqdm
import time


# Породы (Dog CEO API использует другие названия)
BREEDS = {
    'retriever-golden': 'golden_retriever',
    'germanshepherd': 'german_shepherd',
    'beagle': 'beagle',
    'boxer': 'boxer',
    'poodle-standard': 'poodle'
}

# Пути
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'


def download_breed_images(breed_api_name, breed_name, num_images=120):
    """
    Скачать изображения породы через Dog CEO API
    """
    print(f"\nDownloading {breed_name}...")

    # Создать папку
    breed_dir = RAW_DIR / breed_name
    breed_dir.mkdir(parents=True, exist_ok=True)

    # Получить список URL изображений
    url = f'https://dog.ceo/api/breed/{breed_api_name}/images'

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if data['status'] != 'success':
            print(f"✗ API error for {breed_name}")
            return 0

        image_urls = data['message'][:num_images]
        print(f"Found {len(image_urls)} images")

        # Скачать изображения
        success_count = 0
        for i, img_url in enumerate(tqdm(image_urls, desc=f"  {breed_name}")):
            try:
                img_name = f"{breed_name}_{i:04d}.jpg"
                img_path = breed_dir / img_name

                urllib.request.urlretrieve(img_url, img_path)
                success_count += 1

                # Пауза чтобы не перегружать API
                time.sleep(0.05)

            except Exception as e:
                print(f"  ✗ Failed to download image {i}: {e}")
                continue

        print(f"✓ {breed_name}: {success_count} images downloaded")
        return success_count

    except Exception as e:
        print(f"✗ Failed to fetch {breed_name}: {e}")
        return 0


def main():
    print("=" * 70)
    print("Dog Images Downloader - Fast Method")
    print("Using Dog CEO API: https://dog.ceo/dog-api/")
    print("=" * 70)

    # Создать директорию
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    total_images = 0

    # Скачать каждую породу
    for breed_api, breed_name in BREEDS.items():
        count = download_breed_images(breed_api, breed_name, num_images=120)
        total_images += count

    print("\n" + "=" * 70)
    print(f"✓ Downloaded {total_images} images total")
    print("=" * 70)
    print(f"Saved to: {RAW_DIR}")
    print("\nNext step:")
    print("  python scripts/preprocess.py")


if __name__ == '__main__':
    main()
