"""
Запуск экспериментов для Denoising Autoencoder
Тестирует разные комбинации типов шума и функций потерь
"""

import subprocess
import sys
from pathlib import Path
import json


def check_model_exists(project_root, noise_type, loss_fn):
    """
    Проверить существует ли уже обученная модель для данной конфигурации

    Args:
        project_root: Корень проекта
        noise_type: Тип шума
        loss_fn: Функция потерь

    Returns:
        True если модель существует, False иначе
    """
    models_dir = project_root / 'models'
    if not models_dir.exists():
        return False

    # Ищем файлы модели с данной конфигурацией
    pattern = f"{noise_type}_{loss_fn}_*_best.pth"
    existing_models = list(models_dir.glob(pattern))

    return len(existing_models) > 0


def run_experiment(noise_type, loss_fn, num_epochs=30, batch_size=8, skip_existing=True):
    """
    Запустить один эксперимент

    Args:
        noise_type: Тип шума
        loss_fn: Функция потерь ('mse' или 'mae')
        num_epochs: Количество эпох
        batch_size: Размер батча
        skip_existing: Пропускать ли уже обученные модели

    Returns:
        Код возврата процесса (0 = успех, -1 = пропущен)
    """
    print("\n" + "=" * 80)
    print(f"Эксперимент: {noise_type} + {loss_fn.upper()}")
    print("=" * 80)

    # Запускаем из корня проекта, не из scripts/
    project_root = Path(__file__).parent.parent

    # Проверить существует ли уже модель
    if skip_existing and check_model_exists(project_root, noise_type, loss_fn):
        print(f"⏭️  Модель уже существует. Пропускаем обучение.")
        print(f"Найдена модель: {noise_type}_{loss_fn}_*_best.pth")
        return -1  # Специальный код для "пропущено"

    cmd = [
        sys.executable, 'scripts/train.py',
        '--noise_type', noise_type,
        '--loss_fn', loss_fn,
        '--num_epochs', str(num_epochs),
        '--batch_size', str(batch_size),
        '--lr', '0.001'
    ]

    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def main():
    """
    Запустить все эксперименты согласно ТЗ:
    - Гауссовский шум + MSE
    - Гауссовский шум + MAE
    - Импульсный шум + MSE
    - Импульсный шум + MAE
    - Смешанный шум + MSE
    - Смешанный шум + MAE
    """
    experiments = [
        # Гауссовский шум
        ('gaussian_medium', 'mse'),
        ('gaussian_medium', 'mae'),

        # Импульсный шум (Salt & Pepper)
        ('salt_pepper_medium', 'mse'),
        ('salt_pepper_medium', 'mae'),

        # Смешанный шум
        ('mixed_medium', 'mse'),
        ('mixed_medium', 'mae'),
    ]

    print("Запуск экспериментов для Denoising Autoencoder")
    print(f"Всего экспериментов: {len(experiments)}")
    print()

    results = []

    for noise_type, loss_fn in experiments:
        returncode = run_experiment(
            noise_type=noise_type,
            loss_fn=loss_fn,
            num_epochs=30,
            batch_size=8,
            skip_existing=True
        )

        # -1 = пропущен (уже обучен), 0 = успех, >0 = ошибка
        skipped = returncode == -1
        success = returncode == 0

        results.append({
            'noise_type': noise_type,
            'loss_fn': loss_fn,
            'success': success,
            'skipped': skipped
        })

        if returncode > 0:
            print(f"\n❌ Эксперимент {noise_type} + {loss_fn} завершился с ошибкой!")
            print(f"Код ошибки: {returncode}")
        elif skipped:
            print(f"\n⏭️  Эксперимент {noise_type} + {loss_fn} пропущен (уже обучен)")
        else:
            print(f"\n✓ Эксперимент {noise_type} + {loss_fn} завершен успешно!")

    # Сводка
    print("\n" + "=" * 80)
    print("СВОДКА ЭКСПЕРИМЕНТОВ")
    print("=" * 80)

    successful = sum(1 for r in results if r['success'])
    skipped_count = sum(1 for r in results if r['skipped'])
    failed = sum(1 for r in results if not r['success'] and not r['skipped'])

    print(f"Успешно завершено: {successful}/{len(results)}")
    print(f"Пропущено (уже обучены): {skipped_count}/{len(results)}")
    print(f"Ошибок: {failed}/{len(results)}")
    print()

    for r in results:
        if r['skipped']:
            status = "⏭️"
        elif r['success']:
            status = "✓"
        else:
            status = "❌"
        print(f"{status} {r['noise_type']} + {r['loss_fn'].upper()}")

    # Сохранить сводку
    project_root = Path(__file__).parent.parent
    summary_path = project_root / 'results' / 'experiments_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nСводка сохранена: {summary_path}")


if __name__ == '__main__':
    main()
