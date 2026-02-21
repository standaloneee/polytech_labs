"""
Скрипт для запуска всех 6 экспериментов семантической сегментации.

Эксперименты:
1. U-Net + BCE (без аугментаций)
2. U-Net + Dice (без аугментаций)
3. U-Net + BCE+Dice (без аугментаций)
4. U-Net + BCE+Dice (с аугментациями)
5. U-Net+ResNet34 + BCE+Dice (с аугментациями)
6. U-Net+ResNet34 + Dice (с аугментациями)
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime


# Конфигурации экспериментов
EXPERIMENTS = [
    {
        'name': 'exp1_unet_bce_noaug',
        'arch': 'unet',
        'loss': 'bce',
        'augment': False,
        'description': 'U-Net + BCE, без аугментаций',
    },
    {
        'name': 'exp2_unet_dice_noaug',
        'arch': 'unet',
        'loss': 'dice',
        'augment': False,
        'description': 'U-Net + Dice, без аугментаций',
    },
    {
        'name': 'exp3_unet_combined_noaug',
        'arch': 'unet',
        'loss': 'combined',
        'augment': False,
        'description': 'U-Net + BCE+Dice, без аугментаций',
    },
    {
        'name': 'exp4_unet_combined_aug',
        'arch': 'unet',
        'loss': 'combined',
        'augment': True,
        'description': 'U-Net + BCE+Dice, с аугментациями',
    },
    {
        'name': 'exp5_resnet34_combined_aug',
        'arch': 'resnet34',
        'loss': 'combined',
        'augment': True,
        'description': 'U-Net+ResNet34 + BCE+Dice, с аугментациями',
    },
    {
        'name': 'exp6_resnet34_dice_aug',
        'arch': 'resnet34',
        'loss': 'dice',
        'augment': True,
        'description': 'U-Net+ResNet34 + Dice, с аугментациями',
    },
]


def check_model_exists(experiment_name, models_dir):
    """Проверяет, существует ли уже обученная модель."""
    model_path = models_dir / f"{experiment_name}_best.pth"
    return model_path.exists()


def run_experiment(exp_config, scripts_dir, models_dir):
    """
    Запускает один эксперимент.

    Args:
        exp_config: словарь с конфигурацией
        scripts_dir: путь к папке scripts/
        models_dir: путь к папке models/

    Returns:
        int: код возврата (0 = успех, -1 = пропущен)
    """
    name = exp_config['name']

    # Проверяем, есть ли уже модель
    if check_model_exists(name, models_dir):
        print(f"  [ПРОПУСК] Модель уже существует: {name}_best.pth")
        return -1

    # Формируем команду
    cmd = [
        sys.executable, str(scripts_dir / 'train.py'),
        '--arch', exp_config['arch'],
        '--loss', exp_config['loss'],
        '--name', name,
        '--epochs', '50',
        '--batch-size', '4',
        '--lr', '1e-4',
        '--patience', '10',
    ]

    if exp_config['augment']:
        cmd.append('--augment')

    print(f"\n  Команда: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"  [ОШИБКА] {e}")
        return 1


def run_evaluations(scripts_dir, models_dir):
    """
    Запускает оценку всех обученных моделей.

    Args:
        scripts_dir: путь к папке scripts/
        models_dir: путь к папке models/
    """
    print("\n" + "=" * 60)
    print("ОЦЕНКА МОДЕЛЕЙ НА TEST SET")
    print("=" * 60)

    for exp in EXPERIMENTS:
        name = exp['name']
        model_path = models_dir / f"{name}_best.pth"

        if not model_path.exists():
            print(f"\n  [ПРОПУСК] Модель не найдена: {model_path}")
            continue

        print(f"\n--- Оценка: {exp['description']} ---")

        cmd = [
            sys.executable, str(scripts_dir / 'evaluate.py'),
            '--model', str(model_path),
        ]

        try:
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"  [ОШИБКА] {e}")


def main():
    """Запуск всех экспериментов."""
    project_dir = Path(__file__).parent.parent
    scripts_dir = Path(__file__).parent
    models_dir = project_dir / 'models'
    results_dir = project_dir / 'results' / 'metrics'

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ЗАПУСК ЭКСПЕРИМЕНТОВ СЕМАНТИЧЕСКОЙ СЕГМЕНТАЦИИ")
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Всего экспериментов: {len(EXPERIMENTS)}")
    print("=" * 60)

    results_summary = []

    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n{'='*60}")
        print(f"Эксперимент {i}/{len(EXPERIMENTS)}: {exp['description']}")
        print(f"  Архитектура: {exp['arch']}")
        print(f"  Loss: {exp['loss']}")
        print(f"  Аугментации: {'да' if exp['augment'] else 'нет'}")
        print(f"{'='*60}")

        returncode = run_experiment(exp, scripts_dir, models_dir)

        status = 'success' if returncode == 0 else ('skipped' if returncode == -1 else 'failed')
        results_summary.append({
            'experiment': exp['name'],
            'description': exp['description'],
            'status': status,
            'returncode': returncode,
        })

    # Запускаем оценку всех моделей
    run_evaluations(scripts_dir, models_dir)

    # Сохраняем сводку
    summary_path = results_dir / 'experiments_summary.json'
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experiments': results_summary,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("ИТОГИ")
    print(f"{'='*60}")
    for r in results_summary:
        status_icon = '✓' if r['status'] == 'success' else ('→' if r['status'] == 'skipped' else '✗')
        print(f"  {status_icon} {r['description']}: {r['status']}")

    print(f"\nСводка сохранена: {summary_path}")


if __name__ == '__main__':
    main()
