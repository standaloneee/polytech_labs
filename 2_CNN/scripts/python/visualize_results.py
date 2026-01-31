"""
Визуализация всех результатов экспериментов CNN

Создает комплексные графики и таблицы для анализа:
- Сравнение всех архитектур
- Влияние аугментации
- Confusion matrix
- Примеры предсказаний
- Сводная таблица результатов
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import json
import os
import medmnist
from medmnist import INFO


# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Определение устройства
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f'Device: {device}\n')


# ===========================================================================
# Загрузка всех результатов
# ===========================================================================

print('=' * 70)
print('ЗАГРУЗКА РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТОВ')
print('=' * 70)
print()

results_dir = 'results/experiments_results'

# Загрузка результатов
baseline_results = {'name': 'Baseline CNN', 'test_acc': 89.96}

try:
    with open(f'{results_dir}/architecture_experiments.json', 'r') as f:
        arch_results = json.load(f)
    print(f'✓ Загружены результаты архитектурных экспериментов: {len(arch_results)} моделей')
except FileNotFoundError:
    print('⚠ Архитектурные эксперименты не найдены')
    arch_results = []

try:
    with open(f'{results_dir}/augmentation_experiments.json', 'r') as f:
        aug_results = json.load(f)
    print(f'✓ Загружены результаты экспериментов с аугментацией: {len(aug_results)} моделей')
except FileNotFoundError:
    print('⚠ Эксперименты с аугментацией не найдены')
    aug_results = []

try:
    with open(f'{results_dir}/ensemble_results.json', 'r') as f:
        ensemble_data = json.load(f)
    print(f'✓ Загружены результаты ансамбля: {ensemble_data["num_models"]} моделей')
    ensemble_results = {'name': 'Ensemble', 'test_acc': ensemble_data['test_ensemble_acc']}
except FileNotFoundError:
    print('⚠ Результаты ансамбля не найдены')
    ensemble_results = None

print()


# ===========================================================================
# 1. Сравнение всех архитектур
# ===========================================================================

if arch_results:
    print('Создание графика сравнения архитектур...')

    fig, ax = plt.subplots(figsize=(14, 8))

    # Сортировка по test accuracy
    sorted_results = sorted(arch_results, key=lambda x: x['test_acc'], reverse=True)

    names = [r['name'] for r in sorted_results]
    test_accs = [r['test_acc'] for r in sorted_results]
    val_accs = [r['best_val_acc'] for r in sorted_results]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, val_accs, width, label='Validation', color='steelblue')
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test', color='darkgreen')

    # Baseline линия
    ax.axhline(y=89.96, color='red', linestyle='--', linewidth=2,
              label='Baseline CNN (89.96%)', alpha=0.7)
    ax.axhline(y=87.7, color='orange', linestyle='--', linewidth=2,
              label='ResNet-18 Paper (87.7%)', alpha=0.7)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Сравнение CNN архитектур', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{results_dir}/all_architectures_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ Сохранено: all_architectures_comparison.png')


# ===========================================================================
# 2. Влияние аугментации
# ===========================================================================

if aug_results:
    print('Создание графика влияния аугментации...')

    # Группировка по типу аугментации
    aug_types = {}
    for res in aug_results:
        aug_type = res['aug_type']
        if aug_type not in aug_types:
            aug_types[aug_type] = []
        aug_types[aug_type].append(res['test_acc'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # График 1: Средняя точность по типу аугментации
    aug_names = list(aug_types.keys())
    aug_means = [np.mean(aug_types[name]) for name in aug_names]
    aug_stds = [np.std(aug_types[name]) if len(aug_types[name]) > 1 else 0
                for name in aug_names]

    bars = ax1.bar(aug_names, aug_means, yerr=aug_stds, capsize=5, color='teal', alpha=0.7)
    ax1.axhline(y=89.96, color='red', linestyle='--', linewidth=2, label='Baseline (89.96%)')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Влияние аугментации на точность', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Добавить значения
    for bar, mean in zip(bars, aug_means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{mean:.2f}%', ha='center', va='bottom')

    # График 2: Все модели с аугментацией
    sorted_aug = sorted(aug_results, key=lambda x: x['test_acc'], reverse=True)
    names = [r['name'] for r in sorted_aug[:10]]  # Top 10
    accs = [r['test_acc'] for r in sorted_aug[:10]]
    colors = ['lightcoral' if 'No Aug' in name else 'steelblue' for name in names]

    bars = ax2.barh(names, accs, color=colors)
    ax2.axvline(x=89.96, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax2.set_xlabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Топ-10 моделей с аугментацией', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)

    # Добавить значения
    for bar, acc in zip(bars, accs):
        width = bar.get_width()
        ax2.text(width + 0.2, bar.get_y() + bar.get_height()/2.,
                f'{acc:.2f}%', ha='left', va='center')

    plt.tight_layout()
    plt.savefig(f'{results_dir}/augmentation_impact.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ Сохранено: augmentation_impact.png')


# ===========================================================================
# 3. Сводная таблица всех результатов
# ===========================================================================

print('Создание сводной таблицы результатов...')

all_results = [baseline_results]

if arch_results:
    all_results.extend(arch_results[:5])  # Top 5 архитектур

if aug_results:
    top_aug = sorted(aug_results, key=lambda x: x['test_acc'], reverse=True)[:5]
    all_results.extend(top_aug)

if ensemble_results:
    all_results.append(ensemble_results)

# Создание DataFrame
df_data = []
for res in all_results:
    df_data.append({
        'Model': res['name'],
        'Test Accuracy': f"{res['test_acc']:.2f}%",
        'Params': f"{res.get('params', 421771):,}" if 'params' in res else 'N/A',
        'Improvement': f"+{res['test_acc'] - 89.96:.2f}%" if res['test_acc'] > 89.96 else f"{res['test_acc'] - 89.96:.2f}%"
    })

df = pd.DataFrame(df_data)

# Сохранение таблицы
df.to_csv(f'{results_dir}/summary_table.csv', index=False)
print(f'  ✓ Сохранено: summary_table.csv')

# Печать таблицы
print('\n' + '=' * 70)
print('СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ')
print('=' * 70)
print(df.to_string(index=False))
print()


# ===========================================================================
# 4. График прогресса от MLP к CNN Ensemble
# ===========================================================================

print('Создание графика прогресса...')

milestones = [
    {'name': 'MLP Baseline', 'acc': 74.84, 'color': 'lightcoral'},
    {'name': 'MLP Best', 'acc': 79.93, 'color': 'coral'},
    {'name': 'MLP Ensemble', 'acc': 80.83, 'color': 'orangered'},
    {'name': 'ResNet-18 (Paper)', 'acc': 87.7, 'color': 'gold'},
    {'name': 'CNN Baseline', 'acc': 89.96, 'color': 'steelblue'},
]

if arch_results:
    best_arch = max(arch_results, key=lambda x: x['test_acc'])
    milestones.append({'name': f"Best CNN\n({best_arch['name'][:20]})",
                      'acc': best_arch['test_acc'], 'color': 'dodgerblue'})

if aug_results:
    best_aug = max(aug_results, key=lambda x: x['test_acc'])
    milestones.append({'name': f"Best with Aug\n({best_aug['name'][:20]})",
                      'acc': best_aug['test_acc'], 'color': 'royalblue'})

if ensemble_results:
    milestones.append({'name': 'CNN Ensemble', 'acc': ensemble_results['test_acc'],
                      'color': 'darkgreen'})

fig, ax = plt.subplots(figsize=(14, 8))

names = [m['name'] for m in milestones]
accs = [m['acc'] for m in milestones]
colors = [m['color'] for m in milestones]

bars = ax.bar(names, accs, color=colors, edgecolor='black', linewidth=1.5)

# Линии улучшения
for i in range(len(accs) - 1):
    ax.plot([i, i+1], [accs[i], accs[i+1]], 'k--', alpha=0.3, linewidth=2)

ax.set_ylabel('Test Accuracy (%)', fontsize=14)
ax.set_title('Прогресс: От MLP к CNN Ensemble', fontsize=16, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(70, max(accs) + 3)

# Добавить значения и улучшения
for i, (bar, acc) in enumerate(zip(bars, accs)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
           f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    if i > 0:
        improvement = acc - accs[i-1]
        ax.text(bar.get_x() + bar.get_width()/2., height - 2,
               f'+{improvement:.2f}%', ha='center', va='top',
               fontsize=9, color='white', fontweight='bold')

plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(f'{results_dir}/progress_mlp_to_cnn.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✓ Сохранено: progress_mlp_to_cnn.png')


# ===========================================================================
# 5. Confusion Matrix (для лучшей модели, если доступна)
# ===========================================================================

# Эта часть требует обученную модель, пока создаем placeholder
print('Создание placeholder для confusion matrix...')

fig, ax = plt.subplots(figsize=(12, 10))

# Placeholder confusion matrix
data_flag = 'organcmnist'
info = INFO[data_flag]
NUM_CLASSES = len(info['label'])

# Создаем случайную матрицу для демонстрации структуры
np.random.seed(42)
cm = np.random.randint(0, 100, (NUM_CLASSES, NUM_CLASSES))
np.fill_diagonal(cm, np.random.randint(800, 1000, NUM_CLASSES))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=[info['label'][str(i)] for i in range(NUM_CLASSES)],
           yticklabels=[info['label'][str(i)] for i in range(NUM_CLASSES)],
           ax=ax)

ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('True', fontsize=12)
ax.set_title('Confusion Matrix (Placeholder)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{results_dir}/confusion_matrix_placeholder.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✓ Сохранено: confusion_matrix_placeholder.png')


# ===========================================================================
# 6. Сравнение с бенчмарками из статьи
# ===========================================================================

print('Создание графика сравнения с бенчмарками...')

benchmarks = [
    {'name': 'ResNet-18\n(Paper)', 'acc': 87.7, 'source': 'MedMNIST v2'},
    {'name': 'OrganAMNIST\n(Paper)', 'acc': 90.7, 'source': 'MedMNIST v2'},
    {'name': 'CNN Baseline\n(Ours)', 'acc': 89.96, 'source': 'This Work'},
]

if arch_results:
    best_arch = max(arch_results, key=lambda x: x['test_acc'])
    benchmarks.append({'name': f"Best CNN\n(Ours)", 'acc': best_arch['test_acc'],
                      'source': 'This Work'})

if ensemble_results:
    benchmarks.append({'name': 'CNN Ensemble\n(Ours)',
                      'acc': ensemble_results['test_acc'],
                      'source': 'This Work'})

fig, ax = plt.subplots(figsize=(12, 7))

names = [b['name'] for b in benchmarks]
accs = [b['acc'] for b in benchmarks]
colors = ['gold' if b['source'] == 'MedMNIST v2' else 'steelblue' for b in benchmarks]

bars = ax.bar(names, accs, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

ax.set_ylabel('Test Accuracy (%)', fontsize=14)
ax.set_title('Сравнение с бенчмарками MedMNIST v2', fontsize=16, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(85, max(accs) + 2)

# Добавить значения
for bar, acc in zip(bars, accs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
           f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Легенда
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='gold', edgecolor='black', label='MedMNIST v2 Paper'),
    Patch(facecolor='steelblue', edgecolor='black', label='This Work')
]
ax.legend(handles=legend_elements, fontsize=12)

plt.tight_layout()
plt.savefig(f'{results_dir}/benchmark_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✓ Сохранено: benchmark_comparison.png')


# ===========================================================================
# Итоговая статистика
# ===========================================================================

print('\n' + '=' * 70)
print('ИТОГОВАЯ СТАТИСТИКА')
print('=' * 70)

print('\nРезультаты проекта:')
print(f'  MLP (предыдущая работа):  80.83%')
print(f'  ResNet-18 (статья):       87.7%')
print(f'  CNN Baseline:             89.96%')

if arch_results:
    best_arch = max(arch_results, key=lambda x: x['test_acc'])
    print(f'  Best Architecture:        {best_arch["test_acc"]:.2f}%')

if aug_results:
    best_aug = max(aug_results, key=lambda x: x['test_acc'])
    print(f'  Best with Augmentation:   {best_aug["test_acc"]:.2f}%')

if ensemble_results:
    print(f'  CNN Ensemble:             {ensemble_results["test_acc"]:.2f}%')

print('\nУлучшения:')
print(f'  CNN vs MLP:               +{89.96 - 80.83:.2f}%')
print(f'  CNN vs ResNet-18:         +{89.96 - 87.7:.2f}%')

if ensemble_results:
    print(f'  Ensemble vs Baseline:     +{ensemble_results["test_acc"] - 89.96:.2f}%')
    print(f'  Ensemble vs MLP:          +{ensemble_results["test_acc"] - 80.83:.2f}%')

print('\n' + '=' * 70)
print('ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА')
print('=' * 70)
print(f'\nВсе графики сохранены в: {results_dir}/')
print('\nСозданные файлы:')
if arch_results:
    print('  ✓ all_architectures_comparison.png')
if aug_results:
    print('  ✓ augmentation_impact.png')
print('  ✓ summary_table.csv')
print('  ✓ progress_mlp_to_cnn.png')
print('  ✓ confusion_matrix_placeholder.png')
print('  ✓ benchmark_comparison.png')
