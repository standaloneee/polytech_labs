"""
Скрипт для создания итоговой визуализации результатов всех экспериментов
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Загрузка результатов
results_file = 'experiments_results/all_experiments.json'

if not Path(results_file).exists():
    print(f"Файл {results_file} не найден!")
    print("Сначала запустите experiments.py")
    exit(1)

with open(results_file, 'r') as f:
    data = json.load(f)

experiments = data['experiments']

print(f"Загружено {len(experiments)} экспериментов")
print(f"Время создания: {data['timestamp']}")
print()

# ==================== ОБЩАЯ СТАТИСТИКА ====================

print("="*70)
print("ОБЩАЯ СТАТИСТИКА")
print("="*70)

test_accs = [exp['test_acc'] for exp in experiments]
val_accs = [exp['best_val_acc'] for exp in experiments]

print(f"Количество экспериментов: {len(experiments)}")
print(f"\nTest Accuracy:")
print(f"  Мин: {min(test_accs):.2f}%")
print(f"  Макс: {max(test_accs):.2f}%")
print(f"  Среднее: {np.mean(test_accs):.2f}%")
print(f"  Медиана: {np.median(test_accs):.2f}%")
print(f"  Std: {np.std(test_accs):.2f}%")

print(f"\nValidation Accuracy:")
print(f"  Мин: {min(val_accs):.2f}%")
print(f"  Макс: {max(val_accs):.2f}%")
print(f"  Среднее: {np.mean(val_accs):.2f}%")
print(f"  Медиана: {np.median(val_accs):.2f}%")

# ==================== ТОП-10 МОДЕЛЕЙ ====================

print("\n" + "="*70)
print("ТОП-10 МОДЕЛЕЙ ПО TEST ACCURACY")
print("="*70)

sorted_exp = sorted(experiments, key=lambda x: x['test_acc'], reverse=True)
top10 = sorted_exp[:10]

for i, exp in enumerate(top10, 1):
    print(f"\n{i}. {exp['experiment_name']}")
    print(f"   Test Acc: {exp['test_acc']:.2f}%, Val Acc: {exp['best_val_acc']:.2f}%")
    print(f"   Params: {exp['num_parameters']:,}")

# ==================== ВИЗУАЛИЗАЦИЯ ====================

print("\n" + "="*70)
print("СОЗДАНИЕ ГРАФИКОВ")
print("="*70)

# Создаем директорию для графиков
Path('experiments_results/plots').mkdir(exist_ok=True, parents=True)

# 1. Сравнение всех моделей
fig = plt.figure(figsize=(16, 10))

# График 1: Val vs Test Accuracy для всех моделей
ax1 = plt.subplot(2, 2, 1)
ax1.scatter(val_accs, test_accs, alpha=0.6, s=50)
ax1.plot([min(val_accs), max(val_accs)], [min(val_accs), max(val_accs)],
         'r--', label='Val = Test', alpha=0.5)
ax1.set_xlabel('Validation Accuracy (%)')
ax1.set_ylabel('Test Accuracy (%)')
ax1.set_title('Val Acc vs Test Acc (все эксперименты)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# График 2: Топ-10 моделей
ax2 = plt.subplot(2, 2, 2)
names_top10 = [exp['experiment_name'][:30] for exp in top10]
test_accs_top10 = [exp['test_acc'] for exp in top10]
colors = plt.cm.viridis(np.linspace(0, 1, 10))
bars = ax2.barh(range(10), test_accs_top10, color=colors)
ax2.set_yticks(range(10))
ax2.set_yticklabels(names_top10, fontsize=8)
ax2.set_xlabel('Test Accuracy (%)')
ax2.set_title('Топ-10 моделей')
ax2.grid(True, alpha=0.3, axis='x')

# Добавляем значения на столбцы
for i, (bar, acc) in enumerate(zip(bars, test_accs_top10)):
    ax2.text(acc + 0.2, bar.get_y() + bar.get_height()/2,
             f'{acc:.2f}%', va='center', fontsize=8)

# График 3: Распределение Test Accuracy
ax3 = plt.subplot(2, 2, 3)
ax3.hist(test_accs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
ax3.axvline(np.mean(test_accs), color='red', linestyle='--',
            linewidth=2, label=f'Среднее: {np.mean(test_accs):.2f}%')
ax3.axvline(np.median(test_accs), color='green', linestyle='--',
            linewidth=2, label=f'Медиана: {np.median(test_accs):.2f}%')
ax3.set_xlabel('Test Accuracy (%)')
ax3.set_ylabel('Количество моделей')
ax3.set_title('Распределение Test Accuracy')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# График 4: Количество параметров vs Test Acc
ax4 = plt.subplot(2, 2, 4)
params = [exp['num_parameters'] for exp in experiments]
ax4.scatter(params, test_accs, alpha=0.6, s=50)
ax4.set_xlabel('Количество параметров')
ax4.set_ylabel('Test Accuracy (%)')
ax4.set_title('Параметры модели vs Test Accuracy')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experiments_results/plots/overall_summary.png', dpi=300, bbox_inches='tight')
print("✓ Сохранен: experiments_results/plots/overall_summary.png")

# 2. Детальное сравнение по категориям экспериментов
categories = {
    'Architecture': [],
    'Regularization': [],
    'Batch Norm': [],
    'Activation': [],
    'LR': []
}

for exp in experiments:
    for cat in categories:
        if cat in exp['experiment_name']:
            categories[cat].append(exp)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, (cat_name, cat_experiments) in enumerate(categories.items()):
    if not cat_experiments:
        continue

    ax = axes[idx]
    names = [exp['experiment_name'].replace(f'{cat_name}: ', '')[:20] for exp in cat_experiments]
    val_acc = [exp['best_val_acc'] for exp in cat_experiments]
    test_acc = [exp['test_acc'] for exp in cat_experiments]

    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width/2, val_acc, width, label='Val', alpha=0.8)
    ax.bar(x + width/2, test_acc, width, label='Test', alpha=0.8)
    ax.set_xlabel('Конфигурация')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{cat_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

# Удаляем лишний subplot
fig.delaxes(axes[-1])

plt.tight_layout()
plt.savefig('experiments_results/plots/categories_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Сохранен: experiments_results/plots/categories_comparison.png")

# 3. Heatmap корреляций
fig, ax = plt.subplots(figsize=(10, 8))

# Извлекаем ключевые параметры
data_for_corr = []
for exp in experiments:
    row = {
        'test_acc': exp['test_acc'],
        'val_acc': exp['best_val_acc'],
        'params': exp['num_parameters'],
        'dropout': exp['config'].get('dropout', 0.0),
        'weight_decay': exp['config'].get('weight_decay', 0.0),
        'lr': exp['config'].get('lr', 0.001),
        'use_bn': 1 if exp['config'].get('use_batch_norm', False) else 0,
    }
    data_for_corr.append(row)

df_corr = pd.DataFrame(data_for_corr)
corr_matrix = df_corr.corr()

im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(np.arange(len(corr_matrix.columns)))
ax.set_yticks(np.arange(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
ax.set_yticklabels(corr_matrix.columns)

# Добавляем значения
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=9)

ax.set_title('Корреляция между параметрами и точностью')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('experiments_results/plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Сохранен: experiments_results/plots/correlation_heatmap.png")

plt.show()

# ==================== СОХРАНЕНИЕ ТАБЛИЦЫ ====================

print("\n" + "="*70)
print("СОЗДАНИЕ ИТОГОВОЙ ТАБЛИЦЫ")
print("="*70)

# Полная таблица в CSV
all_results = []
for exp in experiments:
    row = {
        'Experiment': exp['experiment_name'],
        'Test_Acc': exp['test_acc'],
        'Val_Acc': exp['best_val_acc'],
        'Params': exp['num_parameters'],
        'Dropout': exp['config'].get('dropout', 0.0),
        'Weight_Decay': exp['config'].get('weight_decay', 0.0),
        'LR': exp['config'].get('lr', 0.001),
        'BatchNorm': exp['config'].get('use_batch_norm', False),
        'Activation': exp['config'].get('activation', 'relu'),
    }
    all_results.append(row)

df_all = pd.DataFrame(all_results)
df_all = df_all.sort_values('Test_Acc', ascending=False)
df_all.to_csv('experiments_results/all_results.csv', index=False)
print("✓ Сохранена таблица: experiments_results/all_results.csv")

print("\n" + "="*70)
print("ВСЕ ВИЗУАЛИЗАЦИИ СОЗДАНЫ!")
print("="*70)
print("\nФайлы:")
print("  - experiments_results/plots/overall_summary.png")
print("  - experiments_results/plots/categories_comparison.png")
print("  - experiments_results/plots/correlation_heatmap.png")
print("  - experiments_results/all_results.csv")
