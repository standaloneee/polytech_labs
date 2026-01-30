"""
Визуализация результатов продвинутых экспериментов и генерация финального отчета
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Загрузка результатов
with open('results/experiments_results/all_experiments.json', 'r') as f:
    basic_data = json.load(f)

with open('results/experiments_results/advanced_experiments.json', 'r') as f:
    advanced_data = json.load(f)

# ==================== СРАВНЕНИЕ БАЗОВЫХ И ПРОДВИНУТЫХ ====================

print("="*70)
print("СРАВНЕНИЕ БАЗОВЫХ И ПРОДВИНУТЫХ ЭКСПЕРИМЕНТОВ")
print("="*70)

# Лучшие из каждой группы
best_basic = max(basic_data['experiments'], key=lambda x: x['test_acc'])
best_advanced = max(advanced_data['experiments'], key=lambda x: x['test_acc'])

print(f"\nЛучший из базовых экспериментов (20 epochs):")
print(f"  {best_basic['experiment_name']}")
print(f"  Test Acc: {best_basic['test_acc']:.2f}%")
print(f"  Val Acc: {best_basic['best_val_acc']:.2f}%")

print(f"\nЛучший из продвинутых экспериментов (40+ epochs):")
print(f"  {best_advanced['experiment_name']}")
print(f"  Test Acc: {best_advanced['test_acc']:.2f}%")
print(f"  Val Acc: {best_advanced['best_val_acc']:.2f}%")

improvement = best_advanced['test_acc'] - best_basic['test_acc']
print(f"\nУлучшение: +{improvement:.2f}%")

# ==================== ВИЗУАЛИЗАЦИЯ ====================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Сравнение топ-5 из обеих групп
ax1 = axes[0, 0]
basic_sorted = sorted(basic_data['experiments'], key=lambda x: x['test_acc'], reverse=True)[:5]
advanced_sorted = sorted(advanced_data['experiments'], key=lambda x: x['test_acc'], reverse=True)[:5]

basic_accs = [exp['test_acc'] for exp in basic_sorted]
advanced_accs = [exp['test_acc'] for exp in advanced_sorted]

x = np.arange(5)
width = 0.35
ax1.bar(x - width/2, basic_accs, width, label='Базовые (20 эпох)', alpha=0.8, color='skyblue')
ax1.bar(x + width/2, advanced_accs, width, label='Продвинутые (40+ эпох)', alpha=0.8, color='orange')
ax1.set_xlabel('Место в топе')
ax1.set_ylabel('Test Accuracy (%)')
ax1.set_title('Сравнение топ-5 моделей')
ax1.set_xticks(x)
ax1.set_xticklabels([f'#{i+1}' for i in range(5)])
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Цель: 80%')

# 2. Распределение результатов
ax2 = axes[0, 1]
all_basic = [exp['test_acc'] for exp in basic_data['experiments']]
all_advanced = [exp['test_acc'] for exp in advanced_data['experiments']]

ax2.hist(all_basic, bins=10, alpha=0.6, label='Базовые', color='skyblue', edgecolor='black')
ax2.hist(all_advanced, bins=10, alpha=0.6, label='Продвинутые', color='orange', edgecolor='black')
ax2.set_xlabel('Test Accuracy (%)')
ax2.set_ylabel('Количество экспериментов')
ax2.set_title('Распределение результатов')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 3. Параметры vs Точность
ax3 = axes[1, 0]
basic_params = [exp['num_parameters'] for exp in basic_data['experiments']]
basic_test = [exp['test_acc'] for exp in basic_data['experiments']]
advanced_params = [exp['num_parameters'] for exp in advanced_data['experiments']]
advanced_test = [exp['test_acc'] for exp in advanced_data['experiments']]

ax3.scatter(basic_params, basic_test, alpha=0.6, s=100, label='Базовые', color='skyblue', edgecolor='black')
ax3.scatter(advanced_params, advanced_test, alpha=0.6, s=100, label='Продвинутые', color='orange', edgecolor='black')
ax3.set_xlabel('Количество параметров')
ax3.set_ylabel('Test Accuracy (%)')
ax3.set_title('Параметры vs Точность')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Val-Test разрыв
ax4 = axes[1, 1]
basic_gap = [exp['best_val_acc'] - exp['test_acc'] for exp in basic_data['experiments']]
advanced_gap = [exp['best_val_acc'] - exp['test_acc'] for exp in advanced_data['experiments']]

bp1 = ax4.boxplot([basic_gap, advanced_gap], labels=['Базовые', 'Продвинутые'],
                   patch_artist=True, widths=0.5)
bp1['boxes'][0].set_facecolor('skyblue')
bp1['boxes'][1].set_facecolor('orange')
ax4.set_ylabel('Val-Test разрыв (%)')
ax4.set_title('Переобучение: Val-Test разрыв')
ax4.grid(True, alpha=0.3, axis='y')
ax4.axhline(y=10, color='red', linestyle='--', alpha=0.5)

plt.suptitle('Анализ базовых vs продвинутых экспериментов', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/experiments_results/comparison_basic_vs_advanced.png', dpi=150, bbox_inches='tight')
print("\n✓ График сохранен: results/experiments_results/comparison_basic_vs_advanced.png")
plt.close()

# ==================== ТОП МОДЕЛЕЙ ВСЕХ ВРЕМЕН ====================

fig, ax = plt.subplots(figsize=(14, 8))

# Объединяем все эксперименты
all_experiments = basic_data['experiments'] + advanced_data['experiments']
all_sorted = sorted(all_experiments, key=lambda x: x['test_acc'], reverse=True)[:10]

names = [exp['experiment_name'][:50] for exp in all_sorted]
test_accs = [exp['test_acc'] for exp in all_sorted]
val_accs = [exp['best_val_acc'] for exp in all_sorted]

y_pos = np.arange(len(names))
colors = ['orange' if exp in advanced_data['experiments'] else 'skyblue' for exp in all_sorted]

bars = ax.barh(y_pos, test_accs, alpha=0.8, color=colors, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel('Test Accuracy (%)', fontsize=12)
ax.set_title('ТОП-10 МОДЕЛЕЙ ВСЕХ ЭКСПЕРИМЕНТОВ', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.axvline(x=80, color='red', linestyle='--', alpha=0.5, linewidth=2)

# Добавляем значения на столбцах
for i, (bar, acc) in enumerate(zip(bars, test_accs)):
    ax.text(acc + 0.2, bar.get_y() + bar.get_height()/2,
            f'{acc:.2f}%', va='center', fontsize=9, fontweight='bold')

# Легенда
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='orange', label='Продвинутые (40+ эпох)'),
                   Patch(facecolor='skyblue', label='Базовые (20 эпох)')]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('results/experiments_results/top10_all_experiments.png', dpi=150, bbox_inches='tight')
print("✓ График сохранен: results/experiments_results/top10_all_experiments.png")
plt.close()

# ==================== СВОДНАЯ ТАБЛИЦА ====================

# Создаем CSV со всеми результатами
all_results = []
for exp in all_experiments:
    source = 'Advanced' if exp in advanced_data['experiments'] else 'Basic'
    all_results.append({
        'Source': source,
        'Experiment': exp['experiment_name'],
        'Test_Acc': round(exp['test_acc'], 2),
        'Val_Acc': round(exp['best_val_acc'], 2),
        'Val_Test_Gap': round(exp['best_val_acc'] - exp['test_acc'], 2),
        'Num_Parameters': exp['num_parameters']
    })

df = pd.DataFrame(all_results)
df = df.sort_values('Test_Acc', ascending=False)
df.to_csv('results/experiments_results/all_results_combined.csv', index=False)
print("✓ Таблица сохранена: results/experiments_results/all_results_combined.csv")

# Печать топ-10
print("\n" + "="*70)
print("ТОП-10 МОДЕЛЕЙ")
print("="*70)
print(df.head(10).to_string(index=False))

print("\n" + "="*70)
print("СТАТИСТИКА")
print("="*70)
print(f"\nБазовые эксперименты:")
print(f"  Количество: {len(basic_data['experiments'])}")
print(f"  Средняя Test Acc: {np.mean(all_basic):.2f}%")
print(f"  Лучшая Test Acc: {max(all_basic):.2f}%")

print(f"\nПродвинутые эксперименты:")
print(f"  Количество: {len(advanced_data['experiments'])}")
print(f"  Средняя Test Acc: {np.mean(all_advanced):.2f}%")
print(f"  Лучшая Test Acc: {max(all_advanced):.2f}%")

print(f"\nВСЕГО проведено экспериментов: {len(all_experiments)}")
print(f"ЛУЧШИЙ результат: {best_advanced['test_acc']:.2f}%")
print("="*70)

print("\n✓ ВСЕ ВИЗУАЛИЗАЦИИ ГОТОВЫ!")
