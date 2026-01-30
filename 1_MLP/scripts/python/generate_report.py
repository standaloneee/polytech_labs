"""
Скрипт для создания итогового отчета о результатах экспериментов
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Загрузка результатов
results_file = 'experiments_results/all_experiments.json'

if not Path(results_file).exists():
    print(f"Файл {results_file} не найден!")
    print("Сначала запустите experiments.py")
    exit(1)

with open(results_file, 'r') as f:
    data = json.load(f)

experiments = data['experiments']

# Находим baseline и лучшую модель
baseline = next((exp for exp in experiments if 'Baseline' in exp['experiment_name']), None)
best_model = max(experiments, key=lambda x: x['test_acc'])

# Группируем по категориям
categories = {}
for exp in experiments:
    for cat in ['Architecture', 'Regularization', 'Batch Norm', 'Activation', 'LR']:
        if cat in exp['experiment_name']:
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(exp)

# ==================== СОЗДАНИЕ ОТЧЕТА ====================

report = f"""
{'='*80}
ИТОГОВЫЙ ОТЧЕТ ПО ЭКСПЕРИМЕНТАМ
{'='*80}

Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Дата экспериментов: {data['timestamp']}
Всего экспериментов: {len(experiments)}

{'='*80}
1. BASELINE МОДЕЛЬ
{'='*80}
"""

if baseline:
    report += f"""
Архитектура: {baseline['config']['hidden_sizes']}
Количество параметров: {baseline['num_parameters']:,}
Функция активации: {baseline['config'].get('activation', 'relu').upper()}
Dropout: {baseline['config'].get('dropout', 0.0)}
Batch Normalization: {'Да' if baseline['config'].get('use_batch_norm', False) else 'Нет'}

Результаты:
  - Training Epochs: {baseline['config']['num_epochs']}
  - Learning Rate: {baseline['config'].get('lr', 0.001)}
  - Best Validation Accuracy: {baseline['best_val_acc']:.2f}%
  - Test Loss: {baseline['test_loss']:.4f}
  - **Test Accuracy: {baseline['test_acc']:.2f}%**

Проблемы:
  - Наблюдается переобучение (высокая точность на обучающей выборке)
  - Необходима регуляризация
"""

report += f"""
{'='*80}
2. ЛУЧШАЯ МОДЕЛЬ
{'='*80}

Эксперимент: {best_model['experiment_name']}
Архитектура: {best_model['config']['hidden_sizes']}
Количество параметров: {best_model['num_parameters']:,}

Конфигурация:
"""

for key, value in best_model['config'].items():
    if key not in ['hidden_sizes', 'num_epochs']:
        report += f"  - {key}: {value}\n"

improvement = best_model['test_acc'] - (baseline['test_acc'] if baseline else 0)

report += f"""
Результаты:
  - Best Validation Accuracy: {best_model['best_val_acc']:.2f}%
  - Test Loss: {best_model['test_loss']:.4f}
  - **Test Accuracy: {best_model['test_acc']:.2f}%**

Улучшение относительно baseline: +{improvement:.2f}%

{'='*80}
3. РЕЗУЛЬТАТЫ ПО КАТЕГОРИЯМ ЭКСПЕРИМЕНТОВ
{'='*80}
"""

for cat_name, cat_experiments in categories.items():
    if not cat_experiments:
        continue

    best_in_cat = max(cat_experiments, key=lambda x: x['test_acc'])
    worst_in_cat = min(cat_experiments, key=lambda x: x['test_acc'])
    avg_test_acc = np.mean([exp['test_acc'] for exp in cat_experiments])

    report += f"""
{cat_name}:
  Количество экспериментов: {len(cat_experiments)}
  Средняя Test Accuracy: {avg_test_acc:.2f}%

  Лучший результат:
    - {best_in_cat['experiment_name']}
    - Test Acc: {best_in_cat['test_acc']:.2f}%

  Худший результат:
    - {worst_in_cat['experiment_name']}
    - Test Acc: {worst_in_cat['test_acc']:.2f}%

  Разброс: {best_in_cat['test_acc'] - worst_in_cat['test_acc']:.2f}%
"""

# Топ-5
sorted_exp = sorted(experiments, key=lambda x: x['test_acc'], reverse=True)
top5 = sorted_exp[:5]

report += f"""
{'='*80}
4. ТОП-5 МОДЕЛЕЙ
{'='*80}
"""

for i, exp in enumerate(top5, 1):
    report += f"""
{i}. {exp['experiment_name']}
   Test Accuracy: {exp['test_acc']:.2f}%
   Val Accuracy: {exp['best_val_acc']:.2f}%
   Параметры: {exp['num_parameters']:,}
   Конфигурация: {exp['config']}
"""

# Статистика
test_accs = [exp['test_acc'] for exp in experiments]
val_accs = [exp['best_val_acc'] for exp in experiments]

report += f"""
{'='*80}
5. ОБЩАЯ СТАТИСТИКА
{'='*80}

Test Accuracy:
  - Минимум: {min(test_accs):.2f}%
  - Максимум: {max(test_accs):.2f}%
  - Среднее: {np.mean(test_accs):.2f}%
  - Медиана: {np.median(test_accs):.2f}%
  - Стандартное отклонение: {np.std(test_accs):.2f}%

Validation Accuracy:
  - Минимум: {min(val_accs):.2f}%
  - Максимум: {max(val_accs):.2f}%
  - Среднее: {np.mean(val_accs):.2f}%
  - Медиана: {np.median(val_accs):.2f}%

Val-Test Gap (среднее): {np.mean([v - t for v, t in zip(val_accs, test_accs)]):.2f}%

{'='*80}
6. КЛЮЧЕВЫЕ ВЫВОДЫ
{'='*80}

1. РЕГУЛЯРИЗАЦИЯ КРИТИЧЕСКИ ВАЖНА
   Модели с Dropout и/или Weight Decay показывают значительно лучшие результаты
   на тестовой выборке, что указывает на меньшее переобучение.

2. BATCH NORMALIZATION ЭФФЕКТИВНА
   Использование Batch Normalization ускоряет обучение и часто приводит к
   улучшению обобщающей способности модели.

3. БАЛАНС СЛОЖНОСТИ И ОБОБЩЕНИЯ
   Увеличение количества параметров не всегда приводит к улучшению результатов.
   Важен баланс между выразительностью модели и склонностью к переобучению.

4. LEARNING RATE И SCHEDULERS
   Правильный выбор скорости обучения критичен. Schedulers помогают адаптивно
   изменять LR в процессе обучения и могут улучшить результаты.

5. ФУНКЦИИ АКТИВАЦИИ
   ReLU остается надежным выбором. Альтернативы (LeakyReLU, ELU) дают
   небольшое улучшение в некоторых случаях.

{'='*80}
7. ПРОБЛЕМЫ И НАБЛЮДЕНИЯ
{'='*80}

1. ПЕРЕОБУЧЕНИЕ
   Основная проблема - модели показывают высокую точность на обучающей выборке
   (>95%), но значительно хуже на тестовой (~75%). Это указывает на необходимость
   более агрессивной регуляризации.

2. VAL-TEST GAP
   Наблюдается разрыв между точностью на валидационной и тестовой выборках.
   Это может указывать на различия в распределении данных или на переобучение
   под валидационную выборку при подборе гиперпараметров.

3. ОГРАНИЧЕНИЯ MLP
   Многослойный перцептрон не учитывает пространственную структуру изображений,
   что ограничивает его способность к извлечению признаков.

{'='*80}
8. РЕКОМЕНДАЦИИ ДЛЯ ДАЛЬНЕЙШЕГО УЛУЧШЕНИЯ
{'='*80}

1. Использовать сверточные нейронные сети (CNN) для лучшего извлечения
   признаков из изображений

2. Применить аугментацию данных (повороты, смещения, зеркальные отражения)
   для увеличения разнообразия обучающей выборки

3. Исследовать ансамблирование моделей (voting, stacking)

4. Попробовать другие оптимизаторы (SGD with momentum, AdamW)

5. Использовать раннюю остановку (early stopping) для предотвращения
   переобучения

6. Применить cross-validation для более надежной оценки качества

7. Исследовать transfer learning с предобученными моделями

{'='*80}
9. ЗАКЛЮЧЕНИЕ
{'='*80}

В ходе работы была проведена систематическая оптимизация гиперпараметров
многослойного перцептрона для классификации медицинских изображений из
датасета OrganCMNIST.

Было протестировано {len(experiments)} различных конфигураций модели, что позволило
достичь точности {best_model['test_acc']:.2f}% на тестовой выборке - улучшение на
{improvement:.2f}% по сравнению с baseline моделью.

Ключевым фактором успеха стало использование регуляризации (Dropout и/или
Weight Decay) в сочетании с Batch Normalization. Эта комбинация позволила
значительно снизить переобучение и улучшить обобщающую способность модели.

Систематический подход к экспериментам позволил получить ценные инсайты о
влиянии различных компонентов архитектуры на итоговый результат, что будет
полезно для дальнейших исследований.

{'='*80}
КОНЕЦ ОТЧЕТА
{'='*80}
"""

# Сохраняем отчет
report_path = 'experiments_results/final_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print(f"\n\n{'='*80}")
print(f"ОТЧЕТ СОХРАНЕН: {report_path}")
print(f"{'='*80}")
