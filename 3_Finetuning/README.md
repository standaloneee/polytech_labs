# Transfer Learning для классификации пород собак

## Описание проекта

Реализация и обучение моделей классификации изображений с использованием transfer learning на датасете Stanford Dogs. Проект демонстрирует эффективность предобученных моделей для задачи классификации пород собак.

**Цель:** Дообучить предобученные модели из библиотеки timm с использованием различных стратегий fine-tuning для достижения высокой точности классификации.

## Задача

Используя transfer learning обучить модели на Stanford Dogs dataset и провести эксперименты с:
- Различными архитектурами (ResNet50, EfficientNet-B0)
- Стратегиями fine-tuning (freeze, partial, full)
- Оптимизацией гиперпараметров
- Экспортом в ONNX для production

## Финальные результаты

**ResNet50:**
- Freeze strategy (только classifier): Val 99.15% / Test **100.0%**
- Partial unfreeze (последние слои): Val 99.15% / Test **100.0%**
- Full fine-tuning (все слои): Val 98.29% / Test **100.0%**

**EfficientNet-B0:**
- Freeze strategy (только classifier): Val 98.29% / Test **96.8%**
- Partial unfreeze (последние слои): Val 98.29% / Test **99.2%**
- Full fine-tuning (все слои): Val 93.16% / Test **92.0%**

**Лучшая модель:** ResNet50 (любая стратегия) - **100.0% test accuracy**

**ONNX экспорт:** 245 KB (0.24 MB) - компактная модель для production

**Вывод:** ResNet50 показывает превосходные результаты на всех стратегиях fine-tuning, достигая идеальной точности 100% на тестовой выборке. EfficientNet-B0 также показывает отличные результаты (до 99.2%), особенно при частичном размораживании слоев.

## Датасет

**Stanford Dogs** (подмножество 5 пород):
- **5 классов**: Beagle, Boxer, German Shepherd, Golden Retriever, Poodle
- **557** обучающих изображений
- **117** валидационных изображений
- **125** тестовых изображений
- **Всего**: 799 изображений
- **Источник**: Stanford Dogs Dataset

**Примечание:** Датасет загружается автоматически при первом запуске Jupyter ноутбука. Если данных нет, ноутбук самостоятельно скачает и предобработает изображения (~30MB).

## Архитектуры моделей

### Модель 1: ResNet50
- **Семейство**: ResNet (Residual Networks)
- **Предобучение**: ImageNet-1k
- **Параметры**: ~25M
- **Особенность**: Skip connections для глубокого обучения
- **Результат**: 100% test accuracy

### Модель 2: EfficientNet-B0
- **Семейство**: EfficientNet
- **Предобучение**: ImageNet-1k
- **Параметры**: ~5M (легковесная)
- **Особенность**: Compound scaling
- **Результат**: До 99.2% test accuracy

## Структура проекта

```
3_Finetuning/
├── data/
│   └── processed/              # Обработанные данные (автозагрузка при запуске ноутбука)
│       ├── train/              # 557 изображений (5 классов)
│       ├── val/                # 117 изображений
│       └── test/               # 125 изображений
├── notebooks/
│   └── dog_classification_results_analysis.ipynb  # Полный анализ с кодом и визуализацией
├── scripts/
│   ├── download_data.py        # Скачивание Stanford Dogs
│   ├── preprocess.py           # Предобработка и разделение данных
│   ├── train.py                # Обучение моделей (все 6 экспериментов)
│   ├── analyze_results.py      # Анализ и визуализация результатов
│   └── export_onnx.py          # Экспорт в ONNX
├── models/
│   ├── resnet50_freeze_*.pth   # Лучшая модель (100% accuracy)
│   └── best_model_resnet50.onnx  # ONNX для production (245 KB)
├── results/
│   ├── plots/                  # Графики, confusion matrices, training curves
│   └── metrics/                # JSON файлы с результатами всех 6 экспериментов
├── docs/
│   └── guide.md                # Технический гайд с детальными объяснениями
├── README.md                   # Этот файл
├── requirements.txt            # Зависимости
└── .gitignore                  # Исключения для git
```

## Технологии

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **timm**: PyTorch Image Models
- **torchvision**: для transforms
- **scikit-learn**: для метрик
- **matplotlib/seaborn**: для визуализации
- **ONNX**: для экспорта моделей
- **Jupyter**: для анализа результатов

## Методология

### 1. Подготовка данных
- Датасет Stanford Dogs (5 пород)
- Разделение: train (557) / val (117) / test (125)
- Аугментация: RandomHorizontalFlip, RandomRotation, ColorJitter
- Нормализация по ImageNet статистикам (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### 2. Transfer Learning стратегии

**Стратегия 1: Freeze (только classifier)**
- Заморозить все слои кроме FC головы
- Обучить только классификатор
- LR: 1e-3, Epochs: 15
- Быстрое обучение для адаптации к новым классам

**Стратегия 2: Partial Unfreeze (последние слои)**
- Разморозить последние 2 блока ResNet или последний блок EfficientNet
- Fine-tune с малым LR
- LR: 1e-4, Epochs: 15
- Адаптация глубоких признаков

**Стратегия 3: Full Fine-tuning (все слои)**
- Разморозить все слои
- Очень малый LR для тонкой настройки
- LR: 1e-5, Epochs: 20
- Полная адаптация к датасету

### 3. Гиперпараметры

**Базовые настройки:**
- Batch size: 32
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Scheduler: ReduceLROnPlateau

**Подобранные значения:**
- Freeze: LR=1e-3, 15 epochs
- Partial: LR=1e-4, 15 epochs
- Full: LR=1e-5, 20 epochs

### 4. Оценка
- **Accuracy** на val/test
- **Loss curves** (train/val)
- **Confusion Matrix**
- **Per-class metrics** (Precision, Recall, F1)

## Установка и запуск

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

Основные зависимости:
- torch
- torchvision
- timm
- scikit-learn
- matplotlib
- seaborn
- jupyter

### 2. Подготовка данных

**Автоматическая загрузка:** Датасет загружается автоматически при первом запуске Jupyter ноутбука. Ноутбук проверяет наличие данных и, если их нет, скачивает и предобрабатывает Stanford Dogs Dataset.

Если хотите подготовить данные вручную:

```bash
# Скачать датасет Stanford Dogs
python scripts/download_data.py

# Предобработать и разделить на train/val/test
python scripts/preprocess.py
```

### 3. Обучение моделей

```bash
# ResNet50 - все три стратегии
python scripts/train.py --model resnet50 --strategy freeze
python scripts/train.py --model resnet50 --strategy partial
python scripts/train.py --model resnet50 --strategy full

# EfficientNet-B0 - все три стратегии
python scripts/train.py --model efficientnet_b0 --strategy freeze
python scripts/train.py --model efficientnet_b0 --strategy partial
python scripts/train.py --model efficientnet_b0 --strategy full
```

Результаты сохраняются в `results/metrics/` и `models/`

### 4. Оценка моделей

```bash
# Оценить лучшую модель
python scripts/evaluate.py --model resnet50 --strategy freeze

# Сравнить все модели
python scripts/evaluate.py --compare-all
```

### 5. Экспорт в ONNX

```bash
# Экспортировать лучшую модель
python scripts/export_onnx.py --model resnet50
```

Результат: `models/best_model_resnet50.onnx` (245 KB)

### 6. Работа с Jupyter Notebook

Для полного анализа результатов откройте ноутбук:

```bash
jupyter notebook notebooks/dog_classification_results_analysis.ipynb
```

Ноутбук содержит:
- Визуализацию learning curves
- Сравнение всех экспериментов
- Confusion matrices
- Per-class метрики
- Примеры предсказаний

## Эксперименты

### ResNet50

| Стратегия | Frozen Layers | LR | Epochs | Best Val Acc | Test Acc |
|-----------|---------------|--------|--------|--------------|----------|
| Freeze | All except classifier | 1e-3 | 15 | 99.15% | **100.0%** |
| Partial | Last 2 blocks unfrozen | 1e-4 | 15 | 99.15% | **100.0%** |
| Full | All unfrozen | 1e-5 | 20 | 98.29% | **100.0%** |

### EfficientNet-B0

| Стратегия | Frozen Layers | LR | Epochs | Best Val Acc | Test Acc |
|-----------|---------------|--------|--------|--------------|----------|
| Freeze | All except classifier | 1e-3 | 15 | 98.29% | 96.8% |
| Partial | Last block unfrozen | 1e-4 | 15 | 98.29% | **99.2%** |
| Full | All unfrozen | 1e-5 | 20 | 93.16% | 92.0% |

### Ключевые наблюдения

**ResNet50:**
- Все три стратегии достигли 100% на тесте
- Freeze и Partial быстрее сходятся (15 эпох)
- Full требует больше эпох (20) но также достигает 100%
- Оптимальная стратегия: Freeze (быстро и эффективно)

**EfficientNet-B0:**
- Partial показала лучший результат: 99.2%
- Full fine-tuning ухудшает результат до 92% (overfitting на малом датасете)
- Freeze дает хороший baseline: 96.8%
- Оптимальная стратегия: Partial

**Общие выводы:**
- ResNet50 превосходит EfficientNet-B0 на данном датасете
- Freeze и Partial стратегии наиболее эффективны для малых датасетов
- Full fine-tuning может привести к overfitting без достаточного количества данных
- Размер модели не всегда коррелирует с точностью (ResNet50 больше, но точнее)

## Выводы

### Достигнутые результаты

1. **ResNet50 достиг идеальной точности 100%** на всех трех стратегиях fine-tuning
   - Freeze: 99.15% val / 100.0% test (15 эпох)
   - Partial: 99.15% val / 100.0% test (15 эпох)
   - Full: 98.29% val / 100.0% test (20 эпох)

2. **EfficientNet-B0 показал отличные результаты** до 99.2% точности
   - Лучшая стратегия: Partial (99.2% test)
   - Full fine-tuning привел к overfitting (92.0% test)

3. **Компактность модели**: ONNX экспорт составил всего 245 KB, что позволяет легко использовать модель в production

### Практические рекомендации

**Для малых датасетов (< 1000 изображений):**
- Используйте Freeze или Partial стратегии
- Избегайте Full fine-tuning (высокий риск overfitting)
- ResNet архитектуры показывают лучшую обобщающую способность

**Для выбора модели:**
- ResNet50: лучшая точность, стабильные результаты
- EfficientNet-B0: легковесная альтернатива, хороша для edge devices

**Для оптимизации обучения:**
- Freeze: быстрое обучение (15 эпох), хорошие результаты
- Partial: небольшое улучшение за ту же цену
- Full: требует больше эпох, не гарантирует улучшения

### Применимость

Модель может быть использована для:
- Автоматической классификации изображений собак
- Мобильных приложений для распознавания пород
- Систем мониторинга в питомниках и приютах
- Образовательных проектов по computer vision

### Дальнейшие улучшения

- Увеличение датасета до 100+ изображений на класс
- Добавление более сложных аугментаций (MixUp, CutMix)
- Тестирование других архитектур (Vision Transformer, ConvNeXt)
- Ансамблирование нескольких моделей
- Квантизация для ускорения inference

## Документация

Для детального понимания проекта см.:
- **[docs/guide.md](docs/guide.md)** - Технический гайд с объяснением всех деталей:
  - Почему выбраны эти стратегии fine-tuning
  - Детальное описание каждого скрипта
  - Объяснение архитектур ResNet50 и EfficientNet-B0
  - Интерпретация результатов (почему ResNet50 показывает 100%)
  - Практические рекомендации по выбору стратегии

## Ссылки

- [timm Documentation](https://timm.fast.ai/)
- [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [ONNX Runtime](https://onnxruntime.ai/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)

## Автор

Проект выполнен как лабораторная работа по курсу "Transfer Learning и Fine-tuning предобученных моделей"

## Лицензия

Проект создан в образовательных целях.
