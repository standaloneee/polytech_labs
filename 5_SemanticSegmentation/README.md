# Лабораторная работа 5: Семантическая сегментация сосудов сетчатки

## Описание

Бинарная сегментация кровеносных сосудов на изображениях сетчатки глаза с использованием архитектуры U-Net. Проект включает сравнение различных конфигураций: архитектур (базовый U-Net, U-Net + ResNet34), loss-функций (BCE, Dice, Combined) и влияния аугментаций.

## Датасет

**DRIVE** (Digital Retinal Images for Vessel Extraction):
- 40 цветных изображений сетчатки (565×584 пикселей)
- 20 изображений для обучения + 20 для тестирования
- Бинарные маски сосудов (ручная разметка экспертами)
- Скачивается автоматически при первом запуске

## Структура проекта

```
5_SemanticSegmentation/
├── README.md                           # Этот файл
├── requirements.txt                    # Зависимости
├── .gitignore
├── docs/
│   └── guide.md                        # Подробный техгайд
├── scripts/
│   ├── dataset.py                      # Dataset DRIVE, аугментации
│   ├── model.py                        # U-Net, U-Net+ResNet34
│   ├── losses.py                       # BCE, Dice, Combined loss
│   ├── metrics.py                      # Dice, IoU, Accuracy, Sensitivity, Specificity
│   ├── train.py                        # Обучение модели
│   ├── evaluate.py                     # Оценка + визуализация
│   └── run_experiments.py              # Запуск всех 6 экспериментов
├── notebooks/
│   └── segmentation_analysis.ipynb     # Анализ результатов (автономный)
├── data/
│   └── DRIVE/                          # Датасет (скачивается автоматически)
├── models/                             # Обученные модели (.pth)
└── results/
    └── metrics/                        # JSON метрики + PNG визуализации
```

## Эксперименты

| № | Архитектура | Loss | Аугментации | Описание |
|---|-------------|------|-------------|----------|
| 1 | U-Net | BCE | Нет | Базовая конфигурация |
| 2 | U-Net | Dice | Нет | Dice loss для несбалансированных классов |
| 3 | U-Net | BCE + Dice | Нет | Комбинированная loss |
| 4 | U-Net | BCE + Dice | Да | Влияние аугментаций |
| 5 | U-Net + ResNet34 | BCE + Dice | Да | Предобученный encoder |
| 6 | U-Net + ResNet34 | Dice | Да | ResNet34 + Dice |

## Метрики

- **Dice Coefficient** — F1-score для сегментации (основная метрика)
- **IoU (Jaccard)** — Intersection over Union
- **Pixel Accuracy** — доля правильно классифицированных пикселей
- **Sensitivity (Recall)** — способность обнаруживать сосуды (TP / (TP + FN))
- **Specificity** — правильное определение фона (TN / (TN + FP))

## Быстрый старт

### 1. Установка зависимостей

```bash
cd 5_SemanticSegmentation
pip install -r requirements.txt
```

### 2. Запуск всех экспериментов

```bash
python scripts/run_experiments.py
```

Скрипт автоматически:
- Скачает датасет DRIVE
- Обучит 6 моделей
- Оценит каждую на test set
- Сохранит метрики и визуализации

### 3. Обучение одной модели

```bash
# Базовый U-Net с BCE loss
python scripts/train.py --arch unet --loss bce

# U-Net + ResNet34 с Dice loss и аугментациями
python scripts/train.py --arch resnet34 --loss dice --augment

# С кастомными параметрами
python scripts/train.py --arch unet --loss combined --augment \
    --epochs 100 --batch-size 8 --lr 5e-5 --patience 15
```

### 4. Оценка модели

```bash
python scripts/evaluate.py --model models/exp1_unet_bce_noaug_best.pth
```

### 5. Jupyter Notebook

```bash
jupyter notebook notebooks/segmentation_analysis.ipynb
```

Ноутбук автономный — скачивает данные, обучает модели и создаёт все визуализации самостоятельно.

## Параметры обучения

| Параметр | Значение |
|----------|----------|
| Размер изображения | 512×512 |
| Batch size | 4 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Early stopping | patience=10 |
| Эпохи | до 50 |

## Аугментации

- Горизонтальное отражение (p=0.5)
- Вертикальное отражение (p=0.5)
- Случайный поворот на 90°/180°/270°
- Случайная яркость (±20%)
- Случайный контраст (±20%)
- Гауссов шум (σ=0.02, p=0.2)

## Архитектуры

### Базовый U-Net (~7.8M параметров)
- Encoder: 4 уровня (64→128→256→512), MaxPool2d
- Bottleneck: 1024 каналов
- Decoder: 4 уровня с ConvTranspose2d + skip connections
- Каждый блок: Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU

### U-Net + ResNet34 (~24M параметров)
- Encoder: предобученный ResNet34 (ImageNet)
- Decoder: 5 уровней с skip connections от каждого слоя encoder
- Использует transfer learning для лучшей инициализации

## Требования

- Python 3.8+
- PyTorch 2.0+
- CUDA / MPS (Apple Silicon) / CPU
- ~2 GB свободного места (данные + модели)

## Визуализация

Для каждой модели создаётся визуализация с 4 столбцами:
1. **Оригинал** — исходное изображение сетчатки
2. **Ground Truth** — экспертная разметка сосудов
3. **Предсказание** — бинарная маска модели
4. **Overlay** — наложение (зелёный=TP, красный=FP, синий=FN)
