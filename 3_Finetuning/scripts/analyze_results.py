"""
Анализ результатов экспериментов с Transfer Learning

Создаёт:
- Графики обучения (loss/accuracy curves)
- Сравнительные таблицы
- Confusion matrices для лучших моделей
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report


# Пути
PROJECT_ROOT = Path(__file__).parent.parent
METRICS_DIR = PROJECT_ROOT / 'results' / 'metrics'
PLOTS_DIR = PROJECT_ROOT / 'results' / 'plots'
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ImageNet нормализация
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_all_results():
    """Загрузить все результаты из JSON файлов"""
    results = {}

    for json_file in METRICS_DIR.glob('*_results.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Извлечь имя модели и стратегию из имени файла
        # Формат: modelname_strategy_timestamp_results.json
        # Для efficientnet_b0: efficientnet_b0_strategy_timestamp_results.json
        filename = json_file.stem  # без .json

        # Убрать _results в конце
        if filename.endswith('_results'):
            filename = filename[:-8]

        # Убрать timestamp (формат: _YYYYMMDD_HHMMSS)
        parts = filename.split('_')

        # Определить model_name и strategy
        if filename.startswith('efficientnet_b0'):
            model_name = 'efficientnet_b0'
            strategy = parts[2] if len(parts) > 2 else 'unknown'
        elif filename.startswith('resnet50'):
            model_name = 'resnet50'
            strategy = parts[1] if len(parts) > 1 else 'unknown'
        else:
            # Fallback
            model_name = parts[0] if len(parts) > 0 else 'unknown'
            strategy = parts[1] if len(parts) > 1 else 'unknown'

        key = f"{model_name}_{strategy}"
        results[key] = data

    return results


def plot_training_curves(results):
    """Построить графики обучения для всех экспериментов"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Train Loss
    ax = axes[0, 0]
    for name, data in results.items():
        history = data.get('history', {})
        if 'train_loss' in history:
            ax.plot(history['train_loss'], label=name, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Val Loss
    ax = axes[0, 1]
    for name, data in results.items():
        history = data.get('history', {})
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label=name, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Train Accuracy
    ax = axes[1, 0]
    for name, data in results.items():
        history = data.get('history', {})
        if 'train_acc' in history:
            ax.plot(history['train_acc'], label=name, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Val Accuracy
    ax = axes[1, 1]
    for name, data in results.items():
        history = data.get('history', {})
        if 'val_acc' in history:
            ax.plot(history['val_acc'], label=name, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'training_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {PLOTS_DIR / 'training_curves.png'}")
    plt.close()


def plot_comparison_table(results):
    """Создать сравнительную таблицу результатов"""

    data = []
    for name, result in results.items():
        data.append({
            'Model': name,
            'Best Val Acc': result.get('best_val_acc', 0),
            'Test Acc': result.get('test_acc', 0),
            'Test Loss': result.get('test_loss', 0),
            'Total Params': result.get('total_params', 0) / 1e6,  # В миллионах
            'Trainable': result.get('trainable_params', 0) / 1e6
        })

    # Сортировать по Test Accuracy
    data.sort(key=lambda x: x['Test Acc'], reverse=True)

    # Создать таблицу
    fig, ax = plt.subplots(figsize=(12, len(data) * 0.6 + 1))
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    headers = ['Model', 'Val Acc (%)', 'Test Acc (%)', 'Test Loss', 'Total Params (M)', 'Trainable (M)']

    for row in data:
        table_data.append([
            row['Model'],
            f"{row['Best Val Acc']:.2f}",
            f"{row['Test Acc']:.2f}",
            f"{row['Test Loss']:.4f}",
            f"{row['Total Params']:.1f}",
            f"{row['Trainable']:.1f}"
        ])

    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Цвета для заголовка
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Выделить лучшую модель
    for i in range(1, len(table_data) + 1):
        if i == 1:  # Первая строка (лучшая модель)
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#d4edda')

    plt.title('Comparison of All Experiments', fontsize=14, weight='bold', pad=20)
    plt.savefig(PLOTS_DIR / 'comparison_table.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {PLOTS_DIR / 'comparison_table.png'}")
    plt.close()


def create_confusion_matrix(model_path, model_type='resnet50'):
    """Создать confusion matrix для модели"""

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Загрузить модель
    from torchvision import models
    import torch.nn as nn

    if model_type == 'resnet50':
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, 5)
    else:  # efficientnet_b0
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)

    checkpoint = torch.load(model_path, map_location=device)
    # Checkpoint содержит словарь с model_state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # Подготовить данные
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    test_dataset = ImageFolder(DATA_DIR / 'test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Получить предсказания
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Создать confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = test_dataset.classes

    return cm, class_names, y_true, y_pred


def plot_confusion_matrix(cm, class_names, title, filename):
    """Построить confusion matrix"""

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'})

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {PLOTS_DIR / filename}")
    plt.close()


def generate_classification_report(y_true, y_pred, class_names, model_name):
    """Создать classification report"""

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    report_file = PLOTS_DIR / f'{model_name}_classification_report.txt'
    with open(report_file, 'w') as f:
        f.write(f"Classification Report: {model_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)

    print(f"✓ Saved: {report_file}")


def main():
    """Главная функция анализа"""

    print("=" * 70)
    print("Results Analysis")
    print("=" * 70)

    # 1. Загрузить все результаты
    print("\n[1/4] Loading results...")
    results = load_all_results()
    print(f"✓ Loaded {len(results)} experiments")

    # 2. Построить графики обучения
    print("\n[2/4] Plotting training curves...")
    plot_training_curves(results)

    # 3. Создать сравнительную таблицу
    print("\n[3/4] Creating comparison table...")
    plot_comparison_table(results)

    # 4. Создать confusion matrices для лучших моделей
    print("\n[4/4] Creating confusion matrices...")

    # Найти лучшие модели для каждого типа
    best_resnet = None
    best_efficientnet = None
    best_resnet_acc = 0
    best_eff_acc = 0

    for name, data in results.items():
        test_acc = data.get('test_acc', 0)
        if 'resnet' in name and test_acc > best_resnet_acc:
            best_resnet_acc = test_acc
            best_resnet = name
        elif 'efficientnet' in name and test_acc > best_eff_acc:
            best_eff_acc = test_acc
            best_efficientnet = name

    # Confusion matrix для ResNet
    if best_resnet:
        model_files = list(MODELS_DIR.glob(f'{best_resnet}*.pth'))
        if model_files:
            print(f"  Creating confusion matrix for {best_resnet}...")
            cm, class_names, y_true, y_pred = create_confusion_matrix(
                model_files[0], model_type='resnet50'
            )
            plot_confusion_matrix(
                cm, class_names,
                f'Confusion Matrix: {best_resnet}',
                f'{best_resnet}_confusion_matrix.png'
            )
            generate_classification_report(y_true, y_pred, class_names, best_resnet)

    # Confusion matrix для EfficientNet
    if best_efficientnet:
        model_files = list(MODELS_DIR.glob(f'{best_efficientnet}*.pth'))
        if model_files:
            print(f"  Creating confusion matrix for {best_efficientnet}...")
            cm, class_names, y_true, y_pred = create_confusion_matrix(
                model_files[0], model_type='efficientnet_b0'
            )
            plot_confusion_matrix(
                cm, class_names,
                f'Confusion Matrix: {best_efficientnet}',
                f'{best_efficientnet}_confusion_matrix.png'
            )
            generate_classification_report(y_true, y_pred, class_names, best_efficientnet)

    print("\n" + "=" * 70)
    print("✓ Analysis complete!")
    print("=" * 70)
    print(f"\nResults saved to: {PLOTS_DIR}")


if __name__ == '__main__':
    main()
