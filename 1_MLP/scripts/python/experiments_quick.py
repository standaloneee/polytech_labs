"""
БЫСТРАЯ ТЕСТОВАЯ ВЕРСИЯ - проверка что все работает
Запускает несколько экспериментов по 10 эпох для быстрой проверки
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO, OrganCMNIST
import random
from torchvision import transforms
import os
import json
import sys
from datetime import datetime

# Отключаем буферизацию вывода
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

# ==================== НАСТРОЙКА ====================

SEED = 42

def set_seed(seed=SEED):
    """Устанавливает seed для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Определение устройства
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f'Device: {device}')
print(f'PyTorch version: {torch.__version__}')
print()

# ==================== ЗАГРУЗКА ДАННЫХ ====================

print("Загрузка датасета...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = OrganCMNIST(split='train', transform=transform, download=False)
val_dataset = OrganCMNIST(split='val', transform=transform, download=False)
test_dataset = OrganCMNIST(split='test', transform=transform, download=False)

data_flag = 'organcmnist'
info = INFO[data_flag]
num_classes = len(info['label'])

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
print()

# ==================== МОДЕЛЬ ====================

class MLP(nn.Module):
    """Многослойный перцептрон"""

    def __init__(self, input_size=28*28, hidden_sizes=[128, 64], num_classes=11,
                 dropout=0.0, use_batch_norm=False, activation='relu'):
        super(MLP, self).__init__()

        self.input_size = input_size

        # Функция активации
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

        # Построение слоев
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        return self.model(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==================== ФУНКЦИИ ОБУЧЕНИЯ ====================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device).squeeze().long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).squeeze().long()
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100. * correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, device, scheduler=None):
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Выводим каждые 2 эпохи
        if (epoch + 1) % 2 == 0:
            print(f'    Epoch {epoch+1}/{num_epochs}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%')
            sys.stdout.flush()

    return history, best_val_acc


# ==================== ЭКСПЕРИМЕНТЫ ====================

def run_experiment(config, train_loader, val_loader, test_loader, experiment_name):
    """Запускает один эксперимент с заданной конфигурацией"""

    set_seed()

    # Создание модели
    model = MLP(
        input_size=28*28,
        hidden_sizes=config['hidden_sizes'],
        num_classes=num_classes,
        dropout=config.get('dropout', 0.0),
        use_batch_norm=config.get('use_batch_norm', False),
        activation=config.get('activation', 'relu')
    ).to(device)

    # Loss и optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 0.0)
    )

    # Scheduler
    scheduler = None
    if config.get('scheduler') == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif config.get('scheduler') == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

    # Обучение
    print(f"\n{experiment_name}")
    print(f"  Params: {model.count_parameters():,}")
    sys.stdout.flush()

    history, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        config['num_epochs'], device, scheduler
    )

    # Тестирование
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"  ✓ Best Val Acc: {best_val_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    sys.stdout.flush()

    return {
        'experiment_name': experiment_name,
        'config': config,
        'num_parameters': model.count_parameters(),
        'history': history,
        'best_val_acc': best_val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc
    }


# ==================== ОСНОВНАЯ ПРОГРАММА ====================

def main():
    # Создаем директорию для результатов
    os.makedirs('experiments_results', exist_ok=True)

    BATCH_SIZE = 128
    NUM_EPOCHS = 10  # Быстрая версия - только 10 эпох

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_results = []

    print("="*70)
    print("БЫСТРАЯ ТЕСТОВАЯ ВЕРСИЯ - 10 эпох на эксперимент")
    print("="*70)
    sys.stdout.flush()

    # Тестовые эксперименты - по одному из каждой категории
    experiments = [
        # Baseline
        {
            'name': 'Baseline',
            'config': {
                'hidden_sizes': [128, 64],
                'num_epochs': NUM_EPOCHS,
                'lr': 0.001,
            }
        },
        # Архитектура - большая модель
        {
            'name': 'Large Architecture',
            'config': {
                'hidden_sizes': [256, 128],
                'num_epochs': NUM_EPOCHS,
                'lr': 0.001,
            }
        },
        # Регуляризация - Dropout
        {
            'name': 'With Dropout 0.3',
            'config': {
                'hidden_sizes': [128, 64],
                'dropout': 0.3,
                'num_epochs': NUM_EPOCHS,
                'lr': 0.001,
            }
        },
        # Batch Normalization
        {
            'name': 'With Batch Norm',
            'config': {
                'hidden_sizes': [128, 64],
                'use_batch_norm': True,
                'num_epochs': NUM_EPOCHS,
                'lr': 0.001,
            }
        },
        # Функция активации
        {
            'name': 'ELU Activation',
            'config': {
                'hidden_sizes': [128, 64],
                'activation': 'elu',
                'num_epochs': NUM_EPOCHS,
                'lr': 0.001,
            }
        },
        # Learning Rate Scheduler
        {
            'name': 'With Cosine Scheduler',
            'config': {
                'hidden_sizes': [128, 64],
                'lr': 0.001,
                'scheduler': 'cosine',
                'num_epochs': NUM_EPOCHS,
            }
        },
    ]

    for exp in experiments:
        result = run_experiment(exp['config'], train_loader, val_loader, test_loader, exp['name'])
        all_results.append(result)

    # ==================== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ====================
    print("\n" + "="*70)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*70)
    sys.stdout.flush()

    # Сохраняем полные результаты в JSON
    results_json = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'type': 'quick_test',
        'experiments': []
    }

    for result in all_results:
        result_copy = result.copy()
        result_copy.pop('history')
        results_json['experiments'].append(result_copy)

    with open('experiments_results/quick_test.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    print("✓ Результаты сохранены в experiments_results/quick_test.json")
    sys.stdout.flush()

    # Создаем сводную таблицу
    print("\n" + "="*70)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*70)
    print(f"{'Эксперимент':<35} {'Params':<12} {'Val Acc':<12} {'Test Acc':<12}")
    print("-"*70)
    sys.stdout.flush()

    summary_lines = []
    for result in all_results:
        line = f"{result['experiment_name']:<35} {result['num_parameters']:<12,} {result['best_val_acc']:<12.2f} {result['test_acc']:<12.2f}"
        print(line)
        sys.stdout.flush()
        summary_lines.append(line)

    # Сохраняем таблицу в файл
    with open('experiments_results/quick_test_summary.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("БЫСТРЫЙ ТЕСТ - СВОДНАЯ ТАБЛИЦА\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Эксперимент':<35} {'Params':<12} {'Val Acc':<12} {'Test Acc':<12}\n")
        f.write("-"*70 + "\n")
        for line in summary_lines:
            f.write(line + "\n")

    print("\n✓ Таблица сохранена в experiments_results/quick_test_summary.txt")
    sys.stdout.flush()

    # Находим лучшую модель
    best_result = max(all_results, key=lambda x: x['test_acc'])
    print("\n" + "="*70)
    print("ЛУЧШАЯ МОДЕЛЬ ИЗ БЫСТРОГО ТЕСТА")
    print("="*70)
    print(f"Эксперимент: {best_result['experiment_name']}")
    print(f"Параметров: {best_result['num_parameters']:,}")
    print(f"Val Accuracy: {best_result['best_val_acc']:.2f}%")
    print(f"Test Accuracy: {best_result['test_acc']:.2f}%")
    sys.stdout.flush()

    print("\n" + "="*70)
    print("БЫСТРЫЙ ТЕСТ ЗАВЕРШЕН!")
    print("="*70)
    print("\nВсе работает корректно. Можно запускать полную версию экспериментов.")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
