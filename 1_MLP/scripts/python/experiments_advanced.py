"""
Продвинутые эксперименты для достижения максимальной точности MLP
Цель: 80-82%+ на тестовой выборке

Новые подходы:
- Глубокие архитектуры (4-5 слоев)
- Усиленные комбинации регуляризации
- Длительное обучение (40-50 epochs)
- Другие оптимизаторы (SGD, AdamW)
- Продвинутые техники (Label Smoothing, Gradient Clipping)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import medmnist
from medmnist import INFO, OrganCMNIST
import random
from torchvision import transforms
import os
import json
from datetime import datetime
import sys

# ==================== НАСТРОЙКА ОКРУЖЕНИЯ ====================

SEED = 42

def set_seed(seed=SEED):
    """Устанавливает seed для воспроизводимости экспериментов"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Определение устройства
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Using CUDA: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using Apple Metal (MPS)')
else:
    device = torch.device('cpu')
    print('Using CPU')

print(f'PyTorch version: {torch.__version__}\n')
sys.stdout.flush()

# ==================== ЗАГРУЗКА ДАННЫХ ====================

data_flag = 'organcmnist'
info = INFO[data_flag]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = OrganCMNIST(split='train', transform=transform, download=False)
val_dataset = OrganCMNIST(split='val', transform=transform, download=False)
test_dataset = OrganCMNIST(split='test', transform=transform, download=False)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}\n")
sys.stdout.flush()

# ==================== МОДЕЛЬ ====================

class MLP(nn.Module):
    """Многослойный перцептрон с продвинутыми возможностями"""

    def __init__(self, input_size=28*28, hidden_sizes=[128, 64], num_classes=11,
                 dropout=0.0, use_batch_norm=False, activation='relu'):
        super(MLP, self).__init__()

        self.input_size = input_size

        # Выбор функции активации
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()

        # Построение слоев
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            layers.append(self.activation)

            # Можем варьировать dropout по слоям
            if dropout > 0:
                # Более сильный dropout на последних слоях
                layer_dropout = dropout if i < len(hidden_sizes) - 1 else min(dropout * 1.2, 0.6)
                layers.append(nn.Dropout(layer_dropout))

            prev_size = hidden_size

        # Выходной слой
        layers.append(nn.Linear(prev_size, num_classes))

        self.model = nn.Sequential(*layers)

        # Инициализация весов
        self._initialize_weights()

    def _initialize_weights(self):
        """Улучшенная инициализация весов"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming initialization для ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        return self.model(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==================== ФУНКЦИИ ОБУЧЕНИЯ ====================

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross Entropy with Label Smoothing"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = torch.nn.functional.log_softmax(pred, dim=-1)

        loss = -log_preds.sum(dim=-1).mean()
        nll = torch.nn.functional.nll_loss(log_preds, target)

        return (1 - self.smoothing) * nll + self.smoothing * (loss / n_classes)


def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad=None):
    """Обучение на одной эпохе с gradient clipping"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device).squeeze().long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Оценка модели"""
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

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def train_model_with_early_stopping(model, train_loader, val_loader, criterion,
                                     optimizer, num_epochs, device, scheduler=None,
                                     patience=10, clip_grad=None):
    """Обучение с early stopping"""
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        # Обучение
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                           optimizer, device, clip_grad)

        # Валидация
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Обновление scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Сохранение истории
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Печать каждые 5 эпох
        if (epoch + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{num_epochs} - '
                  f'Train: {train_acc:.2f}% / Val: {val_acc:.2f}% / '
                  f'LR: {current_lr:.6f}')
            sys.stdout.flush()

        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            sys.stdout.flush()
            break

    return history, best_val_acc


# ==================== КОНФИГУРАЦИИ ЭКСПЕРИМЕНТОВ ====================

BATCH_SIZE = 128

# Создаем DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

experiments_config = [
    # === ГЛУБОКИЕ АРХИТЕКТУРЫ ===
    {
        'name': 'Deep-4L: [256,256,128,64] + BN + Dropout0.3',
        'hidden_sizes': [256, 256, 128, 64],
        'dropout': 0.3,
        'use_batch_norm': True,
        'activation': 'relu',
        'optimizer': 'adam',
        'lr': 0.001,
        'scheduler': 'cosine',
        'num_epochs': 40,
        'weight_decay': 0,
        'label_smoothing': 0,
        'clip_grad': None
    },
    {
        'name': 'Deep-4L: [512,256,128,64] + BN + Dropout0.4',
        'hidden_sizes': [512, 256, 128, 64],
        'dropout': 0.4,
        'use_batch_norm': True,
        'activation': 'relu',
        'optimizer': 'adam',
        'lr': 0.001,
        'scheduler': 'cosine',
        'num_epochs': 40,
        'weight_decay': 1e-4,
        'label_smoothing': 0,
        'clip_grad': None
    },
    {
        'name': 'Deep-5L: [128,128,128,64,32] + BN + Dropout0.3',
        'hidden_sizes': [128, 128, 128, 64, 32],
        'dropout': 0.3,
        'use_batch_norm': True,
        'activation': 'relu',
        'optimizer': 'adam',
        'lr': 0.001,
        'scheduler': 'cosine',
        'num_epochs': 40,
        'weight_decay': 0,
        'label_smoothing': 0,
        'clip_grad': None
    },

    # === УСИЛЕННАЯ РЕГУЛЯРИЗАЦИЯ ===
    {
        'name': 'Best-Arch + Dropout0.5 + BN',
        'hidden_sizes': [128, 64],
        'dropout': 0.5,
        'use_batch_norm': True,
        'activation': 'relu',
        'optimizer': 'adam',
        'lr': 0.001,
        'scheduler': 'cosine',
        'num_epochs': 40,
        'weight_decay': 0,
        'label_smoothing': 0,
        'clip_grad': None
    },
    {
        'name': 'Best-Arch + Dropout0.3 + L2_1e-3 + BN',
        'hidden_sizes': [128, 64],
        'dropout': 0.3,
        'use_batch_norm': True,
        'activation': 'relu',
        'optimizer': 'adam',
        'lr': 0.001,
        'scheduler': 'cosine',
        'num_epochs': 40,
        'weight_decay': 1e-3,
        'label_smoothing': 0,
        'clip_grad': None
    },
    {
        'name': 'Best-Arch + Dropout0.4 + L2_5e-4 + BN',
        'hidden_sizes': [128, 64],
        'dropout': 0.4,
        'use_batch_norm': True,
        'activation': 'relu',
        'optimizer': 'adam',
        'lr': 0.001,
        'scheduler': 'cosine',
        'num_epochs': 40,
        'weight_decay': 5e-4,
        'label_smoothing': 0,
        'clip_grad': None
    },

    # === ДРУГИЕ ОПТИМИЗАТОРЫ ===
    {
        'name': 'SGD-Momentum: [128,64] + BN + Dropout0.3',
        'hidden_sizes': [128, 64],
        'dropout': 0.3,
        'use_batch_norm': True,
        'activation': 'relu',
        'optimizer': 'sgd',
        'lr': 0.01,
        'scheduler': 'cosine',
        'num_epochs': 40,
        'weight_decay': 1e-4,
        'label_smoothing': 0,
        'clip_grad': None
    },
    {
        'name': 'AdamW: [128,64] + BN + Dropout0.3',
        'hidden_sizes': [128, 64],
        'dropout': 0.3,
        'use_batch_norm': True,
        'activation': 'relu',
        'optimizer': 'adamw',
        'lr': 0.001,
        'scheduler': 'cosine',
        'num_epochs': 40,
        'weight_decay': 1e-2,
        'label_smoothing': 0,
        'clip_grad': None
    },

    # === ПРОДВИНУТЫЕ ТЕХНИКИ ===
    {
        'name': 'LabelSmoothing0.1 + [128,64] + BN + Dropout0.3',
        'hidden_sizes': [128, 64],
        'dropout': 0.3,
        'use_batch_norm': True,
        'activation': 'relu',
        'optimizer': 'adam',
        'lr': 0.001,
        'scheduler': 'cosine',
        'num_epochs': 40,
        'weight_decay': 0,
        'label_smoothing': 0.1,
        'clip_grad': None
    },
    {
        'name': 'GradClip1.0 + [128,64] + BN + Dropout0.3',
        'hidden_sizes': [128, 64],
        'dropout': 0.3,
        'use_batch_norm': True,
        'activation': 'relu',
        'optimizer': 'adam',
        'lr': 0.001,
        'scheduler': 'cosine',
        'num_epochs': 40,
        'weight_decay': 0,
        'label_smoothing': 0,
        'clip_grad': 1.0
    },

    # === КОМБО: ВСЕ ЛУЧШИЕ ТЕХНИКИ ===
    {
        'name': 'ULTIMATE: [256,128,64] + BN + Dropout0.4 + L2 + LabelSmooth + GradClip',
        'hidden_sizes': [256, 128, 64],
        'dropout': 0.4,
        'use_batch_norm': True,
        'activation': 'leaky_relu',
        'optimizer': 'adamw',
        'lr': 0.001,
        'scheduler': 'cosine',
        'num_epochs': 50,
        'weight_decay': 1e-2,
        'label_smoothing': 0.1,
        'clip_grad': 1.0
    },
]

print("="*70)
print("ПРОДВИНУТЫЕ ЭКСПЕРИМЕНТЫ ДЛЯ МАКСИМИЗАЦИИ ТОЧНОСТИ MLP")
print("="*70)
print(f"Всего экспериментов: {len(experiments_config)}")
print(f"Device: {device}")
print("="*70)
sys.stdout.flush()

# ==================== ЗАПУСК ЭКСПЕРИМЕНТОВ ====================

results = []

for i, config in enumerate(experiments_config, 1):
    print(f"\n{'='*70}")
    print(f"ЭКСПЕРИМЕНТ {i}/{len(experiments_config)}: {config['name']}")
    print(f"{'='*70}")
    sys.stdout.flush()

    # Сброс seed
    set_seed()

    # Создание модели
    model = MLP(
        input_size=28*28,
        hidden_sizes=config['hidden_sizes'],
        num_classes=len(info['label']),
        dropout=config['dropout'],
        use_batch_norm=config['use_batch_norm'],
        activation=config['activation']
    ).to(device)

    print(f"Параметры: {model.count_parameters():,}")
    sys.stdout.flush()

    # Функция потерь
    if config['label_smoothing'] > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config['label_smoothing'])
    else:
        criterion = nn.CrossEntropyLoss()

    # Оптимизатор
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(),
                              lr=config['lr'],
                              weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(),
                               lr=config['lr'],
                               weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                             lr=config['lr'],
                             momentum=0.9,
                             weight_decay=config['weight_decay'])

    # Scheduler
    if config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['num_epochs']
        )
    elif config['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )
    elif config['scheduler'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    else:
        scheduler = None

    # Обучение
    history, best_val_acc = train_model_with_early_stopping(
        model, train_loader, val_loader, criterion, optimizer,
        config['num_epochs'], device, scheduler,
        patience=15,
        clip_grad=config['clip_grad']
    )

    # Тестирование
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\nРезультаты:")
    print(f"  Best Val Acc: {best_val_acc:.2f}%")
    print(f"  Test Acc: {test_acc:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}")
    sys.stdout.flush()

    # Сохранение результатов
    result = {
        'experiment_name': config['name'],
        'config': config,
        'num_parameters': model.count_parameters(),
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'final_train_acc': history['train_acc'][-1],
        'epochs_trained': len(history['train_acc'])
    }
    results.append(result)

# ==================== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ====================

os.makedirs('results/experiments_results', exist_ok=True)

output_data = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'experiments': results
}

with open('results/experiments_results/advanced_experiments.json', 'w') as f:
    json.dump(output_data, f, indent=2)

# ==================== ИТОГОВАЯ СВОДКА ====================

print("\n" + "="*70)
print("ИТОГОВАЯ СВОДКА ПРОДВИНУТЫХ ЭКСПЕРИМЕНТОВ")
print("="*70)

# Сортируем по test_acc
sorted_results = sorted(results, key=lambda x: x['test_acc'], reverse=True)

print("\nТОП-5 МОДЕЛЕЙ:")
for i, result in enumerate(sorted_results[:5], 1):
    print(f"\n{i}. {result['experiment_name']}")
    print(f"   Test Acc: {result['test_acc']:.2f}%")
    print(f"   Val Acc: {result['best_val_acc']:.2f}%")
    print(f"   Параметры: {result['num_parameters']:,}")

best = sorted_results[0]
print(f"\n{'='*70}")
print("ЛУЧШАЯ МОДЕЛЬ:")
print(f"  Эксперимент: {best['experiment_name']}")
print(f"  Test Accuracy: {best['test_acc']:.2f}%")
print(f"  Val Accuracy: {best['best_val_acc']:.2f}%")
print(f"  Параметры: {best['num_parameters']:,}")
print(f"{'='*70}")

# Сохранение сводки
with open('results/experiments_results/advanced_summary.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("ПРОДВИНУТЫЕ ЭКСПЕРИМЕНТЫ - ИТОГИ\n")
    f.write("="*70 + "\n\n")

    for i, result in enumerate(sorted_results, 1):
        f.write(f"{i}. {result['experiment_name']}\n")
        f.write(f"   Test Acc: {result['test_acc']:.2f}%\n")
        f.write(f"   Val Acc: {result['best_val_acc']:.2f}%\n")
        f.write(f"   Параметры: {result['num_parameters']:,}\n")
        f.write(f"   Эпох: {result['epochs_trained']}\n\n")

    f.write("\n" + "="*70 + "\n")
    f.write("ЛУЧШИЙ РЕЗУЛЬТАТ:\n")
    f.write(f"  {best['experiment_name']}\n")
    f.write(f"  Test Accuracy: {best['test_acc']:.2f}%\n")
    f.write("="*70 + "\n")

print("\nРезультаты сохранены в:")
print("  - results/experiments_results/advanced_experiments.json")
print("  - results/experiments_results/advanced_summary.txt")
print("\nГОТОВО! ✓")
