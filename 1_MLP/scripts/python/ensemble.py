"""
Ансамблирование лучших моделей для достижения максимальной точности
Цель: 81-82%+ на тестовой выборке

Метод:
- Обучаем 5 лучших архитектур с разными random seeds
- Усредняем предсказания (soft voting)
- Проверяем accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import medmnist
from medmnist import INFO, OrganCMNIST
import random
from torchvision import transforms
import json
from datetime import datetime
import sys

# ==================== НАСТРОЙКА ОКРУЖЕНИЯ ====================

def set_seed(seed=42):
    """Устанавливает seed для воспроизводимости экспериментов"""
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

BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}\n")
sys.stdout.flush()

# ==================== МОДЕЛЬ ====================

class MLP(nn.Module):
    """Многослойный перцептрон"""

    def __init__(self, input_size=28*28, hidden_sizes=[128, 64], num_classes=11,
                 dropout=0.0, use_batch_norm=False, activation='relu'):
        super(MLP, self).__init__()

        self.input_size = input_size

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()

        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(self.activation)
            if dropout > 0:
                layer_dropout = dropout if i < len(hidden_sizes) - 1 else min(dropout * 1.2, 0.6)
                layers.append(nn.Dropout(layer_dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
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

def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad=None):
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


def train_model_simple(model, train_loader, val_loader, criterion, optimizer,
                       num_epochs, device, scheduler=None, clip_grad=None):
    """Простое обучение без early stopping"""
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                           optimizer, device, clip_grad)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1}/{num_epochs} - Val: {val_acc:.2f}%')
            sys.stdout.flush()

    return best_val_acc


# ==================== АНСАМБЛЬ ====================

print("="*70)
print("АНСАМБЛИРОВАНИЕ ЛУЧШИХ МОДЕЛЕЙ")
print("="*70)
print("Обучаем 5 лучших архитектур с разными random seeds")
print("Каждая модель обучается 40 эпох")
print("="*70)
sys.stdout.flush()

# 5 лучших конфигураций из наших экспериментов
ensemble_configs = [
    {
        'name': 'Model-1: Deep [256,256,128,64]',
        'hidden_sizes': [256, 256, 128, 64],
        'dropout': 0.3,
        'use_batch_norm': True,
        'activation': 'relu',
        'lr': 0.001,
        'weight_decay': 0,
        'clip_grad': None,
        'epochs': 40
    },
    {
        'name': 'Model-2: Deep [512,256,128,64]',
        'hidden_sizes': [512, 256, 128, 64],
        'dropout': 0.4,
        'use_batch_norm': True,
        'activation': 'relu',
        'lr': 0.001,
        'weight_decay': 1e-4,
        'clip_grad': None,
        'epochs': 40
    },
    {
        'name': 'Model-3: [256,128,64] ULTIMATE',
        'hidden_sizes': [256, 128, 64],
        'dropout': 0.4,
        'use_batch_norm': True,
        'activation': 'leaky_relu',
        'lr': 0.001,
        'weight_decay': 1e-2,
        'clip_grad': 1.0,
        'epochs': 40
    },
    {
        'name': 'Model-4: [128,64] Classic',
        'hidden_sizes': [128, 64],
        'dropout': 0.3,
        'use_batch_norm': True,
        'activation': 'relu',
        'lr': 0.001,
        'weight_decay': 0,
        'clip_grad': None,
        'epochs': 40
    },
    {
        'name': 'Model-5: [128,64] with GradClip',
        'hidden_sizes': [128, 64],
        'dropout': 0.3,
        'use_batch_norm': True,
        'activation': 'relu',
        'lr': 0.001,
        'weight_decay': 0,
        'clip_grad': 1.0,
        'epochs': 40
    }
]

# Обучаем все модели
trained_models = []
individual_results = []

for i, config in enumerate(ensemble_configs, 1):
    print(f"\n{'='*70}")
    print(f"Обучение модели {i}/5: {config['name']}")
    print(f"{'='*70}")
    sys.stdout.flush()

    # Уникальный seed для каждой модели
    seed = 42 + i * 10
    set_seed(seed)

    # Создание модели
    model = MLP(
        input_size=28*28,
        hidden_sizes=config['hidden_sizes'],
        num_classes=len(info['label']),
        dropout=config['dropout'],
        use_batch_norm=config['use_batch_norm'],
        activation=config['activation']
    ).to(device)

    print(f"Параметры: {model.count_parameters():,}, Seed: {seed}")
    sys.stdout.flush()

    # Оптимизация
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                          lr=config['lr'],
                          weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # Обучение
    best_val_acc = train_model_simple(
        model, train_loader, val_loader, criterion, optimizer,
        config['epochs'], device, scheduler, config['clip_grad']
    )

    # Тестирование индивидуальной модели
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\nРезультат модели {i}:")
    print(f"  Val Acc: {best_val_acc:.2f}%")
    print(f"  Test Acc: {test_acc:.2f}%")
    sys.stdout.flush()

    trained_models.append(model)
    individual_results.append({
        'name': config['name'],
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'num_parameters': model.count_parameters()
    })

# ==================== ОЦЕНКА АНСАМБЛЯ ====================

print("\n" + "="*70)
print("ОЦЕНКА АНСАМБЛЯ (SOFT VOTING)")
print("="*70)
sys.stdout.flush()

def evaluate_ensemble(models, dataloader, device):
    """Оценка ансамбля с soft voting"""
    for model in models:
        model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Ensemble evaluation'):
            images = images.to(device)
            labels = labels.to(device).squeeze().long()

            # Получаем предсказания от всех моделей
            all_outputs = []
            for model in models:
                outputs = model(images)
                # Применяем softmax для получения вероятностей
                probs = torch.softmax(outputs, dim=1)
                all_outputs.append(probs)

            # Усредняем вероятности
            ensemble_probs = torch.stack(all_outputs).mean(dim=0)

            # Предсказание = класс с максимальной усредненной вероятностью
            _, predicted = ensemble_probs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return accuracy

# Оценка на валидационной выборке
val_ensemble_acc = evaluate_ensemble(trained_models, val_loader, device)
print(f"\nАнсамбль Val Accuracy: {val_ensemble_acc:.2f}%")
sys.stdout.flush()

# Оценка на тестовой выборке
test_ensemble_acc = evaluate_ensemble(trained_models, test_loader, device)
print(f"Ансамбль Test Accuracy: {test_ensemble_acc:.2f}%")
sys.stdout.flush()

# ==================== ИТОГОВАЯ СВОДКА ====================

print("\n" + "="*70)
print("ИТОГОВАЯ СВОДКА")
print("="*70)

print("\nИндивидуальные модели:")
for i, result in enumerate(individual_results, 1):
    print(f"{i}. {result['name']}")
    print(f"   Test Acc: {result['test_acc']:.2f}%")

avg_individual = np.mean([r['test_acc'] for r in individual_results])
best_individual = max([r['test_acc'] for r in individual_results])

print(f"\nСредняя Test Acc (индивидуальные): {avg_individual:.2f}%")
print(f"Лучшая Test Acc (индивидуальная): {best_individual:.2f}%")

print(f"\n{'='*70}")
print(f"АНСАМБЛЬ Test Accuracy: {test_ensemble_acc:.2f}%")
print(f"{'='*70}")

improvement_vs_best = test_ensemble_acc - best_individual
improvement_vs_avg = test_ensemble_acc - avg_individual

print(f"\nУлучшение vs лучшая индивидуальная: {improvement_vs_best:+.2f}%")
print(f"Улучшение vs средняя индивидуальная: {improvement_vs_avg:+.2f}%")

# Сохранение результатов
results = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'individual_models': individual_results,
    'ensemble_val_acc': val_ensemble_acc,
    'ensemble_test_acc': test_ensemble_acc,
    'avg_individual_test_acc': avg_individual,
    'best_individual_test_acc': best_individual,
    'improvement_vs_best': improvement_vs_best,
    'improvement_vs_avg': improvement_vs_avg
}

import os
os.makedirs('results/experiments_results', exist_ok=True)

with open('results/experiments_results/ensemble_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Результаты сохранены: results/experiments_results/ensemble_results.json")

# Сохранение моделей
print("\nСохранение моделей...")
for i, model in enumerate(trained_models, 1):
    torch.save(model.state_dict(), f'results/experiments_results/ensemble_model_{i}.pth')
    print(f"  ✓ Модель {i} сохранена")

print("\n" + "="*70)
print("ГОТОВО!")
print("="*70)
