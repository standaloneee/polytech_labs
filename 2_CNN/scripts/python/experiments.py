"""
Эксперименты с различными CNN архитектурами для OrganCMNIST

Цель: Улучшить baseline результат (89.96%) через:
- Более глубокие архитектуры
- Batch Normalization
- Skip connections (ResNet-like)
- Различные dropout rates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import medmnist
from medmnist import INFO

# Воспроизводимость
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
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


# ===========================================================================
# Архитектуры CNN
# ===========================================================================

class DeepCNN(nn.Module):
    """
    Глубокая CNN с 4 conv слоями.

    Архитектура:
    - Conv(1->32) -> ReLU -> MaxPool
    - Conv(32->64) -> ReLU -> MaxPool
    - Conv(64->128) -> ReLU -> MaxPool
    - Conv(128->256) -> ReLU
    - Flatten -> FC(256*3*3, 256) -> ReLU -> Dropout -> FC(256, 11)
    """
    def __init__(self, num_classes=11, dropout=0.5):
        super(DeepCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28->14
        x = self.pool(F.relu(self.conv2(x)))  # 14->7
        x = self.pool(F.relu(self.conv3(x)))  # 7->3
        x = F.relu(self.conv4(x))             # 3->3

        x = x.view(-1, 256 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class BatchNormCNN(nn.Module):
    """
    CNN с Batch Normalization после каждого conv слоя.

    Batch Normalization помогает:
    - Ускорить обучение
    - Улучшить стабильность
    - Снизить переобучение
    """
    def __init__(self, num_classes=11, dropout=0.3):
        super(BatchNormCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28->14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 14->7
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 7->3

        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block с skip connection.

    y = F(x) + x (skip connection)
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out


class ResNetLikeCNN(nn.Module):
    """
    ResNet-like архитектура с skip connections.

    Skip connections помогают:
    - Обучать более глубокие сети
    - Избежать vanishing gradients
    - Улучшить gradient flow
    """
    def __init__(self, num_classes=11, dropout=0.3):
        super(ResNetLikeCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(128)

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling

        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # 28x28
        x = self.pool(x)                      # 14x14

        x = self.res_block1(x)                # 14x14
        x = self.res_block2(x)                # 14x14

        x = F.relu(self.bn2(self.conv2(x)))  # 7x7

        x = self.global_pool(x)               # 1x1
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ===========================================================================
# Загрузка данных
# ===========================================================================

print('=' * 70)
print('Загрузка датасета OrganCMNIST')
print('=' * 70)

data_flag = 'organcmnist'
info = INFO[data_flag]
NUM_CLASSES = len(info['label'])
BATCH_SIZE = 128

print(f'Датасет: {info["python_class"]}')
print(f'Классы: {NUM_CLASSES}\n')

# Трансформации
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

DataClass = getattr(medmnist, info['python_class'])

train_dataset = DataClass(split='train', download=False, transform=data_transform, as_rgb=False)
val_dataset = DataClass(split='val', download=False, transform=data_transform, as_rgb=False)
test_dataset = DataClass(split='test', download=False, transform=data_transform, as_rgb=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}\n')


# ===========================================================================
# Функции обучения
# ===========================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Обучение на одной эпохе"""
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
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100. * correct / total


def evaluate(model, dataloader, criterion, device):
    """Оценка модели"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating', leave=False):
            images, labels = images.to(device), labels.to(device).squeeze().long()

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100. * correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, device, verbose=True):
    """Полный цикл обучения"""
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs}')

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if verbose:
            print(f'Train: {train_acc:.2f}% | Val: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return history, best_val_acc


# ===========================================================================
# Эксперименты
# ===========================================================================

print('=' * 70)
print('ЗАПУСК ЭКСПЕРИМЕНТОВ С CNN АРХИТЕКТУРАМИ')
print('=' * 70)
print()

experiments = [
    {
        'name': 'Baseline CNN',
        'model_class': 'SimpleCNN',  # Используем baseline для сравнения
        'params': {'dropout': 0.5},
        'lr': 0.001
    },
    {
        'name': 'Deep CNN (4 layers)',
        'model_class': DeepCNN,
        'params': {'dropout': 0.5},
        'lr': 0.001
    },
    {
        'name': 'Deep CNN + Lower Dropout',
        'model_class': DeepCNN,
        'params': {'dropout': 0.3},
        'lr': 0.001
    },
    {
        'name': 'CNN + Batch Normalization',
        'model_class': BatchNormCNN,
        'params': {'dropout': 0.3},
        'lr': 0.001
    },
    {
        'name': 'ResNet-like (Skip Connections)',
        'model_class': ResNetLikeCNN,
        'params': {'dropout': 0.3},
        'lr': 0.001
    },
    {
        'name': 'ResNet-like + Higher LR',
        'model_class': ResNetLikeCNN,
        'params': {'dropout': 0.3},
        'lr': 0.002
    },
    {
        'name': 'BatchNorm CNN + Longer Training',
        'model_class': BatchNormCNN,
        'params': {'dropout': 0.3},
        'lr': 0.001,
        'epochs': 30
    },
]

NUM_EPOCHS = 20
results = []

for i, exp in enumerate(experiments, 1):
    print(f'[{i}/{len(experiments)}] {exp["name"]}')
    print('-' * 70)

    # Создание модели
    if exp['model_class'] == 'SimpleCNN':
        # Baseline из предыдущего скрипта (для сравнения)
        from train_cnn import SimpleCNN
        model = SimpleCNN(**exp['params']).to(device)
    else:
        model = exp['model_class'](**exp['params']).to(device)

    # Параметры
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {total_params:,}')

    # Оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=exp['lr'])

    # Обучение
    epochs = exp.get('epochs', NUM_EPOCHS)
    history, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        epochs, device, verbose=False
    )

    # Тестирование
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f'Best Val Acc: {best_val_acc:.2f}%')
    print(f'Test Acc: {test_acc:.2f}%')
    print()

    # Сохранение результатов
    results.append({
        'name': exp['name'],
        'params': total_params,
        'dropout': exp['params'].get('dropout'),
        'lr': exp['lr'],
        'epochs': epochs,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'history': history
    })


# ===========================================================================
# Сохранение результатов
# ===========================================================================

os.makedirs('results/experiments_results', exist_ok=True)

# JSON с результатами
with open('results/experiments_results/architecture_experiments.json', 'w') as f:
    # Сохраняем без history (слишком большой)
    results_to_save = [{k: v for k, v in r.items() if k != 'history'} for r in results]
    json.dump(results_to_save, f, indent=2)

print('=' * 70)
print('СВОДКА РЕЗУЛЬТАТОВ')
print('=' * 70)
print()

# Сортировка по test accuracy
results_sorted = sorted(results, key=lambda x: x['test_acc'], reverse=True)

print(f'{"Rank":<6}{"Model":<40}{"Test Acc":<12}{"Val Acc":<12}{"Params":<10}')
print('-' * 80)

for rank, res in enumerate(results_sorted, 1):
    print(f'{rank:<6}{res["name"]:<40}{res["test_acc"]:<12.2f}{res["best_val_acc"]:<12.2f}{res["params"]:<10,}')

print()
print(f'Baseline CNN (предыдущий запуск): 89.96%')
print(f'Лучший результат: {results_sorted[0]["name"]} - {results_sorted[0]["test_acc"]:.2f}%')
print(f'Улучшение над baseline: {results_sorted[0]["test_acc"] - 89.96:+.2f}%')
print()

# Визуализация сравнения
fig, ax = plt.subplots(figsize=(12, 6))

names = [r['name'] for r in results_sorted]
test_accs = [r['test_acc'] for r in results_sorted]

bars = ax.barh(names, test_accs, color='steelblue')
ax.axvline(x=89.96, color='red', linestyle='--', linewidth=2, label='Baseline (89.96%)')
ax.set_xlabel('Test Accuracy (%)')
ax.set_title('Сравнение CNN архитектур')
ax.legend()
ax.grid(axis='x', alpha=0.3)

# Добавить значения на графике
for i, (bar, acc) in enumerate(zip(bars, test_accs)):
    ax.text(acc + 0.2, i, f'{acc:.2f}%', va='center')

plt.tight_layout()
plt.savefig('results/experiments_results/architecture_comparison.png', dpi=150, bbox_inches='tight')

print('✓ Результаты сохранены:')
print('  - results/experiments_results/architecture_experiments.json')
print('  - results/experiments_results/architecture_comparison.png')
print()

print('=' * 70)
print('ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ')
print('=' * 70)
