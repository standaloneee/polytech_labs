"""
Эксперименты с аугментацией данных для OrganCMNIST

Цель: Улучшить результаты через аугментацию и более длительное обучение
- Различные виды аугментации
- Комбинации трансформаций
- Обучение на 30-50 эпох
- Early stopping
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
# Лучшие архитектуры из предыдущих экспериментов
# ===========================================================================

class BatchNormCNN(nn.Module):
    """CNN с Batch Normalization"""
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
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block"""
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
        out += residual
        out = F.relu(out)
        return out


class ResNetLikeCNN(nn.Module):
    """ResNet-like с skip connections"""
    def __init__(self, num_classes=11, dropout=0.3):
        super(ResNetLikeCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(128)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.res_block1(x)
        x = self.res_block2(x)

        x = F.relu(self.bn2(self.conv2(x)))

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ===========================================================================
# Различные аугментации
# ===========================================================================

def get_augmentation_transforms(aug_type='basic'):
    """
    Получить трансформации с аугментацией

    Args:
        aug_type: тип аугментации
            - 'none': без аугментации
            - 'basic': базовая (flip + rotation)
            - 'strong': усиленная (flip + rotation + erasing)
            - 'all': все трансформации
    """

    if aug_type == 'none':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    elif aug_type == 'basic':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    elif aug_type == 'strong':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))
        ])

    elif aug_type == 'all':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))
        ])


# ===========================================================================
# Загрузка данных
# ===========================================================================

data_flag = 'organcmnist'
info = INFO[data_flag]
NUM_CLASSES = len(info['label'])
BATCH_SIZE = 128

print('=' * 70)
print('Загрузка датасета OrganCMNIST')
print('=' * 70)
print(f'Датасет: {info["python_class"]}')
print(f'Классы: {NUM_CLASSES}\n')

# Без аугментации для val/test
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

DataClass = getattr(medmnist, info['python_class'])


# ===========================================================================
# Функции обучения с early stopping
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


def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer,
                              num_epochs, device, patience=10, verbose=False):
    """Обучение с early stopping"""
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        if verbose and epoch % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}')

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if verbose and epoch % 5 == 0:
            print(f'  Train: {train_acc:.2f}% | Val: {val_acc:.2f}%')

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f'Early stopping at epoch {epoch+1}')
            break

    return best_val_acc


# ===========================================================================
# Эксперименты с аугментацией
# ===========================================================================

print('=' * 70)
print('ЭКСПЕРИМЕНТЫ С АУГМЕНТАЦИЕЙ ДАННЫХ')
print('=' * 70)
print()

experiments = [
    # BatchNorm CNN без аугментации (baseline)
    {
        'name': 'BatchNorm CNN (No Aug)',
        'model': BatchNormCNN,
        'aug_type': 'none',
        'epochs': 30,
        'lr': 0.001,
        'dropout': 0.3
    },
    # BatchNorm CNN с базовой аугментацией
    {
        'name': 'BatchNorm CNN + Basic Aug',
        'model': BatchNormCNN,
        'aug_type': 'basic',
        'epochs': 40,
        'lr': 0.001,
        'dropout': 0.3
    },
    # BatchNorm CNN с сильной аугментацией
    {
        'name': 'BatchNorm CNN + Strong Aug',
        'model': BatchNormCNN,
        'aug_type': 'strong',
        'epochs': 40,
        'lr': 0.001,
        'dropout': 0.3
    },
    # ResNet-like без аугментации
    {
        'name': 'ResNet-like (No Aug)',
        'model': ResNetLikeCNN,
        'aug_type': 'none',
        'epochs': 30,
        'lr': 0.001,
        'dropout': 0.3
    },
    # ResNet-like с базовой аугментацией
    {
        'name': 'ResNet-like + Basic Aug',
        'model': ResNetLikeCNN,
        'aug_type': 'basic',
        'epochs': 40,
        'lr': 0.001,
        'dropout': 0.3
    },
    # ResNet-like с сильной аугментацией
    {
        'name': 'ResNet-like + Strong Aug',
        'model': ResNetLikeCNN,
        'aug_type': 'strong',
        'epochs': 40,
        'lr': 0.001,
        'dropout': 0.3
    },
    # ResNet-like со всеми аугментациями + длительное обучение
    {
        'name': 'ResNet-like + All Aug + Long',
        'model': ResNetLikeCNN,
        'aug_type': 'all',
        'epochs': 50,
        'lr': 0.001,
        'dropout': 0.2
    },
]

results = []

for i, exp in enumerate(experiments, 1):
    print(f'[{i}/{len(experiments)}] {exp["name"]}')
    print('-' * 70)

    # Создание датасетов с соответствующей аугментацией
    train_transform = get_augmentation_transforms(exp['aug_type'])

    train_dataset = DataClass(split='train', download=False,
                            transform=train_transform, as_rgb=False)
    val_dataset = DataClass(split='val', download=False,
                           transform=test_transform, as_rgb=False)
    test_dataset = DataClass(split='test', download=False,
                            transform=test_transform, as_rgb=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Создание модели
    model = exp['model'](num_classes=NUM_CLASSES, dropout=exp['dropout']).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {total_params:,}')
    print(f'Augmentation: {exp["aug_type"]}')
    print(f'Epochs: {exp["epochs"]} (with early stopping)')

    # Оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=exp['lr'])

    # Обучение с early stopping
    best_val_acc = train_with_early_stopping(
        model, train_loader, val_loader, criterion, optimizer,
        exp['epochs'], device, patience=15, verbose=True
    )

    # Тестирование
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f'Best Val Acc: {best_val_acc:.2f}%')
    print(f'Test Acc: {test_acc:.2f}%')
    print()

    # Сохранение результатов
    results.append({
        'name': exp['name'],
        'model': exp['model'].__name__,
        'aug_type': exp['aug_type'],
        'params': total_params,
        'epochs': exp['epochs'],
        'best_val_acc': best_val_acc,
        'test_acc': test_acc
    })


# ===========================================================================
# Сохранение результатов
# ===========================================================================

os.makedirs('results/experiments_results', exist_ok=True)

# JSON с результатами
with open('results/experiments_results/augmentation_experiments.json', 'w') as f:
    json.dump(results, f, indent=2)

print('=' * 70)
print('СВОДКА РЕЗУЛЬТАТОВ АУГМЕНТАЦИИ')
print('=' * 70)
print()

# Сортировка по test accuracy
results_sorted = sorted(results, key=lambda x: x['test_acc'], reverse=True)

print(f'{"Rank":<6}{"Model":<45}{"Aug":<12}{"Test Acc":<12}')
print('-' * 80)

for rank, res in enumerate(results_sorted, 1):
    print(f'{rank:<6}{res["name"]:<45}{res["aug_type"]:<12}{res["test_acc"]:<12.2f}')

print()
print(f'Baseline (без аугментации): 89.96%')
print(f'Лучший результат: {results_sorted[0]["name"]} - {results_sorted[0]["test_acc"]:.2f}%')
print(f'Улучшение: {results_sorted[0]["test_acc"] - 89.96:+.2f}%')
print()

# Визуализация
fig, ax = plt.subplots(figsize=(12, 6))

names = [r['name'] for r in results_sorted]
test_accs = [r['test_acc'] for r in results_sorted]
colors = ['lightcoral' if 'No Aug' in name else 'steelblue' for name in names]

bars = ax.barh(names, test_accs, color=colors)
ax.axvline(x=89.96, color='red', linestyle='--', linewidth=2, label='Baseline (89.96%)')
ax.set_xlabel('Test Accuracy (%)')
ax.set_title('Влияние аугментации данных на точность')
ax.legend()
ax.grid(axis='x', alpha=0.3)

# Добавить значения
for i, (bar, acc) in enumerate(zip(bars, test_accs)):
    ax.text(acc + 0.2, i, f'{acc:.2f}%', va='center')

plt.tight_layout()
plt.savefig('results/experiments_results/augmentation_comparison.png', dpi=150, bbox_inches='tight')

print('✓ Результаты сохранены:')
print('  - results/experiments_results/augmentation_experiments.json')
print('  - results/experiments_results/augmentation_comparison.png')
print()

print('=' * 70)
print('ЭКСПЕРИМЕНТЫ С АУГМЕНТАЦИЕЙ ЗАВЕРШЕНЫ')
print('=' * 70)
