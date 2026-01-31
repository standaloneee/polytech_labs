"""
Увеличенный ансамбль для достижения >92% accuracy

Стратегия:
- 10 моделей вместо 5
- Лучшая архитектура: ResNet-like
- Разные seeds и конфигурации
- Легкая аугментация
- Longer training с early stopping
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


def set_seed(seed):
    """Установка seed для воспроизводимости"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
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
# Лучшая архитектура - ResNet-like
# ===========================================================================

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
    def __init__(self, num_classes=11, dropout=0.3, hidden_dim=64):
        super(ResNetLikeCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.pool = nn.MaxPool2d(2, 2)

        self.res_block1 = ResidualBlock(hidden_dim)
        self.res_block2 = ResidualBlock(hidden_dim)

        self.conv2 = nn.Conv2d(hidden_dim, 128, 3, padding=1, stride=2)
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
# Загрузка данных
# ===========================================================================

data_flag = 'organcmnist'
info = INFO[data_flag]
NUM_CLASSES = len(info['label'])
BATCH_SIZE = 128

print('=' * 70)
print('Загрузка датасета OrganCMNIST')
print('=' * 70)

# Легкая аугментация для обучения
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Без аугментации для val/test
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

DataClass = getattr(medmnist, info['python_class'])

val_dataset = DataClass(split='val', download=False, transform=test_transform, as_rgb=False)
test_dataset = DataClass(split='test', download=False, transform=test_transform, as_rgb=False)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f'Val samples: {len(val_dataset)}')
print(f'Test samples: {len(test_dataset)}\n')


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


def train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer,
                                     num_epochs, device, patience=15, verbose=False):
    """Обучение модели с early stopping"""
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        if verbose and epoch % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}')

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

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


def get_predictions_proba(model, dataloader, device):
    """Получить вероятности предсказаний"""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Getting predictions', leave=False):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.squeeze().numpy())

    return np.vstack(all_probs), np.concatenate(all_labels)


# ===========================================================================
# Обучение увеличенного ансамбля (10 моделей)
# ===========================================================================

print('=' * 70)
print('ОБУЧЕНИЕ УВЕЛИЧЕННОГО АНСАМБЛЯ (10 МОДЕЛЕЙ)')
print('=' * 70)
print()

# Конфигурация ансамбля - 10 моделей
ensemble_config = [
    # Разные learning rates
    {'name': 'ResNet-LR0.002-1', 'seed': 42, 'lr': 0.002, 'dropout': 0.3, 'hidden_dim': 64},
    {'name': 'ResNet-LR0.002-2', 'seed': 52, 'lr': 0.002, 'dropout': 0.3, 'hidden_dim': 64},
    {'name': 'ResNet-LR0.002-3', 'seed': 62, 'lr': 0.002, 'dropout': 0.3, 'hidden_dim': 64},

    # Разные dropout rates
    {'name': 'ResNet-Drop0.2', 'seed': 72, 'lr': 0.002, 'dropout': 0.2, 'hidden_dim': 64},
    {'name': 'ResNet-Drop0.4', 'seed': 82, 'lr': 0.002, 'dropout': 0.4, 'hidden_dim': 64},

    # Разные hidden dimensions
    {'name': 'ResNet-Hid80', 'seed': 92, 'lr': 0.002, 'dropout': 0.3, 'hidden_dim': 80},
    {'name': 'ResNet-Hid96', 'seed': 102, 'lr': 0.002, 'dropout': 0.3, 'hidden_dim': 96},

    # Дополнительные комбинации
    {'name': 'ResNet-Mix1', 'seed': 112, 'lr': 0.0015, 'dropout': 0.25, 'hidden_dim': 64},
    {'name': 'ResNet-Mix2', 'seed': 122, 'lr': 0.0025, 'dropout': 0.35, 'hidden_dim': 64},
    {'name': 'ResNet-Mix3', 'seed': 132, 'lr': 0.002, 'dropout': 0.3, 'hidden_dim': 72},
]

NUM_EPOCHS = 40
models = []
results = []

for i, config in enumerate(ensemble_config, 1):
    print(f'[{i}/{len(ensemble_config)}] Training {config["name"]} (seed={config["seed"]}, lr={config["lr"]}, dropout={config["dropout"]}, hidden_dim={config["hidden_dim"]})')
    print('-' * 70)

    # Установка seed
    set_seed(config['seed'])

    # Создание датасета с текущим seed
    train_dataset = DataClass(split='train', download=False,
                            transform=train_transform, as_rgb=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Создание модели
    model = ResNetLikeCNN(
        num_classes=NUM_CLASSES,
        dropout=config['dropout'],
        hidden_dim=config['hidden_dim']
    ).to(device)

    # Оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Обучение с early stopping
    best_val_acc = train_model_with_early_stopping(
        model, train_loader, val_loader, criterion, optimizer,
        NUM_EPOCHS, device, patience=15, verbose=True
    )

    # Тестирование
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f'Best Val Acc: {best_val_acc:.2f}%')
    print(f'Test Acc: {test_acc:.2f}%')
    print()

    # Сохранение модели
    models.append(model)
    results.append({
        'name': config['name'],
        'seed': config['seed'],
        'lr': config['lr'],
        'dropout': config['dropout'],
        'hidden_dim': config['hidden_dim'],
        'best_val_acc': best_val_acc,
        'test_acc': test_acc
    })


# ===========================================================================
# Ансамблирование - Soft Voting
# ===========================================================================

print('=' * 70)
print('АНСАМБЛИРОВАНИЕ - SOFT VOTING (10 МОДЕЛЕЙ)')
print('=' * 70)
print()

def ensemble_predict(models, dataloader, device):
    """Ансамблированное предсказание через soft voting"""
    all_probs_ensemble = []
    all_labels = None

    # Получаем предсказания от каждой модели
    for i, model in enumerate(models):
        print(f'Getting predictions from model {i+1}/{len(models)}...')
        probs, labels = get_predictions_proba(model, dataloader, device)
        all_probs_ensemble.append(probs)
        if all_labels is None:
            all_labels = labels

    # Усреднение вероятностей
    avg_probs = np.mean(all_probs_ensemble, axis=0)
    predictions = np.argmax(avg_probs, axis=1)

    # Подсчет точности
    accuracy = 100. * np.mean(predictions == all_labels)

    return accuracy, predictions, all_labels


# Ансамбль на validation
print('Evaluating ensemble on validation set...')
val_ensemble_acc, _, _ = ensemble_predict(models, val_loader, device)
print(f'Validation Ensemble Accuracy: {val_ensemble_acc:.2f}%')
print()

# Ансамбль на test
print('Evaluating ensemble on test set...')
test_ensemble_acc, test_preds, test_labels = ensemble_predict(models, test_loader, device)
print(f'Test Ensemble Accuracy: {test_ensemble_acc:.2f}%')
print()


# ===========================================================================
# Сохранение результатов
# ===========================================================================

os.makedirs('results/experiments_results', exist_ok=True)

# Сохранение результатов ансамбля
ensemble_results = {
    'individual_models': results,
    'val_ensemble_acc': val_ensemble_acc,
    'test_ensemble_acc': test_ensemble_acc,
    'num_models': len(models),
    'ensemble_type': 'large_10_models'
}

with open('results/experiments_results/ensemble_large_results.json', 'w') as f:
    json.dump(ensemble_results, f, indent=2)

# Сохранение весов моделей
for i, model in enumerate(models):
    torch.save(model.state_dict(),
              f'results/experiments_results/ensemble_large_model_{i+1}.pth')

print('=' * 70)
print('ИТОГОВЫЕ РЕЗУЛЬТАТЫ')
print('=' * 70)
print()

print('Индивидуальные модели:')
for res in results:
    print(f'  {res["name"]:<25} Test Acc: {res["test_acc"]:.2f}%')

print()
print(f'Среднее по индивидуальным: {np.mean([r["test_acc"] for r in results]):.2f}%')
print(f'Лучшая индивидуальная:     {max([r["test_acc"] for r in results]):.2f}%')
print()
print(f'Ансамбль 10 моделей:       {test_ensemble_acc:.2f}%')
print(f'Улучшение:                 +{test_ensemble_acc - max([r["test_acc"] for r in results]):.2f}%')
print()

print('Сравнение с предыдущими результатами:')
print(f'  MLP Ensemble (работа 1):          80.83%')
print(f'  ResNet-18 (статья):               87.7%')
print(f'  CNN Baseline (работа 2):          89.96%')
print(f'  CNN Ensemble 5 моделей:           91.16%')
print(f'  CNN Ensemble 10 моделей:          {test_ensemble_acc:.2f}%')
print()

if test_ensemble_acc > 92.0:
    print('🎉 ЦЕЛЬ ДОСТИГНУТА: >92% accuracy!')
else:
    print(f'📊 Текущий результат: {test_ensemble_acc:.2f}% (цель: >92%)')

print()
print('✓ Результаты сохранены:')
print('  - results/experiments_results/ensemble_large_results.json')
print('  - results/experiments_results/ensemble_large_model_*.pth (10 моделей)')
print()

print('=' * 70)
print('ОБУЧЕНИЕ УВЕЛИЧЕННОГО АНСАМБЛЯ ЗАВЕРШЕНО')
print('=' * 70)
