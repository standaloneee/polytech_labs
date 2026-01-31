"""
Baseline CNN модель для классификации OrganCMNIST

Простая сверточная нейронная сеть с 2 conv слоями и 2 FC слоями.
Цель: Превзойти результат MLP (80.83%) из предыдущей работы.
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

# Определение устройства (GPU/CPU)
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Using CUDA: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using Apple Metal (MPS)')
else:
    device = torch.device('cpu')
    print('Using CPU')

print(f'PyTorch version: {torch.__version__}')
print()

# ===========================================================================
# Архитектура Baseline CNN
# ===========================================================================

class SimpleCNN(nn.Module):
    """
    Простая сверточная нейронная сеть.

    Архитектура:
    - Conv2d(1, 32, 3) -> ReLU -> MaxPool(2) [28x28 -> 14x14]
    - Conv2d(32, 64, 3) -> ReLU -> MaxPool(2) [14x14 -> 7x7]
    - Flatten -> FC(64*7*7, 128) -> ReLU -> Dropout
    - FC(128, 11)

    Параметры:
        num_classes: количество классов (11 для OrganCMNIST)
        dropout: вероятность dropout (по умолчанию 0.5)
    """
    def __init__(self, num_classes=11, dropout=0.5):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Conv block 1: 28x28x1 -> 14x14x32
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Conv block 2: 14x14x32 -> 7x7x64
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten: 7x7x64 -> 3136
        x = x.view(-1, 64 * 7 * 7)

        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# ===========================================================================
# Загрузка данных
# ===========================================================================

print('=' * 70)
print('Загрузка датасета OrganCMNIST')
print('=' * 70)

# Параметры датасета
data_flag = 'organcmnist'
info = INFO[data_flag]
NUM_CLASSES = len(info['label'])
BATCH_SIZE = 128

print(f'Датасет: {info["python_class"]}')
print(f'Классы: {NUM_CLASSES}')
print(f'Размер изображений: {info["n_channels"]}×{28}×{28}')
print()

# Трансформации для преобразования PIL Image в Tensor
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Нормализация для grayscale
])

# Загрузка датасета
DataClass = getattr(medmnist, info['python_class'])

train_dataset = DataClass(split='train', download=False, transform=data_transform, as_rgb=False)
val_dataset = DataClass(split='val', download=False, transform=data_transform, as_rgb=False)
test_dataset = DataClass(split='test', download=False, transform=data_transform, as_rgb=False)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f'Train samples: {len(train_dataset)}')
print(f'Val samples: {len(val_dataset)}')
print(f'Test samples: {len(test_dataset)}')
print(f'Batch size: {BATCH_SIZE}')
print()


# ===========================================================================
# Функции обучения и оценки
# ===========================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Обучение модели на одной эпохе"""
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

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Оценка модели на валидационном/тестовом наборе"""
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

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, device):
    """Полный цикл обучения модели"""
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0

    print('Начало обучения...\n')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 70)

        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f'✓ New best validation accuracy: {val_acc:.2f}%')

        print()

    return history


# ===========================================================================
# Обучение Baseline CNN
# ===========================================================================

print('=' * 70)
print('Конфигурация Baseline CNN')
print('=' * 70)
print('Архитектура:')
print('  - Conv2d(1, 32, 3) + ReLU + MaxPool(2)')
print('  - Conv2d(32, 64, 3) + ReLU + MaxPool(2)')
print('  - FC(3136, 128) + ReLU + Dropout(0.5)')
print('  - FC(128, 11)')
print()
print('Гиперпараметры:')
print(f'  - Epochs: 20')
print(f'  - Batch size: {BATCH_SIZE}')
print(f'  - Learning rate: 0.001')
print(f'  - Optimizer: Adam')
print(f'  - Dropout: 0.5')
print()

# Создание модели
baseline_model = SimpleCNN(num_classes=NUM_CLASSES, dropout=0.5).to(device)

# Количество параметров
total_params = sum(p.numel() for p in baseline_model.parameters())
trainable_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
print(f'Total parameters: {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')
print()

# Оптимизатор и функция потерь
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)

# Параметры обучения
NUM_EPOCHS = 20

# Обучение
history = train_model(
    baseline_model, train_loader, val_loader, criterion, optimizer,
    NUM_EPOCHS, device
)

# ===========================================================================
# Оценка на тестовом наборе
# ===========================================================================

print('=' * 70)
print('Оценка на тестовом наборе')
print('=' * 70)

test_loss, test_acc = evaluate(baseline_model, test_loader, criterion, device)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_acc:.2f}%')
print()

print('Сравнение с MLP (предыдущая работа):')
print(f'  MLP (ансамбль 5 моделей): 80.83%')
print(f'  CNN (baseline):           {test_acc:.2f}%')
print(f'  Улучшение:                {test_acc - 80.83:+.2f}%')
print()


# ===========================================================================
# Сохранение результатов
# ===========================================================================

# Создание директории для результатов
os.makedirs('results/baseline_results', exist_ok=True)

# Сохранение модели
torch.save(baseline_model.state_dict(), 'results/baseline_results/baseline_cnn_model.pth')
print('✓ Модель сохранена: results/baseline_results/baseline_cnn_model.pth')

# Визуализация обучения
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(history['train_loss'], label='Train Loss', marker='o')
ax1.plot(history['val_loss'], label='Val Loss', marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy
ax2.plot(history['train_acc'], label='Train Acc', marker='o')
ax2.plot(history['val_acc'], label='Val Acc', marker='s')
ax2.axhline(y=80.83, color='r', linestyle='--', label='MLP Best (80.83%)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/baseline_results/baseline_cnn_training.png', dpi=150, bbox_inches='tight')
print('✓ Графики сохранены: results/baseline_results/baseline_cnn_training.png')

# Сохранение текстового отчета
with open('results/baseline_results/baseline_cnn_results.txt', 'w', encoding='utf-8') as f:
    f.write('=' * 70 + '\n')
    f.write('BASELINE CNN - РЕЗУЛЬТАТЫ ОБУЧЕНИЯ\n')
    f.write('=' * 70 + '\n\n')

    f.write('АРХИТЕКТУРА:\n')
    f.write('  Conv2d(1, 32, 3) + ReLU + MaxPool(2)\n')
    f.write('  Conv2d(32, 64, 3) + ReLU + MaxPool(2)\n')
    f.write('  Flatten + FC(3136, 128) + ReLU + Dropout(0.5)\n')
    f.write('  FC(128, 11)\n\n')

    f.write('ГИПЕРПАРАМЕТРЫ:\n')
    f.write(f'  Epochs: {NUM_EPOCHS}\n')
    f.write(f'  Batch size: {BATCH_SIZE}\n')
    f.write(f'  Learning rate: 0.001\n')
    f.write(f'  Optimizer: Adam\n')
    f.write(f'  Dropout: 0.5\n\n')

    f.write(f'ПАРАМЕТРЫ МОДЕЛИ: {total_params:,}\n\n')

    f.write('РЕЗУЛЬТАТЫ:\n')
    f.write(f'  Test Loss: {test_loss:.4f}\n')
    f.write(f'  Test Accuracy: {test_acc:.2f}%\n\n')

    f.write('СРАВНЕНИЕ С MLP:\n')
    f.write(f'  MLP (ансамбль): 80.83%\n')
    f.write(f'  CNN (baseline): {test_acc:.2f}%\n')
    f.write(f'  Улучшение: {test_acc - 80.83:+.2f}%\n\n')

    f.write('ИСТОРИЯ ОБУЧЕНИЯ:\n')
    f.write('Epoch | Train Loss | Train Acc | Val Loss | Val Acc\n')
    f.write('-' * 60 + '\n')
    for i in range(NUM_EPOCHS):
        f.write(f'{i+1:5d} | {history["train_loss"][i]:10.4f} | '
                f'{history["train_acc"][i]:9.2f} | {history["val_loss"][i]:8.4f} | '
                f'{history["val_acc"][i]:7.2f}\n')

print('✓ Отчет сохранен: results/baseline_results/baseline_cnn_results.txt')
print()

print('=' * 70)
print('ОБУЧЕНИЕ ЗАВЕРШЕНО')
print('=' * 70)
print(f'Финальная точность на тесте: {test_acc:.2f}%')
print(f'Улучшение по сравнению с MLP: {test_acc - 80.83:+.2f}%')
print()
print('Следующие шаги:')
print('  1. Эксперименты с глубокими CNN архитектурами')
print('  2. Аугментация данных')
print('  3. Batch Normalization')
print('  4. Ансамблирование')
print('=' * 70)
