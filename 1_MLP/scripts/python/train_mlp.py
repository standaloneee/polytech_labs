"""
Скрипт для обучения многослойного перцептрона на датасете OrganCMNIST
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
print(f'MedMNIST version: {medmnist.__version__}\n')

# ==================== ЗАГРУЗКА ДАННЫХ ====================

print("=" * 60)
print("ЗАГРУЗКА ДАТАСЕТА")
print("=" * 60)

data_flag = 'organcmnist'
info = INFO[data_flag]

print(f"Датасет: {info['task']}")
print(f"Количество классов: {len(info['label'])}")
print(f"Классы: {info['label']}")
print(f"Размер изображения: {info['n_channels']} x 28 x 28\n")

# Трансформации для данных
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Загружаем train, validation и test наборы
# download=False т.к. датасет уже должен быть скачан в ~/.medmnist/
print("Загрузка данных...")
train_dataset = OrganCMNIST(split='train', transform=transform, download=False)
val_dataset = OrganCMNIST(split='val', transform=transform, download=False)
test_dataset = OrganCMNIST(split='test', transform=transform, download=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}\n")

# ==================== ОПРЕДЕЛЕНИЕ МОДЕЛИ ====================

class MLP(nn.Module):
    """Многослойный перцептрон для классификации"""

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

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Выходной слой
        layers.append(nn.Linear(prev_size, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten изображения
        x = x.view(-1, self.input_size)
        return self.model(x)

    def count_parameters(self):
        """Подсчитывает количество обучаемых параметров"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==================== ФУНКЦИИ ОБУЧЕНИЯ ====================

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

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Оценка модели на валидационной/тестовой выборке"""
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


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, device, scheduler=None):
    """Полный цикл обучения модели"""
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 40)

        # Обучение
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Валидация
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Обновление scheduler
        if scheduler is not None:
            scheduler.step()

        # Сохранение истории
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Вывод результатов
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f'✓ New best validation accuracy: {best_val_acc:.2f}%')

    return history, best_val_acc


def plot_training_history(history, title='Training History', save_path=None):
    """Визуализация истории обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранен: {save_path}")

    plt.show()


# ==================== ОБУЧЕНИЕ BASELINE МОДЕЛИ ====================

print("=" * 60)
print("ОБУЧЕНИЕ BASELINE МОДЕЛИ")
print("=" * 60)

# Параметры
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Инициализация модели
set_seed()
baseline_model = MLP(
    input_size=28*28,
    hidden_sizes=[128, 64],
    num_classes=len(info['label']),
    dropout=0.0,
    use_batch_norm=False,
    activation='relu'
).to(device)

print(f"\nArchitecture: {baseline_model}")
print(f"\nКоличество параметров: {baseline_model.count_parameters():,}")

# Loss и optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(baseline_model.parameters(), lr=LEARNING_RATE)

print(f"\nПараметры обучения:")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Epochs: {NUM_EPOCHS}")
print(f"  - Learning rate: {LEARNING_RATE}")
print(f"  - Optimizer: Adam")
print(f"  - Loss: CrossEntropyLoss")

# Обучение
print("\nНачало обучения...")
baseline_history, baseline_best_acc = train_model(
    baseline_model, train_loader, val_loader, criterion, optimizer,
    NUM_EPOCHS, device
)

# ==================== ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ ====================

print("\n" + "=" * 60)
print("ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
print("=" * 60)

test_loss, test_acc = evaluate(baseline_model, test_loader, criterion, device)
print(f'\nBaseline Model Results:')
print(f'  - Best Val Accuracy: {baseline_best_acc:.2f}%')
print(f'  - Test Loss: {test_loss:.4f}')
print(f'  - Test Accuracy: {test_acc:.2f}%')

# ==================== ВИЗУАЛИЗАЦИЯ ====================

print("\n" + "=" * 60)
print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 60)

# Создаем директорию для результатов
os.makedirs('results', exist_ok=True)

# Сохраняем графики
plot_training_history(baseline_history, 'Baseline Model Training',
                     save_path='results/baseline_training.png')

# Сохраняем модель
torch.save(baseline_model.state_dict(), 'results/baseline_model.pth')
print("Модель сохранена: results/baseline_model.pth")

# Сохраняем результаты в текстовый файл
with open('results/baseline_results.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("BASELINE MODEL RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Architecture: MLP with hidden layers [128, 64]\n")
    f.write(f"Parameters: {baseline_model.count_parameters():,}\n")
    f.write(f"Activation: ReLU\n")
    f.write(f"Dropout: 0.0\n")
    f.write(f"Batch Normalization: False\n\n")
    f.write(f"Training:\n")
    f.write(f"  - Batch size: {BATCH_SIZE}\n")
    f.write(f"  - Epochs: {NUM_EPOCHS}\n")
    f.write(f"  - Learning rate: {LEARNING_RATE}\n")
    f.write(f"  - Optimizer: Adam\n\n")
    f.write(f"Results:\n")
    f.write(f"  - Best Val Accuracy: {baseline_best_acc:.2f}%\n")
    f.write(f"  - Test Loss: {test_loss:.4f}\n")
    f.write(f"  - Test Accuracy: {test_acc:.2f}%\n\n")
    f.write(f"Training History:\n")
    f.write(f"  Final Train Loss: {baseline_history['train_loss'][-1]:.4f}\n")
    f.write(f"  Final Train Acc: {baseline_history['train_acc'][-1]:.2f}%\n")
    f.write(f"  Final Val Loss: {baseline_history['val_loss'][-1]:.4f}\n")
    f.write(f"  Final Val Acc: {baseline_history['val_acc'][-1]:.2f}%\n")

print("Результаты сохранены: results/baseline_results.txt")

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)
print("\nВсе результаты сохранены в директории 'results/'")
print("  - baseline_training.png - графики обучения")
print("  - baseline_model.pth - веса модели")
print("  - baseline_results.txt - текстовые результаты")
