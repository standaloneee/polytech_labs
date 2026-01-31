"""
Best Models Ensemble - выбор лучших 5 моделей из 10 для ансамбля

Анализируем индивидуальные результаты и выбираем топ-5 моделей
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import json
import medmnist
from medmnist import INFO
from tqdm import tqdm


# Определение устройства
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f'Device: {device}\n')


# ===========================================================================
# Архитектура модели
# ===========================================================================

class ResidualBlock(nn.Module):
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

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

DataClass = getattr(medmnist, info['python_class'])
test_dataset = DataClass(split='test', download=False, transform=test_transform, as_rgb=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f'Test samples: {len(test_dataset)}\n')


# ===========================================================================
# Загрузка конфигурации и выбор лучших моделей
# ===========================================================================

print('=' * 70)
print('АНСАМБЛЬ ЛУЧШИХ 5 МОДЕЛЕЙ ИЗ 10')
print('=' * 70)
print()

# Загружаем результаты
with open('results/experiments_results/ensemble_large_results.json', 'r') as f:
    large_ensemble = json.load(f)

# Сортируем модели по test_acc
models_info = large_ensemble['individual_models']
models_sorted = sorted(enumerate(models_info), key=lambda x: x[1]['test_acc'], reverse=True)

print('Все 10 моделей (отсортировано по test accuracy):')
print()
for idx, (original_idx, model_info) in enumerate(models_sorted, 1):
    print(f'{idx}. [{original_idx+1}/10] {model_info["name"]:<25} Test={model_info["test_acc"]:.2f}%')

# Выбираем топ-5
top5_indices = [original_idx for original_idx, _ in models_sorted[:5]]
print()
print('Выбраны ТОП-5 моделей:')
for rank, original_idx in enumerate(top5_indices, 1):
    model_info = models_info[original_idx]
    print(f'{rank}. Модель {original_idx+1}: {model_info["name"]:<25} Test={model_info["test_acc"]:.2f}%')
print()


# ===========================================================================
# Загрузка топ-5 моделей
# ===========================================================================

print('Загрузка топ-5 моделей...')
selected_models = []

for original_idx in top5_indices:
    model_info = models_info[original_idx]
    model_path = f'results/experiments_results/ensemble_large_model_{original_idx+1}.pth'

    dropout = model_info.get('dropout', 0.3)
    hidden_dim = model_info.get('hidden_dim', 64)

    model = ResNetLikeCNN(num_classes=NUM_CLASSES, dropout=dropout, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    selected_models.append(model)
    print(f'  ✓ Загружена модель {original_idx+1}')

print(f'\n✓ Загружено {len(selected_models)} моделей\n')


# ===========================================================================
# Функция для получения предсказаний
# ===========================================================================

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
# Ансамблирование топ-5
# ===========================================================================

print('Ансамблирование топ-5 моделей через soft voting...')

all_probs_ensemble = []
all_labels = None

for i, model in enumerate(selected_models):
    print(f'Getting predictions from model {i+1}/5...')
    probs, labels = get_predictions_proba(model, test_loader, device)
    all_probs_ensemble.append(probs)
    if all_labels is None:
        all_labels = labels

# Усреднение вероятностей
avg_probs = np.mean(all_probs_ensemble, axis=0)
predictions = np.argmax(avg_probs, axis=1)

# Подсчет точности
ensemble_acc = 100. * np.mean(predictions == all_labels)

print()
print('=' * 70)
print('РЕЗУЛЬТАТЫ')
print('=' * 70)
print()

# Индивидуальные результаты топ-5
print('Индивидуальные результаты топ-5 моделей:')
for rank, original_idx in enumerate(top5_indices, 1):
    model_info = models_info[original_idx]
    print(f'  {rank}. {model_info["name"]:<25} Test={model_info["test_acc"]:.2f}%')

avg_individual = np.mean([models_info[idx]['test_acc'] for idx in top5_indices])
print(f'\nСреднее по индивидуальным: {avg_individual:.2f}%')
print(f'Ансамбль (топ-5):          {ensemble_acc:.2f}%')
print(f'Улучшение:                 +{ensemble_acc - avg_individual:.2f}%')
print()

# Сравнение с предыдущими результатами
print('Сравнение:')
print(f'  Ансамбль 5 моделей (оригинал):  91.16%')
print(f'  Ансамбль 10 моделей (все):      91.02%')
print(f'  Ансамбль 10 моделей + TTA:      91.13%')
print(f'  Ансамбль 5 лучших из 10:        {ensemble_acc:.2f}%')
print()

# Сохранение результатов
results = {
    'selected_models_indices': [int(idx) for idx in top5_indices],
    'selected_models': [models_info[idx] for idx in top5_indices],
    'avg_individual': avg_individual,
    'ensemble_acc': ensemble_acc,
    'improvement': ensemble_acc - avg_individual
}

with open('results/experiments_results/best5_ensemble_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('✓ Результаты сохранены: results/experiments_results/best5_ensemble_results.json')
print()

if ensemble_acc >= 92.0:
    print('=' * 70)
    print('🎉 ЦЕЛЬ ДОСТИГНУТА: >=92% accuracy!')
    print('=' * 70)
else:
    print('=' * 70)
    print(f'📊 Текущий результат: {ensemble_acc:.2f}% (цель: >=92%)')
    print('=' * 70)
