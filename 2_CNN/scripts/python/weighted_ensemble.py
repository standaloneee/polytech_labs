"""
Weighted Ensemble - оптимизация весов для достижения максимальной точности

Использует validation set для поиска оптимальных весов каждой модели
вместо простого усреднения (soft voting)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from scipy.optimize import minimize
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
# Архитектура модели (та же что и в ансамблях)
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

val_dataset = DataClass(split='val', download=False, transform=test_transform, as_rgb=False)
test_dataset = DataClass(split='test', download=False, transform=test_transform, as_rgb=False)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f'Val samples: {len(val_dataset)}')
print(f'Test samples: {len(test_dataset)}\n')


# ===========================================================================
# Загрузка обученных моделей
# ===========================================================================

print('=' * 70)
print('WEIGHTED ENSEMBLE')
print('=' * 70)
print()

# Попробуем загрузить модели из увеличенного ансамбля
import os

model_files = []
ensemble_results_file = None

# Проверяем какой ансамбль доступен
if os.path.exists('results/experiments_results/ensemble_large_results.json'):
    ensemble_results_file = 'results/experiments_results/ensemble_large_results.json'
    model_pattern = 'results/experiments_results/ensemble_large_model_{}.pth'
elif os.path.exists('results/experiments_results/ensemble_results.json'):
    ensemble_results_file = 'results/experiments_results/ensemble_results.json'
    model_pattern = 'results/experiments_results/ensemble_model_{}.pth'
else:
    print('Ошибка: Не найдены результаты ансамбля!')
    exit(1)

# Загружаем конфигурацию
with open(ensemble_results_file, 'r') as f:
    ensemble_config = json.load(f)

num_models = ensemble_config['num_models']
print(f'Загрузка {num_models} моделей из ансамбля...')

# Загружаем модели
models = []
for i in range(1, num_models + 1):
    model_path = model_pattern.format(i)

    # Определяем параметры модели из конфигурации
    if 'individual_models' in ensemble_config and len(ensemble_config['individual_models']) >= i:
        model_info = ensemble_config['individual_models'][i-1]
        dropout = model_info.get('dropout', 0.3)
        hidden_dim = model_info.get('hidden_dim', 64)
    else:
        dropout = 0.3
        hidden_dim = 64

    model = ResNetLikeCNN(num_classes=NUM_CLASSES, dropout=dropout, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    models.append(model)
    print(f'  ✓ Загружена модель {i}/{num_models}')

print(f'\nУспешно загружено {len(models)} моделей\n')


# ===========================================================================
# Получение предсказаний от всех моделей
# ===========================================================================

def get_all_predictions(models, dataloader, device):
    """Получить предсказания от всех моделей"""
    all_model_probs = []
    all_labels = None

    for i, model in enumerate(models):
        model.eval()
        model_probs = []
        labels = []

        with torch.no_grad():
            for images, label_batch in tqdm(dataloader, desc=f'Model {i+1}/{len(models)}', leave=False):
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                model_probs.append(probs.cpu().numpy())
                labels.append(label_batch.squeeze().numpy())

        all_model_probs.append(np.vstack(model_probs))
        if all_labels is None:
            all_labels = np.concatenate(labels)

    # Shape: (num_models, num_samples, num_classes)
    return np.array(all_model_probs), all_labels


print('Получение предсказаний на validation set...')
val_probs, val_labels = get_all_predictions(models, val_loader, device)
print(f'Val predictions shape: {val_probs.shape}\n')

print('Получение предсказаний на test set...')
test_probs, test_labels = get_all_predictions(models, test_loader, device)
print(f'Test predictions shape: {test_probs.shape}\n')


# ===========================================================================
# Оптимизация весов на validation set
# ===========================================================================

print('Оптимизация весов на validation set...')

def weighted_ensemble_accuracy(weights, probs, labels):
    """Вычислить accuracy для заданных весов"""
    # Нормализуем веса
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Взвешенное усреднение вероятностей
    # probs shape: (num_models, num_samples, num_classes)
    weighted_probs = np.tensordot(weights, probs, axes=([0], [0]))

    # Предсказания
    predictions = np.argmax(weighted_probs, axis=1)

    # Accuracy
    accuracy = np.mean(predictions == labels)

    return accuracy


def objective(weights, probs, labels):
    """Функция потерь для оптимизации (минимизируем negative accuracy)"""
    return -weighted_ensemble_accuracy(weights, probs, labels)


# Начальные веса (равномерное распределение)
initial_weights = np.ones(num_models) / num_models

# Ограничения: веса должны быть положительными и суммироваться в 1
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
bounds = [(0, 1) for _ in range(num_models)]

# Оптимизация
print('Запуск оптимизации...')
result = minimize(
    objective,
    initial_weights,
    args=(val_probs, val_labels),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000, 'ftol': 1e-9}
)

optimal_weights = result.x
optimal_weights = optimal_weights / optimal_weights.sum()  # Нормализуем

print('\nОптимальные веса:')
for i, w in enumerate(optimal_weights):
    print(f'  Модель {i+1}: {w:.4f}')

# Валидация с оптимальными весами
val_acc_uniform = weighted_ensemble_accuracy(initial_weights, val_probs, val_labels) * 100
val_acc_weighted = weighted_ensemble_accuracy(optimal_weights, val_probs, val_labels) * 100

print(f'\nValidation Accuracy:')
print(f'  Uniform weights:  {val_acc_uniform:.2f}%')
print(f'  Optimal weights:  {val_acc_weighted:.2f}%')
print(f'  Improvement:      +{val_acc_weighted - val_acc_uniform:.2f}%')


# ===========================================================================
# Тестирование с оптимальными весами
# ===========================================================================

print('\nТестирование на test set...')

test_acc_uniform = weighted_ensemble_accuracy(initial_weights, test_probs, test_labels) * 100
test_acc_weighted = weighted_ensemble_accuracy(optimal_weights, test_probs, test_labels) * 100

print(f'\nTest Accuracy:')
print(f'  Uniform weights (Soft Voting):  {test_acc_uniform:.2f}%')
print(f'  Optimal weights (Weighted):      {test_acc_weighted:.2f}%')
print(f'  Improvement:                     +{test_acc_weighted - test_acc_uniform:.2f}%')


# ===========================================================================
# Сохранение результатов
# ===========================================================================

results = {
    'num_models': num_models,
    'optimal_weights': optimal_weights.tolist(),
    'val_acc_uniform': val_acc_uniform,
    'val_acc_weighted': val_acc_weighted,
    'test_acc_uniform': test_acc_uniform,
    'test_acc_weighted': test_acc_weighted,
    'improvement_val': val_acc_weighted - val_acc_uniform,
    'improvement_test': test_acc_weighted - test_acc_uniform
}

output_file = 'results/experiments_results/weighted_ensemble_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n✓ Результаты сохранены: {output_file}')

print('\n' + '=' * 70)
if test_acc_weighted > 92.0:
    print('🎉 ЦЕЛЬ ДОСТИГНУТА: >92% accuracy!')
else:
    print(f'📊 Текущий результат: {test_acc_weighted:.2f}% (цель: >92%)')
print('=' * 70)
