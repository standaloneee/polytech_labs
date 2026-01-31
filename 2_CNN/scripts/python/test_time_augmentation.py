"""
Test-Time Augmentation (TTA) - усреднение предсказаний с аугментацией

Применяет легкую аугментацию при тестировании и усредняет результаты
для повышения робастности предсказаний
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import json
import medmnist
from medmnist import INFO
from tqdm import tqdm
import os


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
# TTA Transforms
# ===========================================================================

def get_tta_transforms():
    """Создать набор трансформаций для TTA"""
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Легкие аугментации для TTA
    tta_transforms = [
        # Оригинал
        base_transform,

        # Horizontal flip
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),

        # Small rotation +5
        transforms.Compose([
            transforms.RandomRotation(degrees=(5, 5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),

        # Small rotation -5
        transforms.Compose([
            transforms.RandomRotation(degrees=(-5, -5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),

        # Flip + rotation
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(degrees=(5, 5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
    ]

    return tta_transforms


# ===========================================================================
# Загрузка данных
# ===========================================================================

data_flag = 'organcmnist'
info = INFO[data_flag]
NUM_CLASSES = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

print('=' * 70)
print('TEST-TIME AUGMENTATION (TTA)')
print('=' * 70)
print()


# ===========================================================================
# Загрузка моделей
# ===========================================================================

print('Загрузка моделей...')

model_files = []
ensemble_results_file = None

# Проверяем какой ансамбль доступен (предпочитаем большой)
if os.path.exists('results/experiments_results/ensemble_large_results.json'):
    ensemble_results_file = 'results/experiments_results/ensemble_large_results.json'
    model_pattern = 'results/experiments_results/ensemble_large_model_{}.pth'
    print('Используем увеличенный ансамбль (10 моделей)')
elif os.path.exists('results/experiments_results/ensemble_results.json'):
    ensemble_results_file = 'results/experiments_results/ensemble_results.json'
    model_pattern = 'results/experiments_results/ensemble_model_{}.pth'
    print('Используем базовый ансамбль (5 моделей)')
else:
    print('Ошибка: Не найдены результаты ансамбля!')
    exit(1)

# Загружаем конфигурацию
with open(ensemble_results_file, 'r') as f:
    ensemble_config = json.load(f)

num_models = ensemble_config['num_models']

# Загружаем модели
models = []
for i in range(1, num_models + 1):
    model_path = model_pattern.format(i)

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

print(f'✓ Загружено {len(models)} моделей\n')


# ===========================================================================
# TTA Prediction
# ===========================================================================

def predict_with_tta(models, dataset, tta_transforms, batch_size=128):
    """Предсказание с TTA"""
    all_labels = []

    # Получаем labels
    for _, label in dataset:
        all_labels.append(label)
    all_labels = np.array(all_labels).squeeze()

    num_samples = len(dataset)
    num_classes = NUM_CLASSES
    num_tta = len(tta_transforms)

    # Массив для накопления вероятностей
    accumulated_probs = np.zeros((num_samples, num_classes))

    print(f'Применение {num_tta} TTA трансформаций...')

    for tta_idx, transform in enumerate(tta_transforms):
        print(f'\nTTA {tta_idx + 1}/{num_tta}')

        # Создаем dataset с текущей трансформацией
        class TTADataset(Dataset):
            def __init__(self, base_dataset, transform):
                self.base_dataset = base_dataset
                self.transform = transform

            def __len__(self):
                return len(self.base_dataset)

            def __getitem__(self, idx):
                # Получаем PIL изображение из базового датасета
                # OrganCMNIST возвращает (img, label) где img - PIL Image
                from PIL import Image
                # Получаем сырые данные
                data_sample = self.base_dataset[idx]
                if len(data_sample) == 2:
                    img_tensor, label = data_sample
                    # Преобразуем тензор обратно в PIL Image для трансформаций
                    img_array = img_tensor.squeeze().numpy()
                    img_array = ((img_array * 0.5 + 0.5) * 255).astype('uint8')  # денормализация
                    img = Image.fromarray(img_array, mode='L')
                    # Применяем трансформацию
                    img = self.transform(img)
                    return img, label
                else:
                    raise ValueError(f"Unexpected data format: {type(data_sample)}")

        tta_dataset = TTADataset(dataset, transform)
        tta_loader = DataLoader(tta_dataset, batch_size=batch_size, shuffle=False)

        # Получаем предсказания от всех моделей для этой TTA
        tta_probs = np.zeros((num_samples, num_classes))
        sample_idx = 0

        for model_idx, model in enumerate(models):
            model.eval()
            batch_start = 0

            with torch.no_grad():
                for images, _ in tqdm(tta_loader, desc=f'  Model {model_idx+1}/{len(models)}', leave=False):
                    images = images.to(device)
                    outputs = model(images)
                    probs = F.softmax(outputs, dim=1).cpu().numpy()

                    batch_size_actual = probs.shape[0]
                    tta_probs[batch_start:batch_start + batch_size_actual] += probs
                    batch_start += batch_size_actual

        # Усредняем по моделям
        tta_probs /= len(models)

        # Добавляем к общему накоплению
        accumulated_probs += tta_probs

    # Финальное усреднение по всем TTA
    final_probs = accumulated_probs / num_tta

    # Предсказания
    predictions = np.argmax(final_probs, axis=1)
    accuracy = 100. * np.mean(predictions == all_labels)

    return accuracy, predictions, all_labels


# ===========================================================================
# Тестирование
# ===========================================================================

print('Загрузка test dataset...')
test_transform_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_dataset = DataClass(split='test', download=False, transform=test_transform_base, as_rgb=False)
print(f'Test samples: {len(test_dataset)}\n')

# Baseline (без TTA)
print('Baseline (без TTA)...')
test_loader_baseline = DataLoader(test_dataset, batch_size=128, shuffle=False)

baseline_probs = np.zeros((len(test_dataset), NUM_CLASSES))
batch_start = 0

for model in models:
    model.eval()
    batch_idx = 0

    with torch.no_grad():
        for images, _ in tqdm(test_loader_baseline, desc='Baseline', leave=False):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()

            batch_size = probs.shape[0]
            baseline_probs[batch_idx:batch_idx + batch_size] += probs
            batch_idx += batch_size

baseline_probs /= len(models)
baseline_predictions = np.argmax(baseline_probs, axis=1)

# Получаем labels
test_labels = []
for _, label in test_dataset:
    test_labels.append(label)
test_labels = np.array(test_labels).squeeze()

baseline_acc = 100. * np.mean(baseline_predictions == test_labels)
print(f'Baseline Accuracy: {baseline_acc:.2f}%\n')

# TTA
tta_transforms = get_tta_transforms()
print(f'Применение TTA с {len(tta_transforms)} трансформациями...')
tta_acc, _, _ = predict_with_tta(models, test_dataset, tta_transforms)

print('\n' + '=' * 70)
print('РЕЗУЛЬТАТЫ')
print('=' * 70)
print(f'\nBaseline (без TTA):      {baseline_acc:.2f}%')
print(f'С TTA ({len(tta_transforms)} transforms): {tta_acc:.2f}%')
print(f'Улучшение:               +{tta_acc - baseline_acc:.2f}%')

# Сохранение результатов
results = {
    'num_models': len(models),
    'num_tta_transforms': len(tta_transforms),
    'baseline_acc': baseline_acc,
    'tta_acc': tta_acc,
    'improvement': tta_acc - baseline_acc
}

with open('results/experiments_results/tta_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n✓ Результаты сохранены: results/experiments_results/tta_results.json')

print('\n' + '=' * 70)
if tta_acc > 92.0:
    print('🎉 ЦЕЛЬ ДОСТИГНУТА: >92% accuracy!')
else:
    print(f'📊 Текущий результат: {tta_acc:.2f}% (цель: >92%)')
print('=' * 70)
