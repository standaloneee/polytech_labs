"""
Test-Time Augmentation (TTA) для оригинального ансамбля 5 моделей

Применяет TTA к лучшему ансамблю (91.16%)
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


# Определение устройства
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f'Device: {device}\n')


# ===========================================================================
# Архитектуры моделей
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
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)  # Fixed to 64
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


class BatchNormCNN(nn.Module):
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


class DeepCNN(nn.Module):
    def __init__(self, num_classes=11, dropout=0.3):
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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 256 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
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
print('TTA ДЛЯ ОРИГИНАЛЬНОГО АНСАМБЛЯ (5 МОДЕЛЕЙ)')
print('=' * 70)
print()


# ===========================================================================
# Загрузка оригинального ансамбля
# ===========================================================================

print('Загрузка оригинального ансамбля (5 моделей)...')

with open('results/experiments_results/ensemble_results.json', 'r') as f:
    ensemble_config = json.load(f)

# Определяем архитектуры
architectures = [
    ('BatchNormCNN', BatchNormCNN),
    ('BatchNormCNN', BatchNormCNN),
    ('ResNetLikeCNN', ResNetLikeCNN),
    ('ResNetLikeCNN', ResNetLikeCNN),
    ('DeepCNN', DeepCNN),
]

models = []
for i, (arch_name, ArchClass) in enumerate(architectures, 1):
    model_path = f'results/experiments_results/ensemble_model_{i}.pth'
    model = ArchClass(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    models.append(model)
    print(f'  ✓ Загружена модель {i}: {arch_name}')

print(f'\n✓ Загружено {len(models)} моделей')
print(f'Оригинальная точность ансамбля: {ensemble_config["test_ensemble_acc"]:.2f}%')
print()


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
                from PIL import Image
                data_sample = self.base_dataset[idx]
                if len(data_sample) == 2:
                    img_tensor, label = data_sample
                    img_array = img_tensor.squeeze().numpy()
                    img_array = ((img_array * 0.5 + 0.5) * 255).astype('uint8')
                    img = Image.fromarray(img_array, mode='L')
                    img = self.transform(img)
                    return img, label
                else:
                    raise ValueError(f"Unexpected data format: {type(data_sample)}")

        tta_dataset = TTADataset(dataset, transform)
        tta_loader = DataLoader(tta_dataset, batch_size=batch_size, shuffle=False)

        # Получаем предсказания от всех моделей для этой TTA
        tta_probs = np.zeros((num_samples, num_classes))

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

# Baseline (оригинальный ансамбль без TTA)
baseline_acc = ensemble_config['test_ensemble_acc']
print(f'Baseline (оригинальный ансамбль): {baseline_acc:.2f}%\n')

# TTA
tta_transforms = get_tta_transforms()
print(f'Применение TTA с {len(tta_transforms)} трансформациями...')
tta_acc, _, _ = predict_with_tta(models, test_dataset, tta_transforms)

print('\n' + '=' * 70)
print('РЕЗУЛЬТАТЫ')
print('=' * 70)
print(f'\nBaseline (оригинальный ансамбль):  {baseline_acc:.2f}%')
print(f'С TTA ({len(tta_transforms)} transforms):     {tta_acc:.2f}%')
print(f'Улучшение:                        +{tta_acc - baseline_acc:.2f}%')

# Сохранение результатов
results = {
    'num_models': len(models),
    'num_tta_transforms': len(tta_transforms),
    'baseline_acc': baseline_acc,
    'tta_acc': tta_acc,
    'improvement': tta_acc - baseline_acc,
    'note': 'TTA applied to original 5-model ensemble'
}

with open('results/experiments_results/tta_original5_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n✓ Результаты сохранены: results/experiments_results/tta_original5_results.json')

print('\n' + '=' * 70)
if tta_acc >= 92.0:
    print('🎉 ЦЕЛЬ ДОСТИГНУТА: >=92% accuracy!')
else:
    print(f'📊 Текущий результат: {tta_acc:.2f}% (цель: >=92%)')
print('=' * 70)
