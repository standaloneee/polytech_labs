"""
Скрипт для обучения моделей с Transfer Learning

Поддерживаемые модели:
- resnet50
- efficientnet_b0

Стратегии fine-tuning:
1. freeze: только голова (FC layer)
2. partial: последние 2 блока + голова
3. full: все параметры
"""

import argparse
import json
import os
import ssl
from pathlib import Path
from datetime import datetime

# Обход SSL для корпоративных прокси
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from tqdm import tqdm


# Пути
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
LOGS_DIR = PROJECT_ROOT / 'results' / 'logs'
METRICS_DIR = PROJECT_ROOT / 'results' / 'metrics'

# Создать директории
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# ImageNet нормализация
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed=42):
    """Установить seed для воспроизводимости"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(augment=True):
    """Получить трансформации для данных"""

    if augment:
        # Train transforms с аугментацией
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    # Val/test transforms без аугментации
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, val_transform


def get_dataloaders(batch_size=32, augment=True, num_workers=2):
    """Создать DataLoaders"""

    train_transform, val_transform = get_transforms(augment=augment)

    train_dataset = ImageFolder(DATA_DIR / 'train', transform=train_transform)
    val_dataset = ImageFolder(DATA_DIR / 'val', transform=val_transform)
    test_dataset = ImageFolder(DATA_DIR / 'test', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes

    return train_loader, val_loader, test_loader, num_classes, class_names


def create_model(model_name, num_classes, pretrained=True):
    """Создать модель из torchvision"""

    if model_name == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def freeze_backbone(model, model_type='resnet'):
    """Заморозить backbone, оставить только голову"""

    for param in model.parameters():
        param.requires_grad = False

    # Разморозить классификатор
    if model_type == 'resnet':
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_type == 'efficientnet':
        for param in model.classifier.parameters():
            param.requires_grad = True


def unfreeze_last_blocks(model, model_type='resnet', num_blocks=2):
    """Разморозить последние num_blocks блоков"""

    if model_type == 'resnet':
        layers_to_unfreeze = [model.layer4]
        if num_blocks >= 2:
            layers_to_unfreeze.append(model.layer3)

        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

    elif model_type == 'efficientnet':
        # EfficientNet в torchvision имеет структуру features[...блоки...]
        total_blocks = len(model.features)
        for i in range(total_blocks - num_blocks, total_blocks):
            for param in model.features[i].parameters():
                param.requires_grad = True


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Обучение на одной эпохе"""

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc='Training'):
        images, labels = images.to(device), labels.to(device)

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
    """Оценка модели"""

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train(args):
    """Основная функция обучения"""

    # Seed
    set_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    # DataLoaders
    print('Loading data...')
    train_loader, val_loader, test_loader, num_classes, class_names = get_dataloaders(
        batch_size=args.batch_size,
        augment=args.augment,
        num_workers=args.num_workers
    )
    print(f'Classes: {class_names}')
    print(f'Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}')

    # Model
    print(f'\nCreating model: {args.model}')
    model = create_model(args.model, num_classes, pretrained=True)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')

    # Apply freeze strategy
    if args.freeze == 'freeze':
        model_type = 'resnet' if 'resnet' in args.model else 'efficientnet'
        freeze_backbone(model, model_type=model_type)
        print('Freeze strategy: backbone frozen, training head only')
    elif args.freeze == 'partial':
        model_type = 'resnet' if 'resnet' in args.model else 'efficientnet'
        freeze_backbone(model, model_type=model_type)
        unfreeze_last_blocks(model, model_type=model_type, num_blocks=2)
        print('Freeze strategy: last 2 blocks + head unfrozen')
    else:
        print('Freeze strategy: full fine-tuning')

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {trainable_params:,}')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # Scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs//3, gamma=0.1)
    else:
        scheduler = None

    # TensorBoard
    experiment_name = f'{args.model}_{args.freeze}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir=LOGS_DIR / experiment_name)

    # Training loop
    print(f'\n{"="*70}')
    print(f'Training for {args.epochs} epochs')
    print(f'{"="*70}\n')

    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Scheduler step
        if scheduler:
            scheduler.step()

        # Log to TensorBoard
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)

        print(f'  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%')
        print(f'  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = MODELS_DIR / f'{experiment_name}_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names
            }, model_path)
            print(f'  ✓ New best model saved: {val_acc:.2f}%')

        print()

    # Test best model
    print(f'\n{"="*70}')
    print('Evaluating best model on test set')
    print(f'{"="*70}\n')

    best_model_path = MODELS_DIR / f'{experiment_name}_best.pth'
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.2f}%')

    # Save results
    results = {
        'experiment_name': experiment_name,
        'model': args.model,
        'freeze_strategy': args.freeze,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'optimizer': args.optimizer,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'history': history
    }

    results_path = METRICS_DIR / f'{experiment_name}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n✓ Training complete!')
    print(f'Best model: {best_model_path}')
    print(f'Results: {results_path}')

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model with transfer learning')

    # Model
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'efficientnet_b0'],
                       help='Model architecture')

    # Training
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'none'],
                       help='LR scheduler')

    # Freeze strategy
    parser.add_argument('--freeze', type=str, default='partial',
                       choices=['freeze', 'partial', 'full'],
                       help='Freeze strategy: freeze=head only, partial=last 2 blocks, full=all')

    # Data augmentation
    parser.add_argument('--augment', action='store_true', default=True,
                       help='Use data augmentation')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                       help='Disable data augmentation')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of data loading workers')

    args = parser.parse_args()
    train(args)
