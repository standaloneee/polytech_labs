"""
Экспорт лучшей модели в ONNX формат

ONNX (Open Neural Network Exchange) - открытый формат для обмена моделями
между различными фреймворками (PyTorch, TensorFlow, ONNX Runtime, etc.)
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path


# Пути
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'


def export_to_onnx(model_path, output_path, model_type='resnet50', num_classes=5):
    """
    Экспортировать PyTorch модель в ONNX формат

    Args:
        model_path: Путь к .pth файлу модели
        output_path: Путь для сохранения .onnx файла
        model_type: Тип модели ('resnet50' или 'efficientnet_b0')
        num_classes: Количество классов
    """
    print(f"\n{'=' * 70}")
    print(f"Exporting model to ONNX")
    print(f"{'=' * 70}")
    print(f"Model: {model_path.name}")
    print(f"Type: {model_type}")

    # Создать модель с той же архитектурой
    if model_type == 'resnet50':
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == 'efficientnet_b0':
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Загрузить веса
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Создать dummy input для трассировки
    # Размер входа: batch_size=1, channels=3, height=224, width=224
    dummy_input = torch.randn(1, 3, 224, 224)

    # Экспортировать в ONNX
    print(f"\nExporting to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"✓ ONNX model saved: {output_path}")

    # Показать размер файла
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.2f} MB")

    # Проверить ONNX модель
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model verification passed")
    except ImportError:
        print("  (Install 'onnx' package to verify exported model)")
    except Exception as e:
        print(f"  Warning: ONNX verification failed: {e}")

    print(f"\n{'=' * 70}")
    print(f"✓ Export complete!")
    print(f"{'=' * 70}")


def main():
    """Главная функция"""

    # Найти лучшую модель (ResNet50 с любой стратегией показал 100% на тесте)
    # Выберем resnet50_full как итоговую лучшую модель
    best_model_files = list(MODELS_DIR.glob('resnet50_full_*_best.pth'))

    if not best_model_files:
        print("Error: No ResNet50 Full model found!")
        return

    best_model = best_model_files[0]

    # Путь для сохранения ONNX модели
    onnx_path = MODELS_DIR / 'best_model_resnet50.onnx'

    # Экспортировать
    export_to_onnx(
        model_path=best_model,
        output_path=onnx_path,
        model_type='resnet50',
        num_classes=5
    )

    print(f"\nUsage example:")
    print(f"  import onnxruntime as ort")
    print(f"  session = ort.InferenceSession('{onnx_path}')")
    print(f"  outputs = session.run(None, {{'input': your_input_data}})")


if __name__ == '__main__':
    main()
