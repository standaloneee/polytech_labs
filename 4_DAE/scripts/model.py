"""
Denoising Autoencoder Architecture
Основано на статье: "Medical image denoising using convolutional denoising autoencoders"
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder (кодировщик) сжимает изображение в latent representation.

    Архитектура:
    - 3 сверточных блока с увеличением количества каналов
    - Каждый блок: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
    - Уменьшает пространственное разрешение, увеличивает количество каналов
    """

    def __init__(self, in_channels=3, base_channels=32):
        """
        Args:
            in_channels: Количество входных каналов (3 для RGB)
            base_channels: Базовое количество каналов в первом слое
        """
        super(Encoder, self).__init__()

        # Блок 1: 3 -> 32 каналов, 640x480 -> 320x240
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling
        )

        # Блок 2: 32 -> 64 каналов, 320x240 -> 160x120
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Блок 3: 64 -> 128 каналов, 160x120 -> 80x60
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """
        Forward pass через encoder

        Args:
            x: Входное изображение [batch, 3, H, W]

        Returns:
            Latent representation [batch, 128, H/8, W/8]
        """
        x1 = self.enc1(x)  # [batch, 32, H/2, W/2]
        x2 = self.enc2(x1)  # [batch, 64, H/4, W/4]
        x3 = self.enc3(x2)  # [batch, 128, H/8, W/8]
        return x3


class Decoder(nn.Module):
    """
    Decoder (декодировщик) восстанавливает изображение из latent representation.

    Архитектура:
    - 3 транспонированных сверточных блока с уменьшением количества каналов
    - Каждый блок: ConvTranspose2d -> BatchNorm2d -> ReLU -> Conv2d
    - Увеличивает пространственное разрешение, уменьшает количество каналов
    """

    def __init__(self, out_channels=3, base_channels=32):
        """
        Args:
            out_channels: Количество выходных каналов (3 для RGB)
            base_channels: Базовое количество каналов (должно совпадать с encoder)
        """
        super(Decoder, self).__init__()

        # Блок 1: 128 -> 64 каналов, 80x60 -> 160x120
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2,
                             kernel_size=2, stride=2),  # Upsampling
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )

        # Блок 2: 64 -> 32 каналов, 160x120 -> 320x240
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels,
                             kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Блок 3: 32 -> 3 каналов, 320x240 -> 640x480
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2,
                             kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Выход в диапазоне [0, 1]
        )

    def forward(self, x):
        """
        Forward pass через decoder

        Args:
            x: Latent representation [batch, 128, H/8, W/8]

        Returns:
            Восстановленное изображение [batch, 3, H, W]
        """
        x1 = self.dec1(x)  # [batch, 64, H/4, W/4]
        x2 = self.dec2(x1)  # [batch, 32, H/2, W/2]
        x3 = self.dec3(x2)  # [batch, 3, H, W]
        return x3


class DenoisingAutoencoder(nn.Module):
    """
    Полная модель Denoising Autoencoder = Encoder + Decoder

    Принимает зашумленное изображение, кодирует его в latent space,
    затем декодирует обратно в чистое изображение.
    """

    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        """
        Args:
            in_channels: Количество входных каналов (3 для RGB)
            out_channels: Количество выходных каналов (3 для RGB)
            base_channels: Базовое количество каналов в encoder
        """
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = Encoder(in_channels=in_channels, base_channels=base_channels)
        self.decoder = Decoder(out_channels=out_channels, base_channels=base_channels)

    def forward(self, x):
        """
        Forward pass через весь автокодировщик

        Args:
            x: Зашумленное изображение [batch, 3, H, W]

        Returns:
            Восстановленное (очищенное) изображение [batch, 3, H, W]
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def get_num_parameters(self):
        """
        Подсчет количества параметров модели

        Returns:
            Словарь с количеством параметров encoder, decoder и всего
        """
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = encoder_params + decoder_params

        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'total': total_params,
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


def create_model(base_channels=32, device='cpu'):
    """
    Создать модель и переместить на устройство

    Args:
        base_channels: Базовое количество каналов
        device: Устройство для вычислений ('cpu', 'cuda', 'mps')

    Returns:
        Модель DenoisingAutoencoder на указанном устройстве
    """
    model = DenoisingAutoencoder(base_channels=base_channels)
    model = model.to(device)

    # Вывести информацию о модели
    params_info = model.get_num_parameters()
    print(f"Модель создана на устройстве: {device}")
    print(f"Параметры Encoder: {params_info['encoder']:,}")
    print(f"Параметры Decoder: {params_info['decoder']:,}")
    print(f"Всего параметров: {params_info['total']:,}")

    return model


# Пример использования
if __name__ == '__main__':
    # Создать модель
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = create_model(base_channels=32, device=device)

    # Тестовый прогон
    x = torch.randn(1, 3, 480, 640).to(device)  # [batch, channels, height, width]
    print(f"\nВход: {x.shape}")

    output = model(x)
    print(f"Выход: {output.shape}")

    assert x.shape == output.shape, "Размеры входа и выхода не совпадают!"
    print("\n✓ Модель работает корректно!")
