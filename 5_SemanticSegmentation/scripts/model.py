"""
Архитектуры моделей для семантической сегментации сосудов сетчатки.

Реализованы:
- UNet: базовая архитектура U-Net с 4 уровнями encoder/decoder
- UNetResNet34: U-Net с предобученным ResNet34 в качестве encoder
"""

import torch
import torch.nn as nn
import torchvision.models as models


class DoubleConv(nn.Module):
    """Блок из двух свёрточных слоёв: (Conv2d → BN → ReLU) × 2."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Базовая архитектура U-Net для бинарной сегментации.

    Encoder: 4 уровня (64 → 128 → 256 → 512), bottleneck 1024
    Decoder: 4 уровня (512 → 256 → 128 → 64)
    Skip connections между encoder и decoder на каждом уровне.

    Параметры: ~7.8M
    """

    def __init__(self, in_channels=3, out_channels=1, features=None):
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512]

        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        prev_channels = in_channels
        for f in features:
            self.encoder_blocks.append(DoubleConv(prev_channels, f))
            prev_channels = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        reversed_features = list(reversed(features))
        prev_ch = features[-1] * 2  # Выход bottleneck
        for f in reversed_features:
            self.upconvs.append(
                nn.ConvTranspose2d(prev_ch, f, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(DoubleConv(f * 2, f))  # f*2 из-за cat со skip
            prev_ch = f  # Выход decoder блока

        # Выходной слой (без активации — используем BCEWithLogitsLoss)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        skip_connections = []
        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = list(reversed(skip_connections))
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)

            # Обработка разных размеров (если вход не кратен степени 2)
            skip = skip_connections[i]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(
                    x, size=skip.shape[2:], mode='bilinear', align_corners=True
                )

            x = torch.cat([skip, x], dim=1)
            x = self.decoder_blocks[i](x)

        return self.final_conv(x)

    def get_num_parameters(self):
        """Возвращает количество параметров модели."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


class UNetResNet34(nn.Module):
    """
    U-Net с предобученным ResNet34 в качестве encoder.

    Encoder: ResNet34 (предобученный на ImageNet)
        - layer0 (conv1 + bn1 + relu + maxpool): 64 channels
        - layer1: 64 channels
        - layer2: 128 channels
        - layer3: 256 channels
        - layer4: 512 channels

    Decoder: 4 уровня с skip connections от encoder

    Параметры: ~24M
    """

    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super().__init__()

        # Загружаем предобученный ResNet34
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet34(weights=weights)

        # Encoder (используем слои ResNet34)
        self.encoder0 = nn.Sequential(
            resnet.conv1,   # (3, H, W) → (64, H/2, W/2)
            resnet.bn1,
            resnet.relu,
        )
        self.pool0 = resnet.maxpool      # (64, H/2, W/2) → (64, H/4, W/4)
        self.encoder1 = resnet.layer1    # (64, H/4, W/4) → (64, H/4, W/4)
        self.encoder2 = resnet.layer2    # (64, H/4, W/4) → (128, H/8, W/8)
        self.encoder3 = resnet.layer3    # (128, H/8, W/8) → (256, H/16, W/16)
        self.encoder4 = resnet.layer4    # (256, H/16, W/16) → (512, H/32, W/32)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(256 + 256, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(128 + 128, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(64 + 64, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(64 + 64, 64)

        self.upconv0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder0 = DoubleConv(32, 32)

        # Выходной слой
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e0 = self.encoder0(x)       # (64, H/2, W/2)
        e0_pool = self.pool0(e0)     # (64, H/4, W/4)
        e1 = self.encoder1(e0_pool)  # (64, H/4, W/4)
        e2 = self.encoder2(e1)       # (128, H/8, W/8)
        e3 = self.encoder3(e2)       # (256, H/16, W/16)
        e4 = self.encoder4(e3)       # (512, H/32, W/32)

        # Decoder
        d4 = self.upconv4(e4)        # (256, H/16, W/16)
        d4 = self._pad_and_cat(d4, e3)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)        # (128, H/8, W/8)
        d3 = self._pad_and_cat(d3, e2)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)        # (64, H/4, W/4)
        d2 = self._pad_and_cat(d2, e1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)        # (64, H/2, W/2)
        d1 = self._pad_and_cat(d1, e0)
        d1 = self.decoder1(d1)

        d0 = self.upconv0(d1)        # (32, H, W)
        d0 = self.decoder0(d0)

        return self.final_conv(d0)

    @staticmethod
    def _pad_and_cat(x, skip):
        """Выравнивает размеры и конкатенирует skip connection."""
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode='bilinear', align_corners=True
            )
        return torch.cat([x, skip], dim=1)

    def get_num_parameters(self):
        """Возвращает количество параметров модели."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


def create_model(arch='unet', pretrained=True):
    """
    Фабричная функция для создания модели.

    Args:
        arch: 'unet' или 'resnet34'
        pretrained: использовать предобученные веса (для ResNet34)

    Returns:
        модель nn.Module
    """
    if arch == 'unet':
        model = UNet(in_channels=3, out_channels=1)
    elif arch == 'resnet34':
        model = UNetResNet34(in_channels=3, out_channels=1, pretrained=pretrained)
    else:
        raise ValueError(f"Неизвестная архитектура: {arch}. Доступны: unet, resnet34")

    params = model.get_num_parameters()
    print(f"Модель: {arch}")
    print(f"  Всего параметров: {params['total']:,}")
    print(f"  Обучаемых: {params['trainable']:,}")

    return model


if __name__ == '__main__':
    # Тест моделей
    print("=" * 60)
    print("Тестирование архитектур")
    print("=" * 60)

    x = torch.randn(1, 3, 512, 512)

    print("\n1. Базовый U-Net:")
    model1 = create_model('unet')
    out1 = model1(x)
    print(f"  Вход: {x.shape} → Выход: {out1.shape}")

    print("\n2. U-Net + ResNet34:")
    model2 = create_model('resnet34', pretrained=False)
    out2 = model2(x)
    print(f"  Вход: {x.shape} → Выход: {out2.shape}")
