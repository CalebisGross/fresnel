"""
Tiny Depth Model - A minimal encoder-decoder for learning depth estimation fundamentals.

Architecture: ~2.1M parameters
- Encoder: 4 conv blocks (3→32→64→128→256 channels)
- Decoder: 4 upsample blocks (256→128→64→32→1 channels)
- Skip connections for better gradient flow and detail preservation

This is intentionally simple to understand the core concepts:
1. How encoder-decoder works for dense prediction
2. Why skip connections matter
3. What features the network learns at different scales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic conv block: Conv -> BatchNorm -> ReLU"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class EncoderBlock(nn.Module):
    """Encoder block: 2x ConvBlock -> MaxPool (halves resolution)"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        pooled = self.pool(x)
        return pooled, x  # Return both pooled and skip connection


class DecoderBlock(nn.Module):
    """Decoder block: Upsample -> Concat skip -> 2x ConvBlock"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = ConvBlock(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Handle size mismatch (can happen with odd dimensions)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class TinyDepthNet(nn.Module):
    """
    Tiny depth estimation network with U-Net style skip connections.

    Input: RGB image (B, 3, H, W) - any resolution (will be resized internally)
    Output: Relative depth map (B, 1, H, W) - normalized to [0, 1]

    Architecture:
        Encoder (feature extraction):
            3 -> 32 -> 64 -> 128 -> 256 channels
            Each block halves spatial resolution

        Bottleneck:
            256 -> 256 channels at 1/16 resolution

        Decoder (depth reconstruction):
            256 -> 128 -> 64 -> 32 -> 1 channels
            Skip connections from encoder preserve details

    Parameter count: ~2.1M
    """

    def __init__(self, input_size: int = 256):
        super().__init__()
        self.input_size = input_size

        # Encoder (downsampling path)
        self.enc1 = EncoderBlock(3, 32)      # 256 -> 128
        self.enc2 = EncoderBlock(32, 64)     # 128 -> 64
        self.enc3 = EncoderBlock(64, 128)    # 64 -> 32
        self.enc4 = EncoderBlock(128, 256)   # 32 -> 16

        # Bottleneck (deepest features)
        self.bottleneck = nn.Sequential(
            ConvBlock(256, 256),
            ConvBlock(256, 256),
        )

        # Decoder (upsampling path with skip connections)
        self.dec4 = DecoderBlock(256, 256, 128)  # 16 -> 32
        self.dec3 = DecoderBlock(128, 128, 64)   # 32 -> 64
        self.dec2 = DecoderBlock(64, 64, 32)     # 64 -> 128
        self.dec1 = DecoderBlock(32, 32, 32)     # 128 -> 256

        # Final prediction head
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),  # Single channel depth
            nn.Sigmoid()  # Output in [0, 1]
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input RGB image (B, 3, H, W), values in [0, 1] or normalized

        Returns:
            Depth map (B, 1, H, W), values in [0, 1] (relative depth)
        """
        original_size = x.shape[2:]

        # Resize to fixed input size for consistent feature extraction
        if x.shape[2] != self.input_size or x.shape[3] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                            mode='bilinear', align_corners=True)

        # Encoder with skip connections
        x, skip1 = self.enc1(x)  # 256 -> 128, skip at 256
        x, skip2 = self.enc2(x)  # 128 -> 64, skip at 128
        x, skip3 = self.enc3(x)  # 64 -> 32, skip at 64
        x, skip4 = self.enc4(x)  # 32 -> 16, skip at 32

        # Bottleneck
        x = self.bottleneck(x)   # 16x16 at deepest level

        # Decoder with skip connections
        x = self.dec4(x, skip4)  # 16 -> 32
        x = self.dec3(x, skip3)  # 32 -> 64
        x = self.dec2(x, skip2)  # 64 -> 128
        x = self.dec1(x, skip1)  # 128 -> 256

        # Prediction
        depth = self.head(x)

        # Resize back to original size
        if depth.shape[2:] != original_size:
            depth = F.interpolate(depth, size=original_size,
                                mode='bilinear', align_corners=True)

        return depth

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TinyDepthNetWithBackbone(nn.Module):
    """
    Variant using pretrained ResNet-18 encoder for better features.

    This shows how pretrained backbones improve depth estimation
    by providing robust, generalizable features.

    Parameter count: ~14M (but encoder features are much better)
    """

    def __init__(self, input_size: int = 256, pretrained: bool = True):
        super().__init__()
        self.input_size = input_size

        # Load pretrained ResNet-18
        import torchvision.models as models
        resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)

        # Extract encoder layers
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64 ch, /2
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)          # 64 ch, /4
        self.enc3 = resnet.layer2  # 128 ch, /8
        self.enc4 = resnet.layer3  # 256 ch, /16
        self.enc5 = resnet.layer4  # 512 ch, /32

        # Decoder (upsampling with skip connections)
        self.dec5 = DecoderBlock(512, 256, 256)   # /32 -> /16
        self.dec4 = DecoderBlock(256, 128, 128)   # /16 -> /8
        self.dec3 = DecoderBlock(128, 64, 64)     # /8 -> /4
        self.dec2 = DecoderBlock(64, 64, 32)      # /4 -> /2
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(32, 32),
        )  # /2 -> /1

        # Final prediction
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[2:]

        # Resize to fixed size
        if x.shape[2] != self.input_size or x.shape[3] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                            mode='bilinear', align_corners=True)

        # Encoder
        e1 = self.enc1(x)       # /2, 64ch
        e2 = self.enc2(e1)      # /4, 64ch
        e3 = self.enc3(e2)      # /8, 128ch
        e4 = self.enc4(e3)      # /16, 256ch
        e5 = self.enc5(e4)      # /32, 512ch

        # Decoder with skip connections
        d5 = self.dec5(e5, e4)  # /16
        d4 = self.dec4(d5, e3)  # /8
        d3 = self.dec3(d4, e2)  # /4
        d2 = self.dec2(d3, e1)  # /2
        d1 = self.dec1(d2)      # /1

        # Prediction
        depth = self.head(d1)

        # Resize to original
        if depth.shape[2:] != original_size:
            depth = F.interpolate(depth, size=original_size,
                                mode='bilinear', align_corners=True)

        return depth

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(variant: str = 'tiny', pretrained: bool = True, input_size: int = 256):
    """
    Factory function to create depth model.

    Args:
        variant: 'tiny' (~2.1M params) or 'resnet18' (~14M params)
        pretrained: Use ImageNet pretrained weights for resnet18 variant
        input_size: Internal processing resolution

    Returns:
        TinyDepthNet or TinyDepthNetWithBackbone
    """
    if variant == 'tiny':
        model = TinyDepthNet(input_size=input_size)
    elif variant == 'resnet18':
        model = TinyDepthNetWithBackbone(input_size=input_size, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'tiny' or 'resnet18'")

    print(f"Created {variant} model with {model.count_parameters():,} parameters")
    return model


if __name__ == '__main__':
    # Test both variants
    print("=" * 60)
    print("Testing TinyDepthNet (custom encoder)")
    print("=" * 60)

    model = create_model('tiny')
    x = torch.randn(2, 3, 480, 640)  # Batch of 2, RGB, 480x640

    with torch.no_grad():
        y = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.4f}, {y.max():.4f}]")

    print("\n" + "=" * 60)
    print("Testing TinyDepthNetWithBackbone (ResNet-18)")
    print("=" * 60)

    model_r18 = create_model('resnet18', pretrained=False)  # False to avoid download in test

    with torch.no_grad():
        y_r18 = model_r18(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y_r18.shape}")
    print(f"Output range: [{y_r18.min():.4f}, {y_r18.max():.4f}]")

    print("\n" + "=" * 60)
    print("Model comparison")
    print("=" * 60)
    print(f"TinyDepthNet:      {model.count_parameters():>10,} params")
    print(f"ResNet18 backbone: {model_r18.count_parameters():>10,} params")
