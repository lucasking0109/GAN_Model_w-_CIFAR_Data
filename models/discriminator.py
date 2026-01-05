"""
DCGAN Discriminator Network
Determines if images are real or generated
"""
import torch.nn as nn


class Discriminator(nn.Module):
    """
    DCGAN Discriminator

    Architecture (supports 64x64 and 128x128):
        64x64:
            Input: (batch, channels, 64, 64)
            -> Conv2d -> (batch, ndf, 32, 32)
            -> Conv2d -> (batch, ndf*2, 16, 16)
            -> Conv2d -> (batch, ndf*4, 8, 8)
            -> Conv2d -> (batch, ndf*8, 4, 4)
            -> Conv2d -> (batch, 1, 1, 1)

        128x128:
            Additional layer: 128x128 -> 64x64
    """

    def __init__(self, channels: int = 3, ndf: int = 64, image_size: int = 64):
        """
        Args:
            channels: Number of input image channels (RGB=3)
            ndf: Base number of Discriminator feature maps
            image_size: Input image size (64 or 128)
        """
        super().__init__()

        self.channels = channels
        self.ndf = ndf
        self.image_size = image_size

        # Build network layers
        layers = []

        if image_size == 128:
            # Layer 1 (128x128 only): channels -> ndf//2, size 128x128 -> 64x64
            layers.extend([
                nn.Conv2d(channels, ndf // 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.3),
            ])
            # Layer 2: ndf//2 -> ndf, size 64x64 -> 32x32
            layers.extend([
                nn.Conv2d(ndf // 2, ndf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.3),
            ])
        else:
            # Layer 1 (64x64): channels -> ndf, size 64x64 -> 32x32
            layers.extend([
                nn.Conv2d(channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.3),
            ])

        # Following layers are the same for both resolutions
        # Layer 2/3: ndf -> ndf*2, size 32x32 -> 16x16
        layers.extend([
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
        ])

        # Layer 3/4: ndf*2 -> ndf*4, size 16x16 -> 8x8
        layers.extend([
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
        ])

        # Layer 4/5: ndf*4 -> ndf*8, size 8x8 -> 4x4
        layers.extend([
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
        ])

        # Layer 5/6 (output): ndf*8 -> 1, size 4x4 -> 1x1
        layers.extend([
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
        ])

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input image (batch, channels, image_size, image_size)

        Returns:
            Discrimination logits (batch, 1, 1, 1), without Sigmoid
            0 = fake image, 1 = real image
        """
        return self.main(x)
