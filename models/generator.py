"""
DCGAN Generator Network
Converts random noise vector to images
"""
import torch.nn as nn


class Generator(nn.Module):
    """
    DCGAN Generator

    Architecture (supports 64x64 and 128x128):
        64x64:
            Input: (batch, latent_dim, 1, 1)
            -> ConvTranspose2d -> (batch, ngf*8, 4, 4)
            -> ConvTranspose2d -> (batch, ngf*4, 8, 8)
            -> ConvTranspose2d -> (batch, ngf*2, 16, 16)
            -> ConvTranspose2d -> (batch, ngf, 32, 32)
            -> ConvTranspose2d -> (batch, channels, 64, 64)

        128x128:
            Additional layer: 64x64 -> 128x128
    """

    def __init__(self, latent_dim: int = 100, ngf: int = 64, channels: int = 3, image_size: int = 64):
        """
        Args:
            latent_dim: Latent vector dimension (size of z)
            ngf: Base number of Generator feature maps
            channels: Number of output image channels (RGB=3)
            image_size: Output image size (64 or 128)
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.ngf = ngf
        self.channels = channels
        self.image_size = image_size

        # Build network layers
        layers = [
            # Input: (batch, latent_dim, 1, 1)
            # Layer 1: latent_dim -> ngf*8, size 1x1 -> 4x4
            nn.ConvTranspose2d(latent_dim, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            # Output: (batch, ngf*8, 4, 4)

            # Layer 2: ngf*8 -> ngf*4, size 4x4 -> 8x8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            # Output: (batch, ngf*4, 8, 8)

            # Layer 3: ngf*4 -> ngf*2, size 8x8 -> 16x16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            # Output: (batch, ngf*2, 16, 16)

            # Layer 4: ngf*2 -> ngf, size 16x16 -> 32x32
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            # Output: (batch, ngf, 32, 32)
        ]

        if image_size == 64:
            # Layer 5 (output): ngf -> channels, size 32x32 -> 64x64
            layers.extend([
                nn.ConvTranspose2d(ngf, channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh()
            ])
        elif image_size == 128:
            # Layer 5: ngf -> ngf//2, size 32x32 -> 64x64
            layers.extend([
                nn.ConvTranspose2d(ngf, ngf // 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ngf // 2),
                nn.ReLU(inplace=True),
            ])
            # Layer 6 (output): ngf//2 -> channels, size 64x64 -> 128x128
            layers.extend([
                nn.ConvTranspose2d(ngf // 2, channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh()
            ])
        else:
            raise ValueError(f"Unsupported image size: {image_size}. Only 64 or 128 supported")

        self.main = nn.Sequential(*layers)

    def forward(self, z):
        """
        Forward pass

        Args:
            z: Random noise vector (batch, latent_dim, 1, 1)

        Returns:
            Generated images (batch, channels, image_size, image_size), range [-1, 1]
        """
        return self.main(z)
