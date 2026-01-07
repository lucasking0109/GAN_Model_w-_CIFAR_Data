"""
Visualization Tools
For generating image grids and plotting training curves
"""
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from pathlib import Path


def denormalize(tensor):
    """
    Convert tensor from [-1, 1] range back to [0, 1]

    Args:
        tensor: Normalized image tensor

    Returns:
        Tensor in [0, 1] range
    """
    return (tensor + 1) / 2


def save_sample_images(
    generator,
    fixed_noise,
    epoch: int,
    output_dir: str,
    device,
    nrow: int = 8,
):
    """
    Generate and save sample images

    Args:
        generator: Generator model
        fixed_noise: Fixed random noise (for tracking training progress)
        epoch: Current epoch
        output_dir: Output directory
        device: Compute device
        nrow: Number of images per row
    """
    generator.eval()

    with torch.no_grad():
        fake_images = generator(fixed_noise.to(device))
        # Convert to [0, 1] range
        fake_images = denormalize(fake_images)

    # Create image grid
    grid = make_grid(fake_images.cpu(), nrow=nrow, normalize=False, padding=2)

    # Convert to numpy and save
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title(f"Epoch {epoch}")
    plt.imshow(grid.permute(1, 2, 0).numpy())

    output_path = Path(output_dir) / f"epoch_{epoch:04d}.png"
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    generator.train()
    return output_path


def plot_losses(g_losses: list, d_losses: list, output_dir: str):
    """
    Plot training loss curves

    Args:
        g_losses: Generator loss list
        d_losses: Discriminator loss list
        output_dir: Output directory
    """
    plt.figure(figsize=(12, 5))

    # Generator loss
    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label="Generator", color="blue", alpha=0.7)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Generator Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Discriminator loss
    plt.subplot(1, 2, 2)
    plt.plot(d_losses, label="Discriminator", color="orange", alpha=0.7)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Discriminator Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_dir) / "training_losses.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path
