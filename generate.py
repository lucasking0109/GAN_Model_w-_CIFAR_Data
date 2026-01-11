"""
Image Generation Script
Generate new images using trained Generator
"""
import argparse
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from pathlib import Path

from config import Config
from models import Generator
from utils import denormalize


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using DCGAN")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Model checkpoint path (default: use latest)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=64,
        help="Number of images to generate (default: 64)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display generated images",
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Generate latent space interpolation animation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (for reproducible results)",
    )
    return parser.parse_args()


def load_generator(checkpoint_path: str, device):
    """Load trained Generator"""
    config = Config

    generator = Generator(
        latent_dim=config.LATENT_DIM,
        ngf=config.NGF,
        channels=config.CHANNELS,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    print(f"Model loaded: {checkpoint_path}")
    if "epoch" in checkpoint:
        print(f"  Training epoch: {checkpoint['epoch']}")

    return generator


def generate_images(generator, num_images: int, device, seed: int = None):
    """Generate images"""
    if seed is not None:
        torch.manual_seed(seed)
        print(f"Using random seed: {seed}")

    latent_dim = generator.latent_dim

    with torch.no_grad():
        noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
        fake_images = generator(noise)
        fake_images = denormalize(fake_images)

    return fake_images


def generate_interpolation(generator, device, num_steps: int = 10):
    """
    Generate image sequence by interpolating in latent space
    Shows gradual transition between two random points
    """
    latent_dim = generator.latent_dim

    with torch.no_grad():
        # Two random endpoints
        z1 = torch.randn(1, latent_dim, 1, 1, device=device)
        z2 = torch.randn(1, latent_dim, 1, 1, device=device)

        # Linear interpolation
        images = []
        for alpha in torch.linspace(0, 1, num_steps):
            z = z1 * (1 - alpha) + z2 * alpha
            img = generator(z)
            img = denormalize(img)
            images.append(img)

        images = torch.cat(images, dim=0)

    return images


def main():
    args = parse_args()

    # Initialize
    config = Config.init()

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Try to find latest checkpoint
        latest_path = config.CHECKPOINT_DIR / "checkpoint_latest.pth"
        final_path = config.CHECKPOINT_DIR / "checkpoint_final.pth"

        if final_path.exists():
            checkpoint_path = str(final_path)
        elif latest_path.exists():
            checkpoint_path = str(latest_path)
        else:
            print("Error: Checkpoint file not found")
            print(f"Please run training first: python train.py")
            print(f"Or specify checkpoint path with --checkpoint")
            return

    # Load model
    generator = load_generator(checkpoint_path, config.DEVICE)

    if args.interpolate:
        # Generate interpolation sequence
        print(f"\nGenerating latent space interpolation sequence...")
        images = generate_interpolation(generator, config.DEVICE, num_steps=10)
        nrow = 10
    else:
        # Generate random images
        print(f"\nGenerating {args.num_images} images...")
        images = generate_images(
            generator,
            args.num_images,
            config.DEVICE,
            seed=args.seed,
        )
        nrow = 8

    # Create image grid
    grid = make_grid(images.cpu(), nrow=nrow, padding=2, normalize=False)

    # Save images
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = config.OUTPUT_DIR / "generated"
        output_dir.mkdir(exist_ok=True)

        # Find next available filename
        idx = 1
        while (output_dir / f"generated_{idx:04d}.png").exists():
            idx += 1
        output_path = output_dir / f"generated_{idx:04d}.png"

    save_image(grid, output_path)
    print(f"\nImages saved: {output_path}")

    # Display images
    if args.show:
        plt.figure(figsize=(12, 12))
        plt.axis("off")
        plt.title("Generated Images")
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.show()


if __name__ == "__main__":
    main()
