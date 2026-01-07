"""
Model Checkpoint Tools
For saving and loading model weights
"""
import torch
from pathlib import Path


def save_checkpoint(
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    epoch: int,
    g_losses: list,
    d_losses: list,
    checkpoint_dir: str,
    filename: str = None,
):
    """
    Save training checkpoint

    Args:
        generator: Generator model
        discriminator: Discriminator model
        optimizer_g: Generator optimizer
        optimizer_d: Discriminator optimizer
        epoch: Current epoch
        g_losses: Generator loss history
        d_losses: Discriminator loss history
        checkpoint_dir: Save directory
        filename: Filename (default: checkpoint_epoch_XXX.pth)
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_epoch_{epoch:04d}.pth"

    checkpoint = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "g_losses": g_losses,
        "d_losses": d_losses,
    }

    save_path = checkpoint_dir / filename
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")

    # Also save as latest checkpoint
    latest_path = checkpoint_dir / "checkpoint_latest.pth"
    torch.save(checkpoint, latest_path)

    return save_path


def load_checkpoint(
    checkpoint_path: str,
    generator,
    discriminator,
    optimizer_g=None,
    optimizer_d=None,
    device="cpu",
):
    """
    Load training checkpoint

    Args:
        checkpoint_path: Checkpoint file path
        generator: Generator model
        discriminator: Discriminator model
        optimizer_g: Generator optimizer (optional)
        optimizer_d: Discriminator optimizer (optional)
        device: Compute device

    Returns:
        epoch: Epoch at save time
        g_losses: Generator loss history
        d_losses: Discriminator loss history
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

    if optimizer_g is not None and "optimizer_g_state_dict" in checkpoint:
        optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])

    if optimizer_d is not None and "optimizer_d_state_dict" in checkpoint:
        optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    g_losses = checkpoint.get("g_losses", [])
    d_losses = checkpoint.get("d_losses", [])

    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"  - Epoch: {epoch}")

    return epoch, g_losses, d_losses
