"""
DCGAN Training Script
Train on CIFAR-10 dataset to generate color images
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import Config
from models import Generator, Discriminator, weights_init
from data import get_dataloader
from utils import save_sample_images, plot_losses, save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="DCGAN Training")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    return parser.parse_args()


def train():
    # Parse command line arguments
    args = parse_args()

    # Initialize configuration
    config = Config.init()

    # Override config with command line arguments if provided
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr

    print("=" * 50)
    print("DCGAN Training Started")
    print("=" * 50)
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Latent Dimension: {config.LATENT_DIM}")
    print("=" * 50)

    # Load data
    dataloader = get_dataloader(
        dataset_name=config.DATASET,
        data_root=str(config.DATA_CACHE_DIR),
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        target_class=config.TARGET_CLASS,
    )

    # Create models
    generator = Generator(
        latent_dim=config.LATENT_DIM,
        ngf=config.NGF,
        channels=config.CHANNELS,
        image_size=config.IMAGE_SIZE,
    ).to(config.DEVICE)

    discriminator = Discriminator(
        channels=config.CHANNELS,
        ndf=config.NDF,
        image_size=config.IMAGE_SIZE,
    ).to(config.DEVICE)

    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    print(f"\nGenerator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Loss function and optimizers
    # Using BCEWithLogitsLoss (more stable, includes Sigmoid internally)
    criterion = nn.BCEWithLogitsLoss()

    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.BETA1, config.BETA2),
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.BETA1, config.BETA2),
    )

    # Fixed noise for tracking training progress
    fixed_noise = torch.randn(config.NUM_SAMPLE_IMAGES, config.LATENT_DIM, 1, 1)

    # Training history
    g_losses = []
    d_losses = []
    start_epoch = 0

    # Resume from checkpoint if specified
    if args.resume:
        start_epoch, g_losses, d_losses = load_checkpoint(
            args.resume,
            generator,
            discriminator,
            optimizer_g,
            optimizer_d,
            config.DEVICE,
        )
        start_epoch += 1  # Start from next epoch

    # Start training
    print("\nStarting training...\n")
    print("=" * 50)
    print("Stabilization settings:")
    print(f"  Label Smoothing: {config.LABEL_SMOOTHING}")
    if config.LABEL_SMOOTHING:
        print(f"  Real label: {config.REAL_LABEL_VALUE} (originally 1.0)")
        print(f"  Fake label: {config.FAKE_LABEL_VALUE} (originally 0.0)")
        print(f"  Label noise: +/-{config.LABEL_NOISE}")
    print(f"  D training threshold: {config.D_TRAIN_THRESHOLD}")
    print("=" * 50 + "\n")

    for epoch in range(start_epoch, config.EPOCHS):
        generator.train()
        discriminator.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        d_skipped = 0  # Count of D training skips

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch [{epoch+1}/{config.EPOCHS}]",
            leave=True,
        )

        for batch_idx, (real_images, _) in enumerate(progress_bar):
            batch_size = real_images.size(0)
            real_images = real_images.to(config.DEVICE)

            # =====================================
            # Train Discriminator
            # =====================================
            # Using Label Smoothing and Noisy Labels
            if config.LABEL_SMOOTHING:
                # Real labels: 0.9 +/- 0.05 (range ~0.85-0.95)
                real_label_tensor = torch.full(
                    (batch_size,),
                    config.REAL_LABEL_VALUE,
                    dtype=torch.float,
                    device=config.DEVICE
                ) + torch.randn(batch_size, device=config.DEVICE) * config.LABEL_NOISE
                # Fake labels: 0.1 +/- 0.05 (range ~0.05-0.15)
                fake_label_tensor = torch.full(
                    (batch_size,),
                    config.FAKE_LABEL_VALUE,
                    dtype=torch.float,
                    device=config.DEVICE
                ) + torch.randn(batch_size, device=config.DEVICE) * config.LABEL_NOISE
            else:
                real_label_tensor = torch.ones(batch_size, dtype=torch.float, device=config.DEVICE)
                fake_label_tensor = torch.zeros(batch_size, dtype=torch.float, device=config.DEVICE)

            discriminator.zero_grad()

            # Train on real images
            output_real = discriminator(real_images).view(-1)
            loss_d_real = criterion(output_real, real_label_tensor)
            loss_d_real.backward()
            d_x = output_real.mean().item()  # D(x): Discriminator output for real images

            # Train on fake images
            noise = torch.randn(batch_size, config.LATENT_DIM, 1, 1, device=config.DEVICE)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach()).view(-1)
            loss_d_fake = criterion(output_fake, fake_label_tensor)
            loss_d_fake.backward()
            d_g_z1 = output_fake.mean().item()  # D(G(z)): Discriminator output for fake images (training D)

            loss_d = loss_d_real + loss_d_fake

            # Calculate D accuracy (proportion of real > 0.5 and fake < 0.5)
            d_accuracy = ((output_real > 0.5).float().mean().item() +
                         (output_fake < 0.5).float().mean().item()) / 2.0

            # Skip D training with 50% probability if D is too strong (accuracy > threshold)
            if d_accuracy > config.D_TRAIN_THRESHOLD and torch.rand(1).item() > 0.5:
                d_skipped += 1
            else:
                optimizer_d.step()

            # =====================================
            # Train Generator
            # =====================================
            generator.zero_grad()

            # Generator wants Discriminator to classify fake images as real
            # Using smoothed real labels
            if config.LABEL_SMOOTHING:
                g_target_label = torch.full(
                    (batch_size,),
                    config.REAL_LABEL_VALUE,
                    dtype=torch.float,
                    device=config.DEVICE
                )
            else:
                g_target_label = torch.ones(batch_size, dtype=torch.float, device=config.DEVICE)

            output = discriminator(fake_images).view(-1)
            loss_g = criterion(output, g_target_label)
            loss_g.backward()
            d_g_z2 = output.mean().item()  # D(G(z)): Discriminator output for fake images (training G)

            # Update Generator
            optimizer_g.step()

            # Record losses
            g_losses.append(loss_g.item())
            d_losses.append(loss_d.item())
            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()

            # Update progress bar
            progress_bar.set_postfix({
                "D_loss": f"{loss_d.item():.4f}",
                "G_loss": f"{loss_g.item():.4f}",
                "D(x)": f"{d_x:.3f}",
                "D(G(z))": f"{d_g_z1:.3f}/{d_g_z2:.3f}",
                "D_acc": f"{d_accuracy:.2f}",
            })

        # Epoch complete
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)

        print(f"Epoch [{epoch+1}/{config.EPOCHS}] Complete - "
              f"Avg G_loss: {avg_g_loss:.4f}, Avg D_loss: {avg_d_loss:.4f}, "
              f"D skipped: {d_skipped}/{len(dataloader)} batches")

        # Save sample images
        if (epoch + 1) % config.SAMPLE_INTERVAL == 0:
            sample_path = save_sample_images(
                generator,
                fixed_noise,
                epoch + 1,
                str(config.SAMPLE_DIR),
                config.DEVICE,
            )
            print(f"  Sample images saved: {sample_path}")

        # Save checkpoint
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            save_checkpoint(
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                epoch + 1,
                g_losses,
                d_losses,
                str(config.CHECKPOINT_DIR),
            )

    # Training complete
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)

    # Save final model
    save_checkpoint(
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        config.EPOCHS,
        g_losses,
        d_losses,
        str(config.CHECKPOINT_DIR),
        filename="checkpoint_final.pth",
    )

    # Plot loss curves
    loss_plot_path = plot_losses(g_losses, d_losses, str(config.OUTPUT_DIR))
    print(f"Loss curves saved: {loss_plot_path}")

    # Generate final samples
    final_sample_path = save_sample_images(
        generator,
        fixed_noise,
        config.EPOCHS,
        str(config.SAMPLE_DIR),
        config.DEVICE,
    )
    print(f"Final samples saved: {final_sample_path}")


if __name__ == "__main__":
    train()
