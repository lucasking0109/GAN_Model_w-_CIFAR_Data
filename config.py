"""
DCGAN Configuration File
Contains all hyperparameters and path settings
"""
import torch
from pathlib import Path


class Config:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_CACHE_DIR = PROJECT_ROOT / "data_cache"

    # Output paths will be set dynamically in init() based on TARGET_CLASS
    OUTPUT_DIR = None
    CHECKPOINT_DIR = None
    SAMPLE_DIR = None

    # Dataset settings
    DATASET = "celeba"  # "cifar10" or "celeba"
    IMAGE_SIZE = 128    # Image size (CelebA: 128x128, CIFAR-10: 64x64)
    CHANNELS = 3        # RGB color images

    # Training class settings (only effective for CIFAR-10)
    # None = train all 10 classes
    # 0-9 = train specific class (0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer, 5=dog, 6=frog, 7=horse, 8=ship, 9=truck)
    TARGET_CLASS = None  # Not needed for CelebA

    # Model hyperparameters
    LATENT_DIM = 100   # Latent vector dimension (z)
    NGF = 64           # Generator feature map count
    NDF = 64           # Discriminator feature map count

    # Training hyperparameters
    BATCH_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 0.0001  # Reduced learning rate to avoid instability in later training
    BETA1 = 0.5        # Adam optimizer beta1
    BETA2 = 0.999      # Adam optimizer beta2

    # GAN stabilization parameters
    LABEL_SMOOTHING = True      # Use label smoothing
    REAL_LABEL_VALUE = 0.9      # Real label reduced from 1.0 to 0.9
    FAKE_LABEL_VALUE = 0.1      # Fake label increased from 0.0 to 0.1
    LABEL_NOISE = 0.05          # Label noise intensity

    # Training balance settings
    D_TRAIN_THRESHOLD = 0.8     # Skip D training when accuracy > 80%
    G_TRAIN_STEPS = 1           # G training steps per iteration
    D_TRAIN_STEPS = 1           # D training steps per iteration

    # Version control (for distinguishing different training versions)
    # Set to None for default path, or specify version name like "v2_dropout"
    VERSION_SUFFIX = "celeba_128"  # CelebA 128x128 high resolution version

    # Device settings
    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    DEVICE = None  # Will be set at runtime
    NUM_WORKERS = 4

    # Save settings
    SAVE_INTERVAL = 10      # Save checkpoint every N epochs
    SAMPLE_INTERVAL = 1     # Generate sample images every N epochs
    NUM_SAMPLE_IMAGES = 64  # Number of sample images to generate

    @classmethod
    def init(cls):
        """Initialize configuration and create necessary directories"""
        cls.DEVICE = cls.get_device()

        # Create different output folders based on dataset and training class
        if cls.DATASET.lower() == "celeba":
            # CelebA dataset
            output_suffix = "celeba"
        elif cls.TARGET_CLASS is not None:
            # CIFAR-10 single class training
            class_names = {
                0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
                5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
            }
            class_name = class_names[cls.TARGET_CLASS]
            output_suffix = f"class_{cls.TARGET_CLASS}_{class_name}"
        else:
            # CIFAR-10 all classes training
            output_suffix = "cifar10_all_classes"

        # Add version suffix if specified
        if cls.VERSION_SUFFIX:
            output_suffix = f"{output_suffix}_{cls.VERSION_SUFFIX}"

        # Set output paths
        cls.OUTPUT_DIR = cls.PROJECT_ROOT / f"outputs_{output_suffix}"
        cls.CHECKPOINT_DIR = cls.OUTPUT_DIR / "checkpoints"
        cls.SAMPLE_DIR = cls.OUTPUT_DIR / "samples"

        # Create directories
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(exist_ok=True)
        cls.SAMPLE_DIR.mkdir(exist_ok=True)
        cls.DATA_CACHE_DIR.mkdir(exist_ok=True)

        # Reduce batch size and workers for CPU
        if cls.DEVICE.type == "cpu":
            cls.BATCH_SIZE = 64
            cls.NUM_WORKERS = 0

        print(f"Device: {cls.DEVICE}")
        print(f"Output directory: {cls.OUTPUT_DIR}")
        return cls
