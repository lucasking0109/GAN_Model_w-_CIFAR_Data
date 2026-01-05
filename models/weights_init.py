"""
DCGAN Weight Initialization
Uses Normal distribution as recommended by the paper
"""
import torch.nn as nn


def weights_init(m):
    """
    DCGAN Weight Initialization Function

    According to DCGAN paper:
    - Conv and ConvTranspose layers use Normal distribution with mean=0, std=0.02
    - BatchNorm layers use mean=1, std=0.02 for weights, bias set to 0

    Usage:
        model.apply(weights_init)

    Args:
        m: Neural network module
    """
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        # Conv2d and ConvTranspose2d layers
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        # BatchNorm2d layers
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
