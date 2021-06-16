import torch.nn as nn
import torch
from torch.nn import functional as F


class VGG(nn.Module):
    """
    VGG model class

    Attributes:
        features (nn.Sequential): Feature extraction layers
        reg_layer (nn.Sequential): Regression layer
    """

    def __init__(self, features):
        """
        Args:
            features (nn.Sequential): Model feature extraction layers
        """
        super(VGG, self).__init__()
        self.features = features
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        """
        Forward pass method

        Args:
            x (tensor): Input tensor

        Returns:
            (float): Result of forward pass
        """
        x = self.features(x)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = self.reg_layer(x)
        return torch.abs(x)


def make_layers(cfg, batch_norm=False):
    """
    Generates VGG-19 Layers

    Args:
        cfg (array): Configuration array
        batch_norm (bool): Flag for batch norm

    Returns:
         (nn.Sequential): Sequential model structure
    """
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
