import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .unet_utils import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_channels=64, depth=4, bilinear=False):
        """
        Initialize the U-Net model with a customizable depth.

        Args:
            n_channels (int): Number of input channels.
            n_classes (int): Number of output classes.
            base_channels (int): Number of channels for the first layer. Subsequent layers double this value.
            depth (int): Number of downsampling steps (controls depth).
            bilinear (bool): If True, use bilinear upsampling; otherwise, use transposed convolutions.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_channels = base_channels
        self.depth = depth
        self.bilinear = bilinear

        in_channels = n_channels # Input channel to the first upsampling layer is the number of input channels
        out_channels = base_channels # Output channel of the first upsampling layer is the base number of channels

        # Initial upsampling layer
        self.inc = ConvBnRelu(
            in_channels=in_channels, 
            out_channels=out_channels, 
            mid_channels=None,
            num_layers=2)
        
        # Contracting path
        contracting_path = []
        for _ in range(depth):
            # set the number of input channels to the output channels of the previous layer
            in_channels = out_channels
            # double the number of output channels for the next layer
            out_channels *= 2
            contracting_path.append(
                Contract(
                    in_channels=in_channels, 
                    out_channels=out_channels
                    )
            )
        self.down = nn.ModuleList(contracting_path)

        # Bottleneck
        factor = 2 if bilinear else 1
        in_channels = out_channels # Input channel to the bottleneck layer is the output channel of the last downsampling layer
        out_channels = in_channels // factor
        self.bottleneck = ConvBnRelu(
            in_channels=in_channels,
            out_channels=out_channels,
            mid_channels=None,
            num_layers=2
        )

        # Expanding path
        expanding_path = []
        for _ in range(depth):
            # input to expanding path has the same dimension as the output of the bottleneck layer  
            in_channels = out_channels
            # half the number of output channels for the next layer
            out_channels = in_channels // 2
            expanding_path.append(
                ## TODO: replace this with the Upsample and SkipConnection modules maybe
                Up(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bilinear=bilinear
                )
            )        
        self.up = nn.ModuleList(expanding_path)

        # Output layer
        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the U-Net model.

        :param x: Input tensor of shape (batch_size, n_channels, height, width).
        :type x: torch.Tensor
        :return: Output tensor of shape (batch_size, n_classes, height, width).
        :rtype: torch.Tensor
        """
        # Contracting path
        x_contracted = []
        x = self.inc(x)
        for down in self.down:
            x_contracted.append(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Expanding path
        for i, up in enumerate(self.up):
            x = up(x, x_contracted[-(i + 1)])

        # Final output
        logits = self.outc(x)
        return logits