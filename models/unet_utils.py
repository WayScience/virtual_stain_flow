from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Components of the U-Net model
"""

class ConvBnRelu(nn.Module):
    """
    A customizable convolutional block: (Convolution => [BN] => ReLU) * N.

    Allows specifying the number of layers and intermediate channels.
    """

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 mid_channels: Optional[List[int]] = None,
                 num_layers: int = 2):
        """
        Initialize the customizable DoubleConv module for upsampling/downsampling the channels.

        :param in_channels: Number of input channels.
        :type in_channels: int
        :param out_channels: Number of output channels.
        :type out_channels: int
        :param mid_channels: List of intermediate channel numbers for each convolutional layer.
                             If unspecified, defaults to [out_channels] * (num_layers - 1). 
                             Order matters: mid_channels[0] corresponds to the first intermediate layer, etc.
        :type mid_channels: Optional[List[int]]
        :param num_layers: Number of convolutional layers in the block.
        :type num_layers: int
        """
        super().__init__()

        # Default intermediate channels if not specified
        if mid_channels is None:
            mid_channels = [out_channels] * (num_layers - 1)

        if len(mid_channels) != num_layers - 1:
            raise ValueError("Length of mid_channels must be equal to num_layers - 1.")

        layers = []

        # Add the first convolution layer
        layers.append(
            nn.Conv2d(in_channels, mid_channels[0], kernel_size=3, padding=1, bias=False)
        )
        layers.append(nn.BatchNorm2d(mid_channels[0]))
        layers.append(nn.ReLU(inplace=True))

        # Add intermediate convolutional layers
        for i in range(1, num_layers - 1):
            layers.append(
                nn.Conv2d(mid_channels[i - 1], mid_channels[i], kernel_size=3, padding=1, bias=False)
            )
            layers.append(nn.BatchNorm2d(mid_channels[i]))
            layers.append(nn.ReLU(inplace=True))

        # Add the final convolution layer
        layers.append(
            nn.Conv2d(mid_channels[-1], out_channels, kernel_size=3, padding=1, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        # Combine layers into a sequential block
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConvBnRelu module.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Processed output tensor.
        :rtype: torch.Tensor
        """
        return self.conv_block(x)
    
class Contract(nn.Module):
    """Downscaling with maxpool then 2 * ConvBnRelu"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # Halves spatial dimensions
            ConvBnRelu(
                in_channels=in_channels, 
                out_channels=out_channels,
                mid_channels=None, 
                num_layers=2) # Refines features with 2 sequential convolutions   
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    """Upscaling then 2 * ConvBnRelu"""
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 bilinear: bool=True):
        """
        Up sampling module that combines the upsampled feature map with the skip connection.
        Upsampling is done via bilinear interpolation or transposed convolution.

        :param in_channels: Number of input channels.
        :type in_channels: int
        :param out_channels: Number of output channels.
        :type out_channels: int
        :param bilinear: If True, use bilinear upsampling
        :type bilinear: bool
        """
        super().__init__()

        # If bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = ConvBnRelu(
                in_channels=in_channels, 
                out_channels=out_channels, 
                mid_channels=[in_channels // 2], 
                num_layers=2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = ConvBnRelu(
                in_channels=in_channels,
                out_channels=out_channels,
                mid_channels=None,
                num_layers=2
            )

    def forward(self, 
                x1: torch.Tensor, 
                x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Up module.
        :param x1: Input tensor to be upsampled.
        :type x1: torch.Tensor
        :param x2: Skip connection tensor.
        :type x2: torch.Tensor
        :return: Processed output tensor.
        """
        x1 = self.up(x1)  # Upsample x1

        # Handle potential mismatches in spatial dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # Concatenate x1 (upsampled) with x2 (skip connection)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    """
    Final output layer that applies a 1x1 convolution followed by a sigmoid activation.
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the OutConv module.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Processed output tensor.
        :rtype: torch.Tensor
        """
        return torch.sigmoid(self.conv(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Upsample module.

        :param x: Input tensor to be upsampled.
        :type x: torch.Tensor
        :return: Upsampled tensor.
        :rtype: torch.Tensor
        """
        return self.up(x)