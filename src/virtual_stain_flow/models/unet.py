from typing import List, Union

import torch
import torch.nn as nn

from .utils import (
    get_activation,
    ActivationType
)
from .encoder import Encoder
from .decoder import Decoder
from .blocks import Conv2DNormActBlock
from .up_down_blocks import (
    IdentityBlock,
    Conv2DDownBlock,
    MaxPool2DDownBlock,
    ConvTrans2DUpBlock
)
"""
UNet model implementation leveraging the modular "block" and "stage", 
and simply wraps a Encoder and Decoder initialization with pre-defined
combination of blocks types and and block configurations.

This model class allows for two different architecture of UNet:

1. A fully convolutional UNet with Conv2D down-sampling and Conv2DTranspose 
    up-sampling blocks. Both the down-sampling and up-sampling blocks are
    immediately followed by a batch normalization. This is equivalent to 
    the FNet architecture used in Ounkomol et. al, 2018.

2. A UNet with MaxPool2D down-sampling and Conv2DTranspose up-sampling blocks.
    This is a more traditional UNet architecture, where the down-sampling
    is done using MaxPool2D blocks, and the up-sampling is done using
    Conv2DTranspose blocks. The down-sampling blocks are not followed by
    a normalization layer, but the up-sampling blocks are followed by a
    batch normalization.

Both architecture uses _num_units repetitions of the 
    Conv2D>BatchNorm>ReLU computation block to follow the down/up-sampling. 
"""
class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        depth: int = 4,
        max_pool_down: bool = False,
        act_type: ActivationType = 'sigmoid',
        _num_units: Union[List[int], int] = 2
    ):
        """
        Initializes the UNet model.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param base_channels: Number of channels in the first layer.
        :param depth: Depth of the UNet, i.e., number of down-sampling and
            up-sampling stages. Must be >= 1.
        :param max_pool_down: If True, use MaxPool2DDownBlock for down-sampling,
            otherwise use Conv2DDownBlock. Default is False.
        :param act_type: Type of activation function to use in the output layer.
            Default is 'sigmoid'.
        :param _num_units: Number of computation units in each stage.
            Can be an integer for uniform number of units in all stages,
            or a list of integers specifying the number of units for each stage.
        """

        super().__init__()

        if isinstance(depth, int):
            if depth < 1:
                raise ValueError(f"Expected depth to be >= 1, got {depth}")
        else:
            raise TypeError(
                f"Expected depth to be int, got {type(depth).__name__}"
            )

        if max_pool_down:
            in_block_handles = [MaxPool2DDownBlock] * (depth - 1)
        else:
            in_block_handles = [Conv2DDownBlock] * (depth - 1)
        in_block_handles = [IdentityBlock] + in_block_handles

        comp_block_handles = [Conv2DNormActBlock] * depth

        # by default add BatchNorm2d normalization to the in blocks
        # if Conv2DDownBlocks and ConvTrans2DUpBlocks are used,
        # this will not do anything to the MaxPool2DDownBlock
        in_block_kwargs = [{'norm_type': 'batch'}] * depth

        if isinstance(_num_units, int):
            comp_block_kwargs = [{'num_units': _num_units}] * depth
        elif isinstance(_num_units, list):
            if len(_num_units) != depth:
                raise ValueError(
                    f"Expected _num_units to be a list of length {depth}, "
                    f"got {len(_num_units)}"
                )
            comp_block_kwargs = [{'num_units': n} for n in _num_units]
        else:
            raise TypeError(
                f"Expected _num_units to be int or list, "
                f"got {type(_num_units).__name__}"
            )
        
        self.in_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.encoder = Encoder(
            in_channels=base_channels,
            in_block_handles=in_block_handles,
            comp_block_handles=comp_block_handles,
            in_block_kwargs=in_block_kwargs,
            comp_block_kwargs=comp_block_kwargs,
            depth=depth,
        )

        self.decoder = Decoder(
            encoder_feature_map_channels=self.encoder.feature_map_channels,
            in_block_handles=[ConvTrans2DUpBlock] * (depth - 1),
            # mirror the comp_block_handles and comp_block_kwargs
            comp_block_handles=comp_block_handles[::-1][1:], 
            in_block_kwargs=in_block_kwargs[::-1][1:],
            comp_block_kwargs=comp_block_kwargs[::-1][1:],
        )

        self.out_conv = nn.Conv2d(
            in_channels=self.decoder.feature_map_channels[-1],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.out_activation = get_activation(act_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the U-Net model.

        :param x: Input tensor of shape (batch_size, in_channels, height, width).
        :type x: torch.Tensor
        :return: Output tensor of shape (batch_size, out_channels, height, width).
        :rtype: torch.Tensor
        """
        x = self.in_conv(x)
        encoder_feature_maps = self.encoder(x)
        decoder_output = self.decoder(encoder_feature_maps)
        x = self.out_conv(decoder_output)

        return self.out_activation(x)