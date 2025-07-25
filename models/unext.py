from typing import List, Union, Literal

import timm
import torch
import torch.nn as nn

from .utils import (
    get_activation,
    ActivationType
)
from .decoder import Decoder
from .blocks import Conv2DConvNeXtBlock, Conv2DNormActBlock
from .up_down_blocks import (
    ConvTrans2DUpBlock,
    PixelShuffle2DUpBlock
)

"""
ConvNeXtUNet model implementation leveraging the modular "block" and "stage",
Simply initializes a un-pretrained ConvNeXtV2_tiny model from timm library,
adapt it as a encoder for the UNet like architecture, and then initialize
a appropriate Decoder. Depth of model is fixed to 4 as the ConvNeXtV2_tiny 
encoder has 4 stages.

This model class allows for 4 different decoder architectures:
1. ConvTrans2DUpBlock with Conv2DNormActBlock
2. ConvTrans2DUpBlock with Conv2DConvNeXtBlock
3. PixelShuffle2DUpBlock with Conv2DNormActBlock
4. PixelShuffle2DUpBlock with Conv2DConvNeXtBlock
"""
class ConvNeXtUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        decoder_up_block: Literal['pixelshuffle', 'convt'] = 'pixelshuffle',
        decoder_compute_block: Literal['convnext', 'conv2d'] = 'convnext',
        act_type: ActivationType = 'sigmoid',
        _num_units: Union[List[int], int] = 2
    ):
        """
        Initializes the ConvNeXtUNet model.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param decoder_up_block: Type of up-sampling block to use in the decoder.
            Can be 'pixelshuffle' for PixelShuffle2DUpBlock or 'convt' for
            ConvTrans2DUpBlock. Default is 'pixelshuffle'.
        :param decoder_compute_block: Type of computation block to use in the
            decoder. Can be 'convnext' for Conv2DConvNeXtBlock or 'conv2d' for
            Conv2DNormActBlock. Default is 'convnext'.
        :param act_type: Type of activation function to use in the output layer.
            Default is 'sigmoid'.
        :param _num_units: Number of computation units in each stage.
            Can be an integer for uniform number of units in all stages,
            or a list of integers specifying the number of units for each stage.
        """

        super().__init__()
        
        # timm implementation of the depth 4 ConvNeXtV2 model
        # with only down-sampling path, originally optimized
        # for tasks like image classification/object detection
        convnextv2_model = timm.create_model(
            "convnextv2_tiny", 
            features_only=True, 
            pretrained=False
        )
        # replace the first convolutional layer to work with (N, in_channels, H, W)
        # images
        # this also serves as the in_conv function for up-sampling the image
        # channel-wise
        convnextv2_model.stem_0 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=convnextv2_model.feature_info.channels()[0],
            kernel_size=1,
            stride=1,
            padding=0
        )
        depth = len(convnextv2_model.feature_info.channels())

        self.encoder = convnextv2_model

        if decoder_up_block == 'pixelshuffle':
            in_block_handles = [PixelShuffle2DUpBlock] * (depth - 1)
        elif decoder_up_block == 'convt':
            in_block_handles = [ConvTrans2DUpBlock] * (depth - 1)
        else:
            raise ValueError(
                f"Unsupported decoder_up_block: {decoder_up_block!r}. "
                "Expected 'pixelshuffle' or 'convt'."
            )
        
        if decoder_compute_block == 'convnext':
            comp_block_handles = [Conv2DConvNeXtBlock] * (depth - 1)
        elif decoder_compute_block == 'conv2d':
            comp_block_handles = [Conv2DNormActBlock] * (depth - 1)
        else:
            raise ValueError(
                f"Unsupported decoder_compute_block: {decoder_compute_block!r}. "
                "Expected 'convnext' or 'conv2d'."
            )
        
        if isinstance(_num_units, int):
            comp_block_kwargs = [{'num_units': _num_units}] * (depth - 1)
        elif isinstance(_num_units, list):
            if len(_num_units) != (depth - 1):
                raise ValueError(
                    f"Expected _num_units to be a list of length {(depth - 1)}, "
                    f"got {len(_num_units)}"
                )
            comp_block_kwargs = [{'num_units': n} for n in _num_units]
        else:
            raise TypeError(
                f"Expected _num_units to be int or list, "
                f"got {type(_num_units).__name__}"
            )

        self.decoder = Decoder(
            encoder_feature_map_channels=convnextv2_model.feature_info.channels(),
            # use convolutional up-sampling blocks
            in_block_handles=in_block_handles,
            in_block_kwargs=[{'norm_type': 'layer'}] * (depth - 1),
            comp_block_handles=comp_block_handles,
            comp_block_kwargs=comp_block_kwargs,
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
        encoder_feature_maps = self.encoder(x)
        decoder_output = self.decoder(encoder_feature_maps)
        x = self.out_conv(decoder_output)

        return self.out_activation(x)