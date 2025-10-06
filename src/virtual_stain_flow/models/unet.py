"""
/models/unet.py

This module defines a read-to-use UNet model, a popular architecture for 
image-to-image translation tasks. 

Classes:
    UNet: Implements the U-Net style model for image-to-image translation.
"""

from typing import List, Union, Dict, Literal, Any

import torch
import torch.nn as nn

from .utils import ActivationType
from .factory import qualname
from .model import BaseGeneratorModel
from .encoder import Encoder
from .decoder import Decoder
from .blocks import Conv2DNormActBlock
from .up_down_blocks import (
    IdentityBlock,
    Conv2DDownBlock,
    MaxPool2DDownBlock,
    ConvTrans2DUpBlock,
    Bilinear2DUpsampleBlock
)

class UNet(BaseGeneratorModel):
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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        depth: int = 4,
        encoder_down_block: Literal["conv", "maxpool"] = "conv",
        decoder_up_block:   Literal["convt", "bilinear"] = "convt",
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

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            out_activation=act_type,
        )

        if isinstance(depth, int):
            if depth < 1:
                raise ValueError(f"Expected depth to be >= 1, got {depth}")
        else:
            raise TypeError(
                f"Expected depth to be int, got {type(depth).__name__}"
            )

        if encoder_down_block == "maxpool":
            in_block_handles = [MaxPool2DDownBlock] * (depth - 1)
        elif encoder_down_block == "conv":
            in_block_handles = [Conv2DDownBlock] * (depth - 1)
        else:
            raise ValueError("encoder_down_block must be 'conv' or 'maxpool'")
        in_block_handles = [IdentityBlock] + in_block_handles
        self._encoder_down_block = encoder_down_block

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
        self._num_units_cfg = _num_units  # Store original type for config
        
        self.in_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self._base_channels = base_channels

        self.encoder = Encoder(
            in_channels=base_channels,
            in_block_handles=in_block_handles,
            comp_block_handles=comp_block_handles,
            in_block_kwargs=in_block_kwargs,
            comp_block_kwargs=comp_block_kwargs,
            depth=depth,
        )
        self._depth = depth

        if decoder_up_block == "bilinear":
            decoder_in_block_handles = [Bilinear2DUpsampleBlock] * (depth - 1)
        elif decoder_up_block == "convt":
            decoder_in_block_handles = [ConvTrans2DUpBlock] * (depth - 1)
        else:
            raise ValueError("decoder_up_block must be 'convt' or 'bilinear'")
        self._decoder_up_block = decoder_up_block

        self.decoder = Decoder(
            encoder_feature_map_channels=self.encoder.feature_map_channels,
            in_block_handles=decoder_in_block_handles,
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

    def to_config(self) -> Dict[str, Any]:
        """
        Produce a JSON-serializable config sufficient to recreate this model.
        Includes class path, torch version, constructor args, and chosen block classes.
        """
        down_block_path = (
            qualname(MaxPool2DDownBlock) if \
                self._encoder_down_block == "maxpool" \
                    else qualname(Conv2DDownBlock)
        )
        up_block_path = (
            qualname(Bilinear2DUpsampleBlock) if \
                self._decoder_up_block == "bilinear" \
                    else qualname(ConvTrans2DUpBlock)
        )

        # Preserve the original type (int or list) for _num_units
        num_units = self._num_units_cfg

        return {
            "class_path": qualname(self.__class__),
            "module_versions": {
                "torch": torch.__version__,
            },
            "blocks": {
                "encoder_down_block": down_block_path,
                "decoder_up_block": up_block_path,
                "comp_block": qualname(Conv2DNormActBlock),
                "identity_block": qualname(IdentityBlock),
            },
            "init": {
                "in_channels": self._in_channels,
                "out_channels": self._out_channels,
                "base_channels": self._base_channels,
                "depth": self._depth,
                "encoder_down_block": self._encoder_down_block,
                "decoder_up_block":   self._decoder_up_block,
                "act_type": self._act_type,
                "_num_units": num_units,
            },
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "UNet":
        """
        Recreate a UNet from a config produced by `to_config()`.
        Accepts either the full dict or just the "init" sub-dict.
        """
        
        init_cfg = config.get("init", config)

        return cls(**init_cfg)
