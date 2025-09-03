"""
blocks.py

Following the conventions of timm.model.convnext 
(https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py), 
we define a block as the smallest modular unit in image-image translation model,
taking in a feature map tensor of shape (B, C, H, W) and returning a 
feature map tensor of shape (B, C', H', W') where the number of
channels C' and spatial dimensions (H', W') is determined by the block's
implementation.

Here we further make the distinction between "computational blocks"  (this file) 
and "spatial dimension altering blocks", where the former does not change
the spatial dimensions of the input tensor, but may change the number of channels,
while the latter does change the spatial dimensions.

This file Contains the definition of the AbstractBlock class defining the 
behavior of a "block" and centralizing type check for Type[AbstractBlock]
during runtime.
Also contains the implementation of the spatial dimension preserving 
"computational blocks" that learns from feature map tensors at a specific
resolution to capture the context and local features of the images. This is
commonly achieved by applying dimension preserving convolutional layers
with kernel > 1 (usually 3), as in F/UNet architectures.  
"""
from abc import ABC
from typing import Optional

import timm
import torch.nn as nn
from torch import Tensor

from .utils import (
    get_norm, 
    NormType,
    get_activation,
    ActivationType
)

"""

"""
class AbstractBlock(ABC, nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_units: int = 1,
        **kwargs: dict
    ):
        
        super().__init__()

        # Centralizing input type checking
        if not isinstance(in_channels, int):
            raise TypeError("Expected in_channels to be int, "
                            f"got {type(in_channels).__name__}")
        if in_channels <= 0:
            raise ValueError("Expected in_channels to be positive, "
                             f"got {in_channels}")
        if not isinstance(out_channels, int):
            raise TypeError("Expected out_channels to be int, "
                            f"got {type(out_channels).__name__}")
        if out_channels <= 0:
            raise ValueError("Expected out_channels to be positive, "
                             f"got {out_channels}")
        if not isinstance(num_units, int):
            raise TypeError("Expected num_units to be int, "
                            f"got {type(num_units).__name__}")
        if num_units <= 0:
            raise ValueError("Expected num_units to be positive, "
                             f"got {num_units}")

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._num_units = num_units    
    
    # Centralizing property definitions for blocks
    @property
    def in_channels(self) -> int:
        return self._in_channels
    @property
    def out_channels(self) -> int:
        return self._out_channels
    @property
    def num_units(self) -> int:
        return self._num_units
    
    # These 2 below should be overriden to reflect the actual spatial dimension
    # changes the block applies. By default they indicate spatial preserving
    # blocks, i.e. the height and width of the input tensor remain unchanged.
    @property
    def out_h(self, in_h: int) -> int:
        return in_h
    @property
    def out_w(self, in_w: int) -> int:
        return in_w    

"""
A ConvNeXt block that applies a sequence of ConvNeXt units
with inital 2D convolution to adjust the number of channels if needed.
Mimics the design of timm.models.convnext.ConvNeXtStage but less sophisticated
in implementation. 
"""
class Conv2DConvNeXtBlock(AbstractBlock):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_units: int = 1,
        norm_type: NormType = 'layer',
        conv_kernel_size: int = 1,
        convnext_kernel_size: int = 7
    ):
        """
        Initializes a ConvNeXt block with the specified parameters.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels. If
            None, defaults to in_channels.
        :param num_units: Number of ConvNeXt units in the block.
        :param norm_type: Type of normalization to apply.
            Default is 'layer' (GroupNorm with 1 group).
        :param conv_kernel_size: Kernel size for the initial 2d convolution
            that adjusts the number of channels if in_channels != out_channels.
            Default is 1 (identity convolution, fastest). 
            Though one may want to use larger kernel sizes (3) to allow for 
            some spatial feature extraction prior to the ConvNeXtBlocks. 
        :param convnext_kernel_size: Kernel size for the depth-wise convolution
            layers inside the ConvNeXtBlock. Default is 7, 
            as recommended by Liu et al. (2022), for large receptive fields.
        """
        
        out_channels = out_channels or in_channels

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_units=num_units
        )

        layers = []

        if in_channels != out_channels:
            # insert a spatial dimension preserving convolution
            # operation to adjust the number of channels because
            # ConvNeXtBlock expects same input and output channels
            # if the in/out channels are matched, this won't be added. 
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=conv_kernel_size,
                    stride=1, # fixed
                    padding='same' # ensure spatial dimensions are preserved
                )
            )
            # timm.models.convnext.ConvNeXtStage adds a normalization
            # BEFORE the initial 2D convolution, which in practice has 
            # made models with ConvNeXtStage blocks in decoder hard to train.
            # Here we add the normalization AFTER the initial 2D convolution
            # right before the ConvNeXtBlock units.
            layers.append(
                get_norm(
                    num_features=out_channels,
                    norm_type=norm_type
                )
            )

        for _ in range(num_units):
            # Add ConvNeXtBlock in sequence.
            # Under the hood a single ConvNeXtBlock is defined by:
            # Depthwise Conv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv
            # note that the Depthwise Conv is intended to capture the 
            # channel-wise spatial features with large kernel_size/receptive
            # field recommended by Liu et al. (2022). The 1x1 convs are 
            # responsbile for channel mixing and non-linearity following
            # spatial feature extraction. This effectively separates the
            # spatial and channel-wise feature extraction computations, 
            # contrasting with the standard Conv2D whose kernel does both
            # simultaneously. 
            layers.append(
                timm.models.convnext.ConvNeXtBlock(
                    in_chs=out_channels, # same input/output channels
                    out_chs=out_channels, 
                    kernel_size=convnext_kernel_size,
                    stride=1, # fixed
                    ls_init_value=None,
                    # this is a switch between 2 equivalent implementations
                    # but with contrasting speed <-> model size tradeoffs.
                    # here by setting conv_mlp=True we use the faster but 
                    # larger model implementation 
                    conv_mlp=True, 
                    use_grn=True, # GlobalResponseNorm for mlp layers
                    norm_layer=timm.layers.LayerNorm2d, 
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the block.
        
        :param x: Input tensor. Should have shape (B, C, H, W)
        :return: Output tensor after passing through the block. 
            Should have shape (B, C', H, W) where C' is the output channels,
            and H and W are unchanged input spatial dimensions.
        """
        return self.network(x)    
    
    
    #this is a spatial preserving block, we don't override the out_h and out_w
    #def out_h(self, in_h: int) -> int:
    #def out_w(self, in_w: int) -> int:


"""
A Conv2D block that applies a sequence of Conv2D -> Norm -> Activation
layers, commonly used in UNet architectures.
"""
class Conv2DNormActBlock(AbstractBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_units: int = 1,
        norm_type: NormType = 'batch',
        act_type: ActivationType = 'relu'
    ):
        """
        Initializes a Conv2D block with the specified parameters.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels. If None, defaults to
            in_channels.
        :param num_units: Number of Conv2D>Norm>Activation units in the block.
        :param norm_type: Type of normalization to apply. Default is 'batch'
            (GroupNorm with 1 group).
        :param act_type: Type of activation function to apply. Default is 'relu'.
        """

        out_channels = out_channels or in_channels

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_units=num_units
        )

        mid_channels = [out_channels] * (num_units - 1)

        layers = []

        for _in, _out in zip(
            [in_channels] + mid_channels, 
            mid_channels + [out_channels]
        ):
            # standard Conv2D -> Norm -> Activation sequence
            # used widely in UNets (with BatchNorm and ReLU).
            layers.append(
                nn.Conv2d(
                    in_channels=_in,
                    out_channels=_out,
                    kernel_size=3, # fixed for now
                    stride=1, # fixed
                    padding='same' # indicate spatial preserving unit
                )
            )
            layers.append(
                get_norm(
                    num_features=_out,
                    norm_type=norm_type
                )
            )
            layers.append(
                get_activation(act_type)
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the block.
        
        :param x: Input tensor. Should have shape (B, C, H, W)
        :return: Output tensor after passing through the block. 
            Should have shape (B, C', H, W) where C' is the output channels,
            and H and W are unchanged input spatial dimensions.
        """
        return self.network(x) 
    
    #this is a spatial preserving block, we don't override the out_h and out_w
    #def out_h(self, in_h: int) -> int:
    #def out_w(self, in_w: int) -> int: