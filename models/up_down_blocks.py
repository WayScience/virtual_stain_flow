"""
up_down_blocks.py

Following the conventions of timm.model.convnext 
(https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py), 
we define a block as the smallest modular unit in image-image translation model,
taking in a feature map tensor of shape (B, C, H, W) and returning a 
feature map tensor of shape (B, C', H', W') where the number of
channels C' and spatial dimensions (H', W') is determined by the block's
implementation.

Here we further make the distinction between "computational blocks" and 
"spatial dimension altering blocks" (this file), where the former does not change
the spatial dimensions of the input tensor, but may change the number of channels,
while the latter does change the spatial dimensions.

This file Contains the implementation of the "spatial dimension altering blocks" that
alter the spatial dimension of the input tensor on top of potential channel
count number changes. These blocks are commonly used in UNet-like architectures
to reduce and increase resolution of feature map tensors (images), in conjunction
with the spatial dimension preserving blocks, implemented in blocks.py, to 
capture the context and local features of hte images at differing resolutions.
"""
from typing import Optional

import torch.nn as nn
from torch import Tensor

from .blocks import AbstractBlock

"""
Identity block that does nothing to the input. Imagine it as a placeholder
for a actual spatial dimension altering block only used at the first sampling
stage of the UNet-like architectures, so we don't down-sample the input too
quickly before learning at the initial image resolution. Implemented as
just an Identity layer, which is a no-op operation that returns the input as is.
"""
class IdentityBlock(AbstractBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None
    ):
        """
        Initializes the IdentityBlock.

        :param in_channels: Number of input channels.
        :param out_channels: Not used, kept for consistent block class signature.
        """
        
        super().__init__(
            in_channels=in_channels,
            out_channels=in_channels,
            num_units=1
        )

        self.network = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the block.
        
        :param x: Input tensor. Should have shape (B, C, H, W)
        :return: Output tensor, same shape as input and completely unchanged. 
        """
        return self.network(x)
    

"""
Simple downsampling block that applies a Conv2D with kernel size 2 and stride 2.

This block is commonly used in UNet like architectures. 
No normalization or activation are added around the Conv2D backbone. 
"""
class Conv2DDownBlock(AbstractBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None
    ):
        """
        Initializes the Conv2DDownBlock.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels. If not specified,
            defaults to double the number of input channels.
            This is a common practice in UNet architectures.
        """
        
        out_channels = out_channels or (in_channels * 2)

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_units=1
        )

        # we fix the behavior of this block to 
        # downsample the spatial dimensions by a factor of 2
        self.network = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2, # fixed
            stride=2, # fixed
            padding=0 # spatial downsampling
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the block.
        
        :param x: Input tensor. Should have shape (B, C, H, W)
        :return: Output tensor, shape (B, C', H', W') where C' is out_channels,
            H' and W' are half of the input height and width respectively.
        """
        return self.network(x)
    
    def out_h(self, in_h: int) -> int:
        """
        Computes the output height after downsampling.
        
        :param in_h: Input height.
        :return: Output height after downsampling.
        """
        return in_h // 2
    
    def out_w(self, in_w: int) -> int:
        """
        Computes the output width after downsampling.
        
        :param in_w: Input width.
        :return: Output width after downsampling.
        """
        return in_w // 2    

"""
A MaxPoolDownBlock that applies a MaxPool2D operation with fixed 
kernel size 2 and stride 2. Halves the spatial dimensions of the input tensor.

Lightweight alternative to Conv2DDownBlock due to the non-learnable nature.
Unlike Conv2D, the block does not change the number of channels. 
"""
class MaxPool2DDownBlock(AbstractBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None
    ):
        """
        Initializes the MaxPoolDownBlock.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels. 
            Not used, kept for consistent block class signature.
        """
        
        super().__init__(
            in_channels=in_channels,
            out_channels=in_channels,
            num_units=1
        )

        # we fix the behavior of this block to 
        # downsample the spatial dimensions by a factor of 2
        self.network = nn.MaxPool2d(
            kernel_size=2, # fixed
            stride=2, # fixed
            padding=0 # spatial downsampling
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the block.
        
        :param x: Input tensor. Should have shape (B, C, H, W)
        :return: Output tensor, shape (B, C, H', W') where H' and W' are 
            half of the input height and width respectively. C remains unchanged.
        """
        return self.network(x)
    
    def out_h(self, in_h: int) -> int:
        """
        Computes the output height after downsampling.
        
        :param in_h: Input height.
        :return: Output height after downsampling.
        """
        return in_h // 2
    
    def out_w(self, in_w: int) -> int:
        """
        Computes the output width after downsampling.
        :param in_w: Input width.
        :return: Output width after downsampling.
        """
        return in_w // 2