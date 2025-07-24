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