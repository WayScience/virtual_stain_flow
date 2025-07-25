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

from .utils import (
    get_norm,
    NormType
)
from .blocks import AbstractBlock

"""
Abstract base class for downsampling and upsampling blocks with
predefined output spatial dimensions by a factor of 2.
Added to avoid duplicated implementation of the same methods.
"""
class AbstractDownBlock(AbstractBlock):
    def __init__(self, in_channels, out_channels, num_units, **kwargs):
        super().__init__(in_channels, out_channels, num_units, **kwargs)

    @property
    def out_h(self, in_h: int) -> int:
        return in_h // 2

    @property
    def out_w(self, in_w: int) -> int:
        return in_w // 2

class AbstractUpBlock(AbstractBlock):
    def __init__(self, in_channels, out_channels, num_units, **kwargs):
        super().__init__(in_channels, out_channels, num_units, **kwargs)

    @property
    def out_h(self, in_h: int) -> int:
        return in_h * 2
    
    @property
    def out_w(self, in_w: int) -> int:
        return in_w * 2

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
        out_channels: Optional[int] = None,
        norm_type: NormType = 'none'
    ):
        """
        Initializes the IdentityBlock.

        :param in_channels: Number of input channels.
        :param out_channels: Not used, kept for consistent block class signature.
        :param norm_type: Type of normalization to apply. Default is 'none'.
        """
        
        super().__init__(
            in_channels=in_channels,
            out_channels=in_channels,
            num_units=1
        )

        self.network = nn.Identity()
        self.activation = get_norm(
            num_features=in_channels,
            norm_type=norm_type
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the block.
        
        :param x: Input tensor. Should have shape (B, C, H, W)
        :return: Output tensor, same shape as input and completely unchanged. 
        """
        return self.activation(
            self.network(x)
        )    

"""
Simple downsampling block that applies a Conv2D with kernel size 2 and stride 2.

This block is commonly used in UNet like architectures. 
No normalization or activation are added around the Conv2D backbone. 
"""
class Conv2DDownBlock(AbstractDownBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        norm_type: NormType = 'none'
    ):
        """
        Initializes the Conv2DDownBlock.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels. If not specified,
            defaults to double the number of input channels.
            This is a common practice in UNet architectures.
        :param norm_type: Type of normalization to apply. Default is 'none'.
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
        
        self.activation = get_norm(
            num_features=out_channels,
            norm_type=norm_type
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the block.
        
        :param x: Input tensor. Should have shape (B, C, H, W)
        :return: Output tensor, shape (B, C', H', W') where C' is out_channels,
            H' and W' are half of the input height and width respectively.
        """
        return self.activation(
            self.network(x)
        )

"""
A MaxPoolDownBlock that applies a MaxPool2D operation with fixed 
kernel size 2 and stride 2. Halves the spatial dimensions of the input tensor.

Lightweight alternative to Conv2DDownBlock due to the non-learnable nature.
Unlike Conv2D, the block does not change the number of channels. 
"""
class MaxPool2DDownBlock(AbstractDownBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        **kwargs 
    ):
        """
        Initializes the MaxPoolDownBlock.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels. 
            Not used, kept for consistent block class signature.
        :param kwargs: Additional keyword arguments. Not used, ensures
            compatibility with expanded kwargs for other blocks.
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
    
"""
Simple upsampling block that applies a ConvTranspose2D with 
kernel size 2 and stride 2.

This block is commonly used in UNet like architectures.
No normalization or activation are added around the ConvTranspose2D backbone.
"""
class ConvTrans2DUpBlock(AbstractUpBlock):
    def __init__(
        self,
        in_channels,
        out_channels: Optional[int] = None,
        norm_type: NormType = 'none',
    ):
        """
        Initializes the ConvTrans2DUpBlock.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels. If not specified,
            defaults to half the number of input channels.
            This is a common practice in UNet architectures.
        :param norm_type: Type of normalization to apply. Default is 'none'.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels or in_channels // 2,
            num_units=1
        )

        # we fix the behavior of this block to
        # upsample the spatial dimensions by a factor of 2
        self.network = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=2,
            stride=2,
            padding=0 # spatial upsampling
        )

        self.activation = get_norm(
            num_features=self.out_channels,
            norm_type=norm_type
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the block.
        :param x: Input tensor. Should have shape (B, C, H, W)
        :return: Output tensor, shape (B, C', H', W') where C' is out_channels,
            H' and W' are double the input height and width respectively.
        """
        return self.activation(
            self.network(x)
        )
    
"""
A PixelShuffle2DUpsampleBlock that applies a PixelShuffle operation
with a fixed scale factor of 2 (doubles spatial dimension).

The pixel shuffle operation itself is non-learnable, hence the success of
pixel shuffle upsampling depends on the previous convolutional layer(s) at
learning the correct way to organize channels. Most likely will not work well
with shallow blocks or small number of channels in input featuremaps. 
"""
class PixelShuffle2DUpBlock(AbstractUpBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        **kwargs
    ):
        """
        Initializes the PixelShuffleUpsampleBlock.

        :param in_channels: Number of input channels.
        :param out_channels: Not used, kept for consistent block class signature.
        :param kwargs: Additional keyword arguments. Not used, ensures
            compatibility with expanded kwargs for other blocks.
        """
        
        spatial_dims = 2
        scale_factor = 2
        # out_channel is determined by the number of input channels
        # as the pixel shuffle operation merely rearranges the channels
        # to the spatial dimensions
        out_channels = in_channels // (scale_factor ** spatial_dims)

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_units=1
        )

        # dynamic import here because only this class uses SubpixelUpsample
        from monai.networks.blocks import SubpixelUpsample
        # this initializes a deterministic pixel shuffle upsampling block,
        # adds a leading Conv2D layer to the network to handle mis-aligned
        # channel counts, and peforms the necessary icnr initization to avoid
        # checkerboard artifacts as described in Aitken et al. (2017). 
        # Convenient!
        self.network = SubpixelUpsample(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            scale_factor=scale_factor,
            conv_block='default', # this forces having a icnr initialized conv2d layer
            apply_pad_pool=True,
            bias=True
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the block.
        
        :param x: Input tensor. Should have shape (B, C, H, W)
        :return: Output tensor, shape (B, C', H', W') where C' is out_channels,
            H' and W' are double the input height and width respectively.
        """
        return self.network(x)