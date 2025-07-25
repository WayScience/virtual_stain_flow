"""
stages.py

Following the conventions of timm.model.convnext 
(https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py), 
we define a stage as a packaged sequence of blocks that would be equivalent to 
the a single down-sampling or up-sampling step in a F/UNet with a spatial 
dimension altering block followed by a computation block. A stage object
should take in a feature map tensor of shape (B, C, H, W) and return a
feature map tensor of shape (B, C', H', W'), where the spatial dimensions
(H', W') are altered according to the stage's in_block and comp_block
operations, and the channel dimension C' is defined by the stage's
out_channels parameter.

Here, we implement a generic Stage class that accepts handles to the 
in_block and comp_block, allowing for flexibility in defining the
stage's behavior and quickly creating custom stages for different optimizations.
The stage class handles the matching the output_channels of its in_block to 
the input_channels of its comp_block plus the additional channels from a skip 
connection if applicable. 
"""

from typing import Optional, Sequence, Type
import torch
import torch.nn as nn
from torch import Tensor

from .blocks import (
    AbstractBlock,
    Conv2DNormActBlock
)
from .up_down_blocks import (
    Conv2DDownBlock,
    ConvTrans2DUpBlock
)

"""
Centralizing input type checking, functionality, and property definition of 
the stage class without pre-determinig down or up sampling behavior. 

This generic stage class also implements a skip connection functionality, which
may or may not be used by the subclass. 
"""
class Stage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_block_handle: Type[AbstractBlock],
        comp_block_handle: Type[AbstractBlock],
        skip_channels: Optional[int] = None,
        in_block_kwargs: Optional[dict] = None,
        comp_block_kwargs: Optional[dict] = None
    ):
        """
        Initializes the stage with the specified parameters.

        :param in_channels: Number of input channels. 
        :param out_channels: Number of output channels. 
        :param in_block_handle: Class handle for the input block. 
            Should be a subclass of AbstractBlock.
        :param comp_block_handle: Class handle for the computation block.
            Should be a subclass of AbstractBlock.
        :param skip_channels: Number of channels in the skip connection.
            If None, defaults to 0 (no skip connection).
        :param in_block_kwargs: Additional keyword arguments for the
            in_block_handle. 
        :param comp_block_kwargs: Additional keyword arguments for the
            comp_block_handle.
        """

        super().__init__()

        # channel type checking
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
        
        # default assume no skip connections
        skip_channels = skip_channels or 0
        if not isinstance(skip_channels, int):
            raise TypeError("Expected skip_channels to be int, "
                            f"got {type(skip_channels).__name__}")
        if skip_channels < 0:
            raise ValueError("Expected skip_channels to be non-negative, "
                             f"got {skip_channels}")

        self._in_channels = in_channels
        self._skip_channels = skip_channels
        self._out_channels = out_channels

        # Type checking and instantiation of the in_block.
        if not issubclass(in_block_handle, AbstractBlock):
            raise TypeError(f"Expected in_block_handle to be a subclass of "
                            "AbstractBlock, "
                            f"got {in_block_handle.__name__}")
        self.in_block = in_block_handle(
            in_channels=in_channels,
            out_channels=None,
            **(in_block_kwargs or {})
        )

        # Type checking and instantiation of the comp_block.
        if not issubclass(comp_block_handle, AbstractBlock):
            raise TypeError(f"Expected comp_block_handle to be a subclass of "
                            "AbstractBlock, "
                            f"got {comp_block_handle.__name__}")
        # Define the block input/output channel counts
        # a typical stage will have a user-defined `out_channels` that is 
        # usually different by a factor (usually doubling or havling) from 
        # the `in_channels` (which is the `out_channels` from the previous stage). 
        # 
        # However, the in_block forward pass may or may not get one straight 
        # from the stage `in_channels` to the stage `out_channels`. e.g. 
        # operations like Maxpooling does not change the number of channels.
        # The introduction of a skip connecton, feeding into the computational
        # blocks may also result in channel count mismatch.
        #
        # In such cases, the computation blocks are often used to compensate 
        # for the channel count difference. Here, we will force `comp_block` 
        # to take the concatenated input from the in_block and the skip
        self.comp_block = comp_block_handle(
            in_channels=self.in_block.out_channels + skip_channels,
            out_channels=out_channels,
            **(comp_block_kwargs or {})
        )

    def forward(self, x: Tensor, skip: Optional[Tensor] = None) -> Sequence[Tensor]:
        """
        Forward pass of the downsampling stage.
        
        :param x: Input tensor. Should have shape (B, C, H, W) where:
            - B is the batch size,
            - C is the number of channels, should be exactly self._in_channels,
            - H is the height,
            - W is the width.
        :param skip: Optional skip connection tensor. Only used if the stage
            has skip_channels > 0. 
        :return: List of output tensors after processing through each block.
        """
        
        x = self.in_block(x)

        # only concatenate the skip tensor if the stage has skip channels 
        # configured to be > 0. Otherwise, the skip tensor is not used.
        if self._skip_channels > 0:
            if skip is None:
                raise ValueError("Skip tensor must be provided when "
                                 "stage skip_channels > 0")
            x = torch.cat([x, skip], dim=1)

        return self.comp_block(x)

    @property
    def in_channels(self) -> int:
        return self._in_channels
    
    @property
    def skip_channels(self) -> int:
        return self._skip_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels
    
    @property
    def out_h(self, in_h: int) -> int:
        _out_h = in_h
        for block in self.blocks:
            if isinstance(block, Conv2DDownBlock):
                _out_h = block.out_h(_out_h)
        return _out_h
    
    @property
    def out_w(self, in_w: int) -> int:
        _out_w = in_w
        for block in self.blocks:
            if isinstance(block, Conv2DDownBlock):
                _out_w = block.out_w(_out_w)
        return _out_w

"""
Wrapped Stage class to define a downsampling stage.

Just a Stage class with pre-defined in_block and comp_block handles and default
out_channels set to double the in_channels. Assumes no skip connections. 
"""
class DownStage(Stage):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        in_block_handle: Type[AbstractBlock] = Conv2DDownBlock,
        comp_block_handle: Optional[Type[AbstractBlock]] = None,
        in_block_kwargs: Optional[dict] = None,
        comp_block_kwargs: Optional[dict] = {'num_units': 2}
    ):
        """
        Initializes the downsampling stage with the specified parameters.
        """

        super().__init__(
            in_channels=in_channels,
            skip_channels=0, # the typical down stage does not use skip connections
            out_channels=out_channels or (in_channels * 2),
            in_block_handle=in_block_handle,
            comp_block_handle=comp_block_handle or Conv2DNormActBlock,
            in_block_kwargs=in_block_kwargs,
            comp_block_kwargs=comp_block_kwargs
        )

"""
Wrapped Stage class to define an upsampling stage allowing for skip connections.

Just a Stage class with pre-defined in_block and comp_block handles, and default
out_channels set to half the in_channels.
"""
class UpStage(Stage):
    def __init__(
        self,
        in_channels: int,
        skip_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        in_block_handle: Type[AbstractBlock] = ConvTrans2DUpBlock,
        comp_block_handle: Optional[Type[AbstractBlock]] = None,
        in_block_kwargs: Optional[dict] = None,
        comp_block_kwargs: Optional[dict] = {'num_units': 2}
    ):
        """
        Initializes the upsampling stage with the specified parameters.
        """

        skip_channels = skip_channels or 0

        super().__init__(
            in_channels=in_channels,
            skip_channels=skip_channels,
            out_channels=out_channels or (in_channels // 2),
            in_block_handle=in_block_handle,
            comp_block_handle=comp_block_handle or Conv2DNormActBlock,
            in_block_kwargs=in_block_kwargs,
            comp_block_kwargs=comp_block_kwargs
        )