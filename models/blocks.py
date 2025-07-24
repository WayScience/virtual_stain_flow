"""
blocks.py

Following the conventions of timm.model.convnext 
(https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py), 
we define a block as the smallest modular unit in image-image translation model,
taking in a feature map tensor of shape (B, C, H, W) and returning a 
feature map tensor of shape (B, C', H', W') where the number of
channels C' and spatial dimensions (H', W') is determined by the block's
implementation.

This file Contains the definition of the AbstractBlock class defining the 
behavior of a "block", as well as centralizing type check for Type[AbstractBlock]
during runtime.
"""
from abc import ABC

import torch.nn as nn

def _is_block_handle(
    obj
) -> bool:
    """
    Check if the object is a block handle (a subclass of AbstractBlock).
    
    :param obj: The object to check.
    :return: True if obj is a block handle, False otherwise.
    """
    import inspect
    
    return inspect.isclass(obj) and issubclass(obj, AbstractBlock)

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