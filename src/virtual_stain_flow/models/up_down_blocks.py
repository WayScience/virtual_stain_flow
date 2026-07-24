"""
up_down_blocks.py
Original content refactored under the /blocks subpackage.

Compatibility shim to not break previous import behavior.
"""

from .blocks import (
    IdentityBlock,
    Conv2DDownBlock,
    MaxPool2DDownBlock,
    ConvTrans2DUpBlock,
    PixelShuffle2DUpBlock,
    Bilinear2DUpsampleBlock,
)

__all__ = [
    "IdentityBlock",
    "Conv2DDownBlock",
    "MaxPool2DDownBlock",
    "ConvTrans2DUpBlock",
    "PixelShuffle2DUpBlock",
    "Bilinear2DUpsampleBlock",
]
