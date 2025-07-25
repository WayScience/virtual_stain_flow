# decoder.py

from typing import Optional, Sequence

import torch.nn as nn
from torch import Tensor

from .stages import UpStage
from .handle_type_checking import (
    BlockHandleSequence,
    validate_block_configurations
)

"""
The Decoder class implements the upsampling path of a U-Net style
image-image translation model. It is designed to work with feature maps
from an encoder, allowing for flexible configurations of the input and
computation blocks at each stage.

The decoder stages are constructed based on the channels of the encoder's
feature maps, with each stage performing upsampling and concatenation
with skip connections from the corresponding encoder stage.
"""
class Decoder(nn.Module):
    def __init__(
        self,
        encoder_feature_map_channels: Sequence[int],
        in_block_handles: BlockHandleSequence,
        comp_block_handles: BlockHandleSequence,
        in_block_kwargs: Optional[Sequence[dict]] = None,
        comp_block_kwargs: Optional[Sequence[dict]] = None,
    ):
        super().__init__()

        # channel type/value checking
        if not isinstance(encoder_feature_map_channels, Sequence):
            raise TypeError(
                "Expected encoder_feature_map_channels to be a sequence, "
                f"got {type(encoder_feature_map_channels).__name__}")
        if not all(isinstance(ch, int) and ch > 0 for ch in encoder_feature_map_channels):
            raise TypeError(
                "Expected encoder_feature_map_channels to be a sequence "
                f"of positive ints, got {encoder_feature_map_channels}")
        
        depth = len(encoder_feature_map_channels)
        # block handle type checking        
        (
            in_block_handles,
            comp_block_handles,
            in_block_kwargs,
            comp_block_kwargs,
            _
        ) = validate_block_configurations(
            in_block_handles,
            comp_block_handles,
            in_block_kwargs,
            comp_block_kwargs,
            # we only need depth-1 upsampling stages to pair with depth
            # down sampling stages of the encoder
            depth=depth - 1, 
        )

        self._depth = depth
        
        # initialize decoder (upsampling) stages
        self.stages = nn.ModuleList()
        self._feature_map_channels = []
        # the upsampling stages will just have the same number of channels
        # as the corresponding encoder stage skip connections for convenience
        stage_in_channels = encoder_feature_map_channels[::-1][:-1]
        stage_skip_channels = encoder_feature_map_channels[::-1][1:]
        for _in_ch, _skip_ch, _in_handle, _comp_handle, _in_kwargs, _comp_kwargs in zip(
            stage_in_channels, 
            stage_skip_channels, 
            in_block_handles, 
            comp_block_handles,
            in_block_kwargs,
            comp_block_kwargs
        ):
            self.stages.append(UpStage(
                in_channels=_in_ch,
                skip_channels=_skip_ch,
                out_channels=_skip_ch,
                in_block_handle=_in_handle,
                comp_block_handle=_comp_handle,
                in_block_kwargs=_in_kwargs,
                comp_block_kwargs=_comp_kwargs
            ))

            self._feature_map_channels.append(_skip_ch)

    def forward(self, encoder_feature_maps: Sequence[Tensor]) -> Tensor:
        """
        Forward pass of the decoder.
        
        :param encoder_feature_maps: Sequence of feature maps from the encoder.
        :return: Output tensor after passing through the decoder.
        """
        
        if len(encoder_feature_maps) != self._depth:
            raise ValueError(
                f"Expected {self._depth} feature maps from the encoder, "
                f"got {len(encoder_feature_maps)}"
            )

        x = encoder_feature_maps[-1]
        skip_connections = encoder_feature_maps[::-1][1:]
        for stage, skip in zip(self.stages, skip_connections):
            x = stage(x, skip)
        
        return x
    
    @property
    def feature_map_channels(self) -> Sequence[int]:
        """
        Get the number of channels in the feature maps produced by each stage.
        
        :return: Sequence of integers representing the number of channels.
        """
        return self._feature_map_channels