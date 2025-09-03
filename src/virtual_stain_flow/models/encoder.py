# encoder.py

from typing import Optional, Sequence

import torch.nn as nn
from torch import Tensor

from .stages import DownStage
from .handle_type_checking import (
    BlockHandleSequence,
    validate_block_configurations
)
"""
The Encoder class is a functional equivalent of a U-Net style down-sampling path 
for image-image translation tasks. The implemenation allows for flexible
configuration of the input and computation blocks, allowing for different
downsampling strategies and channel configurations at each depth/stage.

Essentially just instantiating and storing in sequence DownStage objects. 
The complexity coming from parsing the in_block_handles and comp_block_handles
to match the expected input and output channels of the blocks,
and the optional comp_block_kwargs that can be used to pass additional
keyword arguments to the computation blocks.
"""
class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_block_handles: BlockHandleSequence,
        comp_block_handles: BlockHandleSequence,
        in_block_kwargs: Optional[Sequence[dict]] = None,
        comp_block_kwargs: Optional[Sequence[dict]] = None,
        depth: Optional[int] = None,
    ):
        """
        Initializes the Encoder with the specified parameters.

        :param in_channels: Number of input channels.
        :param in_block_handles: Sequence of input block handles. Each handle
            should be a subclass of AbstractBlock. If a sequence is supplied,
            overrides the depth parameter and the encoder will be of depth
            len(comp_block_handles).
        :param comp_block_handles: Sequence of computation block handles.
            should be a subclass of AbstractBlock. If a sequence is supplied,
            overrides the depth parameter and the encoder will be of depth
            len(comp_block_handles).
        :param in_block_kwargs: Optional sequence of dictionaries with
            additional keyword arguments for the input blocks.
        :param comp_block_kwargs: Optional sequence of dictionaries with
            additional keyword arguments for the computation blocks.
            If not provided, defaults to a sequence of empty dictionaries.
        :param depth: Optional depth of the encoder. If provided, it will
            expand the in_block_handles and comp_block_handles to match the
            specified depth. If not provided, it will be inferred from the
            lengths of in_block_handles and comp_block_handles.
        """
        super().__init__()

        # channel type checking 
        if not isinstance(in_channels, int):
            raise TypeError("Expected in_channels to be int, "
                            f"got {type(in_channels).__name__}")
        if in_channels <= 0:
            raise ValueError("Expected in_channels to be positive, "
                             f"got {in_channels}")

        # block handle type checking        
        (
            in_block_handles,
            comp_block_handles,
            in_block_kwargs,
            comp_block_kwargs,
            inferred_depth
        ) = validate_block_configurations(
            in_block_handles,
            comp_block_handles,
            in_block_kwargs,
            comp_block_kwargs,
            depth
        )

        self._depth = inferred_depth
        self._feature_map_channels = []
        # initialize stages
        self.stages = nn.ModuleList()
        stage_in_channels = in_channels
        for _in_handle, _comp_handle, _in_kwargs, _comp_kwargs in zip(
            in_block_handles,
            comp_block_handles,
            in_block_kwargs,
            comp_block_kwargs
        ):
            self.stages.append(
                DownStage(
                    in_channels=stage_in_channels,
                    # default behavior of DownStage is doubling channels
                    in_block_handle=_in_handle,
                    comp_block_handle=_comp_handle,
                    in_block_kwargs=_in_kwargs,
                    comp_block_kwargs=_comp_kwargs
                )
            )
            # dynamically obtain the in_channels for the next stage
            stage_out_channels = self.stages[-1].out_channels
            self._feature_map_channels.append(stage_out_channels)
            stage_in_channels = stage_out_channels

    def forward(self, x: Tensor) -> Sequence[Tensor]:
        """
        Forward pass of the encoder.
        
        :param x: Input tensor. Should have shape (B, C, H, W)
        :return: Encoder (downsampling path) feature maps across all stages.
        """

        feature_maps = []

        for stage in self.stages:
            x = stage(x)
            feature_maps.append(x)

        return feature_maps
    
    @property
    def feature_map_channels(self) -> Sequence[int]:
        """
        Get the number of channels in the feature maps produced by each stage.
        
        :return: Sequence of integers representing the number of channels in the
            feature maps produced by each stage.
        """
        return self._feature_map_channels