"""
utils.py

Utility functions for model normalization and activation layers retrieval,
also centralizing input type checking and property definitions for blocks.
"""
from typing import Literal

import torch.nn as nn

NormType = Literal['batch', 'layer', 'none']
ActivationType = Literal[
    'relu', 
    'gelu', 
    'sigmoid', 
    'softmax', 
    'none'
]

"""
Helper Factory methods for retrieving model normalization and activation layers
"""
def get_norm(
        num_features: int,
        norm_type: NormType = "batch"
    ) -> nn.Module:
    """
    Factory for normalization layers.
    Currently the available types are:
    - "batch" for BatchNorm2d
    - "layer" for LayerNorm (implemented as GroupNorm with 1 group)
    - "none" for Identity (no normalization)

    :param num_features: Number of features (channels) in the input tensor.
        In context of feature maps tenors from 2D Convolutional Networks, this
        corresponds to the number of C from the tensor shape (B, C, H, W).
        This will result in different behavior for specific normalization types.
        See below for details.
    :param norm_type: Type of normalization to apply. Default is "batch".
    :return: An instance of the specified normalization layer.
    """    
    if norm_type == "batch":
        """
        The standard BatchNorm2d layer normalizes across the batch dimension
        and spatial dimensions (H, W), acting per channel C.
        """
        return nn.BatchNorm2d(num_features)
    elif norm_type == "layer":
        """
        GroupNorm(1, num_features) = Apply normalization over (C, H, W)
        per sample (every 1 from B dimension). Reported to be benefitical
        for smaller batch sizes and higher variance in feature maps by
        Liu et al. (2022) - "A ConvNet for the 2020s"
        """
        return nn.GroupNorm(1, num_features)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported norm type: {norm_type!r}")
    
def get_activation(
        act_type: ActivationType = "relu"
    ) -> nn.Module:
    """
    Factory for activation functions.

    Currently the available types are:
    - "relu" for ReLU activation
    - "gelu" for Gaussian Error Linear Unit activation
    - "sigmoid" for Sigmoid activation
    - "softmax" for Softmax activation (dim=1)
    - "none" for Identity (no activation)
    :param act_type: Type of activation to apply. Default is "relu".
    :return: An instance of the specified activation function.
    """
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    elif act_type == "gelu":
        """
        Gaussian Error Linear Unit (GELU) activation.
        This is a smooth approximation of the ReLU activation,
        which has been shown to perform well in transformer architectures.
        """
        return nn.GELU()
    elif act_type == "sigmoid":
        return nn.Sigmoid()
    elif act_type == "softmax":
        return nn.Softmax(dim=1)
    elif act_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation: {act_type!r}")