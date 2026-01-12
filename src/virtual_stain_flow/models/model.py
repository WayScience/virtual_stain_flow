"""
/models/model.py

This module defines the base classes for models, including a base model class
and a base generator model class. These classes provide common functionality
for saving weights, converting to/from configuration dictionaries, and defining
the forward pass of the model.

Classes:
    BaseModel: Abstract base class for all models.
    BaseGeneratorModel: Abstract base class for virtual staining generator 
        models, extending BaseModel but defines more specific properties and
        methods.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Union, Any
import pathlib

import torch

from .encoder import Encoder
from .decoder import Decoder
from .utils import (
    get_activation,
    ActivationType
)

class BaseModel(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model and how
        the model parts are connected.

        :param x: Input tensor.
        :return: Output tensor after passing through the model.
        """
        raise NotImplementedError("Subclasses must implement this method.")   

    def save_weights(
        self, 
        filename: str,
        dir: Union[pathlib.Path, str]
    ) -> pathlib.Path:

        if isinstance(dir, str):
            dir = pathlib.Path(dir)
        elif isinstance(dir, pathlib.Path):
            pass
        else:
            raise TypeError(f"Expected dir to be str or pathlib.Path, "
                            f"got {type(dir)}")        
        
        if not dir.is_dir():
            raise NotADirectoryError(
                f"Expected dir {dir} to be a directory, "
                "but it is not. Please provide a valid directory."
            )
        if not dir.exists():
            raise FileNotFoundError(
                f"Path {dir} does not exist. "
                "Please provide a valid directory."
            )
        
        dir = dir.resolve(strict=True)

        weight_file = dir / filename

        torch.save(
            self.state_dict(),
            weight_file
        )

        return weight_file

    @abstractmethod
    def to_config(self) -> Dict[str, Any]:
        """
        Converts the model configuration to a dictionary format.
        
        :return: Dictionary containing model configuration.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict) -> 'BaseModel':
        """
        Creates a model instance from a configuration dictionary.
        
        :param config: Dictionary containing model configuration.
        :return: An instance of the model.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class BaseGeneratorModel(BaseModel):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        out_activation: ActivationType = 'sigmoid',
    ):
        
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels

        # Required/Optional modules
        self.in_conv: Optional[torch.nn.Module] = None
        self.encoder: Encoder = None
        self.decoder: Decoder = None
        self.out_conv: Optional[torch.nn.Module] = None
        self._act_type = out_activation
        self._out_activation = get_activation(out_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model and how
        the model parts are connected.

        :param x: Input tensor.
        :return: Output tensor after passing through the model.
        """
        if self.in_conv is not None:
            x = self.in_conv(x)        
        x = self.encoder(x)        
        x = self.decoder(x)        
        if self.out_conv is not None:
            x = self.out_conv(x)
        
        return self.out_activation(x)

    @property
    def in_channels(self) -> int:
        """Number of input channels."""
        return self._in_channels
    @property
    def out_channels(self) -> int:
        """Number of output channels."""
        return self._out_channels
    @property
    def out_activation(self) -> torch.nn.Module:
        """Activation function for the output layer."""
        return self._out_activation
