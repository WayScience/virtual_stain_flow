"""
/models/discriminator.py

Implementation of GaN discriminators to use along with UNet or FNet generator.
"""

from typing import Dict, Any, Optional

import torch
from torch import nn

from .factory import qualname
from .model import BaseModel

class PatchBasedDiscriminator(BaseModel):    
    def __init__(
        self,
        n_in_channels: int,
        n_in_filters: int,
        _conv_depth: int=4,
        _leaky_relu_alpha: float=0.2,
        _batch_norm: bool=False
    ):
        """
        A patch-based discriminator for pix2pix GANs that outputs a feature map
          of probabilities

        :param n_in_channels: (int) number of input channels
        :param n_in_filters: (int) number of filters in the first convolutional layer.
            Every subsequent layer will double the number of filters
        :param _conv_depth: (int) depth of the convolutional network
        :param _leaky_relu_alpha: (float) alpha value for leaky ReLU activation.
            Must be between 0 and 1
        :param _batch_norm: (bool) whether to use batch normalization, defaults to False
        """

        super().__init__()

        self._n_in_channels = n_in_channels
        self._n_in_filters = n_in_filters
        self._conv_depth = _conv_depth
        self._leaky_relu_alpha = _leaky_relu_alpha
        self._batch_norm = _batch_norm

        conv_layers = []

        n_channels = n_in_filters
        conv_layers.append(
            nn.Conv2d(n_in_channels, n_channels, kernel_size=4, stride=2, padding=1)
            )
        conv_layers.append(nn.LeakyReLU(_leaky_relu_alpha, inplace=True))

        # Sequentially add convolutional layers
        for _ in range(_conv_depth - 2):
            conv_layers.append(
                nn.Conv2d(n_channels, n_channels * 2, kernel_size=4, stride=2, padding=1)
                )
            conv_layers.append(nn.BatchNorm2d(n_channels * 2))
            conv_layers.append(nn.LeakyReLU(_leaky_relu_alpha, inplace=True))
            n_channels *= 2

        # Another layer of conv without downscaling
        ## TODO: figure out if this is needed
        conv_layers.append(
            nn.Conv2d(n_channels, n_channels * 2, kernel_size=4, stride=1, padding=1)
            )
        
        if _batch_norm:
            conv_layers.append(nn.BatchNorm2d(n_channels * 2))

        conv_layers.append(nn.LeakyReLU(_leaky_relu_alpha, inplace=True))
        n_channels *= 2
        self._conv_layers = nn.Sequential(*conv_layers)        
        
        # Output layer to get the probability map
        self.out = nn.Sequential(
            *[nn.Conv2d(n_channels, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()]
        )        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_layers(x)
        x = self.out(x)

        return x
    
    def to_config(self) -> Dict[str, Any]:
        return {
            "class_path": qualname(self.__class__),
            "module_versions": {
                "torch": torch.__version__,
            },
            "init": {
                "n_in_channels": self._n_in_channels,
                "n_in_filters": self._n_in_filters,
                "_conv_depth": self._conv_depth,
                "_leaky_relu_alpha": self._leaky_relu_alpha,
                "_batch_norm": self._batch_norm,
            },
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PatchBasedDiscriminator":
        init_cfg = config.get("init", config)
        return cls(**init_cfg)

class GlobalDiscriminator(BaseModel):
    def __init__(
        self,
        n_in_channels: int,
        n_in_filters: int,
        out_activation: Optional[torch.nn.Module] = None,
        _conv_depth: int=4,
        _leaky_relu_alpha: float=0.2,
        _batch_norm: bool=False,
        _pool_before_fc: bool=False
    ):
        """
        A global discriminator for pix2pix GANs that outputs a single scalar value as the global probability

        :param n_in_channels: (int) number of input channels
        :param n_in_filters: (int) number of filters in the first convolutional layer. 
            Every subsequent layer will double the number of filters
        :param out_activation: output activation function
        :param _conv_depth: (int) depth of the convolutional network
        :param _leaky_relu_alpha: (float) alpha value for leaky ReLU activation. 
            ust be between 0 and 1
        :param _batch_norm: (bool) whether to use batch normalization, defaults to False
        :param _pool_before_fc: (bool) whether to pool before the fully connected network
            Pooling before the fully connected network can reduce the number of parameters
        """       
        
        super().__init__()

        self._n_in_channels = n_in_channels
        self._n_in_filters = n_in_filters
        self._conv_depth = _conv_depth
        self._leaky_relu_alpha = _leaky_relu_alpha
        self._batch_norm = _batch_norm
        
        conv_layers = []
        
        n_channels = n_in_filters
        conv_layers.append(
            nn.Conv2d(n_in_channels, n_channels, kernel_size=4, stride=2, padding=1)
            )
        conv_layers.append(nn.LeakyReLU(_leaky_relu_alpha, inplace=True))

        # Sequentially add convolutional layers
        for _ in range(_conv_depth - 1):
            conv_layers.append(
                nn.Conv2d(n_channels, n_channels * 2, kernel_size=4, stride=2, padding=1)
                )
            
            if _batch_norm:
                conv_layers.append(nn.BatchNorm2d(n_channels * 2))

            conv_layers.append(nn.LeakyReLU(_leaky_relu_alpha, inplace=True))
            n_channels *= 2

        # Flattening
        if _pool_before_fc:
            conv_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        conv_layers.append(nn.Flatten())        
        self._conv_layers = nn.Sequential(*conv_layers)


        # Fully connected network to output probability
        self.fc = nn.Sequential(
            nn.LazyLinear(512),
            nn.LeakyReLU(_leaky_relu_alpha, inplace=True),
            nn.Linear(512, 1),
        )

        self.out_activation = out_activation or torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_layers(x)
        x = self.fc(x)
        x = self.out_activation(x)

        return x
    
    def to_config(self) -> Dict[str, Any]:
        return {
            "class_path": qualname(self.__class__),
            "module_versions": {
                "torch": torch.__version__,
            },
            "init": {
                "n_in_channels": self._n_in_channels,
                "n_in_filters": self._n_in_filters,
                "_conv_depth": self._conv_depth,
                "_leaky_relu_alpha": self._leaky_relu_alpha,
                "_batch_norm": self._batch_norm,
            },
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GlobalDiscriminator":
        init_cfg = config.get("init", config)
        return cls(**init_cfg)
