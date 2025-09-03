import torch
from torch import nn
import torch.nn.functional as F

"""
Implementation of GaN discriminators to use along with UNet or FNet generator.
"""

class PatchBasedDiscriminator(nn.Module):
    
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
        :type n_in_channels: int
        :param n_in_filters: (int) number of filters in the first convolutional layer.
            Every subsequent layer will double the number of filters
        :type n_in_filters: int
        :param _conv_depth: (int) depth of the convolutional network
        :type _conv_depth: int
        :param _leaky_relu_alpha: (float) alpha value for leaky ReLU activation.
            Must be between 0 and 1
        :type _leaky_relu_alpha: float
        :param _batch_norm: (bool) whether to use batch normalization, defaults to False
        :type _batch_norm: bool
        """

        super().__init__()

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

class GlobalDiscriminator(nn.Module):

    def __init__(
            self,
            n_in_channels: int,
            n_in_filters: int,
            _conv_depth: int=4,
            _leaky_relu_alpha: float=0.2,
            _batch_norm: bool=False,
            _pool_before_fc: bool=False
            ):
        """
        A global discriminator for pix2pix GANs that outputs a single scalar value as the global probability

        Parameters:
        :param n_in_channels: (int) number of input channels
        :type n_in_channels: int
        :param n_in_filters: (int) number of filters in the first convolutional layer. 
        Every subsequent layer will double the number of filters
        :type n_in_filters: int
        :param _conv_depth: (int) depth of the convolutional network
        :type _conv_depth: int
        :param _leaky_relu_alpha: (float) alpha value for leaky ReLU activation. 
        Must be between 0 and 1
        :type _leaky_relu_alpha: float
        :param _batch_norm: (bool) whether to use batch normalization, defaults to False
        :type _batch_norm: bool
        :param _pool_before_fc: (bool) whether to pool before the fully connected network
        Pooling before the fully connected network can reduce the number of parameters
        :type _pool_before_fc: bool
        """       
        
        super().__init__()
        
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
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_layers(x)
        x = self.fc(x)

        return x
