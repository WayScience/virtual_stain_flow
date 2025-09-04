from abc import ABC, abstractmethod

import torch
import torch.nn as nn

"""
Adapted from https://github.com/WayScience/nuclear_speckles_analysis
"""
class AbstractLoss(nn.Module, ABC):
    """Abstract class for metrics"""

    def __init__(self, _metric_name: str):
        
        super(AbstractLoss, self).__init__()

        self._metric_name = _metric_name
        self._trainer = None

    @property
    def trainer(self):
        return self._trainer
    
    @trainer.setter
    def trainer(self, value):
        """
        Setter of trainer meant to be called by the trainer class during initialization
        """
        self._trainer = value

    @property
    def metric_name(self, _metric_name: str):
        """Defines the mertic name returned by the class."""
        return self._metric_name

    @abstractmethod
    def forward(self, truth: torch.Tensor, generated: torch.Tensor
                ) -> float:
        """
        Computes the metric given information about the data

        :param truth: The tensor containing the ground truth image, 
            should be of shape [batch_size, channel_number, img_height, img_width].
        :type truth: torch.Tensor
        :param generated: The tensor containing model generated image, 
            should be of shape [batch_size, channel_number, img_height, img_width].
        :type generated: torch.Tensor
        :return: The computed metric as a float value.
        :rtype: float
        """
        pass
