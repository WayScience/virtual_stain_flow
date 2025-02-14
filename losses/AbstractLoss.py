from abc import ABC, abstractmethod
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
        self._trainer = value

    @property
    def metric_name(self, _metric_name: str):
        """Defines the mertic name returned by the class."""
        return self._metric_name

    @abstractmethod
    def forward(self):
        """Computes the metric given information about the data."""
        pass