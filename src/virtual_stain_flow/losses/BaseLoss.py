"""

"""
from __future__ import annotations
from typing import Protocol, runtime_checkable, Any

from torch import Tensor
import torch.nn as nn


@runtime_checkable
class LossLike(Protocol):
    """
    For type checking purposes only.
    """

    def forward(self, *args: Any, **kwargs: Any) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...


class BaseLoss(nn.Module):
    """Base class for loss functions."""

    def __init__(self, _metric_name: str):
        
        super(BaseLoss, self).__init__()

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
    def metric_name(self):
        """Defines the mertic name returned by the class."""
        return self._metric_name
    
    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__}.forward() must be implemented by subclasses."
        )
