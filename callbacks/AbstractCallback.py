from abc import ABC, abstractmethod
from typing import List, Callable, Dict

import torch

class AbstractCallback(ABC):
    """
    Abstract class for callbacks in the training process.
    Callbacks can be used to plot intermediate metrics, log contents, save checkpoints, etc.
    """
    
    def __init__(self, name: str):
        """
        :param name: Name of the callback.
        """        
        self._name = name

    def on_train_start(self, trainer):
        """
        Called at the start of training.
        """
        pass

    def on_epoch_start(self, trainer):
        """
        Called at the start of each epoch.
        """
        pass

    def on_epoch_end(self, trainer):
        """
        Called at the end of each epoch.
        """
        pass
    
    def on_train_end(self, trainer):
        """
        Called at the end of training.
        """
        pass