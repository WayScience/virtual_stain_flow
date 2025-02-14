from typing import List, Union

import torch
import torch.nn as nn

from .AbstractCallback import AbstractCallback
from ..datasets.PatchDataset import PatchDataset
from ..datasets.PatchDataset2 import PatchDataset as PDS2

from ..evaluation.visualization_utils import plot_patches

class IntermediatePatchPlot(AbstractCallback):
    """
    Callback to save the model weights at the end of each epoch.
    """
    
    def __init__(self, 
                 name: str, 
                 path: str, 
                 dataset: PatchDataset | PDS2, 
                 plot_n_patches: int=5,
                 plot_metrics: List[nn.Module]=None,
                 **kwargs):
        """
        :param name: Name of the callback.
        :param path: Path to save the model weights.
        """
        super().__init__(name)
        self._path = path
        if not isinstance(dataset, Union[PatchDataset, PDS2]):
            raise TypeError(f"Expected PatchDataset, got {type(dataset)}")
        self._dataset = dataset

        # Additional kwargs passed to plot_patches
        self.plot_n_patches = plot_n_patches
        self.plot_metrics = plot_metrics
        self.plot_kwargs = kwargs

    def on_epoch_end(self, trainer):
        """
        Plot dataset with model predictions at the end of each epoch.
        """

        original_device = next(trainer.model.parameters()).device

        plot_patches(
            _dataset = self._dataset,
            _n_patches = self.plot_n_patches,
            _model = trainer.model,
            _metrics = self.plot_metrics,
            save_path = f"{self._path}/epoch_{trainer.epoch}.png",
            device=original_device,
            **self.plot_kwargs
        )