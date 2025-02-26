from typing import List, Union

import torch
import torch.nn as nn

from .AbstractCallback import AbstractCallback
from ..datasets.PatchDataset import PatchDataset

from ..evaluation.visualization_utils import plot_patches

class IntermediatePatchPlot(AbstractCallback):
    """
    Callback to plot model generated outputs alongside ground 
    truth and input at the end end of each epoch.
    """
    
    def __init__(self,                
                 name: str, 
                 path: str, 
                 dataset: PatchDataset, 
                 plot_n_patches: int=5,
                 plot_metrics: List[nn.Module]=None,
                 every_n_epochs: int=5,
                 **kwargs):
        """
        Initialize the IntermediatePlot callback.

        :param name: Name of the callback.
        :type name: str
        :param path: Path to save the model weights.
        :type path: str
        :param dataset: Dataset to be used for plotting intermediate results.
        :type dataset: PatchDataset
        :param plot_n_patches: Number of patches to plot, defaults to 5.
        :type plot_n_patches: int, optional
        :param plot_metrics: List of metrics to compute and display in plot title, defaults to None.
        :type plot_metrics: List[nn.Module], optional
        :param kwargs: Additional keyword arguments to be passed to plot_patches.
        :type kwargs: dict
        :param every_n_epochs: How frequent should intermediate plots should be plotted, defaults to 5
        :type every_n_epochs: int
        :raises TypeError: If the dataset is not an instance of PatchDataset.
        """
        super().__init__(name)
        self._path = path
        if not isinstance(dataset, PatchDataset):
            raise TypeError(f"Expected PatchDataset, got {type(dataset)}")
        self._dataset = dataset

        # Additional kwargs passed to plot_patches
        self.plot_n_patches = plot_n_patches
        self.plot_metrics = plot_metrics
        self.every_n_epochs = every_n_epochs
        self.plot_kwargs = kwargs

    def on_epoch_end(self):
        """
        Called at the end of each epoch to plot predictions if the epoch is a multiple of `every_n_epochs`.
        """
        if (self.trainer.epoch + 1) % self.every_n_epochs == 0 or self.trainer.epoch + 1 == self.trainer.total_epochs:
            self._plot()
    
    def on_train_end(self):
        """
        Called at the end of training. Plots if not already done in the last epoch.
        """
        if (self.trainer.epoch + 1) % self.every_n_epochs != 0:
            self._plot()

    def _plot(self):
        """
        Helper method to generate and save plots.
        Plot dataset with model predictions on n random images from dataset at the end of each epoch.
        Called by the on_epoch_end and on_train_end methods
        """
        
        original_device = next(self.trainer.model.parameters()).device

        plot_patches(
            _dataset = self._dataset,
            _n_patches = self.plot_n_patches,
            _model = self.trainer.model,
            _metrics = self.plot_metrics,
            save_path = f"{self._path}/epoch_{self.trainer.epoch}.png",
            device=original_device,
            **self.plot_kwargs
        )