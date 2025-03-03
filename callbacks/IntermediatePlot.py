from typing import List, Union
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .AbstractCallback import AbstractCallback
from ..datasets.PatchDataset import PatchDataset
from ..evaluation.visualization_utils import plot_predictions_grid_from_model

class IntermediatePlot(AbstractCallback):
    """
    Callback to plot model generated outputs, ground 
    truth, and input stained image patches at the end of each epoch.
    """
    
    def __init__(self,                
                 name: str, 
                 path: str, 
                 dataset: Union[Dataset, PatchDataset], 
                 plot_n_patches: int=5,
                 indices: Union[List[int], None]=None,
                 plot_metrics: List[nn.Module]=None,
                 every_n_epochs: int=5,
                 random_seed: int=42,
                 **kwargs):
        """
        Initialize the IntermediatePlot callback.
        Allows plots of predictions to be generated during training for monitoring of training progress.
        Supports both PatchDataset and Dataset classes for plotting.
        This callback, when passed into the trainer, will plot the model predictions on a subset of the provided dataset at the end of each epoch.

        :param name: Name of the callback.
        :type name: str
        :param path: Path to save the model weights.
        :type path: str
        :param dataset: Dataset to be used for plotting intermediate results.
        :type dataset: Union[Dataset, PatchDataset]
        :param plot_n_patches: Number of patches to randomly select and plot, defaults to 5.
        The exact patches/images being plotted may vary due to a difference in seed or dataset size. 
        To ensure best reproducibility and consistency, please use a fixed dataset and indices argument instead.
        :type plot_n_patches: int, optional
        :param indices: Optional list of specific indices to subset the dataset before inference.
        Overrides the plot_n_patches and random_seed arguments and uses the indices list to subset. 
        :type indices: Union[List[int], None]
        :param plot_metrics: List of metrics to compute and display in plot title, defaults to None.
        :type plot_metrics: List[nn.Module], optional
        :param kwargs: Additional keyword arguments to be passed to plot_patches.
        :type kwargs: dict
        :param every_n_epochs: How frequent should intermediate plots should be plotted, defaults to 5
        :type every_n_epochs: int
        :param random_seed: Random seed for reproducibility for random patch/image selection, defaults to 42.
        :type random_seed: int
        :raises TypeError: If the dataset is not an instance of PatchDataset.
        """
        super().__init__(name)
        self._path = path
        if isinstance(dataset, Dataset):
            pass
        if isinstance(dataset, PatchDataset):
            pass
        else:
            raise TypeError(f"Expected PatchDataset, got {type(dataset)}")
        
        self._dataset = dataset

        # Additional kwargs passed to plot_patches
        self.plot_metrics = plot_metrics
        self.every_n_epochs = every_n_epochs
        self.plot_kwargs = kwargs
        
        if indices is not None:
            # Check if indices are within bounds
            for i in indices:
                if i >= len(self._dataset):
                    raise ValueError(f"Index {i} out of bounds for dataset of size {len(self._dataset)}")
            self._dataset_subset_indices = indices
        else:
            # Generate random indices to subset given seed and plot_n_patches
            plot_n_patches = min(plot_n_patches, len(self._dataset))
            random.seed(random_seed)
            self._dataset_subset_indices = random.sample(range(len(self._dataset)), plot_n_patches)

    def on_epoch_end(self):
        """
        Called at the end of each epoch to plot predictions if the epoch is a multiple of `every_n_epochs`.
        """
        if (self.trainer.epoch + 1) % self.every_n_epochs == 0 or self.trainer.epoch + 1 == self.trainer.epoch:
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

        plot_predictions_grid_from_model(
            model=self.trainer.model,
            dataset=self._dataset,
            indices=self._dataset_subset_indices,
            metrics=self.plot_metrics,
            save_path=f"{self._path}/epoch_{self.trainer.epoch}.png",
            device=original_device,
            show=False,
            **self.plot_kwargs
        )