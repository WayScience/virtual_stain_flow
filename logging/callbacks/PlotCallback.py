from typing import Union, List, Tuple
import pathlib
import random

import torch.nn as nn

from .LoggerCallback import (
    AbstractLoggerCallback,
    log_artifact_type
)
from ...evaluation.visualization_utils import plot_predictions_grid_from_model
from ...datasets.PatchDataset import PatchDataset

"""
Prototype Notes: This is an example implementation of a logger callback that plot predictions of the model on a subset of the dataset.
Mostly similar to the `virtual_stain_flow.callbacks.IntermediatePlotCallback`, except interfaces with the MlflowLoggerV2 to have the 
 plots logged as artifacts
"""
class PlotPredictionCallback(AbstractLoggerCallback):
    def __init__(
        self,
        name: str,
        save_path: Union[pathlib.Path, str],
        dataset: PatchDataset, 
        plot_n_patches: int=5,
        indices: Union[List[int], None]=None,
        plot_metrics: List[nn.Module]=None,
        every_n_epochs: int=5,
        random_seed: int=42,
        tag: str = 'plot_predictions',
        **kwargs
    ):
        """
        Initialize the PlotPredictionCallback callback.
        Allows plots of predictions to be generated during training for monitoring of training progress.
        Supports both PatchDataset and Dataset classes for plotting.

        :param name: Name of the callback.
        :param save_path: Path to save the model weights.
        :param dataset: Dataset to be used for plotting intermediate results.
        :param plot_n_patches: Number of patches to randomly select and plot, defaults to 5.
        The exact patches/images being plotted may vary due to a difference in seed or dataset size. 
        To ensure best reproducibility and consistency, please use a fixed dataset and indices argument instead.
        :param indices: Optional list of specific indices to subset the dataset before inference.
        Overrides the plot_n_patches and random_seed arguments and uses the indices list to subset. 
        :param plot_metrics: List of metrics to compute and display in plot title, defaults to None.
        :param every_n_epochs: How frequent should intermediate plots should be plotted, defaults to 5
        :param random_seed: Random seed for reproducibility for random patch/image selection, defaults to 42.
        :param kwargs: Additional keyword arguments to be passed to plot_patches.
        :raises TypeError: If the dataset is not an instance of PatchDataset.
        """

        super().__init__(name)

        self._path = save_path
        if isinstance(dataset, PatchDataset):
            pass
        else:
            raise TypeError(f"Expected PatchDataset, got {type(dataset)}")
        
        self._dataset = dataset

        # Additional kwargs passed to plot_patches
        self.plot_metrics = plot_metrics if plot_metrics is not None else []
        self.every_n_epochs = every_n_epochs
        self.plot_kwargs = kwargs
        self.tag = tag

        self._epoch = None
        
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
        
        return None
    
    """
    Life cycle methods for the callback
    """

    def on_epoch_start(self) -> Tuple[str, None]:
        """
        Called at the start of each epoch.
        Increments the epoch counter.
        """

        self._epoch = self.get_epoch()
        return ('', None)
    
    def on_epoch_end(self) -> Tuple[str, List[log_artifact_type]]:
        """
        Called at the end of each epoch. 
        Plot predictions if the epoch is a multiple of `every_n_epochs`.
        """

        if (self._epoch + 1) % self.every_n_epochs == 0:
            plot_file_path = self._plot()
            return (self.tag, [plot_file_path])
        
        return (self.tag, None)
    
    def on_train_end(self) -> Tuple[str, List[log_artifact_type]]:
        """
        Called at the end of training. 
        Plots if not already done in the last epoch.
        Ensures at least a single plot is generated at the end of training.
        """

        if (self._epoch + 1) % self.every_n_epochs != 0:
            plot_file_path = self._plot()
            return (self.tag, [plot_file_path])
        
        return (self.tag, None)
    
    """
    Internal helper method(s)
    """
    def _plot(self) -> pathlib.Path:
        """
        Helper method to generate and save plots.
        Plot dataset with model predictions on n random images from dataset at the end of each epoch.
        Called by the on_epoch_end and on_train_end methods

        :return: Path to the saved plot file.
        """

        model = self.get_model(best_model=False) # try to access current model
        if model is None:
            return None
        
        original_device = next(model.parameters()).device

        plot_file_path = f"{self._path}/epoch_{self.get_epoch()}.png"
        plot_predictions_grid_from_model(
            model=model,
            dataset=self._dataset,
            indices=self._dataset_subset_indices,
            metrics=self.plot_metrics,
            save_path=plot_file_path,
            device=original_device,
            **self.plot_kwargs
        )
        plot_file_path = pathlib.Path(plot_file_path)
        if not plot_file_path.exists():
            raise RuntimeError(f"Plot file was not created: {plot_file_path}")

        return plot_file_path        