from typing import List, Union, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ..datasets.PatchDataset import PatchDataset
from ..evaluation.predict_utils import predict_image, process_tensor_image
from ..evaluation.evaluation_utils import evaluate_per_image_metric

def _plot_predictions_grid(
    inputs: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    predictions: Union[np.ndarray, torch.Tensor],
    raw_images: Optional[Union[np.ndarray, torch.Tensor]] = None,
    patch_coords: Optional[List[tuple]] = None,
    metrics_df: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
    **kwargs
):
    """
    Generalized function to plot a grid of images with predictions and optional raw images.
    The Batch dimensions of (raw_image), input, target, and prediction should match and so should the length of metrics_df.

    :param inputs: Input images (N, C, H, W) or (N, H, W).
    :param targets: Target images (N, C, H, W) or (N, H, W).
    :param predictions: Model predictions (N, C, H, W) or (N, H, W).
    :param raw_images: Optional raw images for PatchDataset (N, H, W).
    :param patch_coords: Optional list of (x, y) coordinates for patches. 
        Only used if raw_images is provided. Length match the first dimension of inputs/targets/predictions.
    :param metrics_df: Optional DataFrame with per-image metrics.
    :param save_path: If provided, saves figure.
    :param kwargs: Additional keyword arguments to pass to plt.subplots.
    """

    cmap = kwargs.get("cmap", "gray")
    panel_width = kwargs.get("panel_width", 5)
    show_plot = kwargs.get("show_plot", True)
    fig_size = kwargs.get("fig_size", None)

    num_samples = len(inputs)
    is_patch_dataset = raw_images is not None
    num_cols = 4 if is_patch_dataset else 3  # (Raw | Input | Target | Prediction) vs (Input | Target | Prediction)

    fig_size = (panel_width * num_cols, panel_width * num_samples) if fig_size is None else fig_size
    fig, axes = plt.subplots(num_samples, num_cols, figsize=fig_size)
    column_titles = ["Raw Image", "Input", "Target", "Prediction"] if is_patch_dataset else ["Input", "Target", "Prediction"]

    for row_idx in range(num_samples):
        img_set = [raw_images[row_idx]] if is_patch_dataset else []
        img_set.extend([inputs[row_idx], targets[row_idx], predictions[row_idx]])

        for col_idx, img in enumerate(img_set):
            ax = axes[row_idx, col_idx]
            ax.imshow(img.squeeze(), cmap=cmap)
            ax.set_title(column_titles[col_idx])
            ax.axis("off")

            # Draw rectangle on raw image if PatchDataset
            if is_patch_dataset and col_idx == 0 and patch_coords is not None:
                patch_x, patch_y = patch_coords[row_idx]  # (x, y) coordinates
                patch_size = targets.shape[-1]  # Assume square patches from target size
                rect = Rectangle((patch_x, patch_y), patch_size, patch_size, linewidth=2, edgecolor="r", facecolor="none")
                ax.add_patch(rect)

        # Display metrics if provided
        if metrics_df is not None:
            metric_values = metrics_df.iloc[row_idx]
            metric_text = "\n".join([f"{key}: {value:.3f}" for key, value in metric_values.items()])
            axes[row_idx, -1].set_title(
                axes[row_idx, -1].get_title() + "\n" + metric_text, fontsize=10, pad=10)

    # Save and/or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_predictions_grid_from_eval(
    dataset: Dataset,
    predictions: Union[torch.Tensor, np.ndarray],
    indices: List[int],
    metrics_df: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
    **kwargs
):
    """
    Wrapper function to extract dataset samples and call `_plot_predictions_grid`.
    This function operates on the outputs downstream of `evaluate_per_image_metric` 
    and `predict_image` to avoid unecessary forward pass.

    :param dataset: Dataset (either normal or PatchDataset).
    :param predictions: Subsetted tensor/NumPy array of predictions.
    :param indices: Indices corresponding to the subset.
    :param metrics_df: DataFrame with per-image metrics for the subset.
    :param save_path: If provided, saves figure.
    :param kwargs: Additional keyword arguments to pass to `_plot_predictions_grid`.
    """

    is_patch_dataset = isinstance(dataset, PatchDataset)

    # Extract input, target, and (optional) raw images & patch coordinates
    raw_images, inputs, targets, patch_coords = [], [], [], []
    for i in indices:
        inputs.append(dataset[i][0])
        targets.append(dataset[i][1])
        if is_patch_dataset:
            raw_images.append(dataset.raw_input)
            patch_coords.append(dataset.patch_coords)  # Get patch location

    inputs_numpy = process_tensor_image(torch.stack(inputs), invert_function=dataset.input_transform.invert)
    targets_numpy = process_tensor_image(torch.stack(targets), invert_function=dataset.target_transform.invert)

    # Pass everything to the core grid function
    _plot_predictions_grid(
        inputs_numpy, targets_numpy, predictions[indices], 
        raw_images if is_patch_dataset else None,
        patch_coords if is_patch_dataset else None,
        metrics_df, save_path, **kwargs
    )

def plot_predictions_grid_from_model(
    model: torch.nn.Module,
    dataset: Dataset,
    indices: List[int],
    metrics: List[torch.nn.Module],
    device: str = "cuda",
    save_path: Optional[str] = None,
    **kwargs
):
    """
    Wrapper plot function that internally performs inference and evaluation with the following steps:
    1. Perform inference on a subset of the dataset given the model.
    2. Compute per-image metrics on that subset.
    3. Plot the results with core `_plot_predictions_grid` function.

    :param model: PyTorch model for inference.
    :param dataset: The dataset to use for evaluation and plotting.
    :param indices: List of dataset indices to evaluate and visualize.
    :param metrics: List of metric functions to evaluate.
    :param device: Device to run inference on ("cpu" or "cuda").
    :param save_path: Optional path to save the plot.
    :param kwargs: Additional keyword arguments to pass to `_plot_predictions_grid`.
    """
    # Step 1: Run inference on the selected subset
    targets, predictions = predict_image(dataset, model, indices=indices, device=device)

    # Step 2: Compute per-image metrics for the subset
    metrics_df = evaluate_per_image_metric(predictions, targets, metrics)

    # Step 3: Extract subset of inputs & targets and plot
    is_patch_dataset = isinstance(dataset, PatchDataset)
    raw_images, inputs, targets, patch_coords = [], [], [], []
    for i in indices:
        inputs.append(dataset[i][0])
        targets.append(dataset[i][1])
        if is_patch_dataset:
            raw_images.append(dataset.raw_input)
            patch_coords.append(dataset.patch_coords)  # Get patch location

    _plot_predictions_grid(
        torch.stack(inputs), 
        torch.stack(targets), 
        predictions, 
        raw_images=raw_images if is_patch_dataset else None,
        patch_coords=patch_coords if is_patch_dataset else None,
        metrics_df=metrics_df, 
        save_path=save_path,
        **kwargs)
