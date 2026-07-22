"""
visualization.py

Core visualization tools for plotting model predictions alongside inputs and targets.
Supports both BaseImageDataset and CropImageDataset with optional metrics display.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ..datasets.base_dataset import BaseImageDataset
from ..datasets.crop_dataset import CropImageDataset
from ..datasets.base_wrapper_dataset import BaseWrapperDataset
from .evaluation_utils import evaluate_per_image_metric
from .predict_utils import predict_image
from .visualization_utils import extract_samples_from_dataset


def _select_channels(
        images: np.ndarray, channel_indices: Optional[List[int]], name: str
    ) -> Tuple[np.ndarray, List[int]]:
    """
    Helper for selecting channel by index working with (N, C, H, W) image stacks.

    :param images: Image stack with shape (N, C, H, W).
    :param channel_indices: Optional list of channel indices to select.
        If None, all channels are selected.
    :param name: Name of the image stack for error messages.
    :return: Image stack with selected channels and their original channel indices.
    """

    if images.ndim != 4:
        raise ValueError(f"{name} must have shape (N, C, H, W), received {images.shape}.")

    indices = channel_indices if channel_indices is not None else list(range(images.shape[1]))
    if not indices:
        raise ValueError(f"{name} channel indices cannot be empty.")
    if any(index < 0 or index >= images.shape[1] for index in indices):
        raise ValueError(
            f"Channel indices out of range for {name} with {images.shape[1]} channels."
        )

    return images[:, indices, :, :], indices


def _build_titles(
    prefix: str,
    channel_indices: List[int],
    channel_names: Optional[List[str]],
) -> List[str]:
    """
    Build display titles with optional dataset channel names.
    """
    return [
        f"{channel_names[channel_index]} ({prefix} {channel_index + 1})"
        if channel_names is not None and channel_index < len(channel_names)
        else f"{prefix} {position + 1}"
        for position, channel_index in enumerate(channel_indices)
    ]


def plot_predictions_grid(
    inputs: np.ndarray,
    targets: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    *,
    sample_indices: Optional[List[int]] = None,
    row_label_prefix: str = "",
    raw_images: Optional[np.ndarray] = None,
    patch_coords: Optional[List[Tuple[int, int, int, int]]] = None,
    metrics_df: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
    cmap: str = "gray",
    panel_width: float = 4.0,
    show_plot: bool = True,
    wspace: float = 0.05,
    hspace: float = 0.15,
    raw_channel_names: Optional[List[str]] = None,
    input_channel_names: Optional[List[str]] = None,
    target_channel_names: Optional[List[str]] = None,
    prediction_channel_names: Optional[List[str]] = None,
    raw_channel_indices: Optional[List[int]] = None,
    input_channel_indices: Optional[List[int]] = None,
    target_channel_indices: Optional[List[int]] = None,
    prediction_channel_indices: Optional[List[int]] = None,
) -> plt.Figure:
    """
    Core visualization function for visualizing grid of input/target and predictions. 
    Plot a grid of images comparing inputs, targets, and optionally predictions.

    Each row represents one sample. Columns are:
    - [Raw Image] (only if raw_images provided) with optional crop bounding box
    - Input
    - Target
    - [Prediction] (only if predictions provided, with optional metrics in title)

    :param inputs: Input images with shape (N, C, H, W).
    :param targets: Target images with shape (N, C, H, W).
    :param predictions: Optional prediction images with shape (N, C, H, W).
        If None, only inputs and targets are displayed.
    :param sample_indices: Optional list of indices to display as row labels.
        If None, uses 0-based sequential indices.
    :param row_label_prefix: Prefix for row labels (default: "").
        E.g., "Sample " would give labels "Sample 0", "Sample 1", etc.
    :param raw_images: Optional original uncropped images with shape (N, C, H, W).
    :param patch_coords: Optional list of (x, y, width, height) crop rectangles.
        Only used when raw_images is provided.
    :param metrics_df: Optional DataFrame with per-image metrics to display.
        Each row corresponds to a sample; column names become metric labels.
        Only used when predictions is provided.
    :param save_path: If provided, saves the figure to this path.
    :param cmap: Matplotlib colormap for image display (default: "gray").
    :param panel_width: Width of each panel in inches (default: 4.0).
    :param show_plot: Whether to display the plot (default: True).
    :param wspace: Horizontal spacing between subplots (default: 0.05).
    :param hspace: Vertical spacing between subplots (default: 0.15).
    :param raw_channel_names: Optional names for raw image channels.
    :param input_channel_names: Optional names for input image channels.
    :param target_channel_names: Optional names for target image channels.
    :param prediction_channel_names: Optional names for prediction image channels.
    :param raw_channel_indices: Optional list of channel indices to display for raw images.
    :param input_channel_indices: Optional list of channel indices to display for inputs.
    :param target_channel_indices: Optional list of channel indices to display for targets.
    :param prediction_channel_indices: Optional list of channel indices to display for predictions.
    """
    if inputs.ndim != 4:
        raise ValueError(f"Inputs must have shape (N, C, H, W), received {inputs.shape}.")

    num_samples = inputs.shape[0]
    if num_samples == 0:
        raise ValueError("No samples provided to plot.")

    if targets.ndim != 4 or targets.shape[0] != num_samples:
        raise ValueError(
            f"Inputs and targets must have matching batch sizes; received "
            f"{inputs.shape} and {targets.shape}."
        )

    has_predictions = predictions is not None
    if has_predictions and (predictions.ndim != 4 or predictions.shape[0] != num_samples):
        raise ValueError(
            f"Inputs and predictions must have matching batch sizes; received "
            f"{inputs.shape} and {predictions.shape}."
        )

    has_raw_images = raw_images is not None
    if has_raw_images and (raw_images.ndim != 4 or raw_images.shape[0] != num_samples):
        raise ValueError(
            f"Inputs and raw images must have matching batch sizes; received "
            f"{inputs.shape} and {raw_images.shape}."
        )

    if has_raw_images and patch_coords is not None and len(patch_coords) != num_samples:
        raise ValueError(
            f"Length mismatch: inputs ({num_samples}), patch_coords ({len(patch_coords)})"
        )

    if sample_indices is not None and len(sample_indices) != num_samples:
        raise ValueError(
            f"Length mismatch: inputs ({num_samples}), sample_indices ({len(sample_indices)})"
        )

    if has_predictions and prediction_channel_indices is None:
        prediction_channel_indices = target_channel_indices
    if has_predictions and prediction_channel_names is None:
        prediction_channel_names = target_channel_names

    raw_images, raw_indices = (
        _select_channels(raw_images, raw_channel_indices, "Raw images")
        if has_raw_images
        else (None, [])
    )
    inputs, input_indices = _select_channels(inputs, input_channel_indices, "Inputs")
    targets, target_indices = _select_channels(targets, target_channel_indices, "Targets")
    predictions, prediction_indices = (
        _select_channels(predictions, prediction_channel_indices, "Predictions")
        if has_predictions
        else (None, [])
    )

    if has_predictions and predictions.shape[1] != targets.shape[1]:
        raise ValueError(
            "Target and prediction channel counts must match for paired display."
        )

    raw_titles = _build_titles("Raw Input", raw_indices, raw_channel_names)
    input_titles = _build_titles("Input", input_indices, input_channel_names)
    target_titles = _build_titles("Target", target_indices, target_channel_names)
    pred_titles = _build_titles("Prediction", prediction_indices, prediction_channel_names)

    if has_predictions:
        interleaved_titles = [
            title
            for i in range(len(target_titles))
            for title in (target_titles[i], pred_titles[i])
        ]
    else:
        interleaved_titles = target_titles

    column_titles = raw_titles + input_titles + interleaved_titles
    num_cols = len(column_titles)

    # Default sample indices if not provided
    if sample_indices is None:
        sample_indices = list(range(num_samples))

    # Create figure
    fig_width = panel_width * num_cols
    fig_height = panel_width * num_samples
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(fig_width, fig_height))

    # Handle single-row case where axes is 1D
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for row_idx in range(num_samples):
        raw_row = list(raw_images[row_idx]) if has_raw_images else []
        input_row = list(inputs[row_idx])
        target_row = list(targets[row_idx])
        pred_row = list(predictions[row_idx]) if has_predictions else []

        target_pred_row = [
            image
            for target_image, prediction_image in zip(target_row, pred_row)
            for image in (target_image, prediction_image)
        ] if has_predictions else target_row

        img_set = raw_row + input_row + target_pred_row

        if len(img_set) != num_cols:
            raise ValueError(
                f"Row {row_idx} has {len(img_set)} columns, expected {num_cols}."
            )

        for col_idx, img in enumerate(img_set):
            ax = axes[row_idx, col_idx]

            # Squeeze to 2D for display (handles (1, H, W) or (H, W))
            img_2d = np.squeeze(img)
            ax.imshow(img_2d, cmap=cmap)

            # Column title only on first row
            if row_idx == 0:
                ax.set_title(column_titles[col_idx], fontsize=12)

            ax.axis("off")

            # Draw bounding box on raw image
            if has_raw_images and col_idx < len(raw_titles) and patch_coords is not None:
                patch_x, patch_y, patch_w, patch_h = patch_coords[row_idx]
                rect = Rectangle(
                    (patch_x, patch_y),
                    patch_w,
                    patch_h,
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)

            # Row label on the first input column
            input_col_idx = len(raw_titles)
            if col_idx == input_col_idx:
                row_label = f"{row_label_prefix}{sample_indices[row_idx]}"
                # Determine text color based on top-left corner brightness
                # Sample a small region (e.g., top-left 10% of image)
                sample_size = max(1, int(min(img_2d.shape) * 0.1))
                corner_region = img_2d[:sample_size, :sample_size]
                # Normalize to 0-1 range for brightness check
                img_min, img_max = img_2d.min(), img_2d.max()
                if img_max > img_min:
                    normalized_brightness = (corner_region.mean() - img_min) / (img_max - img_min)
                else:
                    normalized_brightness = 0.5
                text_color = "black" if normalized_brightness > 0.5 else "white"
                ax.text(
                    0.02, 0.98,  # Top-left corner in axes coordinates
                    row_label,
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight="bold",
                    color=text_color,
                    verticalalignment="top",
                    horizontalalignment="left",
                )

        # Metrics on last prediction column (only if predictions provided)
        if has_predictions and metrics_df is not None and row_idx < len(metrics_df):
            metric_values = metrics_df.iloc[row_idx]
            metric_text = "\n".join(
                [f"{key}: {value:.3f}" for key, value in metric_values.items()]
            )
            current_title = axes[row_idx, -1].get_title() if row_idx == 0 else ""
            if current_title:
                axes[row_idx, -1].set_title(f"{current_title}\n{metric_text}", fontsize=10)
            else:
                axes[row_idx, -1].set_title(metric_text, fontsize=10)

    # Adjust layout
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    # Save and/or show
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig


def plot_dataset_grid(
    dataset: Union[BaseImageDataset, CropImageDataset, BaseWrapperDataset],
    indices: List[int],
    save_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot a grid of dataset samples (inputs and targets) without model predictions.

    Convenience wrapper around `plot_predictions_grid` for visualizing dataset contents.

    :param dataset: BaseImageDataset, CropImageDataset, or BaseWrapperDataset to visualize.
        Note that if using a wrapper dataset, the raw images and patch coordinates
        if applicable will be based on the wrapped __get_item__ and metadata.
    :param indices: List of dataset indices to display.
    :param save_path: Optional path to save the figure.
    :param kwargs: Additional arguments passed to `plot_predictions_grid`.
        Supported: row_label_prefix, cmap, panel_width, show_plot, wspace, hspace,
        raw_channel_indices, input_channel_indices, target_channel_indices, prediction_channel_indices,
        raw_channel_names, input_channel_names, target_channel_names, prediction_channel_names.
    """
    # Extract samples from dataset
    (
        inputs, targets, raw_images, patch_coords, input_channel_names, target_channel_names
    ) = extract_samples_from_dataset(dataset, indices)

    # Plot without predictions
    return plot_predictions_grid(
        inputs=inputs,
        targets=targets,
        predictions=None,
        sample_indices=indices,
        raw_images=raw_images,
        patch_coords=patch_coords,
        raw_channel_names=input_channel_names,
        input_channel_names=input_channel_names,
        target_channel_names=target_channel_names,
        metrics_df=None,
        save_path=save_path,
        **kwargs,
    )


def plot_predictions_grid_from_model(
    model: torch.nn.Module,
    dataset: Union[BaseImageDataset, CropImageDataset, BaseWrapperDataset],
    indices: List[int],
    metrics: List[torch.nn.Module],
    device: str = "cuda",
    save_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot predictions grid by running inference on a model.

    Performs the following steps:
    1. Run inference on the specified dataset indices.
    2. Compute per-image metrics.
    3. Extract samples and plot using `plot_predictions_grid`.

    :param model: PyTorch model for inference.
    :param dataset: BaseImageDataset, CropImageDataset, or BaseWrapperDataset to visualize.
        Note that if using a wrapper dataset, the predictions will be based on the
        wrapper __get_item__ input whereas the raw images and patch coordinates
        if applicable will be based on the wrapped __get_item__ and metadata.
    :param indices: List of dataset indices to evaluate and visualize.
    :param metrics: List of metric modules to compute (can be empty).
    :param device: Device for inference ("cpu" or "cuda").
    :param save_path: Optional path to save the figure.
    :param kwargs: Additional arguments passed to `plot_predictions_grid`.
        Supported: row_label_prefix, cmap, panel_width, show_plot, wspace, hspace,
        raw_channel_indices, input_channel_indices, target_channel_indices, prediction_channel_indices,
        raw_channel_names, input_channel_names, target_channel_names, prediction_channel_names.
    """
    # Step 1: Run inference
    targets_tensor, predictions_tensor, inputs_tensor = predict_image(
        dataset, model, indices=indices, device=device
    )

    # Step 2: Compute metrics (if any)
    metrics_df = None
    if metrics:
        metrics_df = evaluate_per_image_metric(predictions_tensor, targets_tensor, metrics)

    # Step 3: Re-access the dataset for CropImageDataset raw images and crop metadata.
    (
        _, _, raw_images, patch_coords, input_channel_names, target_channel_names
    ) = extract_samples_from_dataset(dataset, indices)
    if isinstance(inputs_tensor, list):
        raise ValueError(
            "Visualization requires a single batched input tensor with shape (N, C, H, W); "
            "multi-input sequences are not supported."
        )

    inputs = inputs_tensor.detach().cpu().numpy()
    targets = targets_tensor.detach().cpu().numpy()
    predictions = predictions_tensor.detach().cpu().numpy()

    # Step 4: Plot
    return plot_predictions_grid(
        inputs=inputs,
        targets=targets,
        predictions=predictions,
        sample_indices=indices,
        raw_images=raw_images,
        patch_coords=patch_coords,
        raw_channel_names=input_channel_names,
        input_channel_names=input_channel_names,
        target_channel_names=target_channel_names,
        prediction_channel_names=target_channel_names,
        metrics_df=metrics_df,
        save_path=save_path,
        **kwargs,
    )
