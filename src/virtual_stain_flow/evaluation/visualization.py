"""
visualization.py

Core visualization tools for plotting model predictions alongside inputs and targets.
Supports both BaseImageDataset and CropImageDataset with optional metrics display.
"""

from typing import List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ..datasets.base_dataset import BaseImageDataset
from ..datasets.crop_dataset import CropImageDataset
from ..datasets.base_wrapper_dataset import BaseWrapperDataset
from .evaluation_utils import evaluate_per_image_metric, extract_samples_from_dataset
from .predict_utils import predict_image


def _to_numpy_image(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_to_list(sample: Any) -> List[np.ndarray]:
    if isinstance(sample, (list, tuple)):
        return [_to_numpy_image(item) for item in sample]
    return [_to_numpy_image(sample)]


def _split_channels(img: np.ndarray, channel_indices: Optional[List[int]]) -> List[np.ndarray]:
    if img.ndim == 2:
        if channel_indices is not None and any(idx != 0 for idx in channel_indices):
            raise ValueError("Channel indices out of range for 2D image.")
        return [img]

    if img.ndim == 3:
        total_channels = img.shape[0]
        indices = channel_indices if channel_indices is not None else list(range(total_channels))
        for idx in indices:
            if idx < 0 or idx >= total_channels:
                raise ValueError(
                    f"Channel index {idx} out of range for image with {total_channels} channels."
                )
        return [img[idx] for idx in indices]

    raise ValueError(f"Unsupported image shape for visualization: {img.shape}")


def _split_sample_channels(sample: Any, channel_indices: Optional[List[int]]) -> List[np.ndarray]:
    split_images: List[np.ndarray] = []
    for item in _normalize_to_list(sample):
        split_images.extend(_split_channels(item, channel_indices))
    return split_images


def _build_titles(prefix: str, count: int) -> List[str]:
    return [f"{prefix} {i + 1}" for i in range(count)]


def plot_predictions_grid(
    inputs: List[Union[np.ndarray, List[np.ndarray]]],
    targets: List[Union[np.ndarray, List[np.ndarray]]],
    predictions: Optional[List[Union[np.ndarray, List[np.ndarray]]]] = None,
    *,
    sample_indices: Optional[List[int]] = None,
    row_label_prefix: str = "",
    raw_images: Optional[List[Union[np.ndarray, List[np.ndarray]]]] = None,
    patch_coords: Optional[List[Tuple[int, int]]] = None,
    metrics_df: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
    cmap: str = "gray",
    panel_width: float = 4.0,
    show_plot: bool = True,
    wspace: float = 0.05,
    hspace: float = 0.15,
    raw_channel_indices: Optional[List[int]] = None,
    input_channel_indices: Optional[List[int]] = None,
    target_channel_indices: Optional[List[int]] = None,
    prediction_channel_indices: Optional[List[int]] = None,
) -> plt.Figure:
    """
    Plot a grid of images comparing inputs, targets, and optionally predictions.

    Each row represents one sample. Columns are:
    - [Raw Image] (only if raw_images provided) with optional crop bounding box
    - Input
    - Target
    - [Prediction] (only if predictions provided, with optional metrics in title)

    :param inputs: List of input images, each (C, H, W) or (H, W).
        Multi-input samples can be provided as a list of arrays per sample.
    :param targets: List of target images, each (C, H, W) or (H, W).
        Multi-target samples can be provided as a list of arrays per sample.
    :param predictions: Optional list of prediction images, each (C, H, W) or (H, W).
        Multi-output samples can be provided as a list of arrays per sample.
        If None, only inputs and targets are displayed.
    :param sample_indices: Optional list of indices to display as row labels.
        If None, uses 0-based sequential indices.
    :param row_label_prefix: Prefix for row labels (default: "").
        E.g., "Sample " would give labels "Sample 0", "Sample 1", etc.
    :param raw_images: Optional list of original uncropped images for CropImageDataset.
    :param patch_coords: Optional list of (x, y) crop coordinates for bounding boxes.
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
    :param raw_channel_indices: Optional list of channel indices to display for raw images.
    :param input_channel_indices: Optional list of channel indices to display for inputs.
    :param target_channel_indices: Optional list of channel indices to display for targets.
    :param prediction_channel_indices: Optional list of channel indices to display for predictions.
    """
    num_samples = len(inputs)
    if num_samples == 0:
        raise ValueError("No samples provided to plot.")

    # Validate lengths match
    if len(targets) != num_samples:
        raise ValueError(
            f"Length mismatch: inputs ({num_samples}), targets ({len(targets)})"
        )
    
    has_predictions = predictions is not None and len(predictions) > 0
    if has_predictions and len(predictions) != num_samples:
        raise ValueError(
            f"Length mismatch: inputs ({num_samples}), predictions ({len(predictions)})"
        )

    has_raw_images = raw_images is not None and len(raw_images) > 0
    if has_raw_images and len(raw_images) != num_samples:
        raise ValueError(
            f"Length mismatch: inputs ({num_samples}), raw_images ({len(raw_images)})"
        )

    if sample_indices is not None and len(sample_indices) != num_samples:
        raise ValueError(
            f"Length mismatch: inputs ({num_samples}), sample_indices ({len(sample_indices)})"
        )

    if has_predictions and prediction_channel_indices is None:
        prediction_channel_indices = target_channel_indices
    
    # Determine columns based on channel splits in the first sample
    raw_first = _split_sample_channels(raw_images[0], raw_channel_indices) if has_raw_images else []
    input_first = _split_sample_channels(inputs[0], input_channel_indices)
    target_first = _split_sample_channels(targets[0], target_channel_indices)
    pred_first = (
        _split_sample_channels(predictions[0], prediction_channel_indices)
        if has_predictions
        else []
    )

    if has_predictions and len(pred_first) != len(target_first):
        raise ValueError(
            "Target and prediction channel counts must match for paired display."
        )

    raw_titles = _build_titles("Raw Input", len(raw_first))
    input_titles = _build_titles("Input", len(input_first))
    target_titles = _build_titles("Target", len(target_first))
    pred_titles = _build_titles("Prediction", len(pred_first))

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
        raw_row = _split_sample_channels(raw_images[row_idx], raw_channel_indices) if has_raw_images else []
        input_row = _split_sample_channels(inputs[row_idx], input_channel_indices)
        target_row = _split_sample_channels(targets[row_idx], target_channel_indices)
        pred_row = (
            _split_sample_channels(predictions[row_idx], prediction_channel_indices)
            if has_predictions
            else []
        )

        if has_predictions and len(pred_row) != len(target_row):
            raise ValueError(
                f"Row {row_idx} has mismatched target/prediction channel counts."
            )

        if has_predictions:
            target_pred_row = [
                img
                for i in range(len(target_row))
                for img in (target_row[i], pred_row[i])
            ]
        else:
            target_pred_row = target_row

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
                patch_x, patch_y = patch_coords[row_idx]
                # Infer patch size from target shape
                target_shape = np.squeeze(target_row[0]).shape
                patch_h, patch_w = target_shape[-2], target_shape[-1]
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
        raw_channel_indices, input_channel_indices, target_channel_indices, prediction_channel_indices.
    """
    # Extract samples from dataset
    (
        inputs, targets, raw_images, patch_coords
    ) = extract_samples_from_dataset(dataset, indices)

    # Plot without predictions
    return plot_predictions_grid(
        inputs=inputs,
        targets=targets,
        predictions=None,
        sample_indices=indices,
        raw_images=raw_images,
        patch_coords=patch_coords,
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
        raw_channel_indices, input_channel_indices, target_channel_indices, prediction_channel_indices.
    """
    # Step 1: Run inference
    targets_tensor, predictions_tensor, inputs_tensor = predict_image(
        dataset, model, indices=indices, device=device
    )

    # Step 2: Compute metrics (if any)
    metrics_df = None
    if metrics:
        metrics_df = evaluate_per_image_metric(predictions_tensor, targets_tensor, metrics)

    # Step 3: Extract samples from dataset (need to re-access for raw images in CropImageDataset)
    _, _, raw_images, patch_coords = extract_samples_from_dataset(dataset, indices)    
    # use the collected input and target stack at prediction time instead of
    # re-extract
    if isinstance(inputs_tensor, list):
        inputs = [
            [inputs_tensor[i][row_idx].numpy() for i in range(len(inputs_tensor))]
            for row_idx in range(len(indices))
        ]
    else:
        inputs = inputs_tensor.numpy()

    targets = targets_tensor.numpy()

    # Convert predictions tensor to list of numpy arrays
    predictions = [predictions_tensor[i].numpy() for i in range(len(indices))]

    # Step 4: Plot
    return plot_predictions_grid(
        inputs=inputs,
        targets=targets,
        predictions=predictions,
        sample_indices=indices,
        raw_images=raw_images,
        patch_coords=patch_coords,
        metrics_df=metrics_df,
        save_path=save_path,
        **kwargs,
    )
