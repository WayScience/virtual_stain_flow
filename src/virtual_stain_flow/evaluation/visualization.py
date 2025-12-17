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
from .evaluation_utils import evaluate_per_image_metric, extract_samples_from_dataset
from .predict_utils import predict_image


def plot_predictions_grid(
    inputs: List[np.ndarray],
    targets: List[np.ndarray],
    predictions: Optional[List[np.ndarray]] = None,
    *,
    sample_indices: Optional[List[int]] = None,
    row_label_prefix: str = "",
    raw_images: Optional[List[np.ndarray]] = None,
    patch_coords: Optional[List[Tuple[int, int]]] = None,
    metrics_df: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
    cmap: str = "gray",
    panel_width: float = 4.0,
    show_plot: bool = True,
    wspace: float = 0.05,
    hspace: float = 0.15,
) -> None:
    """
    Plot a grid of images comparing inputs, targets, and optionally predictions.

    Each row represents one sample. Columns are:
    - [Raw Image] (only if raw_images provided) with optional crop bounding box
    - Input
    - Target
    - [Prediction] (only if predictions provided, with optional metrics in title)

    :param inputs: List of input images, each (C, H, W) or (H, W).
    :param targets: List of target images, each (C, H, W) or (H, W).
    :param predictions: Optional list of prediction images, each (C, H, W) or (H, W).
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
    
    # Determine number of columns based on what's provided
    # Base: Input, Target (2 cols) or Input, Target, Prediction (3 cols)
    # With raw: Raw, Input, Target (3 cols) or Raw, Input, Target, Prediction (4 cols)
    num_cols = 2 + (1 if has_raw_images else 0) + (1 if has_predictions else 0)

    # Default sample indices if not provided
    if sample_indices is None:
        sample_indices = list(range(num_samples))

    # Column titles - build dynamically
    column_titles = []
    if has_raw_images:
        column_titles.append("Raw Input")
    column_titles.extend(["Input", "Target"])
    if has_predictions:
        column_titles.append("Prediction")

    # Create figure
    fig_width = panel_width * num_cols
    fig_height = panel_width * num_samples
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(fig_width, fig_height))

    # Handle single-row case where axes is 1D
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for row_idx in range(num_samples):
        # Build image set for this row dynamically
        img_set = []
        if has_raw_images:
            img_set.append(raw_images[row_idx])
        img_set.extend([inputs[row_idx], targets[row_idx]])
        if has_predictions:
            img_set.append(predictions[row_idx])

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
            if has_raw_images and col_idx == 0 and patch_coords is not None:
                patch_x, patch_y = patch_coords[row_idx]
                # Infer patch size from target shape
                target_shape = np.squeeze(targets[row_idx]).shape
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

            # Row label on the Input column (first column if no raw images, second if raw images)
            input_col_idx = 1 if has_raw_images else 0
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

        # Metrics on prediction column (last column, only if predictions provided)
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


def plot_dataset_grid(
    dataset: Union[BaseImageDataset, CropImageDataset],
    indices: List[int],
    save_path: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Plot a grid of dataset samples (inputs and targets) without model predictions.

    Convenience wrapper around `plot_predictions_grid` for visualizing dataset contents.

    :param dataset: BaseImageDataset or CropImageDataset to visualize.
    :param indices: List of dataset indices to display.
    :param save_path: Optional path to save the figure.
    :param kwargs: Additional arguments passed to `plot_predictions_grid`.
        Supported: row_label_prefix, cmap, panel_width, show_plot, wspace, hspace.
    """
    # Extract samples from dataset
    inputs, targets, raw_images, patch_coords = extract_samples_from_dataset(dataset, indices)

    # Plot without predictions
    plot_predictions_grid(
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
    dataset: Union[BaseImageDataset, CropImageDataset],
    indices: List[int],
    metrics: List[torch.nn.Module],
    device: str = "cuda",
    save_path: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Plot predictions grid by running inference on a model.

    Performs the following steps:
    1. Run inference on the specified dataset indices.
    2. Compute per-image metrics.
    3. Extract samples and plot using `plot_predictions_grid`.

    :param model: PyTorch model for inference.
    :param dataset: BaseImageDataset or CropImageDataset to evaluate.
    :param indices: List of dataset indices to evaluate and visualize.
    :param metrics: List of metric modules to compute (can be empty).
    :param device: Device for inference ("cpu" or "cuda").
    :param save_path: Optional path to save the figure.
    :param kwargs: Additional arguments passed to `plot_predictions_grid`.
        Supported: row_label_prefix, cmap, panel_width, show_plot, wspace, hspace.
    """
    # Step 1: Run inference
    targets_tensor, predictions_tensor = predict_image(
        dataset, model, indices=indices, device=device
    )

    # Step 2: Compute metrics (if any)
    metrics_df = None
    if metrics:
        metrics_df = evaluate_per_image_metric(predictions_tensor, targets_tensor, metrics)

    # Step 3: Extract samples from dataset (need to re-access for raw images in CropImageDataset)
    inputs, targets, raw_images, patch_coords = extract_samples_from_dataset(dataset, indices)

    # Convert predictions tensor to list of numpy arrays
    predictions = [predictions_tensor[i].numpy() for i in range(len(indices))]

    # Step 4: Plot
    plot_predictions_grid(
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
