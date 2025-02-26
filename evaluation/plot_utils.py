import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, List
from PIL import Image
from torch.utils.data import Dataset
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from ..datasets.PatchDataset import PatchDataset
from ..evaluation.predict_utils import predict_image, process_tensor_image
from ..evaluation.evaluation_utils import evaluate_per_image_metric

def plot_single_image(
    image: Union[np.ndarray, torch.Tensor],
    ax: Optional[plt.Axes] = None,
    cmap: str = "gray",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    title_fontsize: int = 10
):
    """
    Plots a single image on the given matplotlib axis or creates a new figure if no axis is provided.

    :param image: The image to plot, either as a NumPy array or a PyTorch tensor.
    :type image: Union[np.ndarray, torch.Tensor]
    :param ax: Optional existing axis to plot on. If None, a new figure is created.
    :type ax: Optional[plt.Axes], default is None
    :param cmap: Colormap for visualization (default: "gray").
    :type cmap: str, optional
    :param vmin: Minimum value for image scaling. If None, defaults to image min.
    :type vmin: Optional[float], optional
    :param vmax: Maximum value for image scaling. If None, defaults to image max.
    :type vmax: Optional[float], optional
    :param title: Optional title for the image.
    :type title: Optional[str], optional
    :param title_fontsize: Font size of the title.
    :type title_fontsize: int, optional
    """

    # Convert tensor to NumPy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    # Handle grayscale vs multi-channel
    if image.ndim == 3 and image.shape[0] in [1, 3]:  # (C, H, W) format
        image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, C)

    # Create a new figure if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Plot the image
    im = ax.imshow(image, cmap=cmap, vmin=vmin or image.min(), vmax=vmax or image.max())

    # Hide axis
    ax.axis("off")

    # Add title if provided
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold")

    return im  # Return image object for further customization if needed

def visualize_images_with_stats(
    images: Union[torch.Tensor, np.ndarray],
    cmap: str = "gray",
    figsize: Optional[tuple] = None,
    panel_width: int = 3,
    show_stats: bool = True,
    channel_names: Optional[list] = None,
    title_fontsize: int = 10,
    axes: Optional[np.ndarray] = None
):
    """
    Visualizes images using matplotlib, handling various shapes:
    - (H, W) → Single grayscale image.
    - (C, H, W) → Multi-channel image.
    - (N, C, H, W) → Multiple images, multiple channels.

    Supports external axes input for easier integration.

    :param images: Input images as PyTorch tensor or NumPy array.
    :param cmap: Colormap for visualization.
    :param figsize: Optional figure size.
    :param panel_width: Width of each panel.
    :param show_stats: Whether to display statistics (μ, σ, ⊥, ⊤) in titles.
    :param channel_names: List of channel names for first row.
    :param title_fontsize: Font size for titles.
    :param axes: Optional pre-existing matplotlib Axes.
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()

    ndim = images.ndim
    if ndim == 2:
        images = images[np.newaxis, np.newaxis, ...]  # Convert to (1, 1, H, W)
    elif ndim == 3:
        images = images[np.newaxis, ...]  # Convert to (1, C, H, W)
    elif ndim != 4:
        raise ValueError(f"Unsupported shape {images.shape}. Expected (H, W), (C, H, W), or (N, C, H, W).")

    n_images, n_channels, _, _ = images.shape

    # Create figure and axes if not provided
    if axes is None:
        figsize = figsize or (n_channels * panel_width, n_images * panel_width)
        fig, axes = plt.subplots(n_images, n_channels, figsize=figsize, squeeze=False)

    for i in range(n_images):
        for j in range(n_channels):
            img = images[i, j]
            title = None

            # Compute statistics if needed
            if show_stats:
                img_mean, img_std, img_min, img_max = np.mean(img), np.std(img), np.min(img), np.max(img)
                title = f"μ: {img_mean:.2f} | σ: {img_std:.2f} | ⊥: {img_min:.2f} | ⊤: {img_max:.2f}"

            # Use the helper function for plotting
            plot_single_image(img, ax=axes[i, j], cmap=cmap, title=title, title_fontsize=title_fontsize)

    plt.tight_layout()
    if axes is None:
        plt.show()

def plot_single_image(
    image: np.ndarray,
    ax: plt.Axes,
    cmap: str = "gray",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    title_fontsize: int = 10
):
    """
    Plots a single image on the given matplotlib axis.

    :param image: The image to plot (NumPy array).
    :param ax: The matplotlib axis to plot on.
    :param cmap: Colormap for visualization.
    :param vmin: Minimum value for scaling.
    :param vmax: Maximum value for scaling.
    :param title: Optional title for the image.
    :param title_fontsize: Font size of the title.
    """
    ax.imshow(np.squeeze(image), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold")

def plot_patches(
    dataset: torch.utils.data.Dataset,
    n_patches: int = 5,
    model: Optional[torch.nn.Module] = None,
    patch_index: Optional[List[int]] = None,
    random_seed: int = 42,
    metrics: Optional[List[torch.nn.Module]] = None,
    device: str = "cpu",
    **kwargs
):
    """
    Plots dataset patches with optional model predictions and evaluation metrics using GridSpec.
    Uses `plot_single_image` to ensure consistency.

    :param dataset: A dataset that returns (input_tensor, target_tensor) tuples.
    :param n_patches: Number of patches to visualize (default: 5).
    :param model: Optional PyTorch model to run inference on patches.
    :param patch_index: List of dataset indices to select specific patches.
    :param random_seed: Random seed for reproducibility.
    :param metrics: List of metric functions to evaluate model predictions.
    :param device: Device to run model inference on, e.g., "cpu" or "cuda".
    :param **kwargs: Additional customization options (e.g., `cmap`, `panel_width`, `show_plot`).
    """

    cmap = kwargs.get("cmap", "gray")
    panel_width = kwargs.get("panel_width", 5)
    show_plot = kwargs.get("show_plot", True)
    save_path = kwargs.get("save_path", None)
    title_fontsize = kwargs.get("title_fontsize", 12)

    # Select patches
    if patch_index is None:
        random.seed(random_seed)
        patch_index = random.sample(range(len(dataset)), n_patches)
    else:
        patch_index = [i for i in patch_index if i < len(dataset)]
        n_patches = len(patch_index)

    inputs, targets, raw_images, patch_coords = [], [], [], []
    for i in patch_index:
        input_tensor, target_tensor = dataset[i]
        inputs.append(input_tensor)
        targets.append(target_tensor)
        patch_coords.append(dataset.patch_coords)  # Extract (x, y) coordinates
        raw_images.append(np.array(Image.open(dataset.input_names[0])))

    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    # Run model predictions (if provided)
    predictions = predict_image(dataset, model, device=device, indices=patch_index) if model else None

    # Convert tensors to NumPy arrays
    inputs_numpy = process_tensor_image(inputs, invert_function=dataset.input_transform.invert)
    targets_numpy = process_tensor_image(targets, dataset=dataset)
    predictions_numpy = process_tensor_image(predictions, dataset=dataset) if predictions is not None else None

    # Compute evaluation metrics (if applicable)
    if metrics and predictions is not None:
        metric_values = evaluate_per_image_metric(
            predictions=predictions,
            targets=targets,
            metrics=metrics
        )
    else:
        metric_values = None

    # Determine number of columns (Raw + Input + Target + Optional Predictions)
    n_predictions = predictions_numpy.shape[1] if predictions_numpy is not None else 0
    n_columns = 3 + n_predictions  # (Raw, Input, Target, Predictions)

    # Compute raw image global vmin/vmax
    raw_vmin, raw_vmax = np.min(raw_images), np.max(raw_images)

    # Set up figure and GridSpec layout with an extra row for column titles
    figsize = (panel_width * n_columns, panel_width * (n_patches + 1))  # Extra space for headers
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_patches + 1, n_columns, figure=fig, height_ratios=[0.05] + [1] * n_patches, hspace=0.05, wspace=0.05)

    # Column headers (Shared Titles)
    column_titles = ["Raw Image", "Input Patch", "Target Patch"] + [f"Predicted {i+1}" for i in range(n_predictions)]
    for j, title in enumerate(column_titles):
        ax = fig.add_subplot(gs[0, j])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=title_fontsize, fontweight="bold")

    # Iterate through patches and plot each column separately
    for i in range(n_patches):
        row_offset = i + 1  # Offset by 1 to account for the title row

        # Extract patch coordinates
        patch_x, patch_y = patch_coords[i]
        patch_size = targets_numpy.shape[-1]  # Infer patch size from target shape

        # Compute per-patch vmin/vmax
        input_vmin, input_vmax = np.min(inputs_numpy[i]), np.max(inputs_numpy[i])
        target_vmin, target_vmax = np.min(targets_numpy[i]), np.max(targets_numpy[i])

        # Plot raw image with patch annotation
        ax = fig.add_subplot(gs[row_offset, 0])
        plot_single_image(raw_images[i], ax, cmap, raw_vmin, raw_vmax)
        rect = Rectangle((patch_x, patch_y), patch_size, patch_size, linewidth=2, edgecolor="r", facecolor="none")
        ax.add_patch(rect)

        # Plot input patch
        ax = fig.add_subplot(gs[row_offset, 1])
        plot_single_image(inputs_numpy[i], ax, cmap, input_vmin, input_vmax)

        # Plot target patch
        ax = fig.add_subplot(gs[row_offset, 2])
        plot_single_image(targets_numpy[i], ax, cmap, target_vmin, target_vmax)

        # Plot prediction patches (if available) with metrics
        if predictions_numpy is not None:
            for j in range(n_predictions):
                ax = fig.add_subplot(gs[row_offset, 3 + j])
                plot_single_image(predictions_numpy[i, j], ax, cmap, target_vmin, target_vmax)

                # Display metric values below prediction
                metric_str = ""
                if metric_values is not None:
                    metric_value_row = metric_values.iloc[i, :]
                    metric_str = "\n".join(
                        [f"{metric_name}: {metric_val:.2f}" for metric_name, metric_val in metric_value_row.items()]
                    )

                ax.set_title(metric_str, fontsize=title_fontsize - 2)

    # Adjust layout and save/show
    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    else:
        plt.close()