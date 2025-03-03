import pathlib
from typing import Tuple, List, Union, Optional
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from albumentations import ImageOnlyTransform
from albumentations.core.composition import Compose

from ..datasets.PatchDataset import PatchDataset
from ..evaluation.predict_utils import predict_image, process_tensor_image
from ..evaluation.evaluation_utils import evaluate_per_image_metric

def invert_transforms(
        numpy_img: np.ndarray,
        transforms: Union[ImageOnlyTransform, Compose] = None
    ) -> np.ndarray:

    if isinstance(transforms, ImageOnlyTransform):
        return transforms.invert(numpy_img)
    elif isinstance(transforms, Compose):
        for transform in reversed(transforms.transforms):
            numpy_img = transform.invert(numpy_img)
    elif transforms is None:
        return numpy_img
    else:
        raise ValueError(f"Invalid transforms type: {type(transforms)}")
        
    return numpy_img

def format_img(
        _tensor_img: torch.Tensor,
        cast_to_type: torch.dtype = None
    ) -> np.ndarray:
    
    if cast_to_type is not None:
        _tensor_img = _tensor_img.to(cast_to_type)

    img = torch.squeeze(_tensor_img).cpu().numpy()
        
    return img

def evaluate_and_format_imgs(
        _input: torch.Tensor, 
        _target: torch.Tensor, 
        model=None, 
        _input_transform: Optional[Union[Compose, ImageOnlyTransform]]=None,
        _target_transform: Optional[Union[Compose, ImageOnlyTransform]]=None,
        device: str='cpu'
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    input_transform = invert_transforms(
        format_img(_input),
        _input_transform
    )
    target_transform = invert_transforms(
        format_img(_target),
        _target_transform
    )

    if model is not None:
        model.to(device)
        model.eval()
        with torch.no_grad():
            # Forward Pass
            output = model(_input.unsqueeze(1).to(device))

        output_transform = invert_transforms(
            format_img(output),
            _target_transform
        )
    else:
        output_transform = None

    return input_transform, target_transform, output_transform

def plot_patch(
        _raw_img: np.ndarray,
        _patch_size: int,
        _patch_coords: Tuple[int, int],
        _input: torch.Tensor,
        _target: torch.Tensor,
        _output: torch.Tensor = None,
        axes: List = None,
        **kwargs
):
    ## Plot keyword arguments    
    cmap = kwargs.get("cmap", "gray")
    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)
    figsize = kwargs.get("figsize", None)   
    if figsize is None:
        panel_width = kwargs.get("panel_width", 5)
        figsize = (panel_width, panel_width * 3 if _output is None else 4)
    else:
        panel_width = None 

    if axes is None:
        fig, ax = plt.subplots(1, 3 if _output is None else 4, figsize=figsize)
    else:
        ax = axes

    # plot image
    ax[0].imshow(_raw_img, cmap=cmap)
    ax[0].set_title("Raw Image")
    ax[0].axis("off")

    rect = Rectangle(
        _patch_coords,
        _patch_size,
        _patch_size,
        linewidth=1,
        edgecolor="r",
        facecolor="none"
    )

    if vmin is None:
        vmin = min(_output.min(), _target.min())
    if vmax is None:
        vmax = max(_output.max(), _target.max())
    
    ax[0].add_patch(rect)

    # plot input
    ax[1].imshow(_input, cmap=cmap)
    ax[1].set_title("Input")
    ax[1].axis("off")

    # plot target
    ax[2].imshow(_target, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[2].set_title("Target")
    ax[2].axis("off")

    if _output is not None:
        ax[3].imshow(_output, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[3].set_title("Output")
        ax[3].axis("off")

def plot_patches(
    _dataset: Dataset,
    _n_patches: int=5,
    _model: torch.nn.Module=None,
    _patch_index: List[int]=None,
    _random_seed: int=42,
    _metrics: List[torch.nn.Module]=None,
    device: str='cpu',
    **kwargs
):
    ## Plot keyword arguments    
    cmap = kwargs.get("cmap", "gray")
    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)
    panel_width = kwargs.get("panel_width", 5)
    save_path = kwargs.get("save_path", None)
    show_plot = kwargs.get("show_plot", True)

    ## Generate random patch indices to visualize
    if _patch_index is None:
        random.seed(_random_seed)
        _patch_index = random.sample(range(len(_dataset)), _n_patches)
    else:
        _patch_index = [i for i in _patch_index if i < len(_dataset)]
        _n_patches = len(_patch_index)

    figsize = kwargs.get("figsize", None)
    if figsize is None:
        figsize = (panel_width * _n_patches, panel_width * 3 if _model is None else 4, )
    fig, axes = plt.subplots(_n_patches, 3 if _model is None else 4, figsize=figsize)

    for i, row_axes in zip(_patch_index, axes):
        _input, _target = _dataset[i]
        _raw_image = np.array(Image.open(
            _dataset.input_names[0]
        ))
        _input, _target, _output = evaluate_and_format_imgs(
            _input,
            _target,
            _model,
            device=device
        )
        
        plot_patch(
            _raw_img=_raw_image,
            _patch_size=_input.shape[-1],
            _patch_coords=_dataset.patch_coords,
            _input=_input,
            _target=_target,
            _output=_output,
            axes=row_axes,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )

        ## Compute metrics for single set of (target, output) pairs and add to subplot title
        metric_str = ""
        if _metrics is not None:
            for _metric in _metrics:
                metric_val = _metric(
                    torch.tensor(_output).unsqueeze(0).unsqueeze(0), 
                    torch.tensor(_target).unsqueeze(0).unsqueeze(0)
                    ).item()
                metric_str = f"{metric_str}\n{_metric.__class__.__name__}: {metric_val:.2f}"
        row_axes[-1].set_title(
            row_axes[-1].get_title() + metric_str
        )
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path)

    plt.close()

def _plot_predictions_grid(
    inputs: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    predictions: Union[np.ndarray, torch.Tensor],
    raw_images: Optional[Union[np.ndarray, torch.Tensor]] = None,
    patch_coords: Optional[List[tuple]] = None,
    metrics_df: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
    show: bool = True,
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
    :param show: Whether to display the plot.
    """

    num_samples = len(inputs)
    is_patch_dataset = raw_images is not None
    num_cols = 4 if is_patch_dataset else 3  # (Raw | Input | Target | Prediction) vs (Input | Target | Prediction)

    fig, axes = plt.subplots(num_samples, num_cols, figsize=(5 * num_cols, 5 * num_samples))
    column_titles = ["Raw Image", "Input", "Target", "Prediction"] if is_patch_dataset else ["Input", "Target", "Prediction"]

    for row_idx in range(num_samples):
        img_set = [raw_images[row_idx]] if is_patch_dataset else []
        img_set.extend([inputs[row_idx], targets[row_idx], predictions[row_idx]])

        for col_idx, img in enumerate(img_set):
            ax = axes[row_idx, col_idx]
            ax.imshow(img.squeeze(), cmap="gray")
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
            axes[row_idx, -1].set_title(metric_text, fontsize=10, pad=10)

    # Save and/or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

def plot_predictions_grid_from_eval(
    dataset: torch.utils.data.Dataset,
    predictions: Union[torch.Tensor, np.ndarray],
    indices: List[int],
    metrics_df: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
    show: bool = True,
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
    :param show: Whether to display the plot.
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
        metrics_df, save_path, show
    )

def plot_predictions_grid_from_model(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    indices: List[int],
    metrics: List[torch.nn.Module],
    device: str = "cuda",
    save_path: Optional[str] = None,
    show: bool = True,
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
    :param show: Whether to display the plot.
    """
    # Step 1: Run inference on the selected subset
    predictions, targets = predict_image(dataset, model, indices=indices, device=device)

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
        show=show)