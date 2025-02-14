import pathlib
from typing import Tuple, List
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from albumentations import ImageOnlyTransform
from albumentations.core.composition import Compose

def invert_transforms(
        numpy_img: np.ndarray,
        transforms: ImageOnlyTransform | Compose = None
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
        _input_transform: ImageOnlyTransform | Compose=None,
        _target_transform: ImageOnlyTransform | Compose=None,
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

        ## Compute metrics for single set of target output pairs and add to subplot title
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