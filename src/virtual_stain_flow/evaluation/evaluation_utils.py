from typing import List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
from torch.nn import Module

from ..datasets.base_dataset import BaseImageDataset
from ..datasets.crop_dataset import CropImageDataset
from virtual_stain_flow.datasets.base_wrapper_dataset import BaseWrapperDataset


def _to_numpy_image(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_to_list(sample: Any) -> List[np.ndarray]:
    if isinstance(sample, (list, tuple)):
        return [_to_numpy_image(item) for item in sample]
    return [_to_numpy_image(sample)]


def extract_samples_from_dataset(
    dataset: Union[BaseImageDataset, CropImageDataset, BaseWrapperDataset],
    indices: List[int],
) -> Tuple[
    List[Union[np.ndarray, List[np.ndarray]]],
    List[Union[np.ndarray, List[np.ndarray]]],
    Optional[List[Union[np.ndarray, List[np.ndarray]]]],
    Optional[List[Tuple[int, int]]],
]:
    """
    Extract input/target samples and optional raw images with patch coordinates from a dataset.

    For CropImageDataset, also extracts the original (uncropped) input images and the
    (x, y) coordinates of each crop for visualization with bounding boxes.

    :param dataset: A BaseImageDataset or CropImageDataset instance.
        :param indices: List of dataset indices to extract.
        :return: Tuple of (inputs, targets, raw_images, patch_coords).
                - inputs: List of numpy arrays, each with shape (C, H, W) or (H, W).
                    Multi-input samples can be provided as a list of arrays per sample.
                - targets: List of numpy arrays, each with shape (C, H, W) or (H, W).
                    Multi-target samples can be provided as a list of arrays per sample.
                - raw_images: List of numpy arrays for CropImageDataset (original uncropped images),
                    or None for BaseImageDataset.
                - patch_coords: List of (x, y) tuples for CropImageDataset, or None for BaseImageDataset.
    """
    is_wrapper_dataset = False
    if isinstance(dataset, BaseWrapperDataset):
        is_crop_dataset = isinstance(dataset.original, CropImageDataset)
        is_wrapper_dataset = True
    elif isinstance(dataset, CropImageDataset):
        is_crop_dataset = True
    elif isinstance(dataset, BaseImageDataset):
        is_crop_dataset = False
    else:
        raise ValueError(
            "Unsupported dataset type. Expected BaseImageDataset, CropImageDataset, or BaseWrapperDataset.")
    
    if not indices:
        raise ValueError("Indices list cannot be empty.")
    
    if max(indices) >= len(dataset):
        raise IndexError(
            f"Index out of range. Dataset length: {len(dataset)}, "
            f"max index requested: {max(indices)}"
        )

    inputs: List[Union[np.ndarray, List[np.ndarray]]] = []
    targets: List[Union[np.ndarray, List[np.ndarray]]] = []
    raw_images: Optional[List[Union[np.ndarray, List[np.ndarray]]]] = [] if is_crop_dataset else None
    patch_coords: Optional[List[Tuple[int, int]]] = [] if is_crop_dataset else None

    for idx in indices:
        # Access dataset item to trigger lazy loading and state update
        input_tensor, target_tensor = dataset[idx]

        # Convert to numpy - handle both Tensor and ndarray inputs
        input_list = _normalize_to_list(input_tensor)
        target_list = _normalize_to_list(target_tensor)

        inputs.append(input_list[0] if len(input_list) == 1 else input_list)
        targets.append(target_list[0] if len(target_list) == 1 else target_list)

        if is_crop_dataset:
            # Access the original uncropped image and crop coordinates
            # These are populated after __getitem__ is called
            if is_wrapper_dataset:
                raw_images.append(dataset.original.original_input_image[0])
                patch_coords.append((dataset.original.crop_info.x, dataset.original.crop_info.y))
            else:
                raw_images.append(dataset.original_input_image[0])
                patch_coords.append((dataset.crop_info.x, dataset.crop_info.y))

    return inputs, targets, raw_images, patch_coords


def evaluate_per_image_metric(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metrics: List[Module],
    indices: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Computes a set of metrics on a per-image basis and returns the results as a pandas DataFrame.

    :param predictions: Predicted images, shape (N, C, H, W).
    :type predictions: torch.Tensor
    :param targets: Target images, shape (N, C, H, W).
    :type targets: torch.Tensor
    :param metrics: List of metric functions to evaluate.
    :type metrics: List[torch.nn.Module]
    :param indices: Optional list of indices to subset the dataset before inference. If None, all images are evaluated.
    :type indices: Optional[List[int]], optional

    :return: A DataFrame where each row corresponds to an image and each column corresponds to a metric.
    :rtype: pd.DataFrame
    """
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")

    results = []

    if indices is None:
        indices = range(predictions.shape[0])

    for i in indices:  # Iterate over images/subset
        pred, target = predictions[i].unsqueeze(0), targets[i].unsqueeze(0)  # Keep batch dimension
        metric_scores = {metric.__class__.__name__: metric.forward(target, pred).item() for metric in metrics}
        results.append(metric_scores)

    return pd.DataFrame(results)
