from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.nn import Module

from ..datasets.base_dataset import BaseImageDataset
from ..datasets.crop_dataset import CropImageDataset


def extract_samples_from_dataset(
    dataset: Union[BaseImageDataset, CropImageDataset],
    indices: List[int],
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    Optional[List[np.ndarray]],
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
        - targets: List of numpy arrays, each with shape (C, H, W) or (H, W).
        - raw_images: List of numpy arrays for CropImageDataset (original uncropped images),
          or None for BaseImageDataset.
        - patch_coords: List of (x, y) tuples for CropImageDataset, or None for BaseImageDataset.
    """
    is_crop_dataset = isinstance(dataset, CropImageDataset)

    inputs: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    raw_images: Optional[List[np.ndarray]] = [] if is_crop_dataset else None
    patch_coords: Optional[List[Tuple[int, int]]] = [] if is_crop_dataset else None

    for idx in indices:
        # Access dataset item to trigger lazy loading and state update
        input_tensor, target_tensor = dataset[idx]

        # Convert to numpy - handle both Tensor and ndarray inputs
        if isinstance(input_tensor, torch.Tensor):
            inputs.append(input_tensor.numpy())
        else:
            inputs.append(np.asarray(input_tensor))

        if isinstance(target_tensor, torch.Tensor):
            targets.append(target_tensor.numpy())
        else:
            targets.append(np.asarray(target_tensor))

        if is_crop_dataset:
            # Access the original uncropped image and crop coordinates
            # These are populated after __getitem__ is called
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
