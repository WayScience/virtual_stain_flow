"""
visualization_utils.py

Utilities for extracting batch-form image data and crop metadata for visualization.
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..datasets.base_dataset import BaseImageDataset
from ..datasets.base_wrapper_dataset import BaseWrapperDataset
from ..datasets.crop_dataset import CropImageDataset

PatchCoords = Tuple[int, int, int, int]
DatasetType = Union[BaseImageDataset, CropImageDataset, BaseWrapperDataset]


def _to_numpy_image(value: Union[np.ndarray, torch.Tensor], name: str) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        raise ValueError(
            f"{name} must be a single channel-first image, not a sequence of images."
        )

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)

    if value.ndim != 3:
        raise ValueError(
            f"{name} must have shape (C, H, W), received shape {value.shape}."
        )

    # The caller must snapshot this view before accessing the next sample.
    return value


class _ImageBatch:
    """Snapshot images directly into batch storage, promoting dtype if needed."""

    def __init__(self, count: int, name: str):
        self.count = count
        self.name = name
        self.batch: Optional[np.ndarray] = None
        self._dtypes: List[np.dtype] = []
        self._position = 0

    def append(self, image: np.ndarray) -> None:
        if self.batch is not None and image.shape != self.batch.shape[1:]:
            raise ValueError(f"{self.name} must have a consistent (C, H, W) shape.")

        # Use source dtypes rather than pairwise promotion to match np.stack.
        if image.dtype not in self._dtypes:
            self._dtypes.append(image.dtype)
        dtype = np.result_type(*self._dtypes)
        if self.batch is None:
            self.batch = np.empty((self.count, *image.shape), dtype=dtype)
        elif dtype != self.batch.dtype:
            promoted = np.empty(self.batch.shape, dtype=dtype)
            promoted[:self._position] = self.batch[:self._position]
            self.batch = promoted

        # Freeze reused NumPy/tensor buffers before accessing the next sample.
        self.batch[self._position] = image
        self._position += 1


def _get_channel_names(
    channel_keys: Optional[Union[str, Sequence[str]]], expected_count: int
) -> Optional[List[str]]:
    if channel_keys is None:
        return None

    names = [channel_keys] if isinstance(channel_keys, str) else list(channel_keys)
    return names if len(names) == expected_count else None


def extract_samples_from_dataset(
    dataset: DatasetType,
    indices: List[int],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[List[PatchCoords]],
    Optional[List[str]],
    Optional[List[str]],
]:
    """
    Extract image batches and optional crop metadata for visualization.
    Primary function of this abstraction is to provide a consistent data
        access interface between Dataset objects and plotting functions by
        extracting input/target with __get_item__ and also accessing raw
        images and crop annotations when available. Each requested sample is
        accessed once and copied immediately into preallocated batch storage,
        including its raw image metadata. Mixed dtypes may require reallocation
        to preserve NumPy stacking promotion semantics.
        Wrappers around crop datasets must keep the underlying crop metadata
        synchronized with the sample they return.

    :param dataset: A BaseImageDataset, CropImageDataset, or BaseWrapperDataset.
    :param indices: Dataset indices to extract, in the displayed order.
    :return: Tuple of inputs, targets, raw images, crop coordinates, input channel
        names, and target channel names. Image arrays have shape (N, C, H, W).
        Channel names are None when unavailable or inconsistent with the image
        channels. Raw images and patch coordinates are None unless the dataset is
        a CropImageDataset or wraps one. Patch coordinates are (x, y, width, height).
    """
    if isinstance(dataset, BaseWrapperDataset):
        metadata_dataset = dataset.original
        crop_dataset = metadata_dataset if isinstance(metadata_dataset, CropImageDataset) else None
    elif isinstance(dataset, CropImageDataset):
        metadata_dataset = dataset
        crop_dataset = dataset
    elif isinstance(dataset, BaseImageDataset):
        metadata_dataset = dataset
        crop_dataset = None
    else:
        raise ValueError(
            "Unsupported dataset type. Expected BaseImageDataset, CropImageDataset, "
            "or BaseWrapperDataset."
        )

    if not indices:
        raise ValueError("Indices list cannot be empty.")
    if min(indices) < 0 or max(indices) >= len(dataset):
        raise IndexError(
            f"Index out of range. Dataset length: {len(dataset)}, "
            f"requested indices: {indices}"
        )

    inputs = _ImageBatch(len(indices), "Dataset inputs")
    targets = _ImageBatch(len(indices), "Dataset targets")
    raw_images = _ImageBatch(len(indices), "Crop raw images") if crop_dataset is not None else None
    patch_coords: Optional[List[PatchCoords]] = [] if crop_dataset is not None else None

    for index in indices:
        # Need to access and cast single image samples of
        # dimension (C, H, W) from torch tensor to np arrays
        input_image, target_image = dataset[index]
        inputs.append(_to_numpy_image(input_image, "Dataset input"))
        targets.append(_to_numpy_image(target_image, "Dataset target"))

        if crop_dataset is not None and raw_images is not None and patch_coords is not None:
            raw_image = crop_dataset.original_input_image
            crop_info = crop_dataset.crop_info
            if raw_image is None or crop_info is None:
                raise RuntimeError("Crop image metadata was not populated after dataset access.")

            raw_images.append(_to_numpy_image(raw_image, "Crop raw image"))
            patch_coords.append(
                (crop_info.x, crop_info.y, crop_info.width, crop_info.height)
            )

    input_batch = inputs.batch
    target_batch = targets.batch
    raw_batch = raw_images.batch if raw_images is not None else None
    # Nonempty indices guarantee that both batches have been allocated.
    assert input_batch is not None and target_batch is not None

    return (
        input_batch,
        target_batch,
        raw_batch,
        patch_coords,
        _get_channel_names(metadata_dataset.input_channel_keys, input_batch.shape[1]),
        _get_channel_names(metadata_dataset.target_channel_keys, target_batch.shape[1]),
    )
