"""
"""

import numpy as np

from .protocol import CropMap
from ...base_dataset import BaseImageDataset
from ..ds_utils import (
    _get_active_channels, 
    _validate_same_dimensions_across_channels
)


def _compute_point_centered_crops(
    image_width: int,
    image_height: int,
    crop_size: int,
    centers: dict[str, np.ndarray]
) -> list[tuple[int, int]]:
    """
    Compute the top-left coordinates of crops centered around specified points,
        ensuring that the crops fit fully within the image boundaries.

    :param image_width: Width of the image.
    :param image_height: Height of the image.
    :param crop_size: Size of the crops to generate.
    :param centers: Dictionary containing 'X' and 'Y' coordinates of the centers.
    :return: List of (x, y) tuples representing the top-left coordinates of the crops.
    """
    
    crops = []

    for x, y in zip(centers['X'], centers['Y']):

        if x <= crop_size // 2 or x >= image_width - crop_size // 2 or \
           y <= crop_size // 2 or y >= image_height - crop_size // 2:
            continue  # Skip points too close to the edge for a full crop

        crop_x = int(x - crop_size // 2)
        crop_y = int(y - crop_size // 2)
        crops.append((crop_x, crop_y))

    return crops


def generate_point_centered_crops(
    dataset: BaseImageDataset,
    crop_size: int | None = None,
    mapping: list[dict[str, np.ndarray]] | None = None,
) -> CropMap:
    """
    Generate crop specifications centered around specified points 
        for each image in the dataset.
    
    :param dataset: BaseImageDataset containing the images and metadata.
    :param crop_size: Size of the crops to generate. Made optional 
        for better error handling when called from the CropImageDataset
        .from_base_dataset class method. 
    :param mapping: List of dictionaries containing the centers for each image.
        Each dictionary should have keys 'X' and 'Y' with corresponding numpy 
            arrays of coordinates.
        Made optional for the same reason as crop_size.
    :return: CropMap containing the generated crop specifications.
    """

    if crop_size is None:
        raise ValueError(
            "crop_size must be provided for point-centered crop generation."
        )
    if mapping is None:
        raise ValueError(
            "mapping must be provided for point-centered crop generation."
        )

    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}.")
    
    active_channels = _get_active_channels(dataset)
    if not active_channels:
        raise ValueError(
            "No active channels configured. Set input_channel_keys and/or "
            "target_channel_keys on the dataset before generating crops."
        )
    
    manifest = dataset.file_state.manifest
    crop_specs: dict[int, list[tuple[tuple[int, int], int, int]]] = {}

    for idx in range(len(dataset)):
        
        dims = manifest.get_image_dimensions(idx, channels=active_channels)

        width, height = _validate_same_dimensions_across_channels(
            dims, active_channels, idx
        )

        centers = mapping[idx]

        crop_lists = _compute_point_centered_crops(width, height, crop_size, centers)
        crop_specs[idx] = [
            (crop_coords, crop_size, crop_size) for crop_coords in crop_lists
        ]

    return crop_specs
