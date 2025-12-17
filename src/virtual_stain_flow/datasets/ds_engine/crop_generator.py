"""
crop_generator.py

Utilities for generating crop coordinates from BaseImageDataset objects.
Designed for easy creation of CropImageDataset instances.
"""

from typing import Dict, List, Tuple, Any, Protocol

from ..base_dataset import BaseImageDataset
from .ds_utils import (
    _get_active_channels, 
    _validate_same_dimensions_across_channels
)

CropSpec = Tuple[Tuple[int, int], int, int]
CropMap = Dict[int, List[CropSpec]]


class CropGenerator(Protocol):
    """
    Protocol for crop generator functions.
    """
    def __call__(
        self, 
        dataset: BaseImageDataset, 
        **kwargs: Any
    ) -> CropMap:
        pass 


def _compute_center_crop(
    image_width: int,
    image_height: int,
    crop_size: int
) -> Tuple[int, int]:
    """
    Compute top-left (x, y) coordinates for a center crop.
    
    :param image_width: Width of the source image.
    :param image_height: Height of the source image.
    :param crop_size: Size of the square crop (width and height).
    :return: Tuple of (x, y) for top-left corner of center crop.
    :raises ValueError: If crop_size exceeds image dimensions.
    """
    if crop_size > image_width or crop_size > image_height:
        raise ValueError(
            f"crop_size ({crop_size}) exceeds image dimensions "
            f"({image_width}x{image_height})."
        )
    
    x = (image_width - crop_size) // 2
    y = (image_height - crop_size) // 2
    return x, y


def generate_center_crops(
    dataset: BaseImageDataset,
    crop_size: int,
) -> CropMap:
    """
    Generate center crop coordinates for each sample in a BaseImageDataset.
    
    :param dataset: A BaseImageDataset instance (or compatible object with
        `file_state.manifest` attribute supporting `get_image_dimensions()`).
    :param crop_size: Size of the square crop (same width and height).
    :return: Dictionary mapping manifest indices to lists of crop specs.
        Format: {manifest_idx: [((x, y), width, height), ...]}
    :raises ValueError: If crop_size is non-positive, if no active channels
        are configured, or if channel dimensions don't match for any sample.
    """
    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}.")
    
    active_channels = _get_active_channels(dataset)
    if not active_channels:
        raise ValueError(
            "No active channels configured. Set input_channel_keys and/or "
            "target_channel_keys on the dataset before generating crops."
        )
    
    manifest = dataset.file_state.manifest
    crop_specs: Dict[int, List[Tuple[Tuple[int, int], int, int]]] = {}
    
    for idx in range(len(dataset)):
        # Get dimensions for all active channels
        dims = manifest.get_image_dimensions(idx, channels=active_channels)
        
        # Validate all channels have matching dimensions
        width, height = _validate_same_dimensions_across_channels(
            dims, active_channels, idx
        )
        
        # Compute center crop coordinates
        x, y = _compute_center_crop(width, height, crop_size)
        
        # Store as crop_specs format: {idx: [((x, y), w, h), ...]}
        crop_specs[idx] = [((x, y), crop_size, crop_size)]
    
    return crop_specs
