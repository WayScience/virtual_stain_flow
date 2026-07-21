"""
crop_generator.py

Utilities for generating crop coordinates from BaseImageDataset objects.
Designed for easy creation of CropImageDataset instances.
Made Facade to account for increased complexity and future expansion.
"""

from .crop_generators.protocol import CropSpec, CropMap, CropGenerator
from .crop_generators.center import (
    generate_center_crops,
    _compute_center_crop,
)
from .crop_generators.point_centered import generate_point_centered_crops
from .crop_generators.tile import generate_tile_crops
__all__ = [
    "CropSpec",
    "CropMap",
    "CropGenerator",
    "generate_center_crops",
    "generate_point_centered_crops",
    "generate_tile_crops",
    "_compute_center_crop",
]
