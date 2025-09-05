"""
/transforms/transform_utils.py

This module defines type aliases and runtime type tuples for various image
transforms used in the virtual_stain_flow package. 
One type alias checks for albumentations-based transforms that does not 
integrate natively with the logging system, and needs to be treated differently.
The other type alias chekcs for loggable transform classes inheriting from
LoggableTransform (but still behaves like albumentations transforms).

This module will also house any future transform utilities that should be
centralized for use across the rest of the package. 
"""

from typing import Union, Sequence, Any

from albumentations import Compose, ImageOnlyTransform, BasicTransform

from .base_transform import LoggableTransform

# Aliases defining acceptable transform types to be exported for type hinting
# purposes by other parts of the package.
ValidAlbumentationType = Union[BasicTransform, ImageOnlyTransform, Compose]
TransformType = Union[
    LoggableTransform,
    ValidAlbumentationType
]

# Tuples defining acceptable transform types to be used for isinstance checks
RuntimeValidAlbumentationType = (BasicTransform, ImageOnlyTransform, Compose)
RuntimeTransformType = (LoggableTransform,) + RuntimeValidAlbumentationType