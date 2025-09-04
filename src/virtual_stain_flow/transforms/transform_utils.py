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