"""
protocol.py

Defines the CropGenerator protocol for crop generator functions.
"""

from typing import Dict, List, Tuple, Any, Protocol

from ...base_dataset import BaseImageDataset


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
