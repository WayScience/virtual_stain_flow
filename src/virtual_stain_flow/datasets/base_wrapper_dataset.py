"""
base_wrapper_dataset.py

Defines a simple BaseWrapperDataset scheme that can wraps any BaseImageDataset 
    and forwards all method calls to it.
"""

from abc import ABC, abstractmethod
from typing import Union

from .base_dataset import BaseImageDataset
from .crop_dataset import CropImageDataset

class BaseWrapperDataset(ABC):

    def __init__(
        self, 
        dataset: Union[BaseImageDataset, CropImageDataset]
    ):
        self._dataset = dataset
        # optionally do something to the dataset

    def __len__(self):
        return len(self._dataset)

    @abstractmethod
    def __getitem__(self, idx):
        # retrieve images from dataset
        input, target = self._dataset[idx]

        # do something to the input and target here
        # (e.g. apply transformations, generate crops, cache in RAM, etc.)
        
        return input, target
    
    @property
    def original(self) -> Union[BaseImageDataset, CropImageDataset]:
        """
        Access the original underlying dataset for metadata etc.
        """
        if isinstance(self._dataset, BaseWrapperDataset):
            return self._dataset.original
        return self._dataset
