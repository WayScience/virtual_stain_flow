"""
monai_aug_adapter_dataset.py
"""

from monai.transforms import (
    Compose
)

from .base_dataset import BaseImageDataset
from .base_wrapper_dataset import BaseWrapperDataset

class MonaiAdapter(BaseWrapperDataset):
    """
    Adapter dataset to wrap any BaseImageDataset and return samples
        in dictionary format compatible with MONAI transforms and pipelines.
    Specifically, each sample is returned as a dictionary with keys "input" and "target",
        containing the input and target tensors respectively, then ran through
        MONAI transforms if provided, and finally returned back as a tuple of 
        (input, target) tensors. It would be meaningless to use this adapter
        without any MONAI transforms as the data just gets wrapped and unwrapped.
    """
    def __init__(
        self, 
        base_dataset: BaseImageDataset, 
        transform: Compose | None = None
    ):
        super().__init__(base_dataset)
        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        x, y = self._dataset[idx]

        sample = {"input": x, "target": y}

        if self._transform is not None:
            sample = self._transform(sample)

        return sample["input"], sample["target"]
