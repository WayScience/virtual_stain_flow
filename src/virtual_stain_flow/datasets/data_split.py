"""
data_split.py

Module for dataset splitting, to be called by trainers during initialization.
"""

from typing import Optional, Tuple

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split


def default_random_split(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: Optional[float] = None,
    batch_size: int = 4,
    shuffle: bool = True,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Randomly split a dataset into train, validation, and test sets.
    :param dataset: The dataset to split.
    :param train_ratio: Fraction of data to use for training (default: 0.7).
    :param val_ratio: Fraction of data to use for validation (default: 0.15).
    :param test_ratio: Fraction of data to use for testing (default: remaining).
    :param batch_size: Batch size for the DataLoaders (default: 4).
    :param shuffle: Whether to shuffle the data in the DataLoaders
        (default: True).
    :return: A tuple of DataLoaders for (train, val, test) splits.    
    """

    for ratio in (train_ratio, val_ratio, test_ratio):
        if ratio is not None and not (0.0 < ratio < 1.0):
            raise ValueError(
                "train_ratio, val_ratio, test_ratio must be in (0.0, 1.0)"
            )   

    if not test_ratio:
        test_ratio = 1.0 - train_ratio - val_ratio
    
    if not train_ratio + val_ratio + test_ratio <= 1.0 + 1e-8:
        raise ValueError(
            "train_ratio + val_ratio + test_ratio must sum to 1.0"
        )
    
    n_train = int(len(dataset) * train_ratio)
    n_val = int(len(dataset) * val_ratio)
    n_test = len(dataset) - n_train - n_val

    train_split, val_split, test_split = random_split(dataset, 
        [n_train, n_val, n_test]
    )
    
    return tuple([
        DataLoader(split, shuffle=shuffle, batch_size=batch_size) for split in [
            train_split, val_split, test_split
        ]
    ])
