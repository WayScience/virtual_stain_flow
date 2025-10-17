"""
data_split.py

Module for dataset splitting, to be called by trainers during initialization.
"""

from typing import Tuple

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split


def default_random_split(
    dataset: Dataset, 
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Randomly split a dataset into train, validation, and test sets.
    :param dataset: The dataset to split.
    :param train_frac: Fraction of data to use for training (default: 0.7).
    :param val_frac: Fraction of data to use for validation (default: 0.15).
    :param test_frac: Fraction of data to use for testing (default: remaining).
    :param batch_size: Batch size for the DataLoaders (default: 4).
    :param shuffle: Whether to shuffle the data in the DataLoaders
        (default: True).
    :return: A tuple of DataLoaders for (train, val, test) splits.    
    """
    
    train_frac = kwargs.get("train_frac", 0.7)
    val_frac = kwargs.get("val_frac", 0.15)
    test_frac = kwargs.get(
        "test_frac", 1.0 - train_frac - val_frac
    )
    
    for frac in (train_frac, val_frac, test_frac):
        if not (0.0 < frac < 1.0):
            raise ValueError(
                "train_frac, val_frac, test_frac must be in (0.0, 1.0)"
            )
    if not train_frac + val_frac + test_frac <= 1.0 + 1e-8:
        raise ValueError(
            "train_frac + val_frac + test_frac must sum to 1.0"
        )
    
    n_train = int(len(dataset) * train_frac)
    n_val = int(len(dataset) * val_frac)
    n_test = len(dataset) - n_train - n_val

    train_split, val_split, test_split = random_split(dataset, 
        [n_train, n_val, n_test]
    )

    batch_size = kwargs.get("batch_size", 4)
    shuffle = kwargs.get("shuffle", True)

    return tuple([
        DataLoader(split, shuffle=shuffle, batch_size=batch_size) for split in [
            train_split, val_split, test_split
        ]
    ])
