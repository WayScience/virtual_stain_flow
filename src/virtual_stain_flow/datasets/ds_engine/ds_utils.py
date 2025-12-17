"""
ds_utils.py

Utility functions for handling datasets.
"""

from typing import List, Tuple, Sequence

from ..base_dataset import BaseImageDataset


def _get_active_channels(dataset: BaseImageDataset) -> List[str]:
    """
    Get the list of active channel keys from a dataset.
    Active channels are the union of input and target channel keys.
    
    :param dataset: A BaseImageDataset instance or subclass
    :return: List of active channel keys.
    """

    channels = []
    if dataset.input_channel_keys:
        channels.extend(dataset.input_channel_keys)
    if dataset.target_channel_keys:
        channels.extend(dataset.target_channel_keys)
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for c in channels:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _validate_same_dimensions_across_channels(
    dims: Tuple[Tuple[int, int], ...],
    channels: Sequence[str],
    idx: int
) -> Tuple[int, int]:
    """
    Validate that all channels have the same dimensions.
    
    :param dims: Tuple of (width, height) tuples for each channel.
    :param channels: List of channel names (for error messages).
    :param idx: Sample index (for error messages).
    :return: The common (width, height) if all match.
    :raises ValueError: If dimensions mismatch across channels.
    """
    if not dims:
        raise ValueError(f"No dimensions returned for sample {idx}.")
    
    # Filter out None values (missing files)
    valid_dims = [(d, c) for d, c in zip(dims, channels) if d is not None]
    if not valid_dims:
        raise ValueError(f"All channel files missing for sample {idx}.")
    
    first_dim, first_channel = valid_dims[0]
    for dim, channel in valid_dims[1:]:
        if dim != first_dim:
            raise ValueError(
                f"Dimension mismatch at sample {idx}: "
                f"channel '{first_channel}' has {first_dim}, "
                f"but channel '{channel}' has {dim}."
            )
    
    return first_dim
