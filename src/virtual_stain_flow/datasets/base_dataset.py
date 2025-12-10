"""
base_dataset.py

This file contains the `BaseImageDataset` class, meant to serve as the foundation
infrastructure for all image datasets. 
Uses a `DatasetManifest` and `FileState` backbone.
"""

from typing import Dict, Sequence, Optional, Tuple, Union, Any
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .ds_engine.manifest import DatasetManifest, IndexState, FileState


class BaseImageDataset(Dataset):
    
    def __init__(
        self,
        *,
        file_index: Optional[pd.DataFrame] = None,
        pil_image_mode: str = "I;16",
        input_channel_keys: Optional[Union[str, Sequence[str]]] = None,
        target_channel_keys: Optional[Union[str, Sequence[str]]] = None,
        cache_capacity: Optional[int] = None,
        file_state: Optional[FileState] = None,
    ):
        
        """
        Initializes the BaseImageDataset.

        :param file_index: Optional DataFrame containing exclusively file paths as pathlikes
            Must be provided if `file_state` is not provided.
        :param pil_image_mode: Mode for PIL images, default is "I;16".
        :param input_channel_keys: Keys for input channels in the file index.
        :param target_channel_keys: Keys for target channels in the file index.
        :param cache_capacity: Optional capacity for caching loaded images. 
            When set to None, default caching behavior of caching at most
            `file_index.shape[0]` images is used. When set to -1, unbounded
            caching without eviction is used. When set to a positive integer,
            the cache will hold at most that many images, evicting the least recently
            used images when the cache is full (LRU cache). Other values are
            invalid.     
        :param file_state: Optional pre-initialized FileState object. If provided,
            it takes precedence over `file_index` and `pil_image_mode`. Intended
            to be used by only .from_config class method and similar deserialization
            utilities.         
        """
        self.index_state = IndexState()

        if file_state is None and file_index is None:
            raise ValueError(
                "Either 'file_state' or 'file_index' must be provided."
            )
        self.file_state = FileState(
            DatasetManifest(
                file_index=file_index, 
                pil_image_mode=pil_image_mode
            ), 
            cache_capacity=cache_capacity
        ) if file_state is None else file_state
        self.manifest = self.file_state.manifest

        self.input_channel_keys = input_channel_keys
        self.target_channel_keys = target_channel_keys

    def get_raw_item(
        self, 
        idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method to get the raw numpy arrays for input and target images
        corresponding to the given index.
        Used also by `__getitem__` to get the data for PyTorch.

        :param idx: Index of the item to retrieve.
        :return: Tuple of numpy arrays (input_image, target_image).
        """

        self.index_state.update(idx)

        # load files lazily given current channel config
        self.file_state.update(
            idx, 
            input_keys=self.input_channel_keys, 
            target_keys=self.target_channel_keys
        )

        return (
            self.file_state.input_image_raw,
            self.file_state.target_image_raw
        )

    def __len__(self) -> int:
        """
        Overridden Dataset `__len__` method so class works with torch DataLoader.
        """
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overridden Dataset `__getitem__` method so class works with torch DataLoader.
        """        
        input_image_raw, target_image_raw = self.get_raw_item(idx)

        return (torch.from_numpy(input_image_raw).float(), 
                torch.from_numpy(target_image_raw).float())

    @property
    def pil_image_mode(self) -> str:
        """
        Returns the PIL image mode.
        """
        return self.manifest.pil_image_mode

    @property
    def file_index(self) -> pd.DataFrame:
        """
        Returns the file index DataFrame.
        The `file_index` attribute of the dataset class is expected to be
            immutable after class initialization, hence no setter is provided.
        """
        return self.manifest.file_index
    
    def _validate_channel_keys(
        self,
        channel_keys: Optional[Union[str, Sequence[str]]],
    ):
        """
        Validates the channel keys against the file index columns.
        
        :param channel_keys: Keys for input or target channels.
        :raises ValueError: If channel_keys is invalid.
        """
        if channel_keys is None:
            return []
        elif isinstance(channel_keys, str):
            channel_keys = [channel_keys]
        elif not isinstance(channel_keys, Sequence):
            raise ValueError("Expected channel_keys to be a string or a "
                             "sequence of strings, "
                             f"got {type(channel_keys)} instead.")
        
        for key in channel_keys:
            if key not in self.manifest.file_index.columns:
                raise ValueError(f"Channel key '{key}' not found in "
                                 "file_index columns.")            
        return channel_keys
    
    @property
    def input_channel_keys(self) -> Optional[Union[str, Sequence[str]]]:
        """
        Returns the input channel keys.
        """
        return self._input_channel_keys
    
    @input_channel_keys.setter
    def input_channel_keys(self, value: Optional[Union[str, Sequence[str]]]=None):
        """
        Sets the input channel keys.
        """
        value = self._validate_channel_keys(value)        
        self._input_channel_keys = value

    @property
    def target_channel_keys(self) -> Optional[Union[str, Sequence[str]]]:
        """
        Returns the target channel keys.
        """
        return self._target_channel_keys
    
    @target_channel_keys.setter
    def target_channel_keys(self, value: Optional[Union[str, Sequence[str]]]=None):
        """
        Sets the target channel keys.
        """
        value = self._validate_channel_keys(value)        
        self._target_channel_keys = value

    def to_config(self) -> Dict[str, Any]:
        """
        Internal method for serializing the dataset as a configuration dictionary.        
        :return: Dictionary containing the serialized configuration.
        """

        return {
            'file_state': self.file_state.to_config(),
            'input_channel_keys': self.input_channel_keys,
            'target_channel_keys': self.target_channel_keys,
        }
    
    def to_json_config(self, filepath: Union[str, Path]) -> None:
        """
        Exposed method for serializing the dataset as a JSON file, 
        facilitating reproducibility by saving all information needed 
        to reconstruct the dataset. At the moment transforms are not
        serializable and hence are ignored. Future development may
        address this limitation.

        :param filepath: Path where to save the JSON configuration file.
        """
        config = self.to_config()
        
        filepath = Path(filepath)
        with open(filepath, 'w', encoding='utf-8') as json_config_file:
            json.dump(config, json_config_file, indent=2, ensure_ascii=False)

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any]
    ) -> 'BaseImageDataset':
        """
        Class method to instantiate a dataset from a configuration dictionary.
        
        :param config: Configuration dictionary.
        :return: An instance of BaseImageDataset or its subclass.
        """
        return cls(
            file_state=FileState.from_config( # heavy lifting handled by FileState
                config['file_state']
            ),
            input_channel_keys=config.get('input_channel_keys', None),
            target_channel_keys=config.get('target_channel_keys', None),
        )
