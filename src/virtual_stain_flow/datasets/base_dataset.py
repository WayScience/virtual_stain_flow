"""
/datasets/base_dataset.py

This file contains the `BaseImageDataset` class, meant to serve as the foundation
infranstructure for all image datasets. 
Uses a `DatasetManifest` and `FileState` backbone.
"""

from typing import Dict, Sequence, Optional, Tuple, Union, Any
import json
from pathlib import Path, PurePath

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .manifest import DatasetManifest, IndexState, FileState

class BaseImageDataset(Dataset):
    
    def __init__(
        self,
        file_index: pd.DataFrame,
        pil_image_mode: str = "I;16",
        input_channel_keys: Optional[Union[str, Sequence[str]]] = None,
        target_channel_keys: Optional[Union[str, Sequence[str]]] = None,
        cache_capacity: Optional[int] = None,
    ):
        
        """
        Initializes the BaseImageDataset.

        :param file_index: DataFrame containing exclusively file paths as pathlikes
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
        """

        self.manifest = DatasetManifest(
            file_index=file_index, 
            pil_image_mode=pil_image_mode
        )
        self.index_state = IndexState()
        self.file_state = FileState(
            manifest=self.manifest, 
            cache_capacity=cache_capacity
        )

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
        input, target = self.get_raw_item(idx)

        return torch.from_numpy(input).float(), torch.from_numpy(target).float()
    
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
        # Convert file_index to records format for JSON serialization
        # Convert Path objects to strings for JSON compatibility
        file_index_for_json = self.file_index.copy()
        for col in file_index_for_json.columns:
            file_index_for_json[col] = file_index_for_json[col].apply(
                lambda x: str(x) if isinstance(x, (Path, PurePath)) else x
            )
        
        file_index_records = file_index_for_json.to_dict('records')
        file_index_columns = list(self.file_index.columns)
        
        config = {
            'file_index': {
                'records': file_index_records,
                'columns': file_index_columns
            },
            'pil_image_mode': self.pil_image_mode,
            'input_channel_keys': self.input_channel_keys,
            'target_channel_keys': self.target_channel_keys,
            'cache_capacity': self.file_state.cache_capacity,
            'dataset_length': len(self)
        }
        
        return config
    
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
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    @classmethod
    def _deserialize_config_core(
        cls,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Internal method for deserializing the core configuration
        from a configuration dictionary.
        Subclass deserialization methods may call this method
            to conveniently extract the common configuration.
        """
        # Reconstruct file_index DataFrame
        file_index_data = config.get('file_index', None)
        if file_index_data is None:
            raise ValueError(
                "Expected 'file_index' in config, "
                "but found none or empty."
            )
        
        file_index = pd.DataFrame(file_index_data['records'])
        # Convert string paths back to Path objects
        for col in file_index.columns:
            file_index[col] = file_index[col].apply(
                lambda x: Path(x) if isinstance(x, str) else x
            )

        pil_image_mode = config.get('pil_image_mode', None)
        if pil_image_mode is None:
            raise ValueError(
                "Expected 'pil_image_mode' in config, "
                "but found none or empty."
            )
        
        input_channel_keys=config.get('input_channel_keys', None)
        target_channel_keys=config.get('target_channel_keys', None)
        cache_capacity=config.get('cache_capacity', None)
                    
        return {
            'file_index': file_index,
            'pil_image_mode': pil_image_mode,
            'input_channel_keys': input_channel_keys,
            'target_channel_keys': target_channel_keys,
            'cache_capacity': cache_capacity
        }

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any]
    ) -> 'BaseImageDataset':
        """
        Class method to instantiate a dataset from a configuration dictionary.
        As this is the base class, only the core configuration deserialization
            is performed here. Subclasses may override this method to handle
            additional configuration parameters.
        
        :param config: Configuration dictionary.
        :return: An instance of BaseImageDataset or its subclass.
        """
        core_ds_kwargs = cls._deserialize_config_core(config)

        return cls(
            **core_ds_kwargs
        )
