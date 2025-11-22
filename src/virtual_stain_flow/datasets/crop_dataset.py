"""
crop_dataset.py
"""

from typing import Any, Dict, List, Sequence, Optional, Tuple, Union

import pandas as pd

from .base_dataset import BaseImageDataset
from .ds_engine.crop_manifest import CropManifest, CropFileState, Crop


class CropImageDataset(BaseImageDataset):
    """
    Dataset that serves image crops based on a CropManifest.
    Utilizes CropIndexState and CropFileState for state management and lazy loading.
    """
    
    def __init__(
        self,
        *,
        file_index: Optional[pd.DataFrame] = None,
        crop_specs: Optional[Dict[int, List[Tuple[Tuple[int, int], int, int]]]] = None,
        pil_image_mode: str = "I;16",
        cache_capacity: Optional[int] = None,
        input_channel_keys: Optional[Union[str, Sequence[str]]] = None,
        target_channel_keys: Optional[Union[str, Sequence[str]]] = None,
        crop_file_state: Optional[CropFileState] = None,
    ):
        """
        Initialize the CropImageDataset.

        :param file_index: Optional DataFrame containing exclusively file paths as pathlikes.
            Must be provided if `crop_file_state` is not provided.
        :param crop_specs: Optional dictionary mapping file index positions to lists of crop
            specifications. Each crop specification is a tuple of ((x, y), width, height).
            Must be provided if `crop_file_state` is not provided.
        :param pil_image_mode: Mode for PIL images, default is "I;16".
        :param cache_capacity: Optional capacity for caching loaded images. 
            When set to None, default caching behavior of caching at most
            `file_index.shape[0]` images is used. When set to -1, unbounded
            caching without eviction is used. When set to a positive integer,
            the cache will hold at most that many images, evicting the least recently
            used images when the cache is full (LRU cache). Other values are
            invalid.     
        :param input_channel_keys: Keys for input channels in the file index.
        :param target_channel_keys: Keys for target channels in the file index.
        :param crop_file_state: Optional pre-initialized CropFileState object. If provided,
            it takes precedence over `file_index` and `crop_specs`. Intended
            to be used by only .from_config class method and similar deserialization
            utilities.
        """
        if not crop_file_state and (file_index is None or crop_specs is None):
            raise ValueError(
                "Either 'crop_file_state' or both 'file_index' and 'crop_specs' must be provided."
            )
        
        self.file_state = crop_file_state or CropFileState(
            CropManifest.from_coord_size(
                crop_specs=crop_specs,
                file_index=file_index,
                pil_image_mode=pil_image_mode
            ),
            cache_capacity=cache_capacity
        )
        self.manifest = self.file_state.crop_collection
        self.index_state = self.file_state.crop_state

        self.input_channel_keys = input_channel_keys
        self.target_channel_keys = target_channel_keys

    @property
    def pil_image_mode(self) -> str:
        return self.manifest.pil_image_mode
    
    @property
    def file_index(self) -> pd.DataFrame:
        return self.manifest.file_index
    
    @property
    def crop_info(self) -> Optional[Crop]:
        return self.file_state.crop_info
    
    def to_config(self) -> Dict[str, Any]:
        """
        Serialize to dict.
        Mirrors design of BaseImageDataset.to_config().
        Serializes the underlying CropFileState.
        """
        return {
            'crop_file_state': self.file_state.to_config(),
            'input_channel_keys': self.input_channel_keys,
            'target_channel_keys': self.target_channel_keys
        }
    
    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any]
    ) -> 'CropImageDataset':
        """
        Deserialize from dict.
        Mirrors design of BaseImageDataset.from_config().
        Deserializes the underlying CropFileState.
        """
        return cls(
            crop_file_state=CropFileState.from_config(
                config['crop_file_state']
            ),
            input_channel_keys=config.get('input_channel_keys', None),
            target_channel_keys=config.get('target_channel_keys', None)
        )
