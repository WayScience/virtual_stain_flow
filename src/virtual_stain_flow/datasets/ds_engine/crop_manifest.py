"""
crop_manifest.py

Engine for datasets defined by not full images, but crops extracted
    from images. Mirrors the design of manifest.py to have dedicated
    classes for:
    - Immutable dataset manifest: CropManifest, collection of crop definitions
    - Mutable dataset state: CropIndexState, tracks current crop index
    - Lazy image loading and dynamic cropping: CropFileState, wraps FileState
        to load full images and extract crops on demand.
"""

from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Sequence
from dataclasses import dataclass, asdict

import numpy as np

from ..manifest import DatasetManifest, FileState


@dataclass
class Crop:
    """Single crop definition."""
    manifest_idx: int # index mapping back to file
    x: int
    y: int
    width: int
    height: int
    
    def to_dict(self) -> Dict[str, int]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'Crop':
        return cls(**data)


class CropManifest:
    """
    Immutable collection of crops.
    Wraps a DatasetManifest for file access.
    """
    
    def __init__(self, crops: List[Crop], manifest: DatasetManifest):
        """
        Initialize a crop manifest.
        
        :param crops: List of Crop objects.
        :param manifest: DatasetManifest containing the file index.
        """
        if not crops:
            raise ValueError("crops list cannot be empty.")
        
        if not all(crop.manifest_idx < len(manifest) for crop in crops):
                raise IndexError("One or more crop.manifest_idx values are out of bounds.")
        
        self.manifest = manifest
        self.crops = list(crops)
    
    def __len__(self) -> int:
        return len(self.crops)
    
    def get_crop(self, crop_idx: int) -> Crop:
        """
        Get crop at logical index.
        
        :param crop_idx: Index into the crop collection.
        :return: Crop object.
        """
        if crop_idx < 0 or crop_idx >= len(self.crops):
            raise IndexError(
                f"crop_idx {crop_idx} out of range [0, {len(self.crops)})"
            )
        return self.crops[crop_idx]
    
    def to_config(self) -> Dict[str, Any]:
        """
        Serialize to dict (not including manifest, which is handled separately).
        """
        return {
            'crops': [crop.to_dict() for crop in self.crops],
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], manifest: DatasetManifest) -> 'CropManifest':
        """
        Deserialize from config dict.
        Requires manifest to be provided (since it's not stored in config).
        """
        crops = [Crop.from_dict(c) for c in config['crops']]
        return cls(crops, manifest)


@dataclass
class CropIndexState:
    """
    Mutable state tracking the currently active crop region.    
    Analogous to IndexState in manifest.py.
    """
    crop_collection: CropManifest
    last_crop_idx: Optional[int] = None
    _last_crop: Optional[Crop] = None
    
    def is_stale(self, crop_idx: int) -> bool:
        """
        Return True if the provided crop_idx differs from the last recorded one.
        
        :param crop_idx: Crop index to check.
        :return: True if stale (different or never set), False otherwise.
        """
        return self.last_crop_idx is None or crop_idx != self.last_crop_idx
    
    def get_and_update(self, crop_idx: int) -> Crop:
        """
        Get the Crop at crop_idx and update internal state if the index changed.
        
        :param crop_idx: Index into the crop collection.
        :return: Crop object.
        :raises IndexError: If crop_idx is out of range.
        """
        crop = self.crop_collection.get_crop(crop_idx)
        self.last_crop_idx = crop_idx
        self._last_crop = crop
        
        return crop
    
    def reset(self) -> None:
        """
        Clear all state
        """
        self.last_crop_idx = None
        self._last_crop = None


class CropFileState:
    """
    FileState behavior for crop-based datasets.    
    Wraps a FileState internally to handle image loading and caching.

    On update:
    1. Maps crop_idx to manifest_idx
    2. Calls internal FileState to load/retreive full image 
    3. Dynamically extracts the crop region from loaded image
    """
    
    def __init__(
        self,
        crop_collection: CropManifest,
        cache_capacity: Optional[int] = None,
    ):
        """
        Initialize CropFileState.
        
        :param crop_collection: CropManifest defining the crops.
        :param cache_capacity: Optional capacity for caching full images.
            - None: Default (caches up to n_channels images).
            - -1: Unbounded cache.
            - N > 0: LRU cache with capacity N.
        """
        self.manifest = crop_collection.manifest
        self._file_state = FileState(self.manifest, cache_capacity=cache_capacity)
        self.crop_state = CropIndexState(crop_collection)
        
        # Public interface (mirrors FileState)
        self.input_crop_raw: Optional[np.ndarray] = None
        self.target_crop_raw: Optional[np.ndarray] = None
        self.input_paths: List[Path] = []
        self.target_paths: List[Path] = []
    
    def update(
        self,
        crop_idx: int,
        input_keys: Sequence[str],
        target_keys: Sequence[str],
    ) -> None:
        """
        Load and crop images for a given crop_idx.
        
        :param crop_idx: Index into the crop collection.
        :param input_keys: Channel keys for input channels.
        :param target_keys: Channel keys for target channels.
        """
        crop = self.crop_state.get_and_update(crop_idx)
        self._file_state.update(crop.manifest_idx, input_keys, target_keys)

        self.input_crop_raw = self._extract_crop(self._file_state.input_image_raw, crop)
        self.target_crop_raw = self._extract_crop(self._file_state.target_image_raw, crop)

        self.input_paths = self._file_state.input_paths
        self.target_paths = self._file_state.target_paths
    
    def _extract_crop(self, image: np.ndarray, crop: Crop) -> Optional[np.ndarray]:
        """
        Extract spatial region from image array.
        
        :param image: Full image array, or None.
        :param crop: Crop specification with x, y, width, height.
        :return: Cropped array
        """

        if image.ndim != 3:  # (C, H, W)
            raise ValueError(
                f"Unexpected image shape: {image.shape}. Expected (C, H, W)."
            )

        x, y, w, h = crop.x, crop.y, crop.width, crop.height
        _, img_h, img_w = image.shape
        if x + w > img_w or y + h > img_h:
            raise ValueError(
                f"Crop ({x}, {y}, {w}, {h}) exceeds image bounds ({img_w}, {img_h})."
            )

        return image[:, y:y + h, x:x + w]
    
    @property
    def crop_info(self) -> Optional[Crop]:
        """
        Provides public access of the last crop information as the Crop dataclass.
        Intended to be wrapped by dataset implementation to retrieve
            crop details following each data access.
        """
        return self.crop_state._last_crop
    
    def reset(self) -> None:
        """Clear all state and caches."""
        self._file_state.reset()
        self.crop_state.reset()
        self.input_crop_raw, self.target_crop_raw = None, None
        self.input_paths, self.target_paths = [], []
