"""
dataset_view.py

Core Dataset Infrastructure for lazy, dynamic image loading from files.

Classes:
- DatasetManifest: Immutable manifest holding file index and image mode. 
    Manages associated files for multiple channels corresponding to the same
    field of view (FOV) or sample. 
- IndexState: 'Memory' of the Dataset's last accessed index. 
    Intended to be used by realized Dataset classes to track last accessed index,
    and corresponding metadata.
- FileState: 'Memory' of the Dataset's last loaded image data and paths.
    Intended to be used by the realized Dataset classes as the backend for
    lazy loading and optional caching of image files to balance memory 
    and access speed.
"""

from __future__ import annotations
from pathlib import Path, PurePath
from collections import OrderedDict
from typing import List, Optional, Sequence, Dict, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from PIL import Image


@dataclass(frozen=True)
class DatasetManifest:
    """
    Minimal manifest defining the immutable component of a dataset.
    Simply holds a file index containing pathlike where each row corresponds to a
    view/fov/sample and columns corresponds to channels associated with that entry,
    and a PIL image mode to use when reading images.

    The DatasetManifest keeps two attributes defining itself:
    - file_index: pd.DataFrame
        Each row corresponds to a view/fov/sample.
        Each column corresponds to a channel associated with that entry.
        Each entry is a path-like (str or Path) to an image file.
        The DataFrame must be non-empty and contain only path-like entries.
    - pil_image_mode: str
        The PIL image mode to use when reading images.
        Must be one of the valid PIL modes (see Image.MODES).
        Default is "I;16" for 16-bit grayscale images.
    """
    file_index: pd.DataFrame
    pil_image_mode: str = "I;16"

    def __post_init__(self) -> None:
        if not isinstance(self.file_index, pd.DataFrame) or self.file_index.empty:
            raise ValueError("file_index must be a non-empty DataFrame.")
        

        if not isinstance(self.pil_image_mode, str):
            raise ValueError("Expected pil_image_mode to be a string, "
                             f"got {type(self.pil_image_mode)} instead.")
        if self.pil_image_mode not in Image.MODES:
            raise ValueError(f"Invalid pil_image_mode: {self.pil_image_mode}. "
                             f"Must be one of {Image.MODES}.")
        
        bad_types = set()
        for _, row in self.file_index.iterrows():
            for x in row:
                if not isinstance(x, (Path, PurePath, str)):
                    bad_types.add(type(x).__name__)
        if bad_types:
            raise TypeError(f"file_index has non-path-like entries: {bad_types}")
    
    def __len__(self) -> int:
        """
        Return the number of rows in the file index.
        """
        return len(self.file_index)
    
    @property
    def n_channels(self) -> int:
        """
        Return the number of columns in the file index.
        """
        return len(self.file_index.columns)
    
    @property
    def channel_keys(self) -> List[str]:
        """
        Return the list of channel keys (column names) in the file index.
        """
        return list(self.file_index.columns)

    def get_paths_for_keys(self, idx: int, keys: Sequence[str]) -> List[Path]:
        """
        Return file paths (ordered to match keys) for a given row index.
        Can be used by the FileState to obtain a list of paths to use as cache keys.
        """
        row = self.file_index.iloc[idx]
        out = []
        for k in (keys or []):
            v = row[k]
            # v is guaranteed path-like after __post_init__
            p = v if isinstance(v, Path) else Path(v)
            out.append(p)
        return out

    def read_image(self, path: Path) -> np.ndarray:
        """
        Read a single image and convert to configured PIL mode, then to ndarray.
        Return as ndarray of shape (H, W) for grayscale or (H, W, K) 
            for multi-channel.
        Can be used by the FileState to load image and cache with key path
            from get_paths_for_keys.
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        try:
            with Image.open(path) as img:
                img = img.convert(self.pil_image_mode)
                arr = np.asarray(img)
        except Exception as e:
            raise RuntimeError(f"Error reading image from {path}: {e}") from e

        if arr.ndim not in (2, 3):
            raise ValueError(f"Unsupported image shape {arr.shape} from {path}")
        return arr
    
    def to_config(self) -> Dict[str, Any]:
        """Serialize to dict"""
        return { # the file_index is automatically serializable by definition
            'file_index': self.file_index.to_dict(orient='records'),
            'pil_image_mode': self.pil_image_mode,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> DatasetManifest:
        """Deserialize from config dict."""
        return cls(
            file_index=pd.DataFrame(config['file_index']),
            pil_image_mode=config.get('pil_image_mode', 'I;16')
        )


@dataclass
class IndexState:
    """
    Minimal tracker for index state to associate with a DatasetManifest.
    More orderly than just maintaining dataset_class._index.
    Intended to be utilized by dataset classes as internal memory
    for last loaded image index to allow for retrieval of relevant
    metadata.

    Has a single attribute `last` for tracking the last index.
    """
    last: Optional[int] = None

    def reset(self) -> None:
        """
        Resets the index state's single attribute.
        Intended to be called by dataset class when the dataset length
        is modified (e.g. subsetted).
        """
        self.last = None

    def is_stale(self, idx: Optional[int]) -> bool:
        """
        Return True if the provided index is different 
            from the last recorded index.
        If idx is None, always return True.
        If last is None, always return True.

        Useful for the dataset class that keeps track of
            metadata associated with the last retrieved 
            images as updatable attributes.
        """
        return self.last is None or idx is None or self.last != idx

    def update(self, idx: int) -> None:
        """
        Update the last index if the provided index is stale.

        Intended to be called by dataset class after retrieving
            images for a given index, to record that the
            new index state. 
        """
        if self.is_stale(idx):
            self.last = idx

@dataclass
class FileState:
    """
    Lazy file state tracker to wrap a DatasetManifest.
    This tracks the current input/target image np arrays and paths and 
        determines when to invoke the DatasetManifest to read images, 
        in a lazy manner minimizing redundant reads. 
    Also centralizes the LRU caching logic evicting the least recently used
        images when a capacity is set.
    """
    manifest: DatasetManifest
    # Current realized state (order corresponds to last request)
    input_paths: List[Path] = field(default_factory=list)
    target_paths: List[Path] = field(default_factory=list)
    input_image_raw: Optional[np.ndarray] = None  # (C, H, W) or (C, H, W, K)
    target_image_raw: Optional[np.ndarray] = None

    # Global cache of decoded channels by path (optional LRU)
    cache_capacity: Optional[int] = None
    _cache: "OrderedDict[Path, np.ndarray]" = field(
        default_factory=OrderedDict, init=False)
    
    def __post_init__(self) -> None:
        if self.cache_capacity is None:
            self.cache_capacity = self.manifest.n_channels
        # Validate sentinel
        if self.cache_capacity < -1:
            raise ValueError("cache_capacity must be -1 (unbounded) or >= 0.")
        # Optional: guard against 0 (almost always a footgun)
        if self.cache_capacity == 0:
            raise ValueError("cache_capacity=0 disables caching; set 1+ or -1.")

    def reset(self) -> None:
        self.input_paths.clear()
        self.target_paths.clear()
        self.input_image_raw = None
        self.target_image_raw = None

    # ---- internal cache helpers ----
    def _touch_cache(self, path: Path, arr: np.ndarray) -> None:
        
        # unbounded cache, since we never evict cached item
        # and the manifest defining all image files constituting
        # the dataset is immutable, eventually all images
        # will be cached and memory usage will persist until
        # process end.
        if self.cache_capacity == -1:
            self._cache[path] = arr
            return
        
        # bounded LRU
        cap = self.cache_capacity
        if path in self._cache:
            # LRU logic: move cache hit to end (most recently used position)
            self._cache.move_to_end(path)
        self._cache[path] = arr
        while len(self._cache) > cap:
            self._cache.popitem(last=False)  # evict least recently used

    def _get_or_load(self, path: Path) -> np.ndarray:
        arr = self._cache.get(path)
        if arr is None:
            arr = self.manifest.read_image(path)
            self._touch_cache(path, arr)
        else:
            # refresh LRU position if using capacity
            if self.cache_capacity is not None and self.cache_capacity > 0:
                self._cache.move_to_end(path)
        return arr

    def _ensure_same_spatial_shape(self, arrays: List[np.ndarray], paths: List[Path]) -> None:
        if not arrays:
            return
        for a, p in zip(arrays, paths):
            if a.shape[:2] != arrays[0].shape[:2]:
                raise ValueError(
                    f"Spatial shape mismatch: {p} has {a.shape[:2]}, expected {arrays[0].shape[:2]}"
                )

    def _stack_channels(self, arrays: List[np.ndarray]) -> np.ndarray:
        """
        Stack as (C, H, W) for 2D planes, 
        or (C, H, W, K) if planes are multi-channel (e.g., RGB).
        """
        if not arrays:
            # Return empty with zero channels? Usually caller ensures non-empty.
            return np.empty((0,))
        sample = arrays[0]
        if sample.ndim == 2:
            # (H, W) -> (C, H, W)
            return np.stack(arrays, axis=0)
        if sample.ndim == 3:
            # (H, W, K) -> (C, H, W, K)
            return np.stack(arrays, axis=0)
        
        raise ValueError(f"Unsupported per-channel array ndim={sample.ndim}")

    # ---- public update API ----
    def update(
        self,
        idx: int,
        input_keys: Sequence[str],
        target_keys: Sequence[str],
    ) -> None:
        """
        Lazily realize current input/target stacks for an index and channel config.
        Only load missing paths; reuse and reorder already-cached ones.
        """
        # Resolve desired paths (order matters)
        desired_input_paths = self.manifest.get_paths_for_keys(idx, input_keys)
        desired_target_paths = self.manifest.get_paths_for_keys(idx, target_keys)

        # Load/reuse inputs
        input_arrays = [self._get_or_load(p) for p in desired_input_paths]

        # Load/reuse targets
        target_arrays = [self._get_or_load(p) for p in desired_target_paths]

        # Validate shapes before stacking (safer)
        self._ensure_same_spatial_shape(input_arrays, desired_input_paths)
        self._ensure_same_spatial_shape(target_arrays, desired_target_paths)

        # Stack in requested order
        self.input_image_raw = self._stack_channels(input_arrays) if input_arrays else None
        self.target_image_raw = self._stack_channels(target_arrays) if target_arrays else None

        # Record the realized order
        self.input_paths = desired_input_paths
        self.target_paths = desired_target_paths

    def to_config(self) -> Dict[str, Any]:
        """
        Serialize to dict
        Responsible for also serializing the underlying manifest.
        """
        return {
            'manifest': self.manifest.to_config(),
            'cache_capacity': self.cache_capacity
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> FileState:
        """
        Deserialize from config dict.
        Responsible for also deserializing the underlying manifest.
        """
        return cls(
            manifest=DatasetManifest.from_config(config['manifest']),
            cache_capacity=config.get('cache_capacity', None)
        )
