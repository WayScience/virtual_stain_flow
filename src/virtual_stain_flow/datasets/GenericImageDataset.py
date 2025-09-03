import logging
import pathlib
import re
from collections import defaultdict
from typing import List, Optional, Union, Tuple, Dict

import numpy as np
import torch
from PIL import Image
from albumentations import ImageOnlyTransform
from albumentations.core.composition import Compose
from torch.utils.data import Dataset


class GenericImageDataset(Dataset):
    """
    A generic image dataset that automatically associates images under a supplied path
    with sites and channels based on two separate regex patterns for site and channel detection.
    """

    def __init__(
            self,
            image_dir: Union[str, pathlib.Path],
            site_pattern: str,
            channel_pattern: str,
            _input_channel_keys: Optional[Union[str, List[str]]] = None,
            _target_channel_keys: Optional[Union[str, List[str]]] = None,
            _input_transform: Optional[Union[Compose, ImageOnlyTransform]] = None,
            _target_transform: Optional[Union[Compose, ImageOnlyTransform]] = None,
            _PIL_image_mode: str = 'I;16',
            verbose: bool = False,
            check_exists: bool = True,
            **kwargs
    ):
        """
        Initialize the dataset.

        :param image_dir: Directory containing the images.
        :param site_pattern: Regex pattern to extract site identifiers.
        :param channel_pattern: Regex pattern to extract channel identifiers.
        :param _input_channel_keys: List of channel names to use as inputs.
        :param _target_channel_keys: List of channel names to use as targets.
        :param _input_transform: Transformations to apply to input images.
        :param _target_transform: Transformations to apply to target images.
        :param _PIL_image_mode: Mode for loading images.
        :param check_exists: Whether to check if all referenced image files exist.
        """

        self._initialize_logger(verbose)
        self.image_dir = pathlib.Path(image_dir).resolve()
        self.site_pattern = re.compile(site_pattern)
        self.channel_pattern = re.compile(channel_pattern)
        self._PIL_image_mode = _PIL_image_mode

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory {self.image_dir} not found")

        # Parse images and organize by site
        self._channel_keys = []
        self.__image_paths = self._get_image_paths(check_exists)

        # Set input and target channel keys
        self._input_channel_keys = self.__check_channel_keys(_input_channel_keys)
        self._target_channel_keys = self.__check_channel_keys(_target_channel_keys)

        self.set_input_transform(_input_transform)
        self.set_target_transform(_target_transform)

        # Index patches and images
        self.__iter_image_id = list(range(len(self.__image_paths)))

        # Initialize cache
        self.__input_cache = {}
        self.__target_cache = {}
        self.__cache_image_id = None

        # Initialize the current input and target names
        self.__current_input_names = None
        self.__current_target_names = None

    """
    Properties
    """

    @property
    def image_paths(self):
        return self.__image_paths

    @property
    def input_transform(self):
        return self._input_transform
    
    @property
    def target_transform(self):
        return self._target_transform
    
    @property
    def input_channel_keys(self):
        return self._input_channel_keys
    
    @property
    def target_channel_keys(self):
        return self._target_channel_keys
    @property
    def input_names(self):
        return self.__current_input_names
    
    @property
    def target_names(self):
        return self.__current_target_names
    
    """
    Setters
    """

    def set_input_transform(self, _input_transform: Optional[Union[Compose, ImageOnlyTransform]] = None):
        """Sets the input image transform."""
        self.logger.debug("Setting input transform ...")
        self._input_transform = _input_transform

    def set_target_transform(self, _target_transform: Optional[Union[Compose, ImageOnlyTransform]] = None):
        """Sets the target image transform."""
        self.logger.debug("Setting target transform ...")
        self._target_transform = _target_transform

    def set_input_channel_keys(self, _input_channel_keys: Union[str, List[str]]):
        """
        Set the input channel keys

        :param _input_channel_keys: The input channel keys
        :type _input_channel_keys: str or list of str
        """
        self._input_channel_keys = self.__check_channel_keys(_input_channel_keys)
        self.logger.debug(f"Set input channel(s) as {self._input_channel_keys}")

        # clear the cache
        self.__cache_image_id = None

    def set_target_channel_keys(self, _target_channel_keys: Union[str, List[str]]):
        """
        Set the target channel keys

        :param _target_channel_keys: The target channel keys
        :type _target_channel_keys: str or list of str
        """
        self._target_channel_keys = self.__check_channel_keys(_target_channel_keys)
        self.logger.debug(f"Set target channel(s) as {self._target_channel_keys}")

        # clear the cache
        self.__cache_image_id = None

    """
    Logging and Debugging
    """

    def _initialize_logger(self, verbose: bool):
        """Initializes the logger."""
        self.logger = logging.getLogger(f"{__name__}.{id(self)}")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    
    """
    Internal helper functions
    """

    def _get_image_paths(self, check_exists: bool):

        # sets for all unique sites and channels
        sites = set()
        channels = set()
        image_files = list(self.image_dir.glob("*"))

        site_to_channels = defaultdict(dict) 
        for file in image_files:
            site_match = self.site_pattern.search(file.name)
            try:
                site = site_match.group(1)
            except:
                continue
            sites.add(site)

            channel_match = self.channel_pattern.search(file.name)
            try:
                channel = channel_match.group(1)
            except:
                continue
            channels.add(channel)

            site_to_channels[site][channel] = file

        # format as list of dicts
        image_paths = []
        for site, channel_to_file in site_to_channels.items():
            ## Keep only sites with all channels
            if all([c in site_to_channels[site] for c in channels]):
                if check_exists and not all(path.exists() for path in channel_to_file.values()):
                    continue            
                image_paths.append(channel_to_file)

        self.logger.debug(f"Channel keys: {channels} detected")
        self._channel_keys = list(channels)

        return image_paths

    def __len__(self):
        return len(self.__image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the input and target images for a given index.

        :param idx: The index of the image.
        :return: Tuple of input and target images as tensors.
        """
        if idx >= len(self) or idx < 0:
            raise IndexError("Index out of bounds")

        site_id = self.__iter_image_id[idx]
        self._cache_image(site_id)

        # Stack input and target images
        input_images = np.stack([self.__input_cache[key] for key in self._input_channel_keys], axis=0)
        target_images = np.stack([self.__target_cache[key] for key in self._target_channel_keys], axis=0)

        # Apply transformations
        if self._input_transform:
            input_images = self._input_transform(image=input_images)['image']
        if self._target_transform:
            target_images = self._target_transform(image=target_images)['image']

        return torch.from_numpy(input_images).float(), torch.from_numpy(target_images).float()

    def _cache_image(self, site_id: str) -> None:
        """
        Loads and caches images for a given site ID.

        :param site_id: The site ID.
        """
        if self.__cache_image_id != site_id:
            self.__cache_image_id = site_id
            self.__input_cache = {}
            self.__target_cache = {}

            ## Update target and input names (which are just file path(s))
            self.__current_input_names = [self.__image_paths[site_id][key] for key in self._input_channel_keys]
            self.__current_target_names = [self.__image_paths[site_id][key] for key in self._target_channel_keys]

            for key in self._input_channel_keys:
                self.__input_cache[key] = self._read_convert_image(self.__image_paths[site_id][key])
            for key in self._target_channel_keys:
                self.__target_cache[key] = self._read_convert_image(self.__image_paths[site_id][key])

    def _read_convert_image(self, image_path: pathlib.Path) -> np.ndarray:
        """
        Reads and converts an image to a numpy array.

        :param image_path: The image file path.
        :return: The image as a numpy array.
        """
        return np.array(Image.open(image_path).convert(self._PIL_image_mode))
    
    def __check_channel_keys(
            self,
            channel_keys: Optional[Union[str, List[str]]]
    ) -> List[str]:
        """
        Checks user supplied channel key against the inferred ones from the file

        :param channel_keys: user supplied list or single object of string channel keys
        :type channel_keys: string or list of strings
        """
        if channel_keys is None:
            self.logger.debug("No channel keys specified, skip")
            return None
        elif isinstance(channel_keys, str):
            channel_keys = [channel_keys]
        elif isinstance(channel_keys, list):
            if not all([isinstance(key, str) for key in channel_keys]):
                raise ValueError('Channel keys must be a string or a list of strings.')
        else:
            raise ValueError('Channel keys must be a string or a list of strings.')
        
        ## Check supplied channel keys against inferred ones
        filtered_channel_keys = []
        for key in channel_keys:
            if not key in self._channel_keys:
                self.logger.debug(
                    f"ignoring channel key {key} as it does not match loaddata csv file"
                )
            else:
                filtered_channel_keys.append(key)

        if len(filtered_channel_keys) == 0:
            raise ValueError(f'None of the supplied channel keys match the loaddata csv file')
            
        return filtered_channel_keys