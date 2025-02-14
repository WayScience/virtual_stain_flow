from typing import Optional

import torch
from torch.utils.data import Dataset
from collections import OrderedDict

class CachedDataset(Dataset):
    """
    A patched dataset that caches data from dataset objects that 
    dynamically loads the data to reduce memory overhead during training
    """

    def __init__(
            self,
            dataset: Dataset,
            cache_size: Optional[int]=None,
            prefill_cache: bool=False,
            **kwargs 
            ):
        """
        Initialize the CachedDataset from a dataset object

        :param dataset: Dataset object to cache data from
        :type dataset: Dataset
        :param cache_size: Size of the cache, if None, the cache 
        size is set to the length of the dataset. 
        :type cache_size: int
        :param prefill_cache: Whether to prefill the cache
        :type prefill_cache: bool
        """

        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
        
        self.__dataset = dataset

        self.__cache_size = cache_size if cache_size is not None else len(dataset)
        self.__cache = OrderedDict()
        
        # cache for metadata
        self.__cache_input_names = OrderedDict()
        self.__cache_target_names = OrderedDict()

        # pointer to the current patch index
        self._current_idx = None

        if prefill_cache:
            self.cache()

    """
    Overriden methods for Dataset class
    """
    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.__dataset)
    
    def __getitem__(self, _idx: int):
        """
        Get the data from the dataset object at the given index
        If the data is not in the cache, load it from the dataset object and update the cache

        :param _idx: Index of the data to get
        :type _idx: int
        """
        self._current_idx = _idx

        if _idx in self.__cache:
            # cache hit
            return self.__cache[_idx]
        else:
            # cache miss, load from parent class method dynamically
            self._update_cache(_idx)
            return self.__cache[_idx]
        
    """
    Setters 
    """
    def set_cache_size(self, cache_size: int):
        """
        Set the cache size. Does not automatically repopulate the cache but 
        will pop the cache if the size is exceeded

        :param cache_size: Size of the cache
        :type cache_size: int
        """
        self.__cache_size = cache_size
        # pop the cache if the size is exceeded
        while len(self.__cache) > self.__cache_size:
            self._pop_cache()
        
    """
    Properties to remain accessible
    """
    @property
    def input_names(self):
        """
        Get the input names from the dataset object
        """
        if self._current_idx is not None:
            ## TODO: need to think over if this is at all necessary
            if self._current_idx in self.__cache_input_names:
                return self.__cache_input_names[self._current_idx]
            else:
                _ = self.__dataset[self._current_idx]
                return self.__dataset.input_names
        else:
            raise ValueError("No current index set")
        
    @property
    def target_names(self):
        """
        Get the target names from the dataset object
        """
        if self._current_idx is not None:
            ## TODO: need to think over if this is at all necessary
            if self._current_idx in self.__cache_target_names:
                return self.__cache_target_names[self._current_idx]
            else:
                _ = self.__dataset[self._current_idx]
                return self.__dataset.target_names
        else:
            raise ValueError("No current index set")
        
    @property
    def input_channel_keys(self):
        """
        Get the input channel keys from the dataset object
        """
        try:
            return self.__dataset.input_channel_keys
        except AttributeError:
            return None
    
    @property
    def target_channel_keys(self):
        """
        Get the target channel keys from the dataset object
        """
        try:
            return self.__dataset.target_channel_keys
        except AttributeError:
            return None
        
    @property
    def input_transform(self):
        """
        Get the input transform from the dataset object
        """
        return self.__dataset.input_transform
    
    @property
    def target_transform(self):
        """
        Get the target transform from the dataset object
        """
        return self.__dataset.target_transform
    
    @property
    def dataset(self):
        """
        Get the dataset object
        """
        return self.__dataset
    
    """
    Cache method
    """
    def cache(self):
        """
        Clears the current cache and re-populate cache with data from the dataset object
        Iteratively calls the update cache method on a sequence of indices to fill the cache
        """
        self._clear_cache()
        for _idx in range(min(self.__cache_size, len(self.__dataset))):
            self._update_cache(_idx)
    
    """
    Internal helper methods
    """

    def _update_cache(self, _idx: int):
        """
        Update the cache with data from the dataset object. 
        Calls the update cache metadata method as well to sync data and metadata
        Pops the cache if the cache size is exceeded on a first in, first out basis

        :param _idx: Index of the data to cache
        :type _idx: int
        """
        self._current_idx = _idx
        self.__cache[_idx] = self.__dataset[_idx]
        if len(self.__cache) >= self.__cache_size:
            self._pop_cache()
        self._update_cache_metadata(_idx)

    def _pop_cache(self):
        """
        Helper method to pop the cache on a first in, first out basis
        """
        self.__cache.popitem(last=False)

    def _update_cache_metadata(self, _idx: int):
        """
        Update the cache metadata with data from the dataset object
        Meant to be called by _update_cache method

        :param _idx: Index of the data to cache
        :type _idx: int
        """
        self.__cache_input_names[_idx] = self.__dataset.input_names
        self.__cache_target_names[_idx] = self.__dataset.target_names 

        if len(self.__cache_input_names) >= self.__cache_size:
            self._pop_cache_metadata()

    def _pop_cache_metadata(self):
        """
        Helper method to pop the cache metadata on a first in, first out basis
        """
        self.__cache_input_names.popitem(last=False)
        self.__cache_target_names.popitem(last=False)

    def _clear_cache(self):
        """
        Clear the cache and cache metadata
        """
        self.__cache.clear()
        self.__cache_input_names.clear()
        self.__cache_target_names.clear()