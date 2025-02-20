import logging
import pathlib
from random import randint
from typing import List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from albumentations import ImageOnlyTransform
from albumentations.core.composition import Compose
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """
    Image Dataset Class from pe2loaddata generated cellprofiler loaddata csv
    """
    def __init__(
            self,
            _loaddata_csv,
            _input_channel_keys: Optional[Union[str, List[str]]] = None, 
            _target_channel_keys: Optional[Union[str, List[str]]] = None,
            _input_transform: Optional[Union[Compose, ImageOnlyTransform]] = None,
            _target_transform: Optional[Union[Compose, ImageOnlyTransform]] = None,
            _PIL_image_mode: str = 'I;16',
            verbose: bool = False,
            file_column_prefix: str = 'FileName_',
            path_column_prefix: str = 'PathName_',
            check_exists: bool = False,
            **kwargs
    ):
        """
        Initialize the ImageDataset.
            
        :param _loaddata_csv: The dataframe or path to a csv file containing the image paths and labels. 
        :type _loaddata_csv: Union[pd.DataFrame, str]
        :param _input_channel_keys: Keys for input channels. Can be a single key or a list of keys.
        :type _input_channel_keys: Optional[Union[str, List[str]]]
        :param _target_channel_keys: Keys for target channels. Can be a single key or a list of keys.
        :type _target_channel_keys: Optional[Union[str, List[str]]]
        :param _input_transform: Transformations to apply to the input images.
        :type _input_transform: Optional[Union[Compose, ImageOnlyTransform]]
        :param _target_transform: Transformations to apply to the target images.
        :type _target_transform: Optional[Union[Compose, ImageOnlyTransform]]
        :param _PIL_image_mode: Mode to use when loading images with PIL. Default is 'I;16'.
        :type _PIL_image_mode: str
        :param kwargs: Additional keyword arguments.
        """

        self._initialize_logger(verbose)
        self._loaddata_df = self._load_loaddata(_loaddata_csv, **kwargs)
        self._channel_keys = list(self.__infer_channel_keys(file_column_prefix, path_column_prefix))

        # Initialize the cache for the input and target images
        self.__input_cache = {}
        self.__target_cache = {}
        self.__cache_image_id = None

        # Set input/target channels
        self.logger.debug("Setting input channel(s) ...")
        self._input_channel_keys = self.__check_channel_keys(_input_channel_keys)
        self.logger.debug("Setting target channel(s) ...")
        self._target_channel_keys = self.__check_channel_keys(_target_channel_keys)

        self.set_input_transform(_input_transform)
        self.set_target_transform(_target_transform)
            
        self._PIL_image_mode = _PIL_image_mode

        # Obtain image paths
        self.__image_paths = self._get_image_paths(
            file_column_prefix=file_column_prefix,
            path_column_prefix=path_column_prefix,
            check_exists=check_exists,
            **kwargs
        )
        # Index patches and images
        self.__iter_image_id = list(range(len(self.__image_paths)))

        # Initialize the current input and target names
        self.__current_input_names = None
        self.__current_target_names = None

    """
    Overridden Iterator functions
    """
    def __len__(self):
        return len(self.__image_paths)
    
    def __getitem__(self, _idx: int)->Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the input and target images
        :param _idx: The index of the image
        :type _idx: int
        :return: The input and target images, each with dimension [n_channels, height, width]
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """

        if _idx >= len(self) or _idx < 0:
            raise IndexError("Index out of bounds")            

        if self._input_channel_keys is None or self._target_channel_keys is None:
                raise ValueError("Input and target channel keys must be set to access data")
        
        image_id = self.__iter_image_id[_idx]
        self._cache_image(image_id)

        ## Retrieve relevant channels as specified by input and target channel keys and stack
        input_images = np.stack(
            [self.__input_cache[key] for key in self._input_channel_keys], 
            axis=0)
        target_images = np.stack(
            [self.__target_cache[key] for key in self._target_channel_keys],
            axis=0)
        
        ## Apply transform
        if self._input_transform:
            input_images = self._input_transform(image=input_images)['image']
        if self._target_transform:
            target_images = self._target_transform(image=target_images)['image']

        ## Cast to torch tensor and return
        return torch.from_numpy(input_images).float(), torch.from_numpy(target_images).float()
    
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

    def set_input_transform(self, _input_transform: Optional[Union[Compose, ImageOnlyTransform]]=None):
        """
        Set the input transform

        :param _input_transform: The input transform
        :type _input_transform: Compose or ImageOnlyTransform
        """
        # Check and set input/target transforms
        self.logger.debug("Setting input transform ...")
        if self.__check_transforms(_input_transform):
            self._input_transform = _input_transform
        

    def set_target_transform(self, _target_transform: Optional[Union[Compose, ImageOnlyTransform]]=None):
        """
        Set the target transform

        :param _target_transform: The target transform
        :type _target_transform: Compose or ImageOnlyTransform
        """
        # Check and set input/target transforms
        self.logger.debug("Setting target transform ...")
        if self.__check_transforms(_target_transform):
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
    Internal Helper functions
    """    
    def _initialize_logger(self, verbose: bool):
        """
        Initialize logger instance
        """
        self.logger = logging.getLogger(f"{__name__}.{id(self)}")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

    def _load_loaddata(
            self,
            _loaddata_csv: Union[pd.DataFrame, pathlib.Path],
            **kwargs
            ) -> pd.DataFrame:
        """
        Read loaddata csv file, also does type checking
        
        :param _loaddata_csv: The path to the loaddata CSV file or a DataFrame.
        :type _loaddata_csv: Union[pd.DataFrame, pathlib.Path]
        :param kwargs: Additional keyword arguments.
        :type kwargs: dict
        :raises ValueError: If no loaddata CSV is supplied or if the file type is not supported.
        :raises FileNotFoundError: If the specified file does not exist.
        :return: The loaded data as a DataFrame.
        :rtype: pd.DataFrame
        """
        
        if _loaddata_csv is None:
            raise ValueError("No loaddata csv supplied")
        elif isinstance(_loaddata_csv, pd.DataFrame):
            self.logger.debug("Dataframe supplied for loaddata_csv, using as is")
            return _loaddata_csv
        else:
            self.logger.debug("Loading loaddata csv from file")
            ## Convert string to pathlib Path 
            if not isinstance(_loaddata_csv, pathlib.Path):
                try:
                    _loaddata_csv = pathlib.Path(_loaddata_csv)
                except e:
                    raise e
                
            ## Handle file not exist
            if not _loaddata_csv.exists():
                raise FileNotFoundError(f"File {_loaddata_csv} not found")
            
            ## Determine file extension and load accordingly
            if _loaddata_csv.suffix == '.csv':
                return pd.read_csv(_loaddata_csv)
            elif _loaddata_csv.suffix == '.parquet':
                return pd.read_parquet(_loaddata_csv)
            else:
                raise ValueError(f"File type {_loaddata_csv.suffix} not supported")
            
    def __infer_channel_keys(
            self,
            file_column_prefix: str,
            path_column_prefix: str
                             ) -> set[str]:
        """
        Infer channel names from the columns of loaddata csv.
        This method identifies and returns the set of channel keys by comparing
        the columns in the dataframe that start with the specified file and path
        column prefixes. The channel keys are the suffixes of these columns after
        removing the prefixes.
        
        :param file_column_prefix: The prefix for columns that indicate filenames.
        :type file_column_prefix: str
        :param path_column_prefix: The prefix for columns that indicate paths.
        :type path_column_prefix: str
        :return: A set of channel keys inferred from the loaddata csv.
        :rtype: set[str]
        :raises ValueError: If no path or file columns are found, or if no matching
                            channel keys are found between file and path columns.
        """

        self.logger.debug("Inferring channel keys from loaddata csv")
        # Retrieve columns that indicate path and filename to image files
        file_columns = [col for col in self._loaddata_df.columns if col.startswith(file_column_prefix)]
        path_columns = [col for col in self._loaddata_df.columns if col.startswith(path_column_prefix)]

        if len(file_columns) == 0 or len(path_columns) == 0:
            raise ValueError('No path or file columns found in loaddata csv.')
        
        # Anything following the prefix should be the channel names
        file_channel_keys = [col.replace(file_column_prefix, '') for col in file_columns]
        path_channel_keys = [col.replace(path_column_prefix, '') for col in path_columns]
        channel_keys = set(file_channel_keys).intersection(set(path_channel_keys))

        if len(channel_keys) == 0:
            raise ValueError('No matching channel keys found between file and path columns.')
        
        self.logger.debug(f"Channel keys: {channel_keys} inferred from loaddata csv")
        
        return channel_keys
    
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
    
    def __check_transforms(
            self,
            transforms: Optional[Union[Compose, ImageOnlyTransform]]
    ) -> bool:
        """
        Checks if supplied iamge only transform is of valid type, if so, return True

        :param transforms: Transform
        :type transforms: ImageOnlyTransform or Compose of ImageOnlyTransforms
        :return: Boolean indicator of success
        :rtype: bool
        """
        if transforms is None:
            pass
        elif isinstance(transforms, Compose):
            pass
        elif isinstance(transforms, ImageOnlyTransform):
            pass
        else:
            raise TypeError('Invalid image transform type')
        
        return True
    
    def _get_image_paths(self, 
                          file_column_prefix: str, 
                          path_column_prefix: str, 
                          check_exists: bool = False,
                          **kwargs,
                          ) -> List[dict]:
        """
        From loaddata csv, extract the paths to all image channels cooresponding to each view/site

        :param check_exists: check if every individual image file exist and remove those that do not
        :type check_exists: bool        
        :return: A list of dictionaries containing the paths to the image channels
        :rtype: List[dict]
        """

        # Define helper function to get the image file paths from all channels 
        # in a single row of loaddata csv (single view/site), organized into a dict
        def get_channel_paths(row: pd.Series) -> Tuple[dict, bool]:

            missing = False

            multi_channel_paths = {}
            for channel_key in self._channel_keys:
                file_column = f"{file_column_prefix}{channel_key}"
                path_column = f"{path_column_prefix}{channel_key}"
                
                if file_column in row and path_column in row:
                    file = pathlib.Path(
                        row[path_column] 
                    ) / row[file_column]
                    if (not check_exists) or file.exists():
                        multi_channel_paths[channel_key] = file
                    else:
                        missing = True
                        
            return multi_channel_paths, missing
        
        image_paths = []
        self.logger.debug(
            "Extracting image channel paths of site/view and associated"\
                  "cell coordinates (if applicable) from loaddata csv")
        
        for _, row in self._loaddata_df.iterrows():
            multi_channel_paths, missing = get_channel_paths(row)
            if not missing:
                image_paths.append(multi_channel_paths)

        self.logger.debug(f"Extracted images of all input and target channels for {len(image_paths)} unique sites/view")
        return image_paths
    
    def _read_convert_image(self, _image_path: pathlib.Path)->np.ndarray:
        """
        Read and convert the image to a numpy array
        
        :param _image_path: The path to the image
        :type _image_path: pathlib.Path
        :return: The image as a numpy array
        :rtype: np.ndarray
        """
        return np.array(Image.open(_image_path).convert(self._PIL_image_mode))
    
    def _cache_image(self, _id: int)->None:
        """
        Determines if cached images need to be updated and updates the self.__input_cache and self.__target_cache
        Meant to be called by __getitem__ method in dynamic patch cropping

        :param _id: The index of the image
        :type _id: int
        :return: None
        :rtype: None 
        """

        if self.__cache_image_id is None or self.__cache_image_id != _id:
            self.__cache_image_id = _id
            self.__input_cache = {}
            self.__target_cache = {}

            ## Update target and input names (which are just file path(s))
            self.__current_input_names = [self.__image_paths[_id][key] for key in self._input_channel_keys]
            self.__current_target_names = [self.__image_paths[_id][key] for key in self._target_channel_keys]

            for key in self._input_channel_keys:
                self.__input_cache[key] = self._read_convert_image(self.__image_paths[_id][key])
            for key in self._target_channel_keys:
                self.__target_cache[key] = self._read_convert_image(self.__image_paths[_id][key])
        else:
            # No need to update the cache
            pass
        
        return None