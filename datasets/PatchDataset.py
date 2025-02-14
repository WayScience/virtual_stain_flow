import logging
import math
import pathlib
import random
from random import randint
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pyarrow import parquet as pq
import torch

from .ImageDataset import ImageDataset

class PatchDataset(ImageDataset):
    """
    Patch Dataset Class from pe2loaddata generated cellprofiler loaddata csv and sc features
    """
    def __init__(
            self,
            _sc_feature: Optional[pd.DataFrame | pathlib.Path] = None,
            patch_size: int = 64,
            patch_generation_method: str = 'random',
            patch_generation_random_seed: Optional[int] = None,
            patch_generation_max_attempts: int = 1_000,
            n_expected_patches_per_img: int = 5,
            candidate_x: str = 'Metadata_Cells_Location_Center_X',
            candidate_y: str = 'Metadata_Cells_Location_Center_Y',
            **kwargs
    ):

        self._patch_size = patch_size
        self._merge_fields = None
        self._x_col = None
        self._y_col = None
        self.__cell_coords = []

        # This intializes the channels keys, loaddata loading, image mode and 
        # the overriden methods further merge the loaddata with sc features
        super().__init__(_sc_feature=_sc_feature,
                          candidate_x=candidate_x,
                          candidate_y=candidate_y,
                          **kwargs)
        
        
        self.__patch_coords = self._generate_patches(
            _patch_size=self._patch_size,
            patch_generation_method=patch_generation_method,
            patch_generation_random_seed=patch_generation_random_seed, 
            n_expected_patches_per_img=n_expected_patches_per_img, 
            max_attempts=patch_generation_max_attempts,
            consistent_img_size=kwargs.get('consistent_img_size', True),
        )
        
        # Index patches and images
        self.__iter_image_id = []
        self.__iter_patch_id = []
        for i, _patch_coords in enumerate(self.__patch_coords):
            for j, _ in enumerate(_patch_coords):
                self.__iter_image_id.append(i)
                self.__iter_patch_id.append(j)

        # Initialize the cache for the input and target images
        self.__input_cache = {}
        self.__target_cache = {}
        self.__cache_image_id = None        

        # Initialize the current input and target names and patch coordinates
        self.__current_input_names = None
        self.__current_target_names = None
        self.__current_patch_coords = None

    """
    Overridden Iterator functions
    """
    def __len__(self):
        return len(self.__patch_coords)
    
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
        patch_id = self.__iter_patch_id[_idx]
        self.__current_patch_coords = self.__patch_coords[image_id][patch_id]

        self._cache_image(image_id)

        ## Retrieve relevant channels as specified by input and target channel keys and stack
        ## And further crop the patches with __current_patch_coords
        input_images = np.stack(
            [self._ImageDataset__input_cache[key][
                self.__current_patch_coords[1]:self.__current_patch_coords[1] + self._patch_size,
                self.__current_patch_coords[0]:self.__current_patch_coords[0] + self._patch_size
            ] for key in self._input_channel_keys], 
            axis=0)
        target_images = np.stack(
            [self._ImageDataset__target_cache[key][
                self.__current_patch_coords[1]:self.__current_patch_coords[1] + self._patch_size,
                self.__current_patch_coords[0]:self.__current_patch_coords[0] + self._patch_size
            ] for key in self._target_channel_keys],
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
    def patch_size(self):
        return self._patch_size
    
    @property
    def cell_coords(self):
        return self.__cell_coords
    
    @property
    def all_patch_coords(self):
        return self.__patch_coords
    
    @property
    def patch_coords(self):
        return self.__current_patch_coords
    
    """
    Internal Helper functions
    """

    def __preload_sc_feature(self, 
                             _sc_feature: pd.DataFrame | pathlib.Path | None) -> List[str]:
        """
        Preload the sc feature dataframe/parquet file limiting only to the column headers
        If a dataframe is supplied, use as is
        If a path to a csv file is supplied, load only the header row
        If a path to a parquet file is supplied, load only the parquet schema name
        :param _sc_feature: The path to a csv file containing the cell profiler sc features
        :type _sc_feature: str or pathlib.Path
        """

        if _sc_feature is None:
            # No sc feature supplied, cell coordinates not available, patch generation will fixed random
            self.logger.debug("No sc feature supplied, patch generation will be random")
            self._patch_generation_method = 'random'
            return None

        elif isinstance(_sc_feature, pd.DataFrame):
            self.logger.debug("Dataframe supplied for sc_feature, using as is")
            return _sc_feature.columns.tolist()
        
        else:
            self.logger.debug("Preloading sc feature from file")
            if not isinstance(_sc_feature, pathlib.Path):
                try:
                    _sc_feature = pathlib.Path(_sc_feature)
                except e:
                    raise e

            if not _sc_feature.exists():
                raise FileNotFoundError(f"File {_sc_feature} not found")
            
            if _sc_feature.suffix == '.csv':
                self.logger.debug("Preloading sc feature from csv file")
                return pd.read_csv(_sc_feature, nrows=0).columns.tolist()
            elif _sc_feature.suffix == '.parquet':
                pq_file = pq.ParquetFile(_sc_feature)
                return pq_file.schema.names
            else:
                raise ValueError(f"File type {_sc_feature.suffix} not supported")
            
    def __infer_merge_fields(self,
                             _loaddata_df,
                             _sc_col_names: List[str]
                             ) -> List[str] | None:
        """
        Find the columns that are common to both dataframes to use in an inner join
        Mean to be used to associate loaddata_csv with sc features

        :param loaddata_csv: The first dataframe
        :type loaddata_csv: pd.DataFrame
        :param sc_feature: The second dataframe
        :type sc_feature: pd.DataFrame
        :return: The columns that are common to both dataframes
        :rtype: List[str]
        """
        if _sc_col_names is None:
            return None

        self.logger.debug("Both loaddata_csv and sc_feature supplied, " \
                          "inferring merge fields to associate the two dataframes")        
        merge_fields = list(set(_loaddata_df.columns).intersection(set(_sc_col_names)))
        if len(merge_fields) == 0:
            raise ValueError("No common columns found between loaddata_csv and sc_feature")
        self.logger.debug(f"Merge fields inferred: {merge_fields}")

        return merge_fields
    
    def __infer_x_y_columns(self,
                            _loaddata_df,
                            _sc_col_names: List[str],
                            candidate_x: str, 
                            candidate_y: str) -> Tuple[str, str]:
        """
        Infer the columns that contain the x and y coordinates of the patches
        :param candidate_x: The candidate column name for the x coordinates
        :type candidate_x: str
        :param candidate_y: The candidate column name for the y coordinates
        :type candidate_y: str
        :return: The columns that contain the x and y coordinates of the patches
        :rtype: Tuple[str, str]
        """

        if _loaddata_df is None:
            return None, None

        if candidate_x not in _sc_col_names or candidate_y not in _sc_col_names:
            self.logger.debug(f"X and Y columns {candidate_x}, {candidate_y} not detected in sc_features, attempting to infer from sc_feature dataframe")

            # infer the columns that contain the x and y coordinates
            x_col_candidates = [col for col in _sc_col_names if col.lower().endswith('_x')]
            y_col_candidates = [col for col in _sc_col_names if col.lower().endswith('_y')]

            if len(x_col_candidates) == 0 or len(y_col_candidates) == 0:
                raise ValueError("No columns found containing the x and y coordinates")
            else:
                # sort x_col and y_col candidates
                x_col_candidates.sort()
                y_col_candidates.sort()
                x_col_detected = x_col_candidates[0]
                y_col_detected = y_col_candidates[0]
                self.logger.debug(f"X and Y columns {x_col_detected}, {y_col_detected} detected in sc_feature dataframe, using as the coordinates for cell centers")
                return x_col_detected, y_col_detected
        else:
            self.logger.debug(f"X and Y columns {candidate_x}, {candidate_y} detected in sc_feature dataframe, using as the coordinates for cell centers")
            return candidate_x, candidate_y
        
    def __load_sc_feature(self, 
                          _sc_feature: pd.DataFrame | pathlib.Path | None,
                          _merge_fields: List[str],
                          _x_col: str,
                          _y_col: str
                          ) -> pd.DataFrame | None:
        """
        Load the actual sc feature as a dataframe, limiting the columns to the merge fields and the x and y coordinates
        :param _sc_feature: The path to a csv file containing the cell profiler sc features
        :type _sc_feature: str or pathlib.Path
        :return: The dataframe containing the cell profiler sc features
        :rtype: pd.DataFrame
        """

        if _sc_feature is None:
            return None
        elif isinstance(_sc_feature, pd.DataFrame):
            self.logger.debug("Dataframe supplied for sc_feature, using as is")
            return _sc_feature
        else:
            self.logger.debug("Loading sc feature from file")
            if not isinstance(_sc_feature, pathlib.Path):
                try:
                    _sc_feature = pathlib.Path(_sc_feature)
                except e:
                    raise e

            if not _sc_feature.exists():
                raise FileNotFoundError(f"File {_sc_feature} not found")
            
            if _sc_feature.suffix == '.csv':
                return pd.read_csv(_sc_feature,
                                   usecols=_merge_fields + [_x_col, _y_col])
            elif _sc_feature.suffix == '.parquet':
                return pq.read_table(_sc_feature, columns=_merge_fields + [_x_col, _y_col]).to_pandas()
            else:
                raise ValueError(f"File type {_sc_feature.suffix} not supported")
            
    """
    Overriden parent class helper functions
    """
            
    def _load_loaddata(self, 
                       _loaddata_csv: pd.DataFrame | pathlib.Path,
                       _sc_feature: Optional[pd.DataFrame | pathlib.Path],
                       candidate_x: str,
                       candidate_y: str,
                       ):
        """
        Overridden function from parent class
        Calls the parent class to get the loaddata df and then merges it with sc_feature
        """

        ## First calls the parent class to get the full loaddata df
        loaddata_df = super()._load_loaddata(_loaddata_csv)

        ## Obtain column names of sc features first to avoid needing to read in the whole
        ## Parquet file as only a very small number of columns are needed
        sc_feature_col_names = self.__preload_sc_feature(_sc_feature)

        ## Infer columns corresponding to x and y coordinates to cells
        self._x_col, self._y_col = self.__infer_x_y_columns(
            loaddata_df, sc_feature_col_names, candidate_x, candidate_y)

        ## Infer merge fields between the sc features and loaddata
        self._merge_fields = self.__infer_merge_fields(loaddata_df, sc_feature_col_names)        

        ## Load sc features
        sc_feature_df = self.__load_sc_feature(
            _sc_feature, self._merge_fields, self._x_col, self._y_col)

        ## Perform the merge and return the merged dataframe (which is loaddata plus columns for x and y coordinates)
        return loaddata_df.merge(sc_feature_df, on=self._merge_fields, how='inner')
    
    def _get_image_paths(self, 
                          file_column_prefix: str, 
                          path_column_prefix: str, 
                          check_exists: bool = False,
                          **kwargs,
                          ) -> List[dict]:
        """
        Overridden function
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
        
        # Define helper function to get the coordinates associated with a condition
        def get_associated_coords(group):

            try:
                return group.loc[:, [self._x_col, self._y_col]].values
            except:
                return None
        
        image_paths = []
        cell_coords = []
        self.logger.debug(
            "Extracting image channel paths of site/view and associated"\
                  "cell coordinates (if applicable) from loaddata csv")
        
        n_cells = 0
        grouped = self._loaddata_df.groupby(self._merge_fields)
        for _, group in grouped:            

            _, row = next(group.iterrows())
            multi_channel_paths, missing = get_channel_paths(row)
            if not missing:
                image_paths.append(multi_channel_paths)
                coords = get_associated_coords(group)
                n_cells += len(coords)
                cell_coords.append(coords)        

        self.logger.debug("Extracted images of all input and target channels for " \
                          f"{len(image_paths)} unique sites/view and {n_cells} cells")
        
        self.__cell_coords = cell_coords

        return image_paths

    """
    Patch generation helper functions 
    """

    def _generate_patches(self,
                          _patch_size: int,
                          patch_generation_method: str,
                          patch_generation_random_seed: int, 
                          n_expected_patches_per_img=5, 
                          max_attempts=1_000,
                          consistent_img_size=True,
                          )->None:
        """
        Generate patches for each image in the dataset
        :param patch_generation_method: The method to use for generating patches
        :type patch_generation_method: str
        :param patch_generation_random_seed: The random seed to use for patch generation
        :type patch_generation_random_seed: int
        :param consistent_img_size: Whether the images are consistent in size.
        If True, the patch generation will be based on the size of the first input channel of first image
        If False, the patch generation will be based on the size of each image
        :type consistent_img_size: bool
        :param n_expected_patches_per_img: The number of patches to generate per image
        :type n_expected_patches_per_img: int
        :param max_attempts: The maximum number of attempts to generate a patch
        :type max_attempts: int
        :return: The coordinates of the patches
        :rtype: List[List[Tuple[int
        """
        if patch_generation_method == 'random_cell':
            if self.__cell_coords is None:
                raise ValueError("Cell coordinates not available for generating cell containing patches")
            else:
                self.logger.debug("Generating patches that contain cells")
                def patch_fn(image_size, patch_size, cell_coords, n_expected_patches_per_img, max_attempts):
                    return self.__generate_cell_containing_patches_unit(
                        image_size, patch_size, cell_coords, n_expected_patches_per_img, max_attempts)
                pass
        elif patch_generation_method == 'random':
            self.logger.debug("Generating random patches")
            def patch_fn(image_size, patch_size, cell_coords, n_expected_patches_per_img, max_attempts):
                # cell_coords is not used in this case
                return self.__generate_random_patches_unit(image_size, patch_size, n_expected_patches_per_img, max_attempts)
            pass
        else:
            raise ValueError("Patch generation method not supported")
        
        # Generate patches for each image
        image_size = None
        patch_count = 0
        patch_coords = []

        # set random seed
        if patch_generation_random_seed is not None:
            random.seed(patch_generation_random_seed)
        for channel_paths, cell_coords in zip(self._ImageDataset__image_paths, self.__cell_coords):
            if consistent_img_size:
                if image_size is not None:
                    pass
                else:
                    try:
                        image_size = self._read_convert_image(channel_paths[self._channel_keys[0]]).shape[0]
                        self.logger.debug(
                            f"Image size inferred: {image_size} for all images "
                            "to force redetect image sizes for each view/site set consistent_img_size=False"
                        )
                    except:
                        raise ValueError("Error reading image size")
                pass
            else:
                try:
                    image_size = self._read_convert_image(channel_paths[self._channel_keys[0]]).shape[0]
                except:
                    raise ValueError("Error reading image size")
                
            coords = patch_fn(
                image_size=image_size, 
                patch_size=_patch_size, 
                cell_coords=cell_coords,
                n_expected_patches_per_img=n_expected_patches_per_img,
                max_attempts=max_attempts        
                )
            patch_coords.append(coords)
            patch_count += len(coords)
        
        self.logger.debug(f"Generated {patch_count} patches for {len(self._ImageDataset__image_paths)} site/view")
        return patch_coords

    @staticmethod
    def __generate_cell_containing_patches_unit(
        image_size, 
        patch_size, 
        cell_coords, 
        expected_n_patches=5, 
        max_attempts=1_000):
        """
        Static helper function to generate patches that contain the cell coordinates
        :param image_size: The size of the image (square)
        :type image_size: int
        :param patch_size: The size of the square patches to generate
        :type patch_size: int
        :param cell_coords: The coordinates of the cells
        :type cell_coords: List[Tuple[int, int]]
        :param expected_n_patches: The number of patches to generate
        :type expected_n_patches: int
        :return: The coordinates of the patches
        """

        unit_size = math.gcd(image_size, patch_size)
        tile_size_units = patch_size // unit_size
        grid_size_units = image_size // unit_size

        cell_containing_units = {(x // unit_size, y // unit_size) for x, y in cell_coords}
        placed_tiles = set()
        retained_tiles = []

        attempts = 0
        n_tiles = 0
        while attempts < max_attempts:
            top_left_x = randint(0, grid_size_units - tile_size_units)
            top_left_y = randint(0, grid_size_units - tile_size_units)

            tile_units = {(x, y) for x in range(top_left_x, top_left_x + tile_size_units)
                          for y in range(top_left_y, top_left_y + tile_size_units)}

            if any(tile_units & placed_tile for placed_tile in placed_tiles):
                attempts += 1
                continue

            if tile_units & cell_containing_units:
                retained_tiles.append((top_left_x * unit_size, top_left_y * unit_size))
                placed_tiles.add(frozenset(tile_units))
                n_tiles += 1

            attempts += 1
            if n_tiles >= expected_n_patches:
                break

        return retained_tiles

    @staticmethod
    def __generate_random_patches_unit(
        image_size, 
        patch_size, 
        expected_n_patches=5, 
        max_attempts=1_000):
            """
            Static helper function to generate random patches
            :param image_size: The size of the image (square)
            :type image_size: int
            :param patch_size: The size of the square patches to generate
            :type patch_size: int
            :param expected_n_patches: The number of patches to generate
            :type expected_n_patches: int
            :return: The coordinates of the patches
            """
            unit_size = math.gcd(image_size, patch_size)
            tile_size_units = patch_size // unit_size
            grid_size_units = image_size // unit_size

            placed_tiles = set()
            retained_tiles = []

            attempts = 0
            n_tiles = 0
            while attempts < max_attempts:
                top_left_x = randint(0, grid_size_units - tile_size_units)
                top_left_y = randint(0, grid_size_units - tile_size_units)

                # check for overlap with already placed tiles
                tile_units = {(x, y) for x in range(top_left_x, top_left_x + tile_size_units)
                            for y in range(top_left_y, top_left_y + tile_size_units)}

                if any(tile_units & placed_tile for placed_tile in placed_tiles):
                    attempts += 1
                    continue
                
                # no overlap, add the tile to the list of retained tiles
                retained_tiles.append((top_left_x * unit_size, top_left_y * unit_size))
                placed_tiles.add(frozenset(tile_units))
                n_tiles += 1

                attempts += 1
                if n_tiles >= expected_n_patches:
                    break

            return retained_tiles