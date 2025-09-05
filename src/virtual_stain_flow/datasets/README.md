# Datasets Subpackage Documentation

This subpackage provides the infrastructure for managing and loading image datasets in a lazy-loaded and efficient manner, and  materialized dataset classes. 
It is designed to handle datasets with multiple channels and fields of view (FOVs), supporting dynamic image loading and caching to optimize memory usage and access speed.

## Overview

The subpackage consists of three main components:

1. **`DatasetManifest`**: Defines the immutable structure of a dataset, including file paths and image modes.
2. **`FileState`**: Defines the backend memory-efficient lazy loading infranstructure bridging the `DatasetManifest` and the image dataset class. 
3. **`BaseImageDataset`**: A PyTorch-compatible dataset class that uses `DatasetManifest` and `FileState` to manage image loading and caching.

---

## `DatasetManifest`

The `DatasetManifest` class is responsible for defining the structure of a dataset. 
It holds a file index (a `pandas.DataFrame`) where each row corresponds to a sample or FOV, and columns represent channels associated with that sample. 
It also specifies the image mode to use when reading images.

## `FileState`

The `FileState` class wraps a constructred `DatasetManifest` object and handles the loading of images from filepaths on-demand. The dataset class should call the `FileState.update(idx, input_keys, target_keys)` method to request for the images corresponding to the `idx`th sample and the input/target keys to be loaded, stacked, and accessible from `FileState.input_image_raw` and `FileState.target_image_raw` attributes.  

---

## `BaseImageDataset`

The `BaseImageDataset` class builds on `DatasetManifest` to provide a PyTorch-compatible dataset. 
It supports lazy loading of images, caching, and efficient handling of input and target channels.
- Returns paired input/target image stack as numpy arrays or torch Tensors 
- Provides methods to save and load dataset configurations as JSON files for reproducibility.

### Usage:

```python
import pandas as pd

from base_dataset import BaseImageDataset

# Example file index
file_index = pd.DataFrame({
    "input_channel": ["path/to/input1.tif", "path/to/input2.tif"],
    "target_channel": ["path/to/target1.tif", "path/to/target2.tif"]
})

dataset = BaseImageDataset(
    file_index=file_index,
    pil_image_mode="I;16",
    input_channel_keys="input_channel",
    target_channel_keys="target_channel",
    cache_capacity=10
)
```
### Serialization for logging
```python
ds_config = dataset.to_config()
# or
dataset.to_json_config('loggable_artifact.json')
```
