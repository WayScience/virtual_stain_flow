"""
Shared fixtures for testing the Dataset engine and classes
"""

import pytest
import numpy as np
import pandas as pd

from virtual_stain_flow.datasets.base_dataset import BaseImageDataset
from virtual_stain_flow.datasets.ds_engine.manifest import DatasetManifest


@pytest.fixture
def file_index(tmp_path):
    """
    Create temporary TIFF files and return a DataFrame with their paths.
    This allows testing with real file I/O using the actual manifest infrastructure.
    Creates deterministic images:
    - input_ch1: images 0, 1, 2 have values 1, 3, 5
    - input_ch2: images 0, 1, 2 have values 2, 4, 6
    - target_ch1: images 0, 1, 2 have values 1, 2, 3
    """
    from PIL import Image
    
    # Create a temporary directory for test images
    test_data_dir = tmp_path / "test_data"
    test_data_dir.mkdir()
    
    # Define the file paths (3 images instead of 2)
    paths = {
        "input_ch1": [
            test_data_dir / "img_0_in1.tif",
            test_data_dir / "img_1_in1.tif",
            test_data_dir / "img_2_in1.tif",
        ],
        "input_ch2": [
            test_data_dir / "img_0_in2.tif",
            test_data_dir / "img_1_in2.tif",
            test_data_dir / "img_2_in2.tif",
        ],
        "target_ch1": [
            test_data_dir / "img_0_tar.tif",
            test_data_dir / "img_1_tar.tif",
            test_data_dir / "img_2_tar.tif",
        ],
    }
    
    # Create deterministic 16-bit TIFF images with specific values
    # input_ch1: 1, 3, 5 for images 0, 1, 2
    for idx, path in enumerate(paths["input_ch1"]):
        value = 1 + idx * 2  # 1, 3, 5
        img_array = np.full((10, 10), value, dtype=np.uint16)
        img = Image.fromarray(img_array, mode='I;16')
        img.save(path)
    
    # input_ch2: 2, 4, 6 for images 0, 1, 2
    for idx, path in enumerate(paths["input_ch2"]):
        value = 2 + idx * 2  # 2, 4, 6
        img_array = np.full((10, 10), value, dtype=np.uint16)
        img = Image.fromarray(img_array, mode='I;16')
        img.save(path)
    
    # target_ch1: 1, 2, 3 for images 0, 1, 2
    for idx, path in enumerate(paths["target_ch1"]):
        value = 1 + idx  # 1, 2, 3
        img_array = np.full((10, 10), value, dtype=np.uint16)
        img = Image.fromarray(img_array, mode='I;16')
        img.save(path)
    
    return pd.DataFrame(paths)


@pytest.fixture
def basic_dataset(file_index):
    """Basic dataset with valid channel keys for testing."""
    return BaseImageDataset(
        file_index=file_index,
        pil_image_mode="I;16",
        check_exists=False,
        input_channel_keys=["input_ch1", "input_ch2"],
        target_channel_keys="target_ch1",
        cache_capacity=8,
    )


@pytest.fixture
def sample_manifest(file_index):
    """Create a simple manifest for testing."""
    return DatasetManifest(file_index=file_index, pil_image_mode="I;16")
