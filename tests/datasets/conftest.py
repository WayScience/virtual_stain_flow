import pytest
import pandas as pd

from virtual_stain_flow.datasets.ds_engine.manifest import DatasetManifest


@pytest.fixture
def sample_manifest():
    """Create a simple manifest for testing."""
    df = pd.DataFrame({
        'input_ch1': ['/path/img_0_in.tif', '/path/img_1_in.tif', '/path/img_2_in.tif'],
        'target_ch1': ['/path/img_0_tar.tif', '/path/img_1_tar.tif', '/path/img_2_tar.tif'],
    })
    return DatasetManifest(file_index=df, pil_image_mode="I;16")
