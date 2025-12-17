# tests/datasets/test_crop_dataset.py
"""
Test suite for CropImageDataset class.
Mirrors structure of test_base_dataset.py but focuses on crop-specific functionality.
Minimal initialization/error catching tests as base dataset class covers shared validation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytest

from virtual_stain_flow.datasets.crop_dataset import CropImageDataset
from virtual_stain_flow.datasets.ds_engine.crop_manifest import (
    Crop
)


class TestCropImageDataset:
    """Test suite for CropImageDataset class."""

    def test_init_valid(self, file_index, crop_specs):
        """Test CropImageDataset initialization with valid inputs."""
        dataset = CropImageDataset(
            file_index=file_index,
            crop_specs=crop_specs,
            pil_image_mode="I;16",
            input_channel_keys=["input_ch1"],
            target_channel_keys="target_ch1",
            cache_capacity=10,
        )
        # 2 crops per image * 3 images = 6 crops total
        assert len(dataset) == 6
        assert dataset.pil_image_mode == "I;16"
        assert dataset.input_channel_keys == ["input_ch1"]
        assert dataset.target_channel_keys == ["target_ch1"]
        pd.testing.assert_frame_equal(dataset.file_index, file_index)

    def test_init_missing_crop_specs(self, file_index):
        """Test CropImageDataset raises ValueError when crop_specs is missing."""
        with pytest.raises(ValueError, match="Either 'crop_file_state' or both"):
            CropImageDataset(file_index=file_index)

    def test_init_missing_file_index(self, crop_specs):
        """Test CropImageDataset raises ValueError when file_index is missing."""
        with pytest.raises(ValueError, match="Either 'crop_file_state' or both"):
            CropImageDataset(crop_specs=crop_specs)

    def test_len(self, crop_dataset):
        """Test CropImageDataset __len__ method returns number of crops."""
        # 2 crops per image * 3 images = 6 crops
        assert len(crop_dataset) == 6

    def test_properties(self, crop_dataset, file_index):
        """Test CropImageDataset properties."""
        assert crop_dataset.pil_image_mode == "I;16"
        pd.testing.assert_frame_equal(crop_dataset.file_index, file_index)

    def test_get_raw_item_returns_numpy_arrays(self, crop_dataset):
        """Test get_raw_item returns numpy arrays with expected crop shape."""
        inp_np, tar_np = crop_dataset.get_raw_item(0)
        assert isinstance(inp_np, np.ndarray)
        assert isinstance(tar_np, np.ndarray)
        # Crop size is 4x4, 2 input channels, 1 target channel
        assert inp_np.shape == (2, 4, 4)
        assert tar_np.shape == (1, 4, 4)

    def test_get_raw_item_returns_correct_values(self, crop_dataset):
        """Test get_raw_item returns correct pixel values for cropped region."""
        # First crop (idx=0) is from image 0 (input_ch1=1, input_ch2=2, target_ch1=1)
        inp_np, tar_np = crop_dataset.get_raw_item(0)
        assert np.all(inp_np[0] == 1)  # input_ch1 for image 0
        assert np.all(inp_np[1] == 2)  # input_ch2 for image 0
        assert np.all(tar_np[0] == 1)  # target_ch1 for image 0

        # Third crop (idx=2) is from image 1 (input_ch1=3, input_ch2=4, target_ch1=2)
        inp_np, tar_np = crop_dataset.get_raw_item(2)
        assert np.all(inp_np[0] == 3)  # input_ch1 for image 1
        assert np.all(inp_np[1] == 4)  # input_ch2 for image 1
        assert np.all(tar_np[0] == 2)  # target_ch1 for image 1

    def test_get_raw_item_triggers_state_updates(self, crop_dataset):
        """Test get_raw_item triggers CropIndexState and CropFileState updates."""
        crop_dataset.get_raw_item(2)
        
        # Check CropIndexState update - it should record the last crop index
        assert crop_dataset.index_state.last_crop_idx == 2
        
        # Check CropFileState update - it should have loaded the correct paths
        assert len(crop_dataset.file_state.input_paths) == 2
        assert len(crop_dataset.file_state.target_paths) == 1
        assert crop_dataset.file_state.input_image_raw is not None
        assert crop_dataset.file_state.target_image_raw is not None

    def test_getitem_returns_tensors(self, crop_dataset):
        """Test __getitem__ returns torch tensors with correct dtype."""
        inp_t, tar_t = crop_dataset[0]
        assert isinstance(inp_t, torch.Tensor)
        assert isinstance(tar_t, torch.Tensor)
        assert inp_t.dtype == torch.float32
        assert tar_t.dtype == torch.float32
        # Crop size is 4x4
        assert inp_t.shape == (2, 4, 4)
        assert tar_t.shape == (1, 4, 4)

    def test_getitem_returns_correct_values(self, crop_dataset):
        """Test __getitem__ returns tensors with correct pixel values."""
        # First crop from image 0
        inp_t, tar_t = crop_dataset[0]
        assert torch.all(inp_t[0] == 1.0)
        assert torch.all(inp_t[1] == 2.0)
        assert torch.all(tar_t[0] == 1.0)

    def test_crop_info_property(self, crop_dataset):
        """Test crop_info property returns correct Crop object after access."""
        crop_dataset.get_raw_item(0)
        crop_info = crop_dataset.crop_info
        
        assert isinstance(crop_info, Crop)
        assert crop_info.manifest_idx == 0
        assert crop_info.x == 0
        assert crop_info.y == 0
        assert crop_info.width == 4
        assert crop_info.height == 4

    def test_crop_info_updates_on_access(self, crop_dataset):
        """Test crop_info updates correctly when accessing different crops."""
        crop_dataset.get_raw_item(0)
        assert crop_dataset.crop_info.manifest_idx == 0
        assert crop_dataset.crop_info.x == 0
        
        crop_dataset.get_raw_item(1)  # Second crop of image 0
        assert crop_dataset.crop_info.manifest_idx == 0
        assert crop_dataset.crop_info.x == 5  # Second crop starts at x=5

    def test_different_crops_from_same_image(self, crop_dataset):
        """Test accessing different crops from the same source image."""
        # Crops 0 and 1 are both from image 0
        inp0, _ = crop_dataset.get_raw_item(0)
        inp1, _ = crop_dataset.get_raw_item(1)
        
        # Both should have same values since source image is uniform
        assert np.all(inp0[0] == 1)
        assert np.all(inp1[0] == 1)
        
        # But they should be different crop regions
        crop_dataset.get_raw_item(0)
        # Re-access to verify state
        assert crop_dataset.crop_info.x == 0
        crop_dataset.get_raw_item(1)
        assert crop_dataset.crop_info.x == 5


class TestCropImageDatasetSerialization:
    """Test suite for CropImageDataset serialization methods."""

    def test_to_config_contains_expected_fields(self, crop_dataset):
        """Test to_config contains all expected fields."""
        cfg = crop_dataset.to_config()
        
        expected_keys = ["crop_file_state", "input_channel_keys", "target_channel_keys"]
        for key in expected_keys:
            assert key in cfg
        
        state_cfg = cfg.get("crop_file_state", {})
        expected_keys_state = ["crop_collection", "cache_capacity"]
        for key in expected_keys_state:
            assert key in state_cfg
        
        crop_collection_cfg = state_cfg.get("crop_collection", {})
        expected_keys_collection = ["crops", "manifest"]
        for key in expected_keys_collection:
            assert key in crop_collection_cfg

    def test_to_config_crops_structure(self, crop_dataset):
        """Test to_config crops have correct structure."""
        cfg = crop_dataset.to_config()
        crops = cfg.get("crop_file_state", {}).get("crop_collection", {}).get("crops", [])
        
        assert len(crops) == 6  # 2 crops per image * 3 images
        
        for crop in crops:
            assert "manifest_idx" in crop
            assert "x" in crop
            assert "y" in crop
            assert "width" in crop
            assert "height" in crop

    def test_to_config_manifest_structure(self, crop_dataset):
        """Test to_config underlying manifest has correct structure."""
        cfg = crop_dataset.to_config()
        manifest = cfg.get("crop_file_state", {}).get("crop_collection", {}).get("manifest", {})
        
        assert "file_index" in manifest
        assert "pil_image_mode" in manifest
        assert len(manifest["file_index"]) == 3  # 3 source images

    def test_from_config_roundtrip(self, crop_dataset):
        """Test from_config can reconstruct dataset from to_config output."""
        cfg = crop_dataset.to_config()
        ds2 = CropImageDataset.from_config(cfg)

        assert isinstance(ds2, CropImageDataset)
        assert ds2.pil_image_mode == crop_dataset.pil_image_mode
        assert ds2.input_channel_keys == crop_dataset.input_channel_keys
        assert ds2.target_channel_keys == crop_dataset.target_channel_keys
        assert len(ds2) == len(crop_dataset)

        # File index paths should be reconstructed as Path objects
        for col in ds2.file_index.columns:
            assert all(isinstance(p, Path) for p in ds2.file_index[col])

    def test_from_config_preserves_crop_definitions(self, crop_dataset):
        """Test from_config preserves all crop definitions."""
        cfg = crop_dataset.to_config()
        ds2 = CropImageDataset.from_config(cfg)
        
        # Verify crops are preserved by checking data access
        for i in range(len(crop_dataset)):
            inp1, tar1 = crop_dataset.get_raw_item(i)
            inp2, tar2 = ds2.get_raw_item(i)
            np.testing.assert_array_equal(inp1, inp2)
            np.testing.assert_array_equal(tar1, tar2)

    def test_from_config_missing_crop_file_state(self):
        """Test from_config raises error when crop_file_state is missing."""
        config = {"input_channel_keys": ["input_ch1"]}
        with pytest.raises(
            ValueError, 
            match="Configuration missing required field 'crop_file_state'."
        ):
            CropImageDataset.from_config(config=config)

        base_dataset_like_config = {
            "input_channel_keys": ["input_ch1"],
            "target_channel_keys": ["target_ch1"],
            "file_state": {}
        }
        with pytest.raises(
            ValueError, 
            match="Likely the received configuration is for BaseImageDataset"
        ):
            CropImageDataset.from_config(config=base_dataset_like_config)

    def test_to_json_config_creates_file(self, tmp_path, crop_dataset):
        """Test to_json_config creates a valid JSON file."""
        output_path = tmp_path / "crop_dataset_config.json"
        crop_dataset.to_json_config(output_path)
        
        assert output_path.exists()
        
        # Verify JSON structure
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert "crop_file_state" in data
        assert "input_channel_keys" in data
        assert "target_channel_keys" in data

    def test_to_json_config_roundtrip(self, tmp_path, crop_dataset):
        """Test full JSON serialization roundtrip."""
        output_path = tmp_path / "crop_dataset_config.json"
        crop_dataset.to_json_config(output_path)
        
        with open(output_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        ds2 = CropImageDataset.from_config(config)
        
        assert len(ds2) == len(crop_dataset)
        assert ds2.pil_image_mode == crop_dataset.pil_image_mode
        assert ds2.input_channel_keys == crop_dataset.input_channel_keys
        assert ds2.target_channel_keys == crop_dataset.target_channel_keys


class TestCropImageDatasetCropBoundary:
    """Test suite for crop boundary handling."""

    def test_crop_out_of_bounds_raises_error(self, file_index):
        """Test that crops exceeding image bounds raise an error on access."""
        out_of_bounds_specs = {
            0: [((8, 8), 5, 5)]  # Image is 10x10, this crop exceeds bounds
        }
        dataset = CropImageDataset(
            file_index=file_index,
            crop_specs=out_of_bounds_specs,
            input_channel_keys=["input_ch1"],
            target_channel_keys=["target_ch1"],
        )
        
        with pytest.raises(ValueError, match="exceeds image bounds"):
            dataset.get_raw_item(0)

    def test_crop_at_boundary(self, file_index):
        """Test crop that exactly fits at image boundary."""
        boundary_specs = {
            0: [((6, 6), 4, 4)]  # Image is 10x10, crop ends exactly at boundary
        }
        dataset = CropImageDataset(
            file_index=file_index,
            crop_specs=boundary_specs,
            input_channel_keys=["input_ch1"],
            target_channel_keys=["target_ch1"],
        )
        
        inp_np, tar_np = dataset.get_raw_item(0)
        assert inp_np.shape == (1, 4, 4)
        assert tar_np.shape == (1, 4, 4)


class TestCropImageDatasetDataLoader:
    """Test suite for DataLoader compatibility."""

    def test_dataloader_iteration(self, crop_dataset):
        """Test CropImageDataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        
        loader = DataLoader(crop_dataset, batch_size=2, shuffle=False)
        
        batches = list(loader)
        assert len(batches) == 3  # 6 crops / batch_size 2 = 3 batches
        
        inp_batch, tar_batch = batches[0]
        assert inp_batch.shape == (2, 2, 4, 4)  # (batch, channels, H, W)
        assert tar_batch.shape == (2, 1, 4, 4)

    def test_dataloader_with_shuffle(self, crop_dataset):
        """Test CropImageDataset works with shuffled DataLoader."""
        from torch.utils.data import DataLoader
        
        loader = DataLoader(crop_dataset, batch_size=2, shuffle=True)
        
        # Just verify it doesn't crash
        for inp_batch, tar_batch in loader:
            assert inp_batch.shape[0] <= 2
            assert tar_batch.shape[0] <= 2
