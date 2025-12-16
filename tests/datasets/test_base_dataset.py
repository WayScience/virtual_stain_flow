# tests/test_base_image_dataset.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytest

from virtual_stain_flow.datasets.base_dataset import BaseImageDataset


class TestBaseImageDataset:
    """Test suite for BaseImageDataset class."""
    
    def test_init_valid(self, file_index):
        """Test BaseImageDataset initialization with valid inputs."""
        dataset = BaseImageDataset(
            file_index=file_index,
            pil_image_mode="RGB",
            input_channel_keys=["input_ch1"],
            target_channel_keys="target_ch1",
            cache_capacity=10,
        )
        assert len(dataset) == len(file_index)
        assert dataset.pil_image_mode == "RGB"
        assert dataset.input_channel_keys == ["input_ch1"]
        assert dataset.target_channel_keys == ["target_ch1"]
        pd.testing.assert_frame_equal(dataset.file_index, file_index)

    def test_init_default_values(self, file_index):
        """Test BaseImageDataset initialization with default values."""
        dataset = BaseImageDataset(file_index=file_index)
        assert dataset.pil_image_mode == "I;16"
        assert dataset.input_channel_keys == []
        assert dataset.target_channel_keys == []
        assert dataset.file_state.cache_capacity == file_index.shape[1]

    def test_init_none_channel_keys(self, file_index):
        """Test BaseImageDataset initialization with None channel keys."""
        dataset = BaseImageDataset(
            file_index=file_index,
            input_channel_keys=None,
            target_channel_keys=None,
        )
        assert dataset.input_channel_keys == []
        assert dataset.target_channel_keys == []

    def test_init_string_channel_keys(self, file_index):
        """Test BaseImageDataset initialization with string channel keys."""
        dataset = BaseImageDataset(
            file_index=file_index,
            input_channel_keys="input_ch1",
            target_channel_keys="target_ch1",
        )
        assert dataset.input_channel_keys == ["input_ch1"]
        assert dataset.target_channel_keys == ["target_ch1"]

    def test_init_invalid_input_channel_keys_type(self, file_index):
        """Test BaseImageDataset raises ValueError for invalid input channel keys type."""
        with pytest.raises(ValueError, match="Expected channel_keys to be a string or a sequence"):
            BaseImageDataset(file_index=file_index, input_channel_keys=123)

    def test_init_invalid_target_channel_keys_type(self, file_index):
        """Test BaseImageDataset raises ValueError for invalid target channel keys type.""" 
        with pytest.raises(ValueError, match="Expected channel_keys to be a string or a sequence"):
            BaseImageDataset(file_index=file_index, target_channel_keys=123)

    def test_init_nonexistent_input_channel_key(self, file_index):
        """Test BaseImageDataset raises ValueError for nonexistent input channel key."""
        with pytest.raises(ValueError, match="Channel key 'nonexistent' not found"):
            BaseImageDataset(file_index=file_index, input_channel_keys=["nonexistent"])

    def test_init_nonexistent_target_channel_key(self, file_index):
        """Test BaseImageDataset raises ValueError for nonexistent target channel key."""
        with pytest.raises(ValueError, match="Channel key 'nonexistent' not found"):
            BaseImageDataset(file_index=file_index, target_channel_keys=["nonexistent"])

    def test_len(self, basic_dataset, file_index):
        """Test BaseImageDataset __len__ method."""
        assert len(basic_dataset) == len(file_index)

    def test_properties(self, basic_dataset, file_index):
        """Test BaseImageDataset properties."""
        assert basic_dataset.pil_image_mode == "I;16"
        pd.testing.assert_frame_equal(basic_dataset.file_index, file_index)

    def test_get_raw_item_returns_numpy_arrays(self, basic_dataset):
        """Test get_raw_item returns numpy arrays with expected values."""
        inp_np, tar_np = basic_dataset.get_raw_item(1)
        assert isinstance(inp_np, np.ndarray)
        assert isinstance(tar_np, np.ndarray)
        # For image 1: input_ch1=3, input_ch2=4, target_ch1=2
        assert inp_np.shape == (2, 10, 10)  # 2 input channels
        assert tar_np.shape == (1, 10, 10)  # 1 target channel
        assert np.all(inp_np[0] == 3)  # input_ch1 for image 1
        assert np.all(inp_np[1] == 4)  # input_ch2 for image 1
        assert np.all(tar_np[0] == 2)  # target_ch1 for image 1

    def test_get_raw_item_triggers_state_updates(self, basic_dataset):
        """Test get_raw_item triggers IndexState and FileState updates."""
        basic_dataset.get_raw_item(1)
        
        # Check IndexState update - it should record the last index
        assert basic_dataset.index_state.last == 1
        
        # Check FileState update - it should have loaded the correct paths
        assert len(basic_dataset.file_state.input_paths) == 2
        assert len(basic_dataset.file_state.target_paths) == 1
        assert basic_dataset.file_state.input_image_raw is not None
        assert basic_dataset.file_state.target_image_raw is not None

    def test_getitem_returns_tensors(self, basic_dataset):
        """Test __getitem__ returns torch tensors with correct dtype."""
        inp_t, tar_t = basic_dataset[1]
        assert isinstance(inp_t, torch.Tensor) 
        assert isinstance(tar_t, torch.Tensor)
        assert inp_t.dtype == torch.float32
        assert tar_t.dtype == torch.float32
        # For image 1: input_ch1=3, input_ch2=4, target_ch1=2
        assert inp_t.shape == (2, 10, 10)
        assert tar_t.shape == (1, 10, 10)
        assert torch.all(inp_t[0] == 3.0)  # input_ch1 for image 1
        assert torch.all(inp_t[1] == 4.0)  # input_ch2 for image 1
        assert torch.all(tar_t[0] == 2.0)  # target_ch1 for image 1

    def test_channel_keys_setter_valid(self, basic_dataset):
        """Test channel keys setter with valid inputs."""
        basic_dataset.input_channel_keys = ["input_ch1"]
        assert basic_dataset.input_channel_keys == ["input_ch1"]
        
        basic_dataset.target_channel_keys = "target_ch1"
        assert basic_dataset.target_channel_keys == ["target_ch1"]

    def test_channel_keys_setter_invalid(self, basic_dataset):
        """Test channel keys setter raises ValueError for invalid inputs."""
        with pytest.raises(ValueError, match="Expected channel_keys to be a string or a sequence"):
            basic_dataset.input_channel_keys = 123
            
        with pytest.raises(ValueError, match="Channel key 'nonexistent' not found"):
            basic_dataset.target_channel_keys = ["nonexistent"]


class TestBaseImageDatasetValidation:
    """Test suite for BaseImageDataset validation methods."""

    def test_validate_channel_keys_none(self, basic_dataset):
        """Test _validate_channel_keys with None input."""
        result = basic_dataset._validate_channel_keys(None)
        assert result == []

    def test_validate_channel_keys_string(self, basic_dataset):
        """Test _validate_channel_keys with string input.""" 
        result = basic_dataset._validate_channel_keys("input_ch1")
        assert result == ["input_ch1"]

    def test_validate_channel_keys_list(self, basic_dataset):
        """Test _validate_channel_keys with list input."""
        keys = ["input_ch1", "input_ch2"]
        result = basic_dataset._validate_channel_keys(keys)
        assert result == keys

    def test_validate_channel_keys_invalid_type(self, basic_dataset):
        """Test _validate_channel_keys raises ValueError for invalid type."""
        with pytest.raises(ValueError, match="Expected channel_keys to be a string or a sequence"):
            basic_dataset._validate_channel_keys(123)

    def test_validate_channel_keys_nonexistent_key(self, basic_dataset):
        """Test _validate_channel_keys raises ValueError for nonexistent key."""
        with pytest.raises(ValueError, match="Channel key 'does_not_exist' not found"):
            basic_dataset._validate_channel_keys(["does_not_exist"])


class TestBaseImageDatasetSerialization:
    """Test suite for BaseImageDataset serialization methods."""

    def test_to_config_contains_expected_fields(self, basic_dataset):
        """Test to_config contains all expected fields."""
        cfg = basic_dataset.to_config()
        
        expected_keys = [
            "file_state",
        ]
        for key in expected_keys:
            assert key in cfg
        
        state_cfg = cfg.get("file_state", {})
        expected_keys_state = [
            "cache_capacity",
            "manifest",
        ]
        for key in expected_keys_state:
            assert key in state_cfg
        
        manifest_cfg = state_cfg.get("manifest", {})
        expected_keys_manifest = [
            "file_index",
            "pil_image_mode",
        ]
        for key in expected_keys_manifest:
            assert key in manifest_cfg

    def test_to_config_dataset_length(self, basic_dataset):
        """Test to_config dataset_length matches actual length."""
        cfg = basic_dataset.to_config()
        assert len(cfg.get("file_state", {}).get("manifest", {}).get("file_index", [])) == len(basic_dataset)

    def test_to_config_file_index_structure(self, basic_dataset):
        """Test to_config file_index has correct structure."""
        cfg = basic_dataset.to_config()
        file_index = cfg.get(
            "file_state", {}).get("manifest", {}).get("file_index", []
        )
        
        assert len(file_index) > 0

        for record in file_index:
            assert "input_ch1" in record
            assert "input_ch2" in record
            assert "target_ch1" in record
        
        # All paths should be converted to strings
        for record in file_index:
            for key, value in record.items():
                assert isinstance(value, str)

    def test_from_config_roundtrip(self, basic_dataset):
        """Test from_config can reconstruct dataset from to_config output."""
        cfg = basic_dataset.to_config()
        ds2 = BaseImageDataset.from_config(cfg)

        assert isinstance(ds2, BaseImageDataset)
        assert ds2.pil_image_mode == basic_dataset.pil_image_mode
        assert ds2.input_channel_keys == basic_dataset.input_channel_keys
        assert ds2.target_channel_keys == basic_dataset.target_channel_keys
        assert ds2.file_state.cache_capacity == basic_dataset.file_state.cache_capacity
        assert len(ds2) == len(basic_dataset)

        # File index paths should be reconstructed as Path objects
        for col in ds2.file_index.columns:
            assert all(isinstance(p, Path) for p in ds2.file_index[col])

    def test_from_config_missing_file_state(self):
        """Test from_config raises ValueError when file_state is missing."""
        
        config = {"pil_image_mode": "I;16"}
        with pytest.raises(ValueError, match="Configuration must include 'file_state'."):
            BaseImageDataset.from_config(config)

    def test_to_json_config_creates_file(self, tmp_path, basic_dataset):
        """Test to_json_config creates a valid JSON file."""
        output_path = tmp_path / "dataset_config.json"
        basic_dataset.to_json_config(output_path)
        
        assert output_path.exists()
        
        # Verify JSON structure
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert "file_state" in data
        assert "input_channel_keys" in data
        assert "target_channel_keys" in data
        assert "manifest" in data["file_state"]
        assert "cache_capacity" in data["file_state"]
        assert "file_index" in data["file_state"]["manifest"]
        assert "pil_image_mode" in data["file_state"]["manifest"]
        assert len(data["file_state"]["manifest"]["file_index"]) == len(basic_dataset)

    def test_to_json_config_roundtrip(self, tmp_path, basic_dataset):
        """Test full JSON serialization roundtrip."""
        output_path = tmp_path / "dataset_config.json"
        basic_dataset.to_json_config(output_path)
        
        # Load from JSON and reconstruct
        with open(output_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        ds2 = BaseImageDataset.from_config(config)
        
        assert len(ds2) == len(basic_dataset)
        assert ds2.pil_image_mode == basic_dataset.pil_image_mode
        assert ds2.input_channel_keys == basic_dataset.input_channel_keys
        assert ds2.target_channel_keys == basic_dataset.target_channel_keys
