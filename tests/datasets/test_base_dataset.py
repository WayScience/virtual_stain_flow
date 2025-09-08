# tests/test_base_image_dataset.py
import json
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import pytest

# Import the module under test so we can monkeypatch its collaborators
import virtual_stain_flow.datasets.base_dataset as base_mod
from virtual_stain_flow.datasets.base_dataset import BaseImageDataset


# Dummy for Manifest, IndexState, and FileState classes so 
# test focuses on BaseImageDataset logic
class DummyManifest:
    def __init__(self, file_index: pd.DataFrame, pil_image_mode: str = "I;16"):
        self.file_index = file_index
        self.pil_image_mode = pil_image_mode

    def __len__(self) -> int:
        return len(self.file_index)


class DummyIndexState:
    def __init__(self):
        self.updated = False
        self.updated_with: Optional[int] = None
        self.update_calls = 0

    def update(self, idx: int):
        self.updated = True
        self.updated_with = idx
        self.update_calls += 1


class DummyFileState:
    def __init__(self, manifest: DummyManifest, cache_capacity: Optional[int] = None):
        self.manifest = manifest
        # Implement the same default logic as real FileState
        if cache_capacity is None:
            self.cache_capacity = manifest.file_index.shape[1]  # equivalent to manifest.n_channels
        else:
            self.cache_capacity = cache_capacity
        self.update_calls = 0
        self.last_update_kwargs = None
        self.input_image_raw = None
        self.target_image_raw = None

    def update(
        self,
        idx: int,
        input_keys: Optional[Union[str, Sequence[str]]] = None,
        target_keys: Optional[Union[str, Sequence[str]]] = None,
    ):
        # record call
        self.update_calls += 1
        self.last_update_kwargs = {
            "idx": idx,
            "input_keys": list(input_keys) if input_keys is not None else [],
            "target_keys": list(target_keys) if target_keys is not None else [],
        }
        # fabricate deterministic "loaded" arrays
        # shape is arbitrary but stable
        self.input_image_raw = np.full((2, 3), float(idx), dtype=np.float32)
        self.target_image_raw = np.full((2, 3), float(idx + 1), dtype=np.float32)


# fixture to auto-patch the .manifest collaborators for all tests
@pytest.fixture(autouse=True)
def patch_manifest_classes(monkeypatch):
    """
    Automatically monkeypatch the .manifest collaborators for all tests.
    """
    monkeypatch.setattr(base_mod, "DatasetManifest", DummyManifest)
    monkeypatch.setattr(base_mod, "IndexState", DummyIndexState)
    monkeypatch.setattr(base_mod, "FileState", DummyFileState)


@pytest.fixture
def file_index():
    """Minimal, pathlike-only DataFrame matching the class contract."""
    return pd.DataFrame(
        {
            "input_ch1": [Path("/data/img_0_in1.tif"), Path("/data/img_1_in1.tif")],
            "input_ch2": [Path("/data/img_0_in2.tif"), Path("/data/img_1_in2.tif")],
            "target_ch1": [Path("/data/img_0_tar.tif"), Path("/data/img_1_tar.tif")],
        }
    )


@pytest.fixture
def basic_dataset(file_index):
    """Basic dataset with valid channel keys for testing."""
    return BaseImageDataset(
        file_index=file_index,
        pil_image_mode="I;16",
        input_channel_keys=["input_ch1", "input_ch2"],
        target_channel_keys="target_ch1",
        cache_capacity=8,
    )


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
        assert np.all(inp_np == 1.0)
        assert np.all(tar_np == 2.0)

    def test_get_raw_item_triggers_state_updates(self, basic_dataset):
        """Test get_raw_item triggers IndexState and FileState updates."""
        basic_dataset.get_raw_item(1)
        
        # Check IndexState update
        assert basic_dataset.index_state.updated is True
        assert basic_dataset.index_state.updated_with == 1
        assert basic_dataset.index_state.update_calls >= 1
        
        # Check FileState update
        assert basic_dataset.file_state.update_calls >= 1
        assert basic_dataset.file_state.last_update_kwargs == {
            "idx": 1,
            "input_keys": ["input_ch1", "input_ch2"],
            "target_keys": ["target_ch1"],
        }

    def test_getitem_returns_tensors(self, basic_dataset):
        """Test __getitem__ returns torch tensors with correct dtype."""
        inp_t, tar_t = basic_dataset[1]
        assert isinstance(inp_t, torch.Tensor) 
        assert isinstance(tar_t, torch.Tensor)
        assert inp_t.dtype == torch.float32
        assert tar_t.dtype == torch.float32
        assert torch.all(inp_t == 1.0)
        assert torch.all(tar_t == 2.0)

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
            "file_index",
            "pil_image_mode", 
            "input_channel_keys",
            "target_channel_keys",
            "cache_capacity",
            "dataset_length",
        ]
        for key in expected_keys:
            assert key in cfg

    def test_to_config_dataset_length(self, basic_dataset):
        """Test to_config dataset_length matches actual length."""
        cfg = basic_dataset.to_config()
        assert cfg["dataset_length"] == len(basic_dataset)

    def test_to_config_file_index_structure(self, basic_dataset):
        """Test to_config file_index has correct structure."""
        cfg = basic_dataset.to_config()
        file_index = cfg["file_index"]
        
        assert set(file_index.keys()) == {"records", "columns"}
        assert file_index["columns"] == list(basic_dataset.file_index.columns)
        
        # All paths should be converted to strings
        for record in file_index["records"]:
            for col in file_index["columns"]:
                assert isinstance(record[col], str)

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

    def test_from_config_missing_file_index(self):
        """Test from_config raises ValueError when file_index is missing."""
        config = {"pil_image_mode": "I;16"}
        with pytest.raises(ValueError, match="Expected 'file_index' in config"):
            BaseImageDataset.from_config(config)

    def test_from_config_missing_pil_image_mode(self, file_index):
        """Test from_config raises ValueError when pil_image_mode is missing."""
        config = {
            "file_index": {
                "records": file_index.to_dict('records'),
                "columns": list(file_index.columns)
            }
        }
        with pytest.raises(ValueError, match="Expected 'pil_image_mode' in config"):
            BaseImageDataset.from_config(config)

    def test_to_json_config_creates_file(self, tmp_path, basic_dataset):
        """Test to_json_config creates a valid JSON file."""
        output_path = tmp_path / "dataset_config.json"
        basic_dataset.to_json_config(output_path)
        
        assert output_path.exists()
        
        # Verify JSON structure
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert "file_index" in data
        assert "records" in data["file_index"]
        assert data["dataset_length"] == len(basic_dataset)

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
