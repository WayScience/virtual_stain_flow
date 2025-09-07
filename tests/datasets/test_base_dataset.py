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
    # Minimal, pathlike-only DataFrame matching the class contract
    return pd.DataFrame(
        {
            "input_ch1": [Path("/data/img_0_in1.tif"), Path("/data/img_1_in1.tif")],
            "input_ch2": [Path("/data/img_0_in2.tif"), Path("/data/img_1_in2.tif")],
            "target_ch1": [Path("/data/img_0_tar.tif"), Path("/data/img_1_tar.tif")],
        }
    )


@pytest.fixture
def dataset(file_index):
    # Provide valid channel keys so __init__ validation passes
    return BaseImageDataset(
        file_index=file_index,
        pil_image_mode="I;16",
        input_channel_keys=["input_ch1", "input_ch2"],
        target_channel_keys="target_ch1",
        cache_capacity=8,
    )


def test_len_and_props(dataset, file_index):
    assert len(dataset) == len(file_index)
    assert dataset.pil_image_mode == "I;16"
    # file_index should be the same object held by the manifest
    pd.testing.assert_frame_equal(dataset.file_index, file_index)


def test_get_raw_item_and_getitem_calls_and_tensors(dataset):
    # get_raw_item triggers IndexState.update and FileState.update,
    # and returns numpy arrays with expected values
    inp_np, tar_np = dataset.get_raw_item(1)
    assert isinstance(inp_np, np.ndarray)
    assert isinstance(tar_np, np.ndarray)
    assert np.all(inp_np == 1.0)
    assert np.all(tar_np == 2.0)

    # __getitem__ wraps those in float tensors
    inp_t, tar_t = dataset[1]
    assert isinstance(inp_t, torch.Tensor) and inp_t.dtype == torch.float32
    assert isinstance(tar_t, torch.Tensor) and tar_t.dtype == torch.float32
    assert torch.all(inp_t == 1.0)
    assert torch.all(tar_t == 2.0)

    # Our DummyIndexState/DummyFileState keep call bookkeeping
    assert dataset.index_state.updated is True
    assert dataset.index_state.updated_with == 1
    assert dataset.index_state.update_calls >= 1
    assert dataset.file_state.update_calls >= 1
    # And last_update_kwargs should reflect channel keys normalized to lists
    assert dataset.file_state.last_update_kwargs == {
        "idx": 1,
        "input_keys": ["input_ch1", "input_ch2"],
        "target_keys": ["target_ch1"],
    }


def test_validate_channel_keys(dataset):
    # None → []
    assert dataset._validate_channel_keys(None) == []
    # str → [str]
    assert dataset._validate_channel_keys("input_ch1") == ["input_ch1"]
    # list passthrough
    keys = ["input_ch1", "input_ch2"]
    assert dataset._validate_channel_keys(keys) == keys


def test_validate_channel_keys_errors(dataset):
    # wrong type
    with pytest.raises(ValueError):
        dataset._validate_channel_keys(123)  # type: ignore[arg-type]
    # missing column
    with pytest.raises(ValueError):
        dataset._validate_channel_keys(["does_not_exist"])


def test_to_config_contains_expected_fields(dataset):
    cfg = dataset.to_config()
    # basic keys present
    for k in [
        "file_index",
        "pil_image_mode",
        "input_channel_keys",
        "target_channel_keys",
        "cache_capacity",
        "dataset_length",
    ]:
        assert k in cfg

    # dataset_length matches
    assert cfg["dataset_length"] == len(dataset)

    # file_index JSON-friendly (records + columns; paths stringified)
    fi = cfg["file_index"]
    assert set(fi.keys()) == {"records", "columns"}
    assert fi["columns"] == list(dataset.file_index.columns)
    # paths must be strings in the records
    for rec in fi["records"]:
        for col in fi["columns"]:
            assert isinstance(rec[col], str)


def test_from_config_roundtrip(dataset):
    cfg = dataset.to_config()
    # instantiate from the raw dict
    ds2 = BaseImageDataset.from_config(cfg)

    assert isinstance(ds2, BaseImageDataset)
    assert ds2.pil_image_mode == dataset.pil_image_mode
    assert ds2.input_channel_keys == dataset.input_channel_keys
    assert ds2.target_channel_keys == dataset.target_channel_keys
    assert ds2.file_state.cache_capacity == dataset.file_state.cache_capacity

    # file_index paths should be reconstructed as Path objects
    assert all(isinstance(p, Path) for p in ds2.file_index["input_ch1"])
    assert list(ds2.file_index.columns) == list(dataset.file_index.columns)
    assert len(ds2) == len(dataset)


def test_to_json_config_writes_file(tmp_path, dataset):
    out = tmp_path / "ds_config.json"
    dataset.to_json_config(out)
    assert out.exists()

    # sanity-check: load it and ensure JSON structure round-trips
    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "file_index" in data and "records" in data["file_index"]
    assert data["dataset_length"] == len(dataset)
