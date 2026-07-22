"""
Comprehensive test suite for crop management infrastructure.

Tests:
- Crop dataclass serialization
- CropManifest creation, validation, and factories
- CropIndexState state tracking
- CropFileState image extraction and integration
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from virtual_stain_flow.datasets.ds_engine.manifest import DatasetManifest
from virtual_stain_flow.datasets.ds_engine.crop_manifest import (
    Crop,
    CropManifest,
    CropIndexState,
    CropFileState,
)


class TestCrop:
    """Tests for Crop dataclass."""
    
    def test_crop_creation(self):
        """Test creating a Crop."""
        crop = Crop(manifest_idx=0, x=10, y=20, width=100, height=100)
        assert crop.manifest_idx == 0
        assert crop.x == 10
        assert crop.y == 20
        assert crop.width == 100
        assert crop.height == 100
    
    def test_crop_to_dict(self):
        """Test Crop serialization to dict."""
        crop = Crop(manifest_idx=1, x=5, y=15, width=64, height=64)
        d = crop.to_dict()
        assert d == {
            'manifest_idx': 1,
            'x': 5,
            'y': 15,
            'width': 64,
            'height': 64,
        }
    
    def test_crop_from_dict(self):
        """Test Crop deserialization from dict."""
        d = {'manifest_idx': 2, 'x': 0, 'y': 0, 'width': 256, 'height': 256}
        crop = Crop.from_dict(d)
        assert crop.manifest_idx == 2
        assert crop.x == 0
        assert crop.y == 0
        assert crop.width == 256
        assert crop.height == 256


class TestCropManifest:
    """Tests for CropManifest."""
    
    def test_init_valid(self, sample_manifest):
        """Test valid CropManifest initialization."""
        crops = [
            Crop(0, 0, 0, 10, 10),
            Crop(1, 0, 0, 10, 10),
        ]
        cc = CropManifest(crops, manifest=sample_manifest)
        assert len(cc) == 2
        assert cc.get_crop(0).manifest_idx == 0
        assert cc.get_crop(1).manifest_idx == 1
    
    def test_init_empty_raises(self, sample_manifest):
        """Test that empty crops list raises ValueError."""
        with pytest.raises(ValueError, match="crops list cannot be empty"):
            CropManifest([], manifest=sample_manifest)
    
    def test_init_out_of_bounds_manifest_idx_raises(self, sample_manifest):
        """Test that out-of-bounds manifest_idx raises IndexError."""
        crops = [Crop(0, 0, 0, 10, 10), Crop(10, 0, 0, 10, 10)]
        with pytest.raises(IndexError, match="One or more crop.manifest_idx values are out of bounds"):
            CropManifest(crops, manifest=sample_manifest)
    
    def test_get_crop_out_of_bounds_raises(self, sample_manifest):
        """Test that accessing invalid crop_idx raises IndexError."""
        crops = [Crop(0, 0, 0, 10, 10)]
        cc = CropManifest(crops, manifest=sample_manifest)
        with pytest.raises(IndexError, match="crop_idx 5 out of range"):
            cc.get_crop(5)
    
    def test_to_config(self, sample_manifest):
        """Test config serialization."""
        crops = [Crop(0, 1, 2, 8, 8), Crop(1, 0, 0, 10, 10)]
        cc = CropManifest(crops, manifest=sample_manifest)
        config = cc.to_config()
        assert 'crops' in config
        assert len(config['crops']) == 2
        assert config['crops'][0]['manifest_idx'] == 0
        assert config['crops'][0]['x'] == 1
        assert config['crops'][1]['width'] == 10
    
    def test_from_config(self, sample_manifest):
        """Test config deserialization."""
        # First create a config from an existing CropManifest
        crops = [Crop(0, 1, 2, 8, 8), Crop(1, 0, 0, 10, 10)]
        cc_original = CropManifest(crops, manifest=sample_manifest)
        config = cc_original.to_config()
        
        # Now deserialize it
        cc = CropManifest.from_config(config)
        assert len(cc) == 2
        assert cc.get_crop(0).x == 1
        assert cc.get_crop(1).height == 10


class TestCropIndexState:
    """Tests for CropIndexState."""
    
    def test_init(self, sample_manifest):
        """Test CropIndexState initialization."""
        crops = [Crop(0, 0, 0, 10, 10)]
        cc = CropManifest(crops, manifest=sample_manifest)
        cs = CropIndexState(cc)
        assert cs.last_crop_idx is None
        assert cs._last_crop is None
    
    def test_is_stale_initially_true(self, sample_manifest):
        """Test that is_stale returns True initially."""
        crops = [Crop(0, 0, 0, 10, 10)]
        cc = CropManifest(crops, manifest=sample_manifest)
        cs = CropIndexState(cc)
        assert cs.is_stale(0) is True
    
    def test_get_and_update(self, sample_manifest):
        """Test get_and_update updates state and returns crop."""
        crops = [Crop(0, 1, 2, 8, 8), Crop(1, 0, 0, 10, 10)]
        cc = CropManifest(crops, manifest=sample_manifest)
        cs = CropIndexState(cc)
        
        # First call should be stale and update
        crop = cs.get_and_update(0)
        assert crop.x == 1
        assert cs.last_crop_idx == 0
        assert cs._last_crop.manifest_idx == 0
        assert cs._last_crop.x == 1
        assert cs._last_crop.y == 2
        assert cs._last_crop.width == 8
        assert cs._last_crop.height == 8
        assert cs.is_stale(0) is False
        
        # Calling with same index shouldn't change state
        cs.get_and_update(0)
        assert cs.last_crop_idx == 0
        
        # Calling with different index should update
        crop = cs.get_and_update(1)
        assert crop.manifest_idx == 1
        assert cs.last_crop_idx == 1
    
    def test_reset(self, sample_manifest):
        """Test reset clears state."""
        crops = [Crop(0, 0, 0, 10, 10)]
        cc = CropManifest(crops, manifest=sample_manifest)
        cs = CropIndexState(cc)
        cs.get_and_update(0)
        
        cs.reset()
        assert cs.last_crop_idx is None
        assert cs._last_crop is None


class TestCropFileState:
    """Tests for CropFileState (requires mocking FileState)."""
    
    def test_init(self, sample_manifest):
        """Test CropFileState initialization."""
        crops = [Crop(0, 0, 0, 10, 10)]
        cc = CropManifest(crops, manifest=sample_manifest)
        cfs = CropFileState(cc, cache_capacity=10)
        
        assert cfs.manifest is sample_manifest
        assert cfs.input_crop_raw is None
        assert cfs.target_crop_raw is None
        assert len(cfs.input_paths) == 0
    
    def test_extract_crop_3d(self, sample_manifest):
        """Test crop extraction from (C, H, W) image."""
        crops = [Crop(0, 1, 2, 5, 5)]
        cc = CropManifest(crops, manifest=sample_manifest)
        cfs = CropFileState(cc)
        
        # Create dummy (3, 10, 10) image
        image = np.arange(300).reshape(3, 10, 10)
        crop = crops[0]
        
        cropped = cfs._extract_crop(image, crop)
        assert cropped.shape == (3, 5, 5)
        assert np.allclose(
            cropped,
            image[:, 2:7, 1:6]
        )
    
    def test_extract_crop_non_3d(self, sample_manifest):
        """Test crop extraction from non (C, H, W) image."""
        crops = [Crop(0, 0, 0, 5, 5)]
        cc = CropManifest(crops, manifest=sample_manifest)
        cfs = CropFileState(cc)
        
        # Create dummy (2, 10, 10, 3) image
        image = np.random.rand(2, 10, 10, 3)
        
        # anticipate error raised for non-3D image
        crop = crops[0]
        with pytest.raises(ValueError, match="Unexpected image shape"):
            cfs._extract_crop(image, crop)
    
    def test_extract_crop_out_of_bounds_raises(self, sample_manifest):
        """Test that out-of-bounds crops raise ValueError."""
        crops = [Crop(0, 5, 5, 10, 10)]  # Crop extends beyond 10x10 image
        cc = CropManifest(crops, manifest=sample_manifest)
        cfs = CropFileState(cc)
        
        image = np.random.rand(2, 10, 10)
        with pytest.raises(ValueError, match="exceeds image bounds"):
            cfs._extract_crop(image, crops[0])
    
    def test_reset(self, sample_manifest):
        """Test reset clears all state."""
        crops = [Crop(0, 0, 0, 10, 10)]
        cc = CropManifest(crops, manifest=sample_manifest)
        cfs = CropFileState(cc)
        
        # Simulate some state
        cfs.input_crop_raw = np.array([1, 2, 3])
        cfs.input_paths = [Path('/some/path')]
        cfs.crop_state.get_and_update(0)
        
        cfs.reset()
        assert cfs.input_crop_raw is None
        assert cfs.target_crop_raw is None
        assert len(cfs.input_paths) == 0
        assert cfs.crop_state.last_crop_idx is None
    
    def test_update_basic_flow(self, sample_manifest, tmp_path):
        """Test update method with real image files."""
        # Create dummy image files
        input_img = np.ones((10, 10), dtype=np.uint16)
        target_img = np.ones((10, 10), dtype=np.uint16) * 2
        
        input_path = tmp_path / 'input.tif'
        target_path = tmp_path / 'target.tif'
        Image.fromarray(input_img, mode='I;16').save(str(input_path))
        Image.fromarray(target_img, mode='I;16').save(str(target_path))
        
        # Create manifest with these files
        df = pd.DataFrame({
            'input_ch1': [str(input_path)],
            'target_ch1': [str(target_path)],
        })
        manifest = DatasetManifest(file_index=df, pil_image_mode="I;16")
        
        # Create crop at top-left 5x5
        crops = [Crop(manifest_idx=0, x=0, y=0, width=5, height=5)]
        crop_manifest = CropManifest(crops, manifest=manifest)
        cfs = CropFileState(crop_manifest)
        
        # Update with crop index 0
        cfs.update(crop_idx=0, input_keys=['input_ch1'], target_keys=['target_ch1'])
        
        # Verify cropped images are loaded
        assert cfs.input_crop_raw is not None
        assert cfs.target_crop_raw is not None
        assert cfs.input_crop_raw.shape == (1, 5, 5)
        assert cfs.target_crop_raw.shape == (1, 5, 5)
        
        # Verify paths are recorded
        assert len(cfs.input_paths) == 1
        assert len(cfs.target_paths) == 1
    
    def test_update_with_offset_crop(self, sample_manifest, tmp_path):
        """Test update method with offset crop region."""
        # Create a patterned image to verify correct crop extraction
        input_img = np.arange(100, dtype=np.uint16).reshape(10, 10)
        target_img = np.arange(100, 200, dtype=np.uint16).reshape(10, 10)
        
        input_path = tmp_path / 'input.tif'
        target_path = tmp_path / 'target.tif'
        Image.fromarray(input_img, mode='I;16').save(str(input_path))
        Image.fromarray(target_img, mode='I;16').save(str(target_path))
        
        df = pd.DataFrame({
            'input_ch1': [str(input_path)],
            'target_ch1': [str(target_path)],
        })
        manifest = DatasetManifest(file_index=df, pil_image_mode="I;16")
        
        # Create crop at (x=1, y=2, width=3, height=3)
        crops = [Crop(manifest_idx=0, x=1, y=2, width=3, height=3)]
        crop_manifest = CropManifest(crops, manifest=manifest)
        cfs = CropFileState(crop_manifest)
        
        cfs.update(crop_idx=0, input_keys=['input_ch1'], target_keys=['target_ch1'])
        
        # Verify shape
        assert cfs.input_crop_raw.shape == (1, 3, 3)
        assert cfs.target_crop_raw.shape == (1, 3, 3)
        
        # Verify values match expected crop region
        full_img = np.arange(100).reshape(10, 10)
        expected_crop = full_img[2:5, 1:4]  # y:y+h, x:x+w
        np.testing.assert_array_equal(cfs.input_crop_raw[0], expected_crop)
    
    def test_update_multiple_crops_from_same_image(self, sample_manifest, tmp_path):
        """Test updating different crops from the same source image."""
        input_img = np.arange(100, dtype=np.uint16).reshape(10, 10)
        target_img = np.ones((10, 10), dtype=np.uint16)
        
        input_path = tmp_path / 'input.tif'
        target_path = tmp_path / 'target.tif'
        Image.fromarray(input_img, mode='I;16').save(str(input_path))
        Image.fromarray(target_img, mode='I;16').save(str(target_path))
        
        df = pd.DataFrame({
            'input_ch1': [str(input_path)],
            'target_ch1': [str(target_path)],
        })
        manifest = DatasetManifest(file_index=df, pil_image_mode="I;16")
        
        # Two crops from the same image
        crops = [
            Crop(manifest_idx=0, x=0, y=0, width=5, height=5),
            Crop(manifest_idx=0, x=5, y=5, width=5, height=5),
        ]
        crop_manifest = CropManifest(crops, manifest=manifest)
        cfs = CropFileState(crop_manifest)
        
        # Update to first crop
        cfs.update(crop_idx=0, input_keys=['input_ch1'], target_keys=['target_ch1'])
        first_crop = cfs.input_crop_raw.copy()
        
        # Update to second crop
        cfs.update(crop_idx=1, input_keys=['input_ch1'], target_keys=['target_ch1'])
        second_crop = cfs.input_crop_raw.copy()
        
        # Crops should be different (different regions)
        assert not np.array_equal(first_crop, second_crop)
        
        # Verify second crop is from the correct region
        full_img = np.arange(100).reshape(10, 10)
        expected = full_img[5:10, 5:10]
        np.testing.assert_array_equal(second_crop[0], expected)
    
    def test_update_maps_crop_idx_to_manifest_idx(self, sample_manifest, tmp_path):
        """Test that update correctly maps crop_idx to manifest_idx."""
        # Create two different image files
        img1 = np.ones((10, 10), dtype=np.uint16)
        img2 = np.ones((10, 10), dtype=np.uint16) * 2
        
        path1 = tmp_path / 'img1.tif'
        path2 = tmp_path / 'img2.tif'
        path_target = tmp_path / 'target.tif'
        
        Image.fromarray(img1, mode='I;16').save(str(path1))
        Image.fromarray(img2, mode='I;16').save(str(path2))
        Image.fromarray(np.ones((10, 10), dtype=np.uint16), mode='I;16').save(str(path_target))
        
        # Manifest with two images
        df = pd.DataFrame({
            'input_ch1': [str(path1), str(path2)],
            'target_ch1': [str(path_target), str(path_target)],
        })
        manifest = DatasetManifest(file_index=df, pil_image_mode="I;16")
        
        # Two crops from different manifest entries
        crops = [
            Crop(manifest_idx=0, x=0, y=0, width=5, height=5),
            Crop(manifest_idx=1, x=0, y=0, width=5, height=5),
        ]
        crop_manifest = CropManifest(crops, manifest=manifest)
        cfs = CropFileState(crop_manifest)
        
        # Update to crop from first manifest entry
        cfs.update(crop_idx=0, input_keys=['input_ch1'], target_keys=['target_ch1'])
        crop_from_img1 = cfs.input_crop_raw.copy()
        
        # Update to crop from second manifest entry
        cfs.update(crop_idx=1, input_keys=['input_ch1'], target_keys=['target_ch1'])
        crop_from_img2 = cfs.input_crop_raw.copy()
        
        # Crops should have different values since they come from different images
        assert np.allclose(crop_from_img1, 1.0)
        assert np.allclose(crop_from_img2, 2.0)
    
    def test_update_with_cache(self, sample_manifest, tmp_path):
        """Test that update leverages FileState's caching."""
        input_img = np.ones((10, 10), dtype=np.uint16)
        target_img = np.ones((10, 10), dtype=np.uint16) * 2
        
        input_path = tmp_path / 'input.tif'
        target_path = tmp_path / 'target.tif'
        Image.fromarray(input_img, mode='I;16').save(str(input_path))
        Image.fromarray(target_img, mode='I;16').save(str(target_path))
        
        df = pd.DataFrame({
            'input_ch1': [str(input_path)],
            'target_ch1': [str(target_path)],
        })
        manifest = DatasetManifest(file_index=df, pil_image_mode="I;16")
        
        crops = [
            Crop(manifest_idx=0, x=0, y=0, width=5, height=5),
            Crop(manifest_idx=0, x=5, y=5, width=5, height=5),
        ]
        crop_manifest = CropManifest(crops, manifest=manifest)
        
        # With cache capacity of 2, both input and target images should remain cached
        cfs = CropFileState(crop_manifest, cache_capacity=2)
        
        # First update
        cfs.update(crop_idx=0, input_keys=['input_ch1'], target_keys=['target_ch1'])
        assert len(cfs._file_state._cache) == 2  # both images in cache
        
        # Second update (same images, different crop)
        cfs.update(crop_idx=1, input_keys=['input_ch1'], target_keys=['target_ch1'])
        assert len(cfs._file_state._cache) == 2  # still both in cache (they weren't evicted)
    
    def test_update_updates_crop_state(self, sample_manifest, tmp_path):
        """Test that update correctly updates CropIndexState."""
        input_img = np.ones((10, 10), dtype=np.uint16)
        target_img = np.ones((10, 10), dtype=np.uint16)
        
        input_path = tmp_path / 'input.tif'
        target_path = tmp_path / 'target.tif'
        Image.fromarray(input_img, mode='I;16').save(str(input_path))
        Image.fromarray(target_img, mode='I;16').save(str(target_path))
        
        df = pd.DataFrame({
            'input_ch1': [str(input_path)],
            'target_ch1': [str(target_path)],
        })
        manifest = DatasetManifest(file_index=df, pil_image_mode="I;16")
        
        crops = [
            Crop(manifest_idx=0, x=1, y=2, width=3, height=4),
            Crop(manifest_idx=0, x=0, y=0, width=5, height=5),
        ]
        crop_manifest = CropManifest(crops, manifest=manifest)
        cfs = CropFileState(crop_manifest)
        
        # Initially stale
        assert cfs.crop_state.is_stale(0) is True
        
        # After first update
        cfs.update(crop_idx=0, input_keys=['input_ch1'], target_keys=['target_ch1'])
        assert cfs.crop_state.last_crop_idx == 0
        assert cfs.crop_state._last_crop.manifest_idx == 0
        assert cfs.crop_state._last_crop.x == 1
        assert cfs.crop_state._last_crop.y == 2
        assert cfs.crop_state._last_crop.width == 3
        assert cfs.crop_state._last_crop.height == 4
        assert cfs.crop_state.is_stale(0) is False
        assert cfs.crop_state.is_stale(1) is True
        
        # After second update
        cfs.update(crop_idx=1, input_keys=['input_ch1'], target_keys=['target_ch1'])
        assert cfs.crop_state.last_crop_idx == 1
        assert cfs.crop_state._last_crop.manifest_idx == 0
        assert cfs.crop_state._last_crop.x == 0
        assert cfs.crop_state._last_crop.y == 0
        assert cfs.crop_state._last_crop.width == 5
        assert cfs.crop_state._last_crop.height == 5
    
    def test_crop_info_property(self, sample_manifest, tmp_path):
        """Test that crop_info property provides public access to last crop."""
        input_img = np.ones((10, 10), dtype=np.uint16)
        target_img = np.ones((10, 10), dtype=np.uint16)
        
        input_path = tmp_path / 'input.tif'
        target_path = tmp_path / 'target.tif'
        Image.fromarray(input_img, mode='I;16').save(str(input_path))
        Image.fromarray(target_img, mode='I;16').save(str(target_path))
        
        df = pd.DataFrame({
            'input_ch1': [str(input_path)],
            'target_ch1': [str(target_path)],
        })
        manifest = DatasetManifest(file_index=df, pil_image_mode="I;16")
        
        crops = [
            Crop(manifest_idx=0, x=1, y=2, width=4, height=4),
            Crop(manifest_idx=0, x=5, y=5, width=3, height=3),
        ]
        crop_manifest = CropManifest(crops, manifest=manifest)
        cfs = CropFileState(crop_manifest)
        
        # Initially None
        assert cfs.crop_info is None
        
        # After first update
        cfs.update(crop_idx=0, input_keys=['input_ch1'], target_keys=['target_ch1'])
        assert cfs.crop_info is not None
        assert cfs.crop_info.manifest_idx == 0
        assert cfs.crop_info.x == 1
        assert cfs.crop_info.y == 2
        assert cfs.crop_info.width == 4
        assert cfs.crop_info.height == 4
        
        # After second update
        cfs.update(crop_idx=1, input_keys=['input_ch1'], target_keys=['target_ch1'])
        assert cfs.crop_info is not None
        assert cfs.crop_info.manifest_idx == 0
        assert cfs.crop_info.x == 5
        assert cfs.crop_info.y == 5
        assert cfs.crop_info.width == 3
        assert cfs.crop_info.height == 3
        
        # After reset
        cfs.reset()
        assert cfs.crop_info is None


class TestCropManifestSerialization:
    """Test suite for CropManifest serialization methods."""

    def test_from_config_invalid_config_type(self):
        """Test CropManifest.from_config raises TypeError when config is not a dict."""
        with pytest.raises(TypeError, match="CropManifest.from_config: expected config to be a dict"):
            CropManifest.from_config("not_a_dict")

    def test_from_config_missing_crops(self):
        """Test CropManifest.from_config raises ValueError when crops is missing."""
        config = {"manifest": {"file_index": [], "pil_image_mode": "I;16"}}
        with pytest.raises(ValueError, match="CropManifest.from_config: missing 'crops' key in config"):
            CropManifest.from_config(config)

    def test_from_config_invalid_crops_type(self):
        """Test CropManifest.from_config raises TypeError when crops is not a list."""
        config = {"crops": "not_a_list", "manifest": {"file_index": [], "pil_image_mode": "I;16"}}
        with pytest.raises(TypeError, match="CropManifest.from_config: expected 'crops' to be a list"):
            CropManifest.from_config(config)

    def test_from_config_missing_manifest(self):
        """Test CropManifest.from_config raises ValueError when manifest is missing."""
        config = {"crops": [{"manifest_idx": 0, "x": 0, "y": 0, "width": 10, "height": 10}]}
        with pytest.raises(ValueError, match="CropManifest.from_config: missing 'manifest' key in config"):
            CropManifest.from_config(config)

    def test_from_config_invalid_manifest_type(self):
        """Test CropManifest.from_config raises TypeError when manifest is not a dict."""
        config = {"crops": [{"manifest_idx": 0, "x": 0, "y": 0, "width": 10, "height": 10}], "manifest": "not_a_dict"}
        with pytest.raises(TypeError, match="CropManifest.from_config: expected 'manifest' to be a dict"):
            CropManifest.from_config(config)


class TestCropFileStateSerialization:
    """Test suite for CropFileState serialization methods."""

    def test_from_config_invalid_config_type(self):
        """Test CropFileState.from_config raises TypeError when config is not a dict."""
        with pytest.raises(TypeError, match="CropFileState.from_config: expected config to be a dict"):
            CropFileState.from_config("not_a_dict")

    def test_from_config_missing_crop_collection(self):
        """Test CropFileState.from_config raises ValueError when crop_collection is missing."""
        config = {"cache_capacity": 10, "crop_collection": None}
        with pytest.raises(ValueError, match="CropFileState.from_config: missing 'crop_collection' key in config"):
            CropFileState.from_config(config)

    def test_from_config_invalid_crop_collection_type(self):
        """Test CropFileState.from_config raises TypeError when crop_collection is not a dict."""
        config = {"cache_capacity": 10, "crop_collection": "not_a_dict"}
        with pytest.raises(TypeError, match="CropFileState.from_config: expected 'crop_collection' to be a dict"):
            CropFileState.from_config(config)
