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
import tempfile
import json
from PIL import Image

from virtual_stain_flow.datasets.ds_engine.crop_manifest import (
    DatasetManifest,
    Crop,
    CropManifest,
    CropIndexState,
    CropFileState,
)


@pytest.fixture
def sample_manifest():
    """Create a simple manifest for testing."""
    df = pd.DataFrame({
        'input_ch1': ['/path/img_0_in.tif', '/path/img_1_in.tif', '/path/img_2_in.tif'],
        'target_ch1': ['/path/img_0_tar.tif', '/path/img_1_tar.tif', '/path/img_2_tar.tif'],
    })
    return DatasetManifest(file_index=df, pil_image_mode="I;16")


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
            Crop(0, 0, 0, 100, 100),
            Crop(1, 50, 50, 100, 100),
        ]
        cc = CropManifest(crops, sample_manifest)
        assert len(cc) == 2
        assert cc.get_crop(0).manifest_idx == 0
        assert cc.get_crop(1).manifest_idx == 1
    
    def test_init_empty_raises(self, sample_manifest):
        """Test that empty crops list raises ValueError."""
        with pytest.raises(ValueError, match="crops list cannot be empty"):
            CropManifest([], sample_manifest)
    
    def test_init_out_of_bounds_manifest_idx_raises(self, sample_manifest):
        """Test that out-of-bounds manifest_idx raises IndexError."""
        crops = [Crop(0, 0, 0, 100, 100), Crop(10, 0, 0, 100, 100)]
        with pytest.raises(IndexError, match="One or more crop.manifest_idx values are out of bounds"):
            CropManifest(crops, sample_manifest)
    
    def test_get_crop_out_of_bounds_raises(self, sample_manifest):
        """Test that accessing invalid crop_idx raises IndexError."""
        crops = [Crop(0, 0, 0, 100, 100)]
        cc = CropManifest(crops, sample_manifest)
        with pytest.raises(IndexError, match="crop_idx 5 out of range"):
            cc.get_crop(5)
    
    def test_to_config(self, sample_manifest):
        """Test config serialization."""
        crops = [Crop(0, 10, 20, 100, 100), Crop(1, 0, 0, 256, 256)]
        cc = CropManifest(crops, sample_manifest)
        config = cc.to_config()
        assert 'crops' in config
        assert len(config['crops']) == 2
        assert config['crops'][0]['manifest_idx'] == 0
        assert config['crops'][0]['x'] == 10
        assert config['crops'][1]['width'] == 256
    
    def test_from_config(self, sample_manifest):
        """Test config deserialization."""
        config = {
            'crops': [
                {'manifest_idx': 0, 'x': 10, 'y': 20, 'width': 100, 'height': 100},
                {'manifest_idx': 1, 'x': 0, 'y': 0, 'width': 256, 'height': 256},
            ]
        }
        cc = CropManifest.from_config(config, sample_manifest)
        assert len(cc) == 2
        assert cc.get_crop(0).x == 10
        assert cc.get_crop(1).height == 256


class TestCropIndexState:
    """Tests for CropIndexState."""
    
    def test_init(self, sample_manifest):
        """Test CropIndexState initialization."""
        crops = [Crop(0, 0, 0, 100, 100)]
        cc = CropManifest(crops, sample_manifest)
        cs = CropIndexState(cc)
        assert cs.last_crop_idx is None
        assert cs._last_crop is None
    
    def test_is_stale_initially_true(self, sample_manifest):
        """Test that is_stale returns True initially."""
        crops = [Crop(0, 0, 0, 100, 100)]
        cc = CropManifest(crops, sample_manifest)
        cs = CropIndexState(cc)
        assert cs.is_stale(0) is True
    
    def test_get_and_update(self, sample_manifest):
        """Test get_and_update updates state and returns crop."""
        crops = [Crop(0, 10, 20, 100, 100), Crop(1, 0, 0, 256, 256)]
        cc = CropManifest(crops, sample_manifest)
        cs = CropIndexState(cc)
        
        # First call should be stale and update
        crop = cs.get_and_update(0)
        assert crop.x == 10
        assert cs.last_crop_idx == 0
        assert cs._last_crop.manifest_idx == 0
        assert cs._last_crop.x == 10
        assert cs._last_crop.y == 20
        assert cs._last_crop.width == 100
        assert cs._last_crop.height == 100
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
        crops = [Crop(0, 0, 0, 100, 100)]
        cc = CropManifest(crops, sample_manifest)
        cs = CropIndexState(cc)
        cs.get_and_update(0)
        
        cs.reset()
        assert cs.last_crop_idx is None
        assert cs._last_crop is None


class TestCropFileState:
    """Tests for CropFileState (requires mocking FileState)."""
    
    def test_init(self, sample_manifest):
        """Test CropFileState initialization."""
        crops = [Crop(0, 0, 0, 100, 100)]
        cc = CropManifest(crops, sample_manifest)
        cfs = CropFileState(cc, cache_capacity=10)
        
        assert cfs.manifest is sample_manifest
        assert cfs.input_crop_raw is None
        assert cfs.target_crop_raw is None
        assert len(cfs.input_paths) == 0
    
    def test_extract_crop_3d(self, sample_manifest):
        """Test crop extraction from (C, H, W) image."""
        crops = [Crop(0, 10, 20, 50, 50)]
        cc = CropManifest(crops, sample_manifest)
        cfs = CropFileState(cc)
        
        # Create dummy (3, 100, 100) image
        image = np.arange(30000).reshape(3, 100, 100)
        crop = crops[0]
        
        cropped = cfs._extract_crop(image, crop)
        assert cropped.shape == (3, 50, 50)
        assert np.allclose(
            cropped,
            image[:, 20:70, 10:60]
        )
    
    def test_extract_crop_non_3d(self, sample_manifest):
        """Test crop extraction from non (C, H, W) image."""
        crops = [Crop(0, 0, 0, 50, 50)]
        cc = CropManifest(crops, sample_manifest)
        cfs = CropFileState(cc)
        
        # Create dummy (2, 100, 100, 3) image
        image = np.random.rand(2, 100, 100, 3)
        
        # anticipate error raised for non-3D image
        crop = crops[0]
        with pytest.raises(ValueError, match="Unexpected image shape"):
            cfs._extract_crop(image, crop)
    
    def test_extract_crop_out_of_bounds_raises(self, sample_manifest):
        """Test that out-of-bounds crops raise ValueError."""
        crops = [Crop(0, 50, 50, 100, 100)]  # Crop extends beyond 100x100 image
        cc = CropManifest(crops, sample_manifest)
        cfs = CropFileState(cc)
        
        image = np.random.rand(2, 100, 100)
        with pytest.raises(ValueError, match="exceeds image bounds"):
            cfs._extract_crop(image, crops[0])
    
    def test_reset(self, sample_manifest):
        """Test reset clears all state."""
        crops = [Crop(0, 0, 0, 100, 100)]
        cc = CropManifest(crops, sample_manifest)
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
        input_img = np.ones((100, 100), dtype=np.uint16)
        target_img = np.ones((100, 100), dtype=np.uint16) * 2
        
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
        
        # Create crop at top-left 50x50
        crops = [Crop(manifest_idx=0, x=0, y=0, width=50, height=50)]
        crop_manifest = CropManifest(crops, manifest)
        cfs = CropFileState(crop_manifest)
        
        # Update with crop index 0
        cfs.update(crop_idx=0, input_keys=['input_ch1'], target_keys=['target_ch1'])
        
        # Verify cropped images are loaded
        assert cfs.input_crop_raw is not None
        assert cfs.target_crop_raw is not None
        assert cfs.input_crop_raw.shape == (1, 50, 50)
        assert cfs.target_crop_raw.shape == (1, 50, 50)
        
        # Verify paths are recorded
        assert len(cfs.input_paths) == 1
        assert len(cfs.target_paths) == 1
    
    def test_update_with_offset_crop(self, sample_manifest, tmp_path):
        """Test update method with offset crop region."""
        # Create a patterned image to verify correct crop extraction
        input_img = np.arange(10000, dtype=np.uint16).reshape(100, 100)
        target_img = np.arange(10000, 20000, dtype=np.uint16).reshape(100, 100)
        
        input_path = tmp_path / 'input.tif'
        target_path = tmp_path / 'target.tif'
        Image.fromarray(input_img, mode='I;16').save(str(input_path))
        Image.fromarray(target_img, mode='I;16').save(str(target_path))
        
        df = pd.DataFrame({
            'input_ch1': [str(input_path)],
            'target_ch1': [str(target_path)],
        })
        manifest = DatasetManifest(file_index=df, pil_image_mode="I;16")
        
        # Create crop at (x=10, y=20, width=30, height=30)
        crops = [Crop(manifest_idx=0, x=10, y=20, width=30, height=30)]
        crop_manifest = CropManifest(crops, manifest)
        cfs = CropFileState(crop_manifest)
        
        cfs.update(crop_idx=0, input_keys=['input_ch1'], target_keys=['target_ch1'])
        
        # Verify shape
        assert cfs.input_crop_raw.shape == (1, 30, 30)
        assert cfs.target_crop_raw.shape == (1, 30, 30)
        
        # Verify values match expected crop region
        full_img = np.arange(10000).reshape(100, 100)
        expected_crop = full_img[20:50, 10:40]  # y:y+h, x:x+w
        np.testing.assert_array_equal(cfs.input_crop_raw[0], expected_crop)
    
    def test_update_multiple_crops_from_same_image(self, sample_manifest, tmp_path):
        """Test updating different crops from the same source image."""
        input_img = np.arange(10000, dtype=np.uint16).reshape(100, 100)
        target_img = np.ones((100, 100), dtype=np.uint16)
        
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
            Crop(manifest_idx=0, x=0, y=0, width=50, height=50),
            Crop(manifest_idx=0, x=50, y=50, width=50, height=50),
        ]
        crop_manifest = CropManifest(crops, manifest)
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
        full_img = np.arange(10000).reshape(100, 100)
        expected = full_img[50:100, 50:100]
        np.testing.assert_array_equal(second_crop[0], expected)
    
    def test_update_maps_crop_idx_to_manifest_idx(self, sample_manifest, tmp_path):
        """Test that update correctly maps crop_idx to manifest_idx."""
        # Create two different image files
        img1 = np.ones((100, 100), dtype=np.uint16)
        img2 = np.ones((100, 100), dtype=np.uint16) * 2
        
        path1 = tmp_path / 'img1.tif'
        path2 = tmp_path / 'img2.tif'
        path_target = tmp_path / 'target.tif'
        
        Image.fromarray(img1, mode='I;16').save(str(path1))
        Image.fromarray(img2, mode='I;16').save(str(path2))
        Image.fromarray(np.ones((100, 100), dtype=np.uint16), mode='I;16').save(str(path_target))
        
        # Manifest with two images
        df = pd.DataFrame({
            'input_ch1': [str(path1), str(path2)],
            'target_ch1': [str(path_target), str(path_target)],
        })
        manifest = DatasetManifest(file_index=df, pil_image_mode="I;16")
        
        # Two crops from different manifest entries
        crops = [
            Crop(manifest_idx=0, x=0, y=0, width=50, height=50),
            Crop(manifest_idx=1, x=0, y=0, width=50, height=50),
        ]
        crop_manifest = CropManifest(crops, manifest)
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
        input_img = np.ones((100, 100), dtype=np.uint16)
        target_img = np.ones((100, 100), dtype=np.uint16) * 2
        
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
            Crop(manifest_idx=0, x=0, y=0, width=25, height=25),
            Crop(manifest_idx=0, x=25, y=25, width=25, height=25),
        ]
        crop_manifest = CropManifest(crops, manifest)
        
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
        input_img = np.ones((100, 100), dtype=np.uint16)
        target_img = np.ones((100, 100), dtype=np.uint16)
        
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
            Crop(manifest_idx=0, x=10, y=20, width=30, height=40),
            Crop(manifest_idx=0, x=0, y=0, width=50, height=50),
        ]
        crop_manifest = CropManifest(crops, manifest)
        cfs = CropFileState(crop_manifest)
        
        # Initially stale
        assert cfs.crop_state.is_stale(0) is True
        
        # After first update
        cfs.update(crop_idx=0, input_keys=['input_ch1'], target_keys=['target_ch1'])
        assert cfs.crop_state.last_crop_idx == 0
        assert cfs.crop_state._last_crop.manifest_idx == 0
        assert cfs.crop_state._last_crop.x == 10
        assert cfs.crop_state._last_crop.y == 20
        assert cfs.crop_state._last_crop.width == 30
        assert cfs.crop_state._last_crop.height == 40
        assert cfs.crop_state.is_stale(0) is False
        assert cfs.crop_state.is_stale(1) is True
        
        # After second update
        cfs.update(crop_idx=1, input_keys=['input_ch1'], target_keys=['target_ch1'])
        assert cfs.crop_state.last_crop_idx == 1
        assert cfs.crop_state._last_crop.manifest_idx == 0
        assert cfs.crop_state._last_crop.x == 0
        assert cfs.crop_state._last_crop.y == 0
        assert cfs.crop_state._last_crop.width == 50
        assert cfs.crop_state._last_crop.height == 50
    
    def test_crop_info_property(self, sample_manifest, tmp_path):
        """Test that crop_info property provides public access to last crop."""
        input_img = np.ones((100, 100), dtype=np.uint16)
        target_img = np.ones((100, 100), dtype=np.uint16)
        
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
            Crop(manifest_idx=0, x=5, y=10, width=40, height=45),
            Crop(manifest_idx=0, x=50, y=50, width=30, height=30),
        ]
        crop_manifest = CropManifest(crops, manifest)
        cfs = CropFileState(crop_manifest)
        
        # Initially None
        assert cfs.crop_info is None
        
        # After first update
        cfs.update(crop_idx=0, input_keys=['input_ch1'], target_keys=['target_ch1'])
        assert cfs.crop_info is not None
        assert cfs.crop_info.manifest_idx == 0
        assert cfs.crop_info.x == 5
        assert cfs.crop_info.y == 10
        assert cfs.crop_info.width == 40
        assert cfs.crop_info.height == 45
        
        # After second update
        cfs.update(crop_idx=1, input_keys=['input_ch1'], target_keys=['target_ch1'])
        assert cfs.crop_info is not None
        assert cfs.crop_info.manifest_idx == 0
        assert cfs.crop_info.x == 50
        assert cfs.crop_info.y == 50
        assert cfs.crop_info.width == 30
        assert cfs.crop_info.height == 30
        
        # After reset
        cfs.reset()
        assert cfs.crop_info is None
