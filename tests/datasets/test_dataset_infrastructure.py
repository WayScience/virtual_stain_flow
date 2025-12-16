import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock
from virtual_stain_flow.datasets.ds_engine.manifest import DatasetManifest, IndexState, FileState

class TestDatasetManifest:
    
    def test_init_valid(self):
        """Test DatasetManifest initialization with valid inputs."""
        df = pd.DataFrame({
            'channel1': ['/path/to/img1.tif', '/path/to/img2.tif'],
            'channel2': ['/path/to/img3.tif', '/path/to/img4.tif']
        })
        manifest = DatasetManifest(file_index=df, pil_image_mode="RGB", check_exists=False)
        assert manifest.file_index.equals(df)
        assert manifest.pil_image_mode == "RGB"
    
    def test_init_default_pil_mode(self):
        """Test DatasetManifest uses default PIL mode."""
        df = pd.DataFrame({'channel1': ['/path/to/img1.tif']})
        manifest = DatasetManifest(file_index=df, check_exists=False)
        assert manifest.pil_image_mode == "I;16"
    
    def test_init_empty_dataframe_raises(self):
        """Test DatasetManifest raises ValueError for empty DataFrame."""
        with pytest.raises(ValueError, match="file_index must be a non-empty DataFrame"):
            DatasetManifest(file_index=pd.DataFrame())
    
    def test_init_non_dataframe_raises(self):
        """Test DatasetManifest raises ValueError for non-DataFrame input."""
        with pytest.raises(ValueError, match="file_index must be a non-empty DataFrame"):
            DatasetManifest(file_index="not_a_dataframe")
    
    def test_init_invalid_pil_mode_raises(self):
        """Test DatasetManifest raises ValueError for invalid PIL mode."""
        df = pd.DataFrame({'channel1': ['/path/to/img1.tif']})
        with pytest.raises(ValueError, match="Invalid pil_image_mode"):
            DatasetManifest(file_index=df, pil_image_mode="INVALID_MODE")
    
    def test_init_non_string_pil_mode_raises(self):
        """Test DatasetManifest raises ValueError for non-string PIL mode."""
        df = pd.DataFrame({'channel1': ['/path/to/img1.tif']})
        with pytest.raises(ValueError, match="Expected pil_image_mode to be a string"):
            DatasetManifest(file_index=df, pil_image_mode=123)
    
    def test_init_non_pathlike_entries_raises(self):
        """Test DatasetManifest raises TypeError for non-path-like entries."""
        df = pd.DataFrame({'channel1': [123, 456]})
        with pytest.raises(TypeError, match="file_index has non-path-like entries"):
            DatasetManifest(file_index=df)
    
    def test_len(self):
        """Test DatasetManifest __len__ method."""
        df = pd.DataFrame({
            'channel1': ['/path/to/img1.tif', '/path/to/img2.tif', '/path/to/img3.tif']
        })
        manifest = DatasetManifest(file_index=df)
        assert len(manifest) == 3
    
    def test_n_channels(self):
        """Test DatasetManifest n_channels property."""
        df = pd.DataFrame({
            'ch1': ['/path/1.tif'], 
            'ch2': ['/path/2.tif'], 
            'ch3': ['/path/3.tif']
        })
        manifest = DatasetManifest(file_index=df)
        assert manifest.n_channels == 3
    
    def test_channel_keys(self):
        """Test DatasetManifest channel_keys property."""
        df = pd.DataFrame({
            'channel_a': ['/path/1.tif'], 
            'channel_b': ['/path/2.tif']
        })
        manifest = DatasetManifest(file_index=df)
        assert manifest.channel_keys == ['channel_a', 'channel_b']
    
    def test_get_paths_for_keys(self):
        """Test DatasetManifest get_paths_for_keys method."""
        df = pd.DataFrame({
            'ch1': ['/path/1.tif', '/path/3.tif'],
            'ch2': ['/path/2.tif', '/path/4.tif']
        })
        manifest = DatasetManifest(file_index=df)
        paths = manifest.get_paths_for_keys(0, ['ch1', 'ch2'])
        assert paths == [Path('/path/1.tif'), Path('/path/2.tif')]
        
        paths = manifest.get_paths_for_keys(1, ['ch2'])
        assert paths == [Path('/path/4.tif')]
    
    def test_read_image_file_not_found(self):
        """Test DatasetManifest read_image raises FileNotFoundError for missing file."""
        df = pd.DataFrame({'ch1': ['/nonexistent.tif']})
        manifest = DatasetManifest(file_index=df)
        with pytest.raises(FileNotFoundError, match="File not found"):
            manifest.read_image(Path('/nonexistent.tif'))
    
    @patch("virtual_stain_flow.datasets.ds_engine.manifest.Image.open")
    def test_read_image_success(self, mock_open):
        """Test DatasetManifest read_image successfully reads image."""
        # Mock PIL Image
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_open.return_value.__enter__.return_value = mock_img
        
        df = pd.DataFrame({'ch1': ['/test.tif']})
        manifest = DatasetManifest(file_index=df)
        
        # Mock numpy array
        mock_array = np.ones((100, 100), dtype=np.uint16)
        with patch("virtual_stain_flow.datasets.ds_engine.manifest.np.asarray", return_value=mock_array):
            
            with tempfile.NamedTemporaryFile(suffix='.tif') as tmp:
                result = manifest.read_image(Path(tmp.name))
                assert np.array_equal(result, mock_array)
    
    @patch("virtual_stain_flow.datasets.ds_engine.manifest.Image.open")
    def test_read_image_invalid_shape(self, mock_open):
        """Test DatasetManifest read_image raises ValueError for invalid shape."""
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_open.return_value.__enter__.return_value = mock_img

        df = pd.DataFrame({'ch1': ['/test.tif']})
        manifest = DatasetManifest(file_index=df)
        
        # Mock 1D array (invalid)
        bad = np.ones((100,), dtype=np.uint16)
        with patch("virtual_stain_flow.datasets.ds_engine.manifest.np.asarray", return_value=bad):            
            with tempfile.NamedTemporaryFile(suffix='.tif') as tmp:
                with pytest.raises(ValueError, match="Unsupported image shape"):
                    manifest.read_image(Path(tmp.name))


class TestIndexState:
    
    def test_init_default(self):
        """Test IndexState initialization with default values."""
        state = IndexState()
        assert state.last is None
    
    def test_init_with_value(self):
        """Test IndexState initialization with specific value."""
        state = IndexState(last=5)
        assert state.last == 5
    
    def test_reset(self):
        """Test IndexState reset method."""
        state = IndexState(last=10)
        state.reset()
        assert state.last is None
    
    def test_is_stale_both_none(self):
        """Test IndexState is_stale when both last and idx are None."""
        state = IndexState()
        assert state.is_stale(None) is True
    
    def test_is_stale_last_none(self):
        """Test IndexState is_stale when last is None."""
        state = IndexState()
        assert state.is_stale(5) is True
    
    def test_is_stale_idx_none(self):
        """Test IndexState is_stale when idx is None."""
        state = IndexState(last=5)
        assert state.is_stale(None) is True
    
    def test_is_stale_different_values(self):
        """Test IndexState is_stale when values are different."""
        state = IndexState(last=5)
        assert state.is_stale(10) is True
    
    def test_is_stale_same_values(self):
        """Test IndexState is_stale when values are the same."""
        state = IndexState(last=5)
        assert state.is_stale(5) is False
    
    def test_update_stale(self):
        """Test IndexState update when index is stale."""
        state = IndexState()
        state.update(10)
        assert state.last == 10
    
    def test_update_not_stale(self):
        """Test IndexState update when index is not stale."""
        state = IndexState(last=5)
        state.update(5)  # Same value, not stale
        assert state.last == 5


class TestFileState:
    
    def create_test_manifest(self):
        """Helper to create a test manifest."""
        df = pd.DataFrame({
            'ch1': ['/path/1.tif', '/path/3.tif'],
            'ch2': ['/path/2.tif', '/path/4.tif']
        })
        return DatasetManifest(file_index=df)
    
    def test_init_default(self):
        """Test FileState initialization with default values."""
        manifest = self.create_test_manifest()
        state = FileState(manifest=manifest)
        assert state.manifest == manifest
        assert state.input_paths == []
        assert state.target_paths == []
        assert state.input_image_raw is None
        assert state.target_image_raw is None
        assert state.cache_capacity == manifest.n_channels
        assert len(state._cache) == 0
    
    def test_init_with_cache_capacity(self):
        """Test FileState initialization with specific cache capacity."""
        manifest = self.create_test_manifest()
        state = FileState(manifest=manifest, cache_capacity=10)
        assert state.cache_capacity == 10
    
    def test_init_unbounded_cache(self):
        """Test FileState initialization with unbounded cache."""
        manifest = self.create_test_manifest()
        state = FileState(manifest=manifest, cache_capacity=-1)
        assert state.cache_capacity == -1
    
    def test_init_invalid_cache_capacity(self):
        """Test FileState raises ValueError for invalid cache capacity."""
        manifest = self.create_test_manifest()
        with pytest.raises(ValueError, match="cache_capacity must be -1"):
            FileState(manifest=manifest, cache_capacity=-2)
    
    def test_init_zero_cache_capacity(self):
        """Test FileState raises ValueError for zero cache capacity."""
        manifest = self.create_test_manifest()
        with pytest.raises(ValueError, match="cache_capacity=0 disables caching"):
            FileState(manifest=manifest, cache_capacity=0)
    
    def test_reset(self):
        """Test FileState reset method."""
        manifest = self.create_test_manifest()
        state = FileState(manifest=manifest)
        
        # Set some state
        state.input_paths = [Path('/test.tif')]
        state.target_paths = [Path('/test2.tif')]
        state.input_image_raw = np.ones((1, 100, 100))
        state.target_image_raw = np.ones((1, 100, 100))
        
        state.reset()
        
        assert state.input_paths == []
        assert state.target_paths == []
        assert state.input_image_raw is None
        assert state.target_image_raw is None
    
    def test_stack_channels_2d(self):
        """Test FileState _stack_channels with 2D arrays."""
        manifest = self.create_test_manifest()
        state = FileState(manifest=manifest)
        
        arrays = [np.ones((100, 100)), np.zeros((100, 100))]
        result = state._stack_channels(arrays)
        
        assert result.shape == (2, 100, 100)
        assert np.array_equal(result[0], np.ones((100, 100)))
        assert np.array_equal(result[1], np.zeros((100, 100)))
    
    def test_stack_channels_3d(self):
        """Test FileState _stack_channels with 3D arrays."""
        manifest = self.create_test_manifest()
        state = FileState(manifest=manifest)
        
        arrays = [np.ones((100, 100, 3)), np.zeros((100, 100, 3))]
        result = state._stack_channels(arrays)
        
        assert result.shape == (2, 100, 100, 3)
    
    def test_stack_channels_empty(self):
        """Test FileState _stack_channels with empty list."""
        manifest = self.create_test_manifest()
        state = FileState(manifest=manifest)
        
        result = state._stack_channels([])
        assert result.shape == (0,)
    
    def test_stack_channels_invalid_ndim(self):
        """Test FileState _stack_channels raises ValueError for invalid ndim."""
        manifest = self.create_test_manifest()
        state = FileState(manifest=manifest)
        
        arrays = [np.ones((100,))]  # 1D array
        with pytest.raises(ValueError, match="Unsupported per-channel array ndim"):
            state._stack_channels(arrays)
    
    def test_ensure_same_spatial_shape_valid(self):
        """Test FileState _ensure_same_spatial_shape with valid shapes."""
        manifest = self.create_test_manifest()
        state = FileState(manifest=manifest)
        
        arrays = [np.ones((100, 100)), np.zeros((100, 100))]
        paths = [Path('/1.tif'), Path('/2.tif')]
        
        # Should not raise
        state._ensure_same_spatial_shape(arrays, paths)
    
    def test_ensure_same_spatial_shape_mismatch(self):
        """Test FileState _ensure_same_spatial_shape raises for shape mismatch."""
        manifest = self.create_test_manifest()
        state = FileState(manifest=manifest)
        
        arrays = [np.ones((100, 100)), np.zeros((50, 50))]
        paths = [Path('/1.tif'), Path('/2.tif')]
        
        with pytest.raises(ValueError, match="Spatial shape mismatch"):
            state._ensure_same_spatial_shape(arrays, paths)
    
    def test_ensure_same_spatial_shape_empty(self):
        """Test FileState _ensure_same_spatial_shape with empty arrays."""
        manifest = self.create_test_manifest()
        state = FileState(manifest=manifest)
        
        # Should not raise
        state._ensure_same_spatial_shape([], [])
    
    @patch.object(DatasetManifest, 'read_image')
    def test_update_basic(self, mock_read_image):
        """Test FileState update method basic functionality."""
        manifest = self.create_test_manifest()
        state = FileState(manifest=manifest, cache_capacity=5)
        
        # Mock image reading
        mock_read_image.side_effect = [
            np.ones((100, 100)),  # input ch1
            np.zeros((100, 100))  # target ch2
        ]
        
        state.update(idx=0, input_keys=['ch1'], target_keys=['ch2'])
        
        assert len(state.input_paths) == 1
        assert len(state.target_paths) == 1
        assert state.input_image_raw.shape == (1, 100, 100)
        assert state.target_image_raw.shape == (1, 100, 100)
        assert len(state._cache) == 2
