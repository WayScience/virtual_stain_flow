import pytest
import numpy as np
from unittest.mock import patch
from virtual_stain_flow.transforms.normalizations import MaxScaleNormalize, ZScoreNormalize

class TestMaxScaleNormalize:
    """Test cases for MaxScaleNormalize transform."""
    
    def test_init_with_float_normalization_factor(self):
        """Test initialization with float normalization factor."""
        transform = MaxScaleNormalize(normalization_factor=255.0)
        assert transform.normalization_factor == 255.0
        assert transform._name == "MaxScaleNormalize"
        assert transform.p == 1.0
    
    def test_init_with_int_normalization_factor(self):
        """Test initialization with int normalization factor."""
        transform = MaxScaleNormalize(normalization_factor=255)
        assert transform.normalization_factor == 255.0
        assert isinstance(transform.normalization_factor, float)
    
    def test_init_with_16bit_literal(self):
        """Test initialization with '16bit' literal."""
        transform = MaxScaleNormalize(normalization_factor='16bit')
        assert transform.normalization_factor == 2**16 - 1
        assert transform.normalization_factor == 65535.0
    
    def test_init_with_8bit_literal(self):
        """Test initialization with '8bit' literal."""
        transform = MaxScaleNormalize(normalization_factor='8bit')
        assert transform.normalization_factor == 2**8 - 1
        assert transform.normalization_factor == 255.0
    
    def test_init_with_invalid_literal(self):
        """Test initialization with invalid literal raises ValueError."""
        with pytest.raises(ValueError, match="Allowed literals of normalization_factor are '16bit' and '8bit'"):
            MaxScaleNormalize(normalization_factor='32bit')
    
    def test_init_with_invalid_type(self):
        """Test initialization with invalid type raises TypeError."""
        with pytest.raises(TypeError, match="Expected normalization factor to be a number"):
            MaxScaleNormalize(normalization_factor=[255])
    
    def test_init_with_zero_normalization_factor(self):
        """Test initialization with zero normalization factor raises ValueError."""
        with pytest.raises(ValueError, match="Normalization factor must be greater than zero"):
            MaxScaleNormalize(normalization_factor=0)
    
    def test_init_with_negative_normalization_factor(self):
        """Test initialization with negative normalization factor raises ValueError."""
        with pytest.raises(ValueError, match="Normalization factor must be greater than zero"):
            MaxScaleNormalize(normalization_factor=-1.0)
    
    def test_init_with_custom_name_and_p(self):
        """Test initialization with custom name and probability."""
        transform = MaxScaleNormalize(
            normalization_factor=100.0,
            name="CustomNormalize",
            p=0.5
        )
        assert transform._name == "CustomNormalize"
        assert transform.p == 0.5
    
    def test_repr(self):
        """Test string representation of the transform."""
        transform = MaxScaleNormalize(normalization_factor=255.0, name="TestNorm", p=0.8)
        expected = "MaxScaleNormalize(name=TestNorm, normalization_factor=255.0, p=0.8)"
        assert repr(transform) == expected
    
    def test_apply_with_valid_image(self):
        """Test apply method with valid numpy array."""
        transform = MaxScaleNormalize(normalization_factor=255.0)
        img = np.array([[100, 200], [50, 255]], dtype=np.uint8)
        result = transform.apply(img)
        
        expected = img / 255.0
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_apply_with_invalid_input(self):
        """Test apply method with invalid input type raises TypeError."""
        transform = MaxScaleNormalize(normalization_factor=255.0)
        with pytest.raises(TypeError, match="Expected input image to be a NumPy array"):
            transform.apply([100, 200, 50, 255])
    
    def test_to_config(self):
        """Test to_config method returns correct configuration."""
        transform = MaxScaleNormalize(normalization_factor=255.0, name="TestNorm", p=0.8)
        config = transform.to_config()
        
        expected = {
            "class": "MaxScaleNormalize",
            "name": "TestNorm",
            "params": {
                "normalization_factor": 255.0,
                "p": 0.8
            }
        }
        assert config == expected


class TestZScoreNormalize:
    """Test cases for ZScoreNormalize transform."""
    
    def test_init_with_default_values(self):
        """Test initialization with default values."""
        transform = ZScoreNormalize()
        assert transform._name == "ZScoreNormalize"
        assert transform.mean is None
        assert transform.std is None
        assert transform.p == 1.0
    
    def test_init_with_custom_mean_and_std(self):
        """Test initialization with custom mean and std."""
        transform = ZScoreNormalize(mean=0.5, std=0.2)
        assert transform.mean == 0.5
        assert transform.std == 0.2
    
    def test_init_with_int_mean_and_std(self):
        """Test initialization with integer mean and std."""
        transform = ZScoreNormalize(mean=1, std=2)
        assert transform.mean == 1
        assert transform.std == 2
    
    def test_init_with_invalid_mean_type(self):
        """Test initialization with invalid mean type raises TypeError."""
        with pytest.raises(TypeError, match="Expected mean to be a number"):
            ZScoreNormalize(mean="invalid")
    
    def test_init_with_invalid_std_type(self):
        """Test initialization with invalid std type raises TypeError."""
        with pytest.raises(TypeError, match="Expected std to be a number"):
            ZScoreNormalize(std="invalid")
    
    def test_init_with_zero_std(self):
        """Test initialization with zero std raises ValueError."""
        with pytest.raises(ValueError, match="Standard deviation must be greater than zero"):
            ZScoreNormalize(std=0)
    
    def test_init_with_negative_std(self):
        """Test initialization with negative std raises ValueError."""
        with pytest.raises(ValueError, match="Standard deviation must be greater than zero"):
            ZScoreNormalize(std=-1.0)
    
    def test_init_with_custom_name_and_p(self):
        """Test initialization with custom name and probability."""
        transform = ZScoreNormalize(name="CustomZScore", p=0.7)
        assert transform._name == "CustomZScore"
        assert transform.p == 0.7
    
    def test_repr(self):
        """Test string representation of the transform."""
        transform = ZScoreNormalize(name="TestZScore", mean=0.5, std=0.2, p=0.9)
        expected = "ZScoreNormalize(name=TestZScore, mean=0.5, std=0.2, p=0.9)"
        assert repr(transform) == expected
    
    def test_apply_with_preset_mean_and_std(self):
        """Test apply method with preset mean and std."""
        transform = ZScoreNormalize(mean=0.5, std=0.2)
        img = np.array([[[100, 200], [50, 255]]], dtype=np.float32)
        result = transform.apply(img)
        
        expected = (img - 0.5) / 0.2
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_apply_with_computed_mean_and_std(self):
        """Test apply method with computed mean and std from image."""
        transform = ZScoreNormalize()
        img = np.random.rand(3, 10, 10).astype(np.float32)
        result = transform.apply(img)
        
        # Check that result has approximately zero mean and unit std per channel
        for i in range(img.shape[0]):
            channel_mean = result[i].mean()
            channel_std = result[i].std()
            assert abs(channel_mean) < 1e-6
            assert abs(channel_std - 1.0) < 1e-6
    
    def test_apply_with_zero_std_image(self):
        """Test apply method with image having zero std raises ValueError."""
        transform = ZScoreNormalize()
        img = np.ones((1, 5, 5), dtype=np.float32)  # Constant image
        
        with pytest.raises(ValueError, match="Standard deviation is zero, cannot normalize"):
            transform.apply(img)
    
    def test_apply_with_invalid_input(self):
        """Test apply method with invalid input type raises TypeError."""
        transform = ZScoreNormalize()
        with pytest.raises(TypeError, match="Expected input image to be a NumPy array"):
            transform.apply([[100, 200], [50, 255]])
    
    def test_to_config(self):
        """Test to_config method returns correct configuration."""
        transform = ZScoreNormalize(name="TestZScore", mean=0.5, std=0.2, p=0.9)
        config = transform.to_config()
        
        expected = {
            "class": "ZScoreNormalize",
            "name": "TestZScore",
            "params": {
                "mean": 0.5,
                "std": 0.2,
                "p": 0.9
            }
        }
        assert config == expected
    
    def test_to_config_with_none_values(self):
        """Test to_config method with None mean and std."""
        transform = ZScoreNormalize()
        config = transform.to_config()
        
        expected = {
            "class": "ZScoreNormalize",
            "name": "ZScoreNormalize",
            "params": {
                "mean": None,
                "std": None,
                "p": 1.0
            }
        }
        assert config == expected