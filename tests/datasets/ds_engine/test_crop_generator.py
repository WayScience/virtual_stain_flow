"""
Tests for crop_generator.py utility functions.
"""

import pytest

from virtual_stain_flow.datasets.base_dataset import BaseImageDataset
from virtual_stain_flow.datasets.ds_engine.crop_generator import (
    _compute_center_crop,
    generate_center_crops,
)


class TestComputeCenterCrop:
    """Tests for _compute_center_crop function."""

    def test_computes_correct_center_for_even_dimensions(self):
        """Should compute correct top-left for centered crop."""
        x, y = _compute_center_crop(100, 100, 50)
        assert (x, y) == (25, 25)

    def test_computes_correct_center_for_odd_remainder(self):
        """Should floor division for odd remainder."""
        x, y = _compute_center_crop(10, 10, 5)
        assert (x, y) == (2, 2)

    def test_computes_correct_center_for_rectangular_image(self):
        """Should handle non-square images."""
        x, y = _compute_center_crop(100, 50, 20)
        assert (x, y) == (40, 15)

    def test_raises_error_when_crop_exceeds_width(self):
        """Should raise ValueError when crop_size > width."""
        with pytest.raises(ValueError, match="exceeds image dimensions"):
            _compute_center_crop(10, 100, 20)

    def test_raises_error_when_crop_exceeds_height(self):
        """Should raise ValueError when crop_size > height."""
        with pytest.raises(ValueError, match="exceeds image dimensions"):
            _compute_center_crop(100, 10, 20)


class TestGenerateCenterCrops:
    """Tests for generate_center_crops function."""

    def test_generates_center_crops_for_all_samples(self, basic_dataset):
        """Should generate one center crop per sample."""
        crop_specs = generate_center_crops(basic_dataset, crop_size=4)
        
        # Should have 3 samples (from file_index fixture)
        assert len(crop_specs) == 3
        assert set(crop_specs.keys()) == {0, 1, 2}

    def test_crop_specs_have_correct_format(self, basic_dataset):
        """Should return crop specs in ((x, y), width, height) format."""
        crop_specs = generate_center_crops(basic_dataset, crop_size=4)
        
        for idx, crops in crop_specs.items():
            assert len(crops) == 1  # One center crop per sample
            (x, y), w, h = crops[0]
            assert w == 4
            assert h == 4
            # For 10x10 images with crop_size=4: center is at (3, 3)
            assert (x, y) == (3, 3)

    def test_raises_error_for_non_positive_crop_size(self, basic_dataset):
        """Should raise ValueError for crop_size <= 0."""
        with pytest.raises(ValueError, match="crop_size must be positive"):
            generate_center_crops(basic_dataset, crop_size=0)
        
        with pytest.raises(ValueError, match="crop_size must be positive"):
            generate_center_crops(basic_dataset, crop_size=-1)

    def test_raises_error_when_no_active_channels(self, file_index):
        """Should raise ValueError when no channels configured."""
        dataset = BaseImageDataset(
            file_index=file_index,
            pil_image_mode="I;16",
            input_channel_keys=None,
            target_channel_keys=None,
        )
        with pytest.raises(ValueError, match="No active channels"):
            generate_center_crops(dataset, crop_size=4)

    def test_raises_error_when_crop_too_large(self, basic_dataset):
        """Should raise ValueError when crop_size exceeds image dimensions."""
        # Images are 10x10, so crop_size=20 should fail
        with pytest.raises(ValueError, match="exceeds image dimensions"):
            generate_center_crops(basic_dataset, crop_size=20)
