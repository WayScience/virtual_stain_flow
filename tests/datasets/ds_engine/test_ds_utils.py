"""
Tests for ds_utils.py utility functions.
"""

import pytest

from virtual_stain_flow.datasets.ds_engine.ds_utils import (
    _get_active_channels,
    _validate_same_dimensions_across_channels,
)


class TestGetActiveChannels:
    """Tests for _get_active_channels function."""

    def test_returns_union_of_input_and_target_channels(self, basic_dataset):
        """Should return all unique channel keys from input and target."""
        result = _get_active_channels(basic_dataset)
        assert result == ["input_ch1", "input_ch2", "target_ch1"]

    def test_removes_duplicates_preserving_order(self, file_index):
        """Should remove duplicates while preserving order."""
        from virtual_stain_flow.datasets.base_dataset import BaseImageDataset

        # Create dataset with overlapping channel keys
        dataset = BaseImageDataset(
            file_index=file_index,
            pil_image_mode="I;16",
            input_channel_keys=["input_ch1", "input_ch2"],
            target_channel_keys=["input_ch1"],  # Duplicate
        )
        result = _get_active_channels(dataset)
        assert result == ["input_ch1", "input_ch2"]


class TestValidateSameDimensionsAcrossChannels:
    """Tests for _validate_same_dimensions_across_channels function."""

    def test_returns_common_dimension_when_all_match(self):
        """Should return the common dimension when all channels match."""
        dims = ((10, 10), (10, 10), (10, 10))
        channels = ["ch1", "ch2", "ch3"]
        result = _validate_same_dimensions_across_channels(dims, channels, idx=0)
        assert result == (10, 10)

    def test_raises_error_on_dimension_mismatch(self):
        """Should raise ValueError when dimensions don't match."""
        dims = ((10, 10), (20, 20))
        channels = ["ch1", "ch2"]
        with pytest.raises(ValueError, match="Dimension mismatch"):
            _validate_same_dimensions_across_channels(dims, channels, idx=0)

    def test_raises_error_on_empty_dims(self):
        """Should raise ValueError when dims is empty."""
        with pytest.raises(ValueError, match="No dimensions returned"):
            _validate_same_dimensions_across_channels((), [], idx=0)

    def test_raises_error_when_all_files_missing(self):
        """Should raise ValueError when all channel files are missing (None)."""
        dims = (None, None)
        channels = ["ch1", "ch2"]
        with pytest.raises(ValueError, match="All channel files missing"):
            _validate_same_dimensions_across_channels(dims, channels, idx=0)

    def test_ignores_none_values_for_missing_files(self):
        """Should skip None values and validate remaining dimensions."""
        dims = ((10, 10), None, (10, 10))
        channels = ["ch1", "ch2", "ch3"]
        result = _validate_same_dimensions_across_channels(dims, channels, idx=0)
        assert result == (10, 10)
