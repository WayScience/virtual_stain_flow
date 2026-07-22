"""
Tests for tile crop generator functions.
"""

import pytest

from virtual_stain_flow.datasets.base_dataset import BaseImageDataset
from virtual_stain_flow.datasets.ds_engine.crop_generator import generate_tile_crops
from virtual_stain_flow.datasets.ds_engine.crop_generators.crop_summary import (
    CropSummaryWarning,
)
from virtual_stain_flow.datasets.ds_engine.crop_generators.tile import (
    _compute_centered_tile_positions,
    _compute_centered_tile_crops,
)


class TestComputeCenteredTilePositions:
    """Tests for 1D centered tile position computation."""

    def test_returns_single_centered_position_when_one_tile_fits(self):
        """Should center one tile when only one fits in the extent."""
        positions = _compute_centered_tile_positions(10, 8)
        assert positions == [1]

    def test_returns_evenly_spaced_non_overlapping_positions(self):
        """Should produce step=crop_size positions with centered margin."""
        positions = _compute_centered_tile_positions(10, 4)
        assert positions == [1, 5]

    def test_returns_zero_margin_positions_when_exact_multiple(self):
        """Should start at zero when extent is an exact multiple."""
        positions = _compute_centered_tile_positions(12, 4)
        assert positions == [0, 4, 8]

    def test_raises_error_when_crop_larger_than_extent(self):
        """Should raise ValueError when no tile can fit."""
        with pytest.raises(ValueError, match="exceeds image dimensions"):
            _compute_centered_tile_positions(3, 4)


class TestComputeCenteredTileCrops:
    """Tests for 2D centered tile crop computation."""

    def test_computes_centered_non_overlapping_grid(self):
        """Should produce Cartesian product of centered x/y positions."""
        crops = _compute_centered_tile_crops(10, 10, 4)
        assert crops == [
            ((1, 1), 4, 4),
            ((5, 1), 4, 4),
            ((1, 5), 4, 4),
            ((5, 5), 4, 4),
        ]

    def test_raises_error_when_crop_larger_than_image(self):
        """Should fail when crop_size exceeds width or height."""
        with pytest.raises(ValueError, match="exceeds image dimensions"):
            _compute_centered_tile_crops(3, 10, 4)


class TestGenerateTileCrops:
    """Tests for generate_tile_crops function."""

    def test_generates_tiled_crops_for_all_samples(self, basic_dataset):
        """Should generate tile crops for each sample index."""
        crop_specs = generate_tile_crops(basic_dataset, crop_size=4)

        assert len(crop_specs) == 3
        assert set(crop_specs.keys()) == {0, 1, 2}

    def test_crop_specs_have_centered_non_overlapping_layout(self, basic_dataset):
        """Should return centered non-overlapping 4x4 tiles for 10x10 images."""
        crop_specs = generate_tile_crops(basic_dataset, crop_size=4)

        expected = [
            ((1, 1), 4, 4),
            ((5, 1), 4, 4),
            ((1, 5), 4, 4),
            ((5, 5), 4, 4),
        ]

        for _, crops in crop_specs.items():
            assert crops == expected

    def test_raises_error_for_non_positive_crop_size(self, basic_dataset):
        """Should raise ValueError for crop_size <= 0."""
        with pytest.raises(ValueError, match="crop_size must be positive"):
            generate_tile_crops(basic_dataset, crop_size=0)

        with pytest.raises(ValueError, match="crop_size must be positive"):
            generate_tile_crops(basic_dataset, crop_size=-1)

    def test_raises_error_when_no_active_channels(self, file_index):
        """Should raise ValueError when no channels are configured."""
        dataset = BaseImageDataset(
            file_index=file_index,
            pil_image_mode="I;16",
            input_channel_keys=None,
            target_channel_keys=None,
        )

        with pytest.raises(ValueError, match="No active channels"):
            generate_tile_crops(dataset, crop_size=4)

    def test_raises_error_when_crop_too_large(self, basic_dataset):
        """Should raise ValueError when crop_size exceeds image dimensions."""
        with pytest.raises(ValueError, match="exceeds image dimensions"):
            generate_tile_crops(basic_dataset, crop_size=20)

    def test_verbose_emits_tile_distribution_summary(self, basic_dataset):
        """Should emit one summary warning with tile-count distribution."""
        with pytest.warns(CropSummaryWarning, match="Tile crop generation statistics") as record:
            generate_tile_crops(basic_dataset, crop_size=4, verbose=True)

        message = str(record[0].message)
        assert "Tiles per FOV = 4" in message
