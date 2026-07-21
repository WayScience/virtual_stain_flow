"""
Tests for point-centered crop generator core functionality.
"""

import warnings

import numpy as np
import pytest

from virtual_stain_flow.datasets.base_dataset import BaseImageDataset
from virtual_stain_flow.datasets.ds_engine.crop_generator import (
    generate_point_centered_crops,
)
from virtual_stain_flow.datasets.ds_engine.crop_generators.crop_summary import (
    CropSummaryWarning,
)
from virtual_stain_flow.datasets.ds_engine.crop_generators.point_centered import (
    _compute_point_centered_crops,
)


class TestComputePointCenteredCrops:
    """Core tests for point-centered crop coordinate computation."""

    def test_computes_crops_and_tracks_counts(self):
        """Should return valid crop coordinates with accepted/rejected counts."""
        centers = {
            "X": np.array([5, 2, 8, 7]),
            "Y": np.array([5, 5, 8, 7]),
        }

        crops, skipped, accepted = _compute_point_centered_crops(
            image_width=10,
            image_height=10,
            crop_size=4,
            centers=centers,
            track_skipped=True,
        )

        assert crops == [(3, 3), (5, 5)]
        assert skipped == 2
        assert accepted == 2


class TestGeneratePointCenteredCrops:
    """Core tests for generate_point_centered_crops."""

    def test_generates_expected_crop_specs(self, basic_dataset):
        """Should generate crop specs in expected format for each sample."""
        mapping = [
            {"X": np.array([5, 7]), "Y": np.array([5, 7])},
            {"X": np.array([6]), "Y": np.array([6])},
            {"X": np.array([5, 2]), "Y": np.array([5, 5])},
        ]

        crop_specs = generate_point_centered_crops(
            dataset=basic_dataset,
            crop_size=4,
            mapping=mapping,
        )

        assert set(crop_specs.keys()) == {0, 1, 2}
        assert crop_specs[0] == [((3, 3), 4, 4), ((5, 5), 4, 4)]
        assert crop_specs[1] == [((4, 4), 4, 4)]
        assert crop_specs[2] == [((3, 3), 4, 4)]

    def test_verbose_emits_summary_warning_when_points_rejected(self, basic_dataset):
        """Should emit one summary warning when any points are rejected."""
        mapping = [
            {"X": np.array([5, 2]), "Y": np.array([5, 5])},
            {"X": np.array([6]), "Y": np.array([6])},
            {"X": np.array([7]), "Y": np.array([7])},
        ]

        with pytest.warns(
            CropSummaryWarning,
            match="Point-centered crop generation statistics",
        ) as record:
            generate_point_centered_crops(
                dataset=basic_dataset,
                crop_size=4,
                mapping=mapping,
                verbose=True,
            )

        message = str(record[0].message)
        assert "Total accepted" in message
        assert "Total rejected" in message

    def test_verbose_no_warning_when_no_points_rejected(self, basic_dataset):
        """Should not emit summary warning when no points are rejected."""
        mapping = [
            {"X": np.array([5, 7]), "Y": np.array([5, 7])},
            {"X": np.array([6]), "Y": np.array([6])},
            {"X": np.array([7]), "Y": np.array([7])},
        ]

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            generate_point_centered_crops(
                dataset=basic_dataset,
                crop_size=4,
                mapping=mapping,
                verbose=True,
            )

        assert len(record) == 0

    @pytest.mark.parametrize(
        "crop_size,mapping,match",
        [
            (None, [{"X": np.array([5]), "Y": np.array([5])}], "crop_size must be provided"),
            (4, None, "mapping must be provided"),
            (1.5, [{"X": np.array([5]), "Y": np.array([5])}], "crop_size must be an integer"),
            (0, [{"X": np.array([5]), "Y": np.array([5])}], "crop_size must be positive"),
        ],
    )
    def test_validates_required_arguments(self, basic_dataset, crop_size, mapping, match):
        """Should validate required inputs and crop_size constraints."""
        with pytest.raises(ValueError, match=match):
            generate_point_centered_crops(
                dataset=basic_dataset,
                crop_size=crop_size,
                mapping=mapping,
            )

    def test_raises_error_when_no_active_channels(self, file_index):
        """Should raise ValueError when no channels are configured."""
        dataset = BaseImageDataset(
            file_index=file_index,
            pil_image_mode="I;16",
            input_channel_keys=None,
            target_channel_keys=None,
        )
        mapping = [
            {"X": np.array([5]), "Y": np.array([5])},
            {"X": np.array([5]), "Y": np.array([5])},
            {"X": np.array([5]), "Y": np.array([5])},
        ]

        with pytest.raises(ValueError, match="No active channels"):
            generate_point_centered_crops(
                dataset=dataset,
                crop_size=4,
                mapping=mapping,
            )
