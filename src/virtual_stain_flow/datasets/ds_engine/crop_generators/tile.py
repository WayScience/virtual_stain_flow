"""
tile.py

Crop generator module for creating non-overlapping tile crops centered within
images in a BaseImageDataset. Provides a best-effort tiling approach that 
maximizes the number of full tiles while centering the grid within the image
boundaries.
"""

from typing import List

from .crop_summary import warn_formatted_crop_summary
from .protocol import CropMap, CropSpec
from ...base_dataset import BaseImageDataset
from ..ds_utils import (
	_get_active_channels,
	_validate_same_dimensions_across_channels
)


def _compute_centered_tile_positions(
	image_extent: int,
	crop_size: int
) -> List[int]:
	"""
	Compute 1D tile start positions for non-overlapping tiles centered
	within an image extent.

	:param image_extent: Size of image extent along one axis (width or height).
	:param crop_size: Size of one tile along the same axis.
	:return: List of start positions for each tile.
	:raises ValueError: If crop_size exceeds image extent.
	"""
	n_tiles = image_extent // crop_size
	if n_tiles == 0:
		raise ValueError(
			f"crop_size ({crop_size}) exceeds image dimensions "
			f"along one axis ({image_extent})."
		)

	covered_extent = n_tiles * crop_size
	margin = image_extent - covered_extent
	start = margin // 2

	return [start + (tile_idx * crop_size) for tile_idx in range(n_tiles)]


def _compute_centered_tile_crops(
	image_width: int,
	image_height: int,
	crop_size: int
) -> List[CropSpec]:
	"""
	Compute non-overlapping square tiles arranged on a centered grid.

	:param image_width: Width of the source image.
	:param image_height: Height of the source image.
	:param crop_size: Size of square tile crop.
	:return: List of crop specs in format ((x, y), width, height).
	:raises ValueError: If crop_size exceeds image width or height.
	"""
	if crop_size > image_width or crop_size > image_height:
		raise ValueError(
			f"crop_size ({crop_size}) exceeds image dimensions "
			f"({image_width}x{image_height})."
		)

	x_positions = _compute_centered_tile_positions(image_width, crop_size)
	y_positions = _compute_centered_tile_positions(image_height, crop_size)

	return [
		((x, y), crop_size, crop_size)
		for y in y_positions
		for x in x_positions
	]


def generate_tile_crops(
	dataset: BaseImageDataset,
	crop_size: int,
	verbose: bool = False,
) -> CropMap:
	"""
	Generate best-effort centered, non-overlapping tiling crops
	for each sample in a BaseImageDataset.

	Tiling is "best-effort" in that it uses as many full, non-overlapping
	tiles of size `crop_size` as possible along each axis, then centers
	the tile grid within the remaining field of view.

	:param dataset: A BaseImageDataset instance (or compatible object with
		`file_state.manifest` attribute supporting `get_image_dimensions()`).
	:param crop_size: Size of the square crop tile (same width and height).
	:param verbose: If True, emit summary statistics for crop generation.
	:return: Dictionary mapping manifest indices to lists of crop specs.
		Format: {manifest_idx: [((x, y), width, height), ...]}
	:raises ValueError: If crop_size is non-positive, if no active channels
		are configured, or if channel dimensions don't match for any sample.
	"""
	if crop_size <= 0:
		raise ValueError(f"crop_size must be positive, got {crop_size}.")

	active_channels = _get_active_channels(dataset)
	if not active_channels:
		raise ValueError(
			"No active channels configured. Set input_channel_keys and/or "
			"target_channel_keys on the dataset before generating crops."
		)

	manifest = dataset.file_state.manifest
	crop_specs: CropMap = {}
	tile_count_distribution: dict[int, int] = {}
	total_samples = len(dataset)

	for idx in range(total_samples):
		dims = manifest.get_image_dimensions(idx, channels=active_channels)

		width, height = _validate_same_dimensions_across_channels(
			dims, active_channels, idx
		)

		crop_specs[idx] = _compute_centered_tile_crops(
			width, height, crop_size
		)
		tile_count = len(crop_specs[idx])
		tile_count_distribution[tile_count] = (
			tile_count_distribution.get(tile_count, 0) + 1
		)

	if verbose:
		distribution_metrics = {
			f"Tiles per FOV = {tile_count}": full_fov_count
			for tile_count, full_fov_count in sorted(tile_count_distribution.items())
		}
		warn_formatted_crop_summary(
			title="Tile crop generation statistics:",
			detail_line=(
				"Tiling criterion: each FOV receives as many full, non-overlapping "
				"tiles as fit along width and height, centered within remaining margins."
			),
			metrics={
				"Total dataset count": total_samples,
				**distribution_metrics,
			},
		)

	return crop_specs
