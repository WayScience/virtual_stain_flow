"""
point_centered.py

Helper module for generating crops centered around specified points within an image.
"""

import numpy as np

from .crop_summary import warn_formatted_crop_summary
from .protocol import CropMap
from ...base_dataset import BaseImageDataset
from ..ds_utils import (
    _get_active_channels, 
    _validate_same_dimensions_across_channels
)


def _compute_point_centered_crops(
    image_width: int,
    image_height: int,
    crop_size: int,
    centers: dict[str, np.ndarray],
    track_skipped: bool = False,
) -> tuple[list[tuple[int, int]], int, int]:
    """
    Compute the top-left coordinates of crops centered around specified points,
        ensuring that the crops fit fully within the image boundaries.

    :param image_width: Width of the image.
    :param image_height: Height of the image.
    :param crop_size: Size of the crops to generate.
    :param centers: Dictionary containing 'X' and 'Y' coordinates of the centers.
    :param track_skipped: Whether to count points skipped due to boundary checks.
    :return: Tuple of:
        - List of (x, y) tuples representing the top-left coordinates of the crops.
        - Number of skipped points.
        - Number of accepted points used for crops.
    """
    
    crops = []
    skipped_count = 0
    accepted_count = 0

    for x, y in zip(centers['X'], centers['Y']):

        if x <= crop_size // 2 or x >= image_width - crop_size // 2 or \
           y <= crop_size // 2 or y >= image_height - crop_size // 2:
            if track_skipped:
                skipped_count += 1
            continue  # Skip points too close to the edge for a full crop

        crop_x = int(x - crop_size // 2)
        crop_y = int(y - crop_size // 2)
        crops.append((crop_x, crop_y))
        if track_skipped:
            accepted_count += 1

    return crops, skipped_count, accepted_count


def generate_point_centered_crops(
    dataset: BaseImageDataset,
    crop_size: int | None = None,
    mapping: list[dict[str, np.ndarray]] | None = None,
    verbose: bool = False,
) -> CropMap:
    """
    Generate crop specifications centered around specified points 
        for each image in the dataset.
    
    :param dataset: BaseImageDataset containing the images and metadata.
    :param crop_size: Size of the crops to generate. Made optional 
        for better error handling when called from the CropImageDataset
        .from_base_dataset class method. 
    :param mapping: List of dictionaries containing the centers for each image.
        Each dictionary should have keys 'X' and 'Y' with corresponding numpy 
            arrays of coordinates.
        Made optional for the same reason as crop_size.
    :param verbose: If True, compute and report skipped-point statistics.
    :return: CropMap containing the generated crop specifications.
    """

    if crop_size is None:
        raise ValueError(
            "crop_size must be provided for point-centered crop generation."
        )
    if mapping is None:
        raise ValueError(
            "mapping must be provided for point-centered crop generation."
        )

    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}.")
    
    active_channels = _get_active_channels(dataset)
    if not active_channels:
        raise ValueError(
            "No active channels configured. Set input_channel_keys and/or "
            "target_channel_keys on the dataset before generating crops."
        )
    
    manifest = dataset.file_state.manifest
    crop_specs: dict[int, list[tuple[tuple[int, int], int, int]]] = {}
    skipped_points_per_image: list[int] = []
    accepted_points_per_image: list[int] = []

    for idx in range(len(dataset)):
        
        dims = manifest.get_image_dimensions(idx, channels=active_channels)

        width, height = _validate_same_dimensions_across_channels(
            dims, active_channels, idx
        )

        centers = mapping[idx]

        crop_lists, skipped_count, accepted_count = _compute_point_centered_crops(
            width,
            height,
            crop_size,
            centers,
            track_skipped=verbose,
        )
        crop_specs[idx] = [
            (crop_coords, crop_size, crop_size) for crop_coords in crop_lists
        ]

        if verbose:
            skipped_points_per_image.append(skipped_count)
            accepted_points_per_image.append(accepted_count)

    if verbose:
        total_points_skipped = sum(skipped_points_per_image)
        total_points_accepted = sum(accepted_points_per_image)
        num_images = len(skipped_points_per_image)

        avg_accepted_per_image = (
            total_points_accepted / num_images if num_images > 0 else 0.0
        )
        images_with_accepted = sum(1 for count in accepted_points_per_image if count > 0)
        avg_accepted_per_affected_image = (
            total_points_accepted / images_with_accepted
            if images_with_accepted > 0
            else 0.0
        )
        max_accepted_single_image = max(accepted_points_per_image, default=0)

        if total_points_skipped > 0:
            avg_skipped_per_image = (
                total_points_skipped / num_images if num_images > 0 else 0.0
            )
            images_with_skips = sum(1 for count in skipped_points_per_image if count > 0)
            avg_skipped_per_affected_image = (
                total_points_skipped / images_with_skips
                if images_with_skips > 0
                else 0.0
            )
            max_skipped_single_image = max(skipped_points_per_image, default=0)

            warn_formatted_crop_summary(
                title="Point-centered crop generation statistics:",
                detail_line=(
                    "Exclusion criterion: a point is rejected when its center "
                    "is within or on half a crop-size from any image border, "
                    "which prevents full in-bounds crop extraction."
                ),
                metrics={
                    "Total accepted": total_points_accepted,
                    "Total rejected": total_points_skipped,
                    "Mean accepted / image": avg_accepted_per_image,
                    "Mean rejected / image": avg_skipped_per_image,
                    "Mean accepted / affected image": avg_accepted_per_affected_image,
                    "Mean rejected / affected image": avg_skipped_per_affected_image,
                    "Images with accepted": f"{images_with_accepted}/{num_images}",
                    "Images with rejected": f"{images_with_skips}/{num_images}",
                    "Max accepted in one image": max_accepted_single_image,
                    "Max rejected in one image": max_skipped_single_image,
                },
            )

    return crop_specs
